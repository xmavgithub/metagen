from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from metagen.specs.schema import ModelSpec
from metagen.synth.modalities import get_handler
from metagen.utils.io import ensure_dir, write_json, write_yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlueprintState:
    """
    Complete blueprint state for model architecture.
    This is the single source of truth for all dimensions and parameters.
    Passed from architecture synthesis to all downstream generators.
    """

    # Core dimensions (always present)
    dims: dict[str, int]  # {hidden_size, layers, heads}

    # Modality-specific parameters
    vocab_size: int | None = None  # Text: vocabulary size
    max_seq_len: int | None = None  # Text: max sequence length
    num_channels: int | None = None  # Image/Video: channels (3=RGB, 1=grayscale)
    image_size: int | None = None  # Image: spatial size (224, 512, etc.)
    sample_rate: int | None = None  # Audio: Hz (16000, 44100, etc.)
    latent_dim: int | None = None  # Diffusion: latent space dimension
    patch_size: int | None = None  # ViT: patch size (16, 32, etc.)
    num_patches: int | None = None  # ViT: total patch count

    # Task-specific parameters (Phase 1: Extended Model Types)
    # Classification/Regression
    num_classes: int | None = None  # Classification: number of output classes
    num_outputs: int | None = None  # Regression: number of output values

    # Time Series
    horizon: int | None = None  # Time series: prediction horizon
    lookback: int | None = None  # Time series: historical window size

    # Reinforcement Learning
    num_actions: int | None = None  # RL: number of discrete actions
    action_dim: int | None = None  # RL: continuous action space dimension
    action_space: str | None = None  # RL: "discrete" or "continuous"

    # Detection/Segmentation
    num_anchors: int | None = None  # Detection: number of anchor boxes
    mask_resolution: int | None = None  # Segmentation: output mask resolution

    # Graph Neural Networks
    node_features: int | None = None  # GNN: node feature dimension
    edge_features: int | None = None  # GNN: edge feature dimension

    # Embedding
    embedding_dim: int | None = None  # Embedding: output vector dimension

    # Architecture metadata
    family: str = "transformer"  # Architecture family from spec
    components: tuple = field(default_factory=tuple)  # Component list (immutable)

    # Parameter estimates
    total_params: int = 0  # Total parameter count
    trainable_params: int = 0  # Trainable params (usually same)
    activation_memory_gb: float = 0.0  # Forward pass memory
    kv_cache_gb: float = 0.0  # Attention cache (transformers only)

    # Seed and determinism
    seed: int = 42  # Random seed used for generation

    # Task type (for downstream code generation)
    task_type: str = "generation"  # Task type from spec


def _choose_dims(spec: ModelSpec) -> dict[str, int]:
    """Derive model dimensions from spec constraints."""
    family = spec.architecture.family.lower()
    latency = spec.constraints.latency
    device = spec.constraints.device

    # Parse parameter budget to get target params
    budget_str = str(spec.constraints.parameter_budget.max).upper().strip()
    target_params = None
    if "B" in budget_str:
        target_params = float(budget_str.replace("B", "")) * 1e9
    elif "M" in budget_str:
        target_params = float(budget_str.replace("M", "")) * 1e6
    elif "K" in budget_str:
        target_params = float(budget_str.replace("K", "")) * 1e3
    elif budget_str.replace(".", "").isdigit():
        target_params = float(budget_str)

    # Default dimensions
    base_hidden = 4096 if family == "transformer" else 2048
    layers = 32 if family == "transformer" else 24

    # Scale dims based on parameter budget if specified
    if target_params is not None:
        # Formula: params ≈ 12 * hidden^2 * layers (for transformers)
        # Solve for hidden given layers: hidden = sqrt(params / (12 * layers))
        import math

        # Start with fewer layers for small models
        if target_params < 100e6:  # < 100M
            layers = 4
        elif target_params < 500e6:  # < 500M
            layers = 8
        elif target_params < 1e9:  # < 1B
            layers = 16
        else:
            layers = 32

        factor = 12 if family == "transformer" else 8
        base_hidden = int(math.sqrt(target_params / (factor * layers)))
        # Ensure hidden is divisible by 64 for efficiency
        base_hidden = max(64, (base_hidden // 64) * 64)

    # Additional adjustments based on latency/device
    if latency == "real-time" or device in {"edge", "cpu"}:
        base_hidden = int(base_hidden * 0.5)
    elif device == "datacenter_gpu":
        base_hidden = int(base_hidden * 1.5)

    # Domain adjustments
    if spec.task.domain == "video":
        layers += 8
    if spec.task.domain == "audio":
        layers -= 4
    if "tiny" in spec.name:
        layers = max(2, int(layers * 0.5))

    heads = max(2, base_hidden // 64)
    return {"hidden_size": base_hidden, "layers": layers, "heads": heads}


def _estimate_params(hidden_size: int, layers: int, family: str) -> float:
    """Approximate parameter count (billions)."""
    if family == "transformer":
        params = 12 * (hidden_size**2) * layers
    elif family == "diffusion":
        params = 8 * (hidden_size**2) * layers
    else:
        params = 10 * (hidden_size**2) * layers
    return round(params / 1e9, 1)


def _parse_context_window(context_window: str | int | None) -> int:
    """Parse context window string to a max sequence length."""
    ctx = str(context_window).lower().strip().replace(" ", "")
    if any(x in ctx for x in ["inf", "∞", "infinite"]):
        return 1048576  # Approx 1M for infinite
    if "m" in ctx:
        return int(float(ctx.replace("m", "")) * 1024 * 1024)
    if "k" in ctx:
        return int(float(ctx.replace("k", "")) * 1024)
    if ctx.isdigit():
        return int(ctx)
    return 2048  # Default fallback


def _legacy_augment_blueprint(
    spec: ModelSpec,
    blueprint: BlueprintState,
    *,
    only_missing: bool,
) -> BlueprintState:
    """Apply legacy modality defaults for backward compatibility."""
    inputs = [m.lower() for m in spec.modality.inputs]
    outputs = [m.lower() for m in spec.modality.outputs]
    updates: dict[str, int] = {}

    if "text" in inputs or "text" in outputs:
        if not only_missing or blueprint.vocab_size is None:
            updates["vocab_size"] = 50257  # GPT-2 tokenizer default
        if not only_missing or blueprint.max_seq_len is None:
            updates["max_seq_len"] = _parse_context_window(spec.constraints.context_window)

    if "image" in inputs or "image" in outputs:
        if not only_missing or blueprint.num_channels is None:
            updates["num_channels"] = 3  # RGB default
        if not only_missing or blueprint.image_size is None:
            updates["image_size"] = 224  # Standard image size

    if not updates:
        return blueprint
    return replace(blueprint, **updates)


def _augment_blueprint_with_handler(
    spec: ModelSpec,
    blueprint: BlueprintState,
    seed: int,
) -> BlueprintState:
    """Augment blueprint using modality handlers with legacy fallback."""
    try:
        handler = get_handler(spec)
    except ValueError as exc:
        logger.warning("No modality handler available: %s. Using legacy defaults.", exc)
        return _legacy_augment_blueprint(spec, blueprint, only_missing=False)

    try:
        augmented = handler.augment_blueprint(spec, blueprint, seed)
    except ValueError as exc:
        logger.warning("Modality handler rejected spec: %s. Using legacy defaults.", exc)
        return _legacy_augment_blueprint(spec, blueprint, only_missing=False)

    return _legacy_augment_blueprint(spec, augmented, only_missing=True)


def _build_blueprint_state(spec: ModelSpec, dims: dict[str, int], seed: int) -> BlueprintState:
    """Build complete BlueprintState from spec and computed dimensions."""
    rnd = random.Random(seed)

    # Estimate parameters
    params_b = _estimate_params(
        dims["hidden_size"], dims["layers"], spec.architecture.family.lower()
    )
    memory_gb = round(params_b * 0.5 + rnd.random(), 2)
    kv_cache = round(dims["layers"] * dims["heads"] * 0.01, 2)

    # Build BlueprintState
    blueprint = BlueprintState(
        dims=dims,
        vocab_size=None,
        max_seq_len=None,
        num_channels=None,
        image_size=None,
        family=spec.architecture.family.lower(),
        components=tuple(c.model_dump() for c in spec.architecture.components),
        total_params=int(params_b * 1e9),
        trainable_params=int(params_b * 1e9),
        activation_memory_gb=memory_gb,
        kv_cache_gb=kv_cache,
        seed=seed,
        task_type=spec.task.type.lower(),
    )

    return _augment_blueprint_with_handler(spec, blueprint, seed)


def build_blueprint_from_dims(spec: ModelSpec, dims: dict[str, int], seed: int) -> BlueprintState:
    """
    Build a BlueprintState from provided dimensions without writing files.

    Args:
        spec: Model specification.
        dims: Precomputed architecture dimensions.
        seed: Random seed for deterministic augmentation.

    Returns:
        BlueprintState with modality-specific parameters filled in.
    """
    return _build_blueprint_state(spec, dims, seed)


def estimate_summary(spec: ModelSpec, seed: int) -> tuple[dict[str, int], dict[str, float]]:
    rnd = random.Random(seed)
    dims = _choose_dims(spec)
    params_b = _estimate_params(
        dims["hidden_size"],
        dims["layers"],
        spec.architecture.family.lower(),
    )
    memory_gb = round(params_b * 0.5 + rnd.random(), 2)
    kv_cache = round(dims["layers"] * dims["heads"] * 0.01, 2)
    summary = {
        "params_billion": params_b,
        "activation_memory_gb": memory_gb,
        "kv_cache_gb": kv_cache,
    }
    return dims, summary


def generate_blueprint(spec: ModelSpec, out_dir: Path, seed: int) -> BlueprintState:
    """
    Generate architecture blueprint and return complete state.

    Returns:
        BlueprintState: Complete blueprint with all dimensions and metadata
    """
    ensure_dir(out_dir)
    dims = _choose_dims(spec)
    blueprint = _build_blueprint_state(spec, dims, seed)
    rnd = random.Random(seed)

    # Write architecture.yaml
    architecture_yaml = {
        "name": spec.name,
        "family": blueprint.family,
        "hidden_size": blueprint.dims["hidden_size"],
        "layers": blueprint.dims["layers"],
        "heads": blueprint.dims["heads"],
        "components": [dict(c) for c in blueprint.components],
    }
    write_yaml(out_dir / "architecture.yaml", architecture_yaml)

    # Write graph.json
    graph = {
        "nodes": [{"id": c["name"], "type": c["type"]} for c in blueprint.components],
        "edges": [
            {"source": "SpecEncoder", "target": "ModelLatent"},
            {"source": "ModelLatent", "target": "ArchitectureSynth"},
            {"source": "ArchitectureSynth", "target": "LossComposer"},
            {"source": "LossComposer", "target": "PaperHead"},
        ],
    }
    with open(out_dir / "graph.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    # Write params_estimate.json
    params_estimate = {
        "total_params": blueprint.total_params,
        "trainable_params": blueprint.trainable_params,
        "activation_memory_gb": blueprint.activation_memory_gb,
        "kv_cache_gb": blueprint.kv_cache_gb,
        "notes": ["Estimated under optimistic assumptions"],
    }
    write_json(out_dir / "params_estimate.json", params_estimate)

    # Write ablations.yaml
    ablations = {
        "remove_loss_composer": {
            "delta_spec_fidelity": -rnd.uniform(0.5, 2.0),
            "delta_latency_ms": rnd.uniform(2, 10),
        },
        "drop_architecture_synth": {
            "delta_spec_fidelity": -rnd.uniform(5, 15),
            "delta_latency_ms": rnd.uniform(-5, 5),
        },
    }
    with open(out_dir / "ablations.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(ablations, f, sort_keys=False)

    return blueprint
