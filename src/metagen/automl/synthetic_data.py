from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if TYPE_CHECKING:
    from metagen.specs.schema import ModelSpec
    from metagen.synth.architecture import BlueprintState


@dataclass(frozen=True)
class ValidationTask:
    """Descriptor for modality-specific validation tasks."""

    name: str
    modality: str
    description: str


def get_validation_task(spec: ModelSpec) -> ValidationTask:
    """Return a default validation task based on spec modality."""
    modalities = {m.lower() for m in spec.modality.inputs + spec.modality.outputs}
    if "text" in modalities:
        return ValidationTask(
            name="next_token_prediction",
            modality="text",
            description="Random token sequences for next-token prediction.",
        )
    if "image" in modalities:
        return ValidationTask(
            name="image_denoising",
            modality="image",
            description="Gaussian noise tensors for denoising.",
        )
    return ValidationTask(
        name="generic_feature_prediction",
        modality="generic",
        description="Synthetic feature tensors for regression.",
    )


def generate_text_tokens(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    rng: random.Random | None = None,
):
    """Generate random token sequences."""
    if torch is not None:
        return torch.randint(0, vocab_size, (batch_size, seq_len))
    rng = rng or random.Random()
    return [[rng.randrange(vocab_size) for _ in range(seq_len)] for _ in range(batch_size)]


def generate_image_noise(
    batch_size: int,
    num_channels: int,
    image_size: int,
    rng: random.Random | None = None,
):
    """Generate gaussian noise tensors for images."""
    if torch is not None:
        return torch.randn(batch_size, num_channels, image_size, image_size)
    rng = rng or random.Random()
    return [
        [
            [[rng.gauss(0.0, 1.0) for _ in range(image_size)] for _ in range(image_size)]
            for _ in range(num_channels)
        ]
        for _ in range(batch_size)
    ]


def generate_generic_features(batch_size: int, feature_dim: int, rng: random.Random | None = None):
    """Generate generic feature tensors."""
    if torch is not None:
        return torch.randn(batch_size, feature_dim)
    rng = rng or random.Random()
    return [[rng.gauss(0.0, 1.0) for _ in range(feature_dim)] for _ in range(batch_size)]


def generate_synthetic_batch(
    spec: ModelSpec,
    blueprint: BlueprintState,
    batch_size: int,
    rng: random.Random | None = None,
):
    """Generate a modality-appropriate synthetic batch and return task metadata."""
    task = get_validation_task(spec)
    if task.modality == "text":
        seq_len = min(128, blueprint.max_seq_len or 128)
        vocab_size = blueprint.vocab_size or 50257
        return task, generate_text_tokens(batch_size, seq_len, vocab_size, rng)
    if task.modality == "image":
        image_size = min(128, blueprint.image_size or 224)
        num_channels = blueprint.num_channels or 3
        return task, generate_image_noise(batch_size, num_channels, image_size, rng)
    feature_dim = blueprint.dims.get("hidden_size", 256)
    return task, generate_generic_features(batch_size, feature_dim, rng)
