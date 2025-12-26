from pathlib import Path

from metagen.specs.schema import ModelSpec
from metagen.synth.architecture import BlueprintState, estimate_summary, generate_blueprint


def test_generate_blueprint_outputs(tmp_path: Path):
    """Test blueprint generation creates all expected files."""
    spec = ModelSpec(name="arch_test")
    blueprint = generate_blueprint(spec, tmp_path, seed=123)

    # Verify return type
    assert isinstance(blueprint, BlueprintState)

    # Verify files created
    assert (tmp_path / "architecture.yaml").exists()
    assert (tmp_path / "graph.json").exists()
    assert (tmp_path / "params_estimate.json").exists()
    assert (tmp_path / "ablations.yaml").exists()

    # Verify blueprint has params
    assert blueprint.total_params > 0


def test_blueprint_state_has_required_fields(tmp_path: Path):
    """Test BlueprintState contains all required fields."""
    spec = ModelSpec(name="test_blueprint", modality={"inputs": ["text"], "outputs": ["text"]})
    blueprint = generate_blueprint(spec, tmp_path, seed=42)

    # Core dimensions
    assert "hidden_size" in blueprint.dims
    assert "layers" in blueprint.dims
    assert "heads" in blueprint.dims

    # Text-specific params (since modality is text)
    assert blueprint.vocab_size == 50257  # GPT-2 tokenizer
    assert blueprint.max_seq_len > 0

    # Parameter estimates
    assert blueprint.total_params > 0
    assert blueprint.trainable_params > 0
    assert blueprint.activation_memory_gb > 0

    # Metadata
    assert blueprint.family in ["transformer", "diffusion", "hybrid", "other"]
    assert blueprint.seed == 42


def test_blueprint_deterministic(tmp_path: Path):
    """Test same spec + seed produces identical blueprint."""
    spec = ModelSpec(name="deterministic_test")

    bp1 = generate_blueprint(spec, tmp_path / "run1", seed=42)
    bp2 = generate_blueprint(spec, tmp_path / "run2", seed=42)

    # Verify dimensions are identical
    assert bp1.dims == bp2.dims
    assert bp1.vocab_size == bp2.vocab_size
    assert bp1.total_params == bp2.total_params
    assert bp1.family == bp2.family


def test_estimate_summary_returns_dims(tmp_path: Path):
    spec = ModelSpec(name="arch_dims")
    dims, summary = estimate_summary(spec, seed=123)
    assert dims["hidden_size"] > 0
    assert summary["params_billion"] > 0


def test_blueprint_image_fields_from_handler(tmp_path: Path):
    spec = ModelSpec(
        name="image_blueprint",
        modality={"inputs": ["image"], "outputs": ["image"]},
        task={"type": "classification", "domain": "image"},
        architecture={"family": "transformer"},
    )
    blueprint = generate_blueprint(spec, tmp_path, seed=42)

    assert blueprint.image_size is not None
    assert blueprint.num_channels == 3
    assert blueprint.patch_size is not None
    assert blueprint.num_patches == (blueprint.image_size // blueprint.patch_size) ** 2


def test_blueprint_fallback_for_unregistered_modality(tmp_path: Path):
    spec = ModelSpec(
        name="unregistered_fallback",
        modality={"inputs": ["3d"], "outputs": ["3d"]},
    )
    blueprint = generate_blueprint(spec, tmp_path, seed=42)

    assert blueprint.dims["hidden_size"] > 0
    assert blueprint.vocab_size is None
