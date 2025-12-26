"""Tests for code generation with BlueprintState."""

from pathlib import Path

from metagen.specs.loader import load_spec
from metagen.synth.architecture import generate_blueprint
from metagen.synth.codegen import generate_code


def test_blueprint_dimensions_in_generated_code(tmp_path: Path):
    """Verify generated model.py uses blueprint dimensions."""
    # Load real spec
    spec, _ = load_spec("examples/specs/text_llm_8b.yaml")

    # Generate blueprint
    blueprint = generate_blueprint(spec, tmp_path / "bp", seed=42)

    # Generate code
    generate_code(spec, tmp_path / "code", blueprint, seed=42)

    # Read generated model.py
    model_code = (tmp_path / "code" / "model.py").read_text()

    # Verify blueprint dimensions appear in code
    assert f"hidden_size: int = {blueprint.dims['hidden_size']}" in model_code
    assert f"layers: int = {blueprint.dims['layers']}" in model_code

    # Verify dimensions are documented in docstring
    assert f"hidden_size: {blueprint.dims['hidden_size']}" in model_code
    assert f"layers: {blueprint.dims['layers']}" in model_code
    assert f"heads: {blueprint.dims['heads']}" in model_code


def test_no_hardcoded_random_dimensions(tmp_path: Path):
    """Verify no hardcoded random dimension values in generated code."""
    spec, _ = load_spec("examples/specs/text_llm_8b.yaml")
    blueprint = generate_blueprint(spec, tmp_path / "bp", seed=42)
    generate_code(spec, tmp_path / "code", blueprint, seed=42)

    model_code = (tmp_path / "code" / "model.py").read_text()

    # The code should NOT have arbitrary hardcoded values
    # (unless they match the blueprint)
    hidden_size = blueprint.dims["hidden_size"]
    layers = blueprint.dims["layers"]

    # Check that if 1024, 2048 appear, they must match blueprint
    if "hidden_size: int = 1024" in model_code:
        assert hidden_size == 1024, "Found hardcoded 1024 but blueprint has different value"
    if "hidden_size: int = 2048" in model_code:
        assert hidden_size == 2048, "Found hardcoded 2048 but blueprint has different value"
    if "layers: int = 12" in model_code:
        assert layers == 12, "Found hardcoded 12 but blueprint has different value"


def test_code_architecture_yaml_consistency(tmp_path: Path):
    """Integration test: architecture.yaml and model.py have consistent dimensions."""
    import yaml

    spec, _ = load_spec("examples/specs/text_llm_8b.yaml")
    blueprint = generate_blueprint(spec, tmp_path / "bp", seed=42)
    generate_code(spec, tmp_path / "code", blueprint, seed=42)

    # Load architecture.yaml
    arch_yaml = yaml.safe_load((tmp_path / "bp" / "architecture.yaml").read_text())

    # Parse model.py for dimensions
    model_code = (tmp_path / "code" / "model.py").read_text()

    # Extract hidden_size from model.py
    import re

    match = re.search(r"hidden_size: int = (\d+)", model_code)
    assert match, "Could not find hidden_size in model.py"
    code_hidden_size = int(match.group(1))

    match_layers = re.search(r"layers: int = (\d+)", model_code)
    assert match_layers, "Could not find layers in model.py"
    code_layers = int(match_layers.group(1))

    # Verify consistency
    assert arch_yaml["hidden_size"] == code_hidden_size, (
        f"Mismatch: architecture.yaml has {arch_yaml['hidden_size']}, "
        f"but model.py has {code_hidden_size}"
    )
    assert arch_yaml["layers"] == code_layers, (
        f"Mismatch: architecture.yaml has {arch_yaml['layers']}, but model.py has {code_layers}"
    )

    # Also verify they match blueprint
    assert blueprint.dims["hidden_size"] == code_hidden_size
    assert blueprint.dims["layers"] == code_layers
