import sys
from pathlib import Path

from metagen.specs.loader import load_spec
from metagen.synth.architecture import generate_blueprint
from metagen.synth.codegen import generate_code


def test_data_generation_validity(tmp_path: Path):
    """Verify generated data.py is valid python and classes exist."""
    spec, _ = load_spec("examples/specs/text/text_llm_8b.yaml")
    blueprint = generate_blueprint(spec, tmp_path / "bp", seed=42)
    generate_code(spec, tmp_path / "code", blueprint, seed=42)

    # Add generated code to sys.path to import it
    sys.path.insert(0, str(tmp_path / "code"))

    try:
        import data

        assert hasattr(data, "SyntheticTextDataset")
        assert hasattr(data, "FileTextDataset")
        assert hasattr(data, "load_data")

        # Test instantiation if possible (requires torch)
        try:
            import torch  # noqa: F401

            loader = data.load_data(batch_size=2)
            # Just check it returns something iterable
            assert iter(loader)
        except ImportError:
            pass

    finally:
        sys.path.pop(0)
