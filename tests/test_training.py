import subprocess
import sys
from pathlib import Path

from metagen.specs.loader import load_spec
from metagen.synth.codegen import generate_code


def test_generated_training_loop_runs(tmp_path: Path):
    """Verify generated train.py runs successfully."""
    # Generate code
    spec, _ = load_spec("examples/specs/text/text_llm_8b.yaml")
    # Use smaller model for fast test
    spec.architecture.family = "transformer"
    # Manually construct a tiny blueprint for speed (CPU friendly)
    from metagen.synth.architecture import BlueprintState

    tiny_dims = {"hidden_size": 64, "layers": 2, "heads": 2}
    blueprint = BlueprintState(
        dims=tiny_dims,
        vocab_size=1000,  # Small vocab
        max_seq_len=64,
        family="transformer",
        components=(),
        seed=42,
    )

    generate_code(spec, tmp_path / "code", blueprint, seed=42)

    # Execute train.py as a script
    # This verifies imports, main block, and actual training loop mechanics
    result = subprocess.run(
        [sys.executable, str(tmp_path / "code" / "train.py")],
        capture_output=True,
        text=True,
        cwd=tmp_path / "code",
    )

    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)

    assert result.returncode == 0
    assert "Training completed" in result.stdout
    assert "Epoch 1 finished" in result.stdout
