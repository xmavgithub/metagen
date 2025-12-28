from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from metagen.specs.loader import load_spec
from metagen.synth.architecture import build_blueprint_from_dims
from metagen.synth.codegen import generate_code


def test_generated_training_with_sample_data(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    spec, _ = load_spec("examples/specs/text/text_classifier_bert.yaml")
    dims = {"hidden_size": 64, "layers": 2, "heads": 2}
    blueprint = build_blueprint_from_dims(spec, dims, seed=42)

    code_dir = tmp_path / "code"
    generate_code(spec, code_dir, blueprint, seed=42)

    result = subprocess.run(
        [
            sys.executable,
            str(code_dir / "train.py"),
            "--sample-data",
            "auto",
            "--sample-size",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "2",
        ],
        capture_output=True,
        text=True,
        cwd=code_dir,
    )

    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)

    assert result.returncode == 0
    assert "Training completed" in result.stdout
