from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from metagen.specs.loader import load_spec
from metagen.synth.architecture import BlueprintState
from metagen.synth.codegen import generate_code


def test_train_raises_value_error_missing_targets(tmp_path: Path) -> None:
    """Verify that generated train.py raises ValueError if targets are missing."""

    # 1. Generate code for a vocab-based model
    spec, _ = load_spec("examples/specs/text/text_llm_8b.yaml")

    # Tiny blueprint
    blueprint = BlueprintState(
        dims={"hidden_size": 32, "layers": 1, "heads": 2},
        vocab_size=100,  # Triggers the vocab branch
        max_seq_len=16,
        family="transformer",
        components=(),
        seed=42,
    )

    code_dir = tmp_path / "code"
    generate_code(spec, code_dir, blueprint, seed=42)

    # 2. Create a test script that imports the generated train function
    # and calls it with a dummy data loader that yields (x, None) and ensures AR logic fails
    # effectively by passing 1D x which won't trigger the AR shifter

    # OVERWRITE model.py with a tolerant mock that accepts 1D inputs
    # so we can bypass AR shifting but still run forward()
    model_file = code_dir / "model.py"
    model_file.write_text("""
import torch.nn as nn
import torch

class MetaGenModel(nn.Module):
    def __init__(self, hidden_size=32, layers=1):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 100) # vocab size
    def forward(self, x):
        # Return dummy output of shape [Batch, Seq, Vocab] or [Batch, Vocab]
        # logic in train.py expects [B, S, V] effectively
        return torch.randn(x.size(0), 16, 100) 
""")

    runner_script = code_dir / "run_test.py"
    runner_script.write_text("""
import torch
import sys
from train import train
from model import MetaGenModel
from unittest.mock import MagicMock

# Mock data loader yielding only x (1D tensor), y=None
# 1D tensor [batch] won't trigger the 2D check for AR shifting [batch, seq]
def bad_loader():
    yield torch.zeros(4, dtype=torch.long) # 1D tensor, so AR shift will skip it

def main():
    model = MetaGenModel(hidden_size=32, layers=1)
    try:
        train(model, bad_loader(), epochs=1, device='cpu')
    except ValueError as e:
        if "requires targets" in str(e):
            print("SUCCESS: Caught expected ValueError")
            sys.exit(0)
        else:
            print(f"FAILURE: Caught unexpected ValueError: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"FAILURE: Caught unexpected exception: {type(e)} {e}")
        sys.exit(1)
        
    print("FAILURE: Did not raise ValueError")
    sys.exit(1)

if __name__ == "__main__":
    main()
""")

    # 3. Run it
    res = subprocess.run(
        [sys.executable, str(runner_script)], cwd=code_dir, capture_output=True, text=True
    )

    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)

    assert res.returncode == 0, "Test runner failed"
    assert "SUCCESS" in res.stdout


def test_train_raises_value_error_missing_targets_no_vocab(tmp_path: Path) -> None:
    """Verify that generated train.py raises ValueError if targets are missing (NO vocab)."""

    # 1. Generate code for a NO-vocab model (e.g. image)
    spec, _ = load_spec("examples/specs/text/text_llm_8b.yaml")  # Reuse spec but force blueprint

    # Tiny blueprint NO VOCAB
    blueprint = BlueprintState(
        dims={"hidden_size": 32, "layers": 1, "heads": 2},
        vocab_size=None,  # Triggers the no-vocab branch
        max_seq_len=16,
        family="transformer",
        components=(),
        seed=42,
    )

    code_dir = tmp_path / "code_novocab"
    generate_code(spec, code_dir, blueprint, seed=42)

    # 2. Runner script
    runner_script = code_dir / "run_test.py"
    runner_script.write_text("""
import torch
import sys
from train import train
from model import MetaGenModel

# Mock data loader yielding (x, None) explicitly
def bad_loader():
    x = torch.randn(4, 32)
    yield x  # y is None implicitly in train fallback logic handling tuple
    
    # Wait, train.py logic for tensor yield is:
    # elif isinstance(batch, torch.Tensor):
    #     x = batch.to(device)
    #     y = None
    
    # So yielding a tensor is enough.

def main():
    model = MetaGenModel(hidden_size=32, layers=1)
    try:
        train(model, bad_loader(), epochs=1, device='cpu')
    except ValueError as e:
        if "requires targets" in str(e):
            print("SUCCESS: Caught expected ValueError")
            sys.exit(0)
        else:
            print(f"FAILURE: Caught unexpected ValueError: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"FAILURE: Caught unexpected exception: {type(e)} {e}")
        sys.exit(1)

    print("FAILURE: Did not raise ValueError")
    sys.exit(1)

if __name__ == "__main__":
    main()
""")

    # 3. Run it
    res = subprocess.run(
        [sys.executable, str(runner_script)], cwd=code_dir, capture_output=True, text=True
    )

    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)

    assert res.returncode == 0, "Test runner failed"
    assert "SUCCESS" in res.stdout
