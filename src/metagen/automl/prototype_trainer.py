from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

from metagen.specs.schema import ModelSpec
from metagen.synth.architecture import BlueprintState
from metagen.synth.codegen import generate_code


@dataclass(frozen=True)
class TrainingMetrics:
    """Metrics produced by prototype training."""

    final_loss: float
    steps: int
    early_stopped: bool
    runtime_sec: float
    device: str

    def to_dict(self) -> dict[str, float | int | str | bool]:
        """Return a JSON-serializable representation."""
        return {
            "final_loss": self.final_loss,
            "steps": self.steps,
            "early_stopped": self.early_stopped,
            "runtime_sec": self.runtime_sec,
            "device": self.device,
        }


class PrototypeTrainer:
    """Trains small functional models to validate architectures."""

    def __init__(
        self,
        max_seq_len_cap: int = 256,
        max_hidden_size: int = 512,
        max_layers: int = 4,
        max_heads: int = 8,
        timeout_seconds: int = 300,
    ) -> None:
        """
        Initialize the prototype trainer.

        Args:
            max_seq_len_cap: Upper bound for sequence length in prototypes.
            max_hidden_size: Upper bound for hidden size in prototypes.
            max_layers: Upper bound for layers in prototypes.
            max_heads: Upper bound for attention heads in prototypes.
            timeout_seconds: Maximum time in seconds for training subprocess.
                Defaults to 300 (5 minutes). Set to 0 or None to disable.
        """
        self.max_seq_len_cap = max_seq_len_cap
        self.max_hidden_size = max_hidden_size
        self.max_layers = max_layers
        self.max_heads = max_heads
        self.timeout_seconds = timeout_seconds if timeout_seconds else None

    def train_prototype(
        self,
        blueprint: BlueprintState,
        spec: ModelSpec,
        budget_steps: int = 1000,
        *,
        batch_size: int = 4,
        lr: float = 1e-4,
        seed: int | None = None,
    ) -> TrainingMetrics:
        """
        Train a small prototype model and return metrics.

        Args:
            blueprint: Blueprint for the candidate architecture.
            spec: Model specification.
            budget_steps: Maximum number of training steps.
            batch_size: Prototype batch size.
            lr: Learning rate for the prototype run.
            seed: Optional seed for deterministic codegen.

        Returns:
            TrainingMetrics with loss and runtime.

        Example:
            >>> trainer = PrototypeTrainer(max_seq_len_cap=128)
            >>> metrics = trainer.train_prototype(blueprint, spec, budget_steps=10)
            >>> metrics.steps <= 10
            True
        """
        if budget_steps < 1:
            raise ValueError("budget_steps must be >= 1")

        blueprint = self._clamp_blueprint(blueprint)
        resolved_seed = seed if seed is not None else blueprint.seed

        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = Path(tmpdir) / "code"
            generate_code(spec, code_dir, blueprint, resolved_seed)
            metrics_path = code_dir / "prototype_metrics.json"

            cmd = [
                sys.executable,
                str(code_dir / "train.py"),
                "--prototype-mode",
                "--output-metrics",
                str(metrics_path),
                "--budget-steps",
                str(budget_steps),
                "--batch-size",
                str(batch_size),
                "--lr",
                str(lr),
            ]

            start_time = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=code_dir,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as e:
                runtime = time.time() - start_time
                raise RuntimeError(
                    f"Prototype training timed out after {self.timeout_seconds}s\n"
                    f"stdout:\n{e.stdout or ''}\n"
                    f"stderr:\n{e.stderr or ''}"
                ) from e
            runtime = time.time() - start_time

            if result.returncode != 0:
                raise RuntimeError(
                    "Prototype training failed:\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )

            if not metrics_path.exists():
                raise RuntimeError("Prototype metrics not produced by train.py")

            metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
            return TrainingMetrics(
                final_loss=float(metrics_data["final_loss"]),
                steps=int(metrics_data["steps"]),
                early_stopped=bool(metrics_data["early_stopped"]),
                runtime_sec=float(metrics_data.get("runtime_sec", runtime)),
                device=str(metrics_data.get("device", "unknown")),
            )

    def _clamp_blueprint(self, blueprint: BlueprintState) -> BlueprintState:
        """Clamp blueprint fields to keep prototypes lightweight."""
        dims = dict(blueprint.dims)
        dims["hidden_size"] = min(dims["hidden_size"], self.max_hidden_size)
        dims["layers"] = min(dims["layers"], self.max_layers)
        dims["heads"] = min(dims["heads"], self.max_heads)

        while dims["heads"] > 1 and dims["hidden_size"] % dims["heads"] != 0:
            dims["heads"] -= 1

        max_seq_len = blueprint.max_seq_len
        if max_seq_len and max_seq_len > self.max_seq_len_cap:
            max_seq_len = self.max_seq_len_cap

        return replace(blueprint, dims=dims, max_seq_len=max_seq_len)
