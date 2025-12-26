#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path


def _find_latest_run() -> Path | None:
    roots = [Path("outputs"), Path("examples/outputs")]
    runs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("run-*"):
            if path.is_dir() and (path / "code").is_dir():
                runs.append(path)
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def _select_device(torch_mod, device_arg: str | None):
    if device_arg:
        return torch_mod.device(device_arg)
    if torch_mod.cuda.is_available():
        return torch_mod.device("cuda")
    if torch_mod.backends.mps.is_available():
        return torch_mod.device("mps")
    return torch_mod.device("cpu")


def _sync_device(torch_mod, device):
    # Sync ensures accurate timing on async backends.
    if device.type == "cuda":
        torch_mod.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch_mod, "mps"):
        torch_mod.mps.synchronize()


def _train_step(model, optimizer, batch, device):
    if isinstance(batch, (list, tuple)):
        x = batch[0].to(device)
    else:
        x = batch.to(device)
    optimizer.zero_grad()
    output = model(x)
    if isinstance(output, (tuple, list)) and len(output) > 1:
        loss = output[0]
    elif hasattr(output, "loss"):
        loss = output.loss
    else:
        loss = output.mean()
    loss.backward()
    optimizer.step()
    return x


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark batch size throughput.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory (contains code/).")
    parser.add_argument("--data", type=str, default="examples/data/train.bin", help="Path to data.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--steps", type=int, default=100, help="Timed steps per batch size.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps per batch size.")
    parser.add_argument("--device", type=str, default=None, help="cuda/mps/cpu (auto if omitted).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _find_latest_run()
    if not run_dir:
        print("Error: No run directory found. Pass --run-dir explicitly.")
        return 1

    code_dir = run_dir / "code"
    if not code_dir.is_dir():
        print(f"Error: code/ not found under {run_dir}")
        return 1

    sys.path.insert(0, str(code_dir))
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"Error: PyTorch import failed: {exc}")
        return 1

    try:
        from data import load_data
        from model import MetaGenModel
    except Exception as exc:
        print(f"Error: Failed to import generated code: {exc}")
        return 1

    device = _select_device(torch, args.device)
    print(f"Run: {run_dir}")
    print(f"Device: {device}")

    for batch_size in args.batch_sizes:
        loader = load_data(data_path=args.data, batch_size=batch_size)
        if not hasattr(loader, "__len__"):
            print("Error: Data loader is not a DataLoader. Aborting.")
            return 1
        total_steps = min(args.steps, len(loader))
        warmup_steps = min(args.warmup, total_steps)
        if total_steps == 0:
            print(f"batch={batch_size} -> no steps (dataset too small)")
            continue

        model = MetaGenModel().to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        it = iter(loader)
        for _ in range(warmup_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            _train_step(model, optimizer, batch, device)

        _sync_device(torch, device)
        start = time.perf_counter()
        seq_len = None
        for step in range(total_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            x = _train_step(model, optimizer, batch, device)
            if seq_len is None:
                seq_len = x.shape[1] if x.ndim > 1 else x.shape[0]
        _sync_device(torch, device)
        elapsed = time.perf_counter() - start

        steps_per_sec = total_steps / max(elapsed, 1e-9)
        tokens_per_sec = steps_per_sec * batch_size * (seq_len or 0)
        print(
            f"batch={batch_size} steps={total_steps} "
            f"sec={elapsed:.2f} steps/s={steps_per_sec:.2f} "
            f"tok/s={tokens_per_sec/1e6:.2f}M"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
