from __future__ import annotations

from pathlib import Path

from metagen.automl.search_engine import ArchitectureSearchEngine
from metagen.specs.loader import load_spec


def main() -> None:
    spec_path = Path("examples/specs/text/text_llm_8b.yaml")
    spec, _ = load_spec(spec_path)

    engine = ArchitectureSearchEngine(seed=42)
    result = engine.search(spec, search_budget=5)

    print(f"AutoML search for {spec.name}")
    for idx, candidate in enumerate(result.top_k(3), start=1):
        metrics = candidate.metrics
        print(
            f"{idx}) hidden={candidate.dims['hidden_size']} "
            f"layers={candidate.dims['layers']} "
            f"heads={candidate.dims['heads']} "
            f"params={metrics['params_billion']:.2f}B "
            f"score={candidate.score:.4f}"
        )


if __name__ == "__main__":
    main()
