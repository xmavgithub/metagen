"""
MetaGen Experiment Runner

Orchestrates running experiments across multiple specs and baselines,
collecting metrics and generating comparison reports.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from metagen.experiments.baselines import (
    Baseline,
    BaselineMetrics,
    get_baseline,
    list_baselines,
)
from metagen.specs.loader import load_spec
from metagen.utils.io import ensure_dir, write_json, write_text

if TYPE_CHECKING:
    from metagen.specs.schema import ModelSpec


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    spec_name: str
    baseline_name: str
    seed: int
    metrics: BaselineMetrics
    architecture_summary: dict
    run_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "spec_name": self.spec_name,
            "baseline_name": self.baseline_name,
            "seed": self.seed,
            "metrics": self.metrics.to_dict(),
            "architecture_summary": self.architecture_summary,
            "run_metadata": self.run_metadata,
        }


@dataclass
class ExperimentSuite:
    """Collection of experiment results for comparison."""

    name: str
    results: list[ExperimentResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_result(self, result: ExperimentResult) -> None:
        """Add a result to the suite."""
        self.results.append(result)

    def get_results_by_spec(self, spec_name: str) -> list[ExperimentResult]:
        """Get all results for a specific spec."""
        return [r for r in self.results if r.spec_name == spec_name]

    def get_results_by_baseline(self, baseline_name: str) -> list[ExperimentResult]:
        """Get all results for a specific baseline."""
        return [r for r in self.results if r.baseline_name == baseline_name]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "num_experiments": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        }

    def generate_comparison_table(self) -> str:
        """Generate a markdown comparison table."""
        if not self.results:
            return "No results available."

        # Group by spec
        specs = sorted(set(r.spec_name for r in self.results))

        lines = ["# Experiment Results\n"]
        lines.append(f"Suite: **{self.name}**\n")
        lines.append(f"Total experiments: {len(self.results)}\n")

        for spec in specs:
            lines.append(f"\n## {spec}\n")
            lines.append("| Baseline | Accuracy | Params (M) | Latency (ms) | Memory (MB) |")
            lines.append("|----------|----------|------------|--------------|-------------|")

            spec_results = self.get_results_by_spec(spec)
            for result in sorted(spec_results, key=lambda r: r.baseline_name):
                m = result.metrics
                lines.append(
                    f"| {result.baseline_name} | {m.accuracy:.4f} | "
                    f"{m.params_million:.1f} | {m.latency_ms:.1f} | {m.memory_mb:.1f} |"
                )

        return "\n".join(lines)


class ExperimentRunner:
    """
    Runs experiments comparing MetaGen against baselines.

    Example:
        >>> runner = ExperimentRunner(output_dir=Path("experiments/"))
        >>> suite = runner.run_suite(
        ...     spec_paths=["examples/specs/text_llm_8b.yaml"],
        ...     baselines=["gpt2", "random_search", "metagen"],
        ...     num_runs=3,
        ...     seed=42,
        ... )
        >>> print(suite.generate_comparison_table())
    """

    def __init__(self, output_dir: Path | None = None):
        """
        Initialize the experiment runner.

        Args:
            output_dir: Directory for saving experiment results.
        """
        self.output_dir = output_dir

    def run_single(
        self,
        spec: ModelSpec,
        baseline: Baseline,
        seed: int,
    ) -> ExperimentResult:
        """
        Run a single experiment.

        Args:
            spec: Model specification.
            baseline: Baseline to evaluate.
            seed: Random seed for reproducibility.

        Returns:
            ExperimentResult with metrics and metadata.
        """
        metrics = baseline.evaluate(spec, seed)
        architecture_summary = baseline.get_architecture_summary(spec)

        return ExperimentResult(
            spec_name=spec.name,
            baseline_name=baseline.name,
            seed=seed,
            metrics=metrics,
            architecture_summary=architecture_summary,
            run_metadata={
                "baseline_category": baseline.category,
                "baseline_description": baseline.description,
            },
        )

    def run_suite(
        self,
        spec_paths: list[str | Path],
        baselines: list[str] | None = None,
        num_runs: int = 3,
        seed: int = 42,
        suite_name: str = "experiment_suite",
    ) -> ExperimentSuite:
        """
        Run a full experiment suite across specs and baselines.

        Args:
            spec_paths: List of paths to spec files.
            baselines: List of baseline names (defaults to all).
            num_runs: Number of runs per spec/baseline combination.
            seed: Base random seed.
            suite_name: Name for the experiment suite.

        Returns:
            ExperimentSuite with all results.
        """
        if baselines is None:
            baselines = list_baselines()

        suite = ExperimentSuite(
            name=suite_name,
            metadata={
                "num_specs": len(spec_paths),
                "num_baselines": len(baselines),
                "num_runs": num_runs,
                "base_seed": seed,
            },
        )

        rnd = random.Random(seed)

        for spec_path in spec_paths:
            spec, _ = load_spec(Path(spec_path))

            for baseline_name in baselines:
                baseline = get_baseline(baseline_name)

                for run_idx in range(num_runs):
                    run_seed = rnd.randint(0, 2**31 - 1)
                    result = self.run_single(spec, baseline, run_seed)
                    result.run_metadata["run_index"] = run_idx
                    suite.add_result(result)

        if self.output_dir:
            self._save_suite(suite)

        return suite

    def _save_suite(self, suite: ExperimentSuite) -> None:
        """Save experiment suite to disk."""
        if not self.output_dir:
            return

        output_dir = ensure_dir(self.output_dir)

        # Save JSON results
        write_json(output_dir / f"{suite.name}.json", suite.to_dict())

        # Save markdown report
        report = suite.generate_comparison_table()
        write_text(output_dir / f"{suite.name}_report.md", report)


def run_experiment(
    spec_path: str | Path,
    baselines: list[str] | None = None,
    seed: int = 42,
    output_dir: Path | None = None,
) -> ExperimentSuite:
    """
    Convenience function to run experiments on a single spec.

    Args:
        spec_path: Path to spec file.
        baselines: List of baseline names (defaults to all).
        seed: Random seed.
        output_dir: Optional output directory.

    Returns:
        ExperimentSuite with results.

    Example:
        >>> suite = run_experiment("examples/specs/text_llm_8b.yaml")
        >>> print(suite.generate_comparison_table())
    """
    runner = ExperimentRunner(output_dir=output_dir)
    return runner.run_suite(
        spec_paths=[spec_path],
        baselines=baselines,
        num_runs=1,
        seed=seed,
        suite_name=Path(spec_path).stem,
    )
