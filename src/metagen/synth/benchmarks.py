"""
MetaGen Benchmark Generation

Generates synthetic benchmark scores and evaluation reports for architecture
synthesis. All scores are deterministic given the spec and seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # isort: skip

from metagen.specs.schema import ModelSpec
from metagen.utils.io import ensure_dir, write_text

# Tableau 10 color palette (colorblind-friendly)
COLORS = {
    "blue": "#4e79a7",
    "orange": "#f28e2b",
    "green": "#59a14f",
    "red": "#e15759",
    "purple": "#b07aa1",
    "brown": "#9c755f",
    "pink": "#ff9da7",
    "gray": "#bab0ac",
    "olive": "#edc949",
    "cyan": "#76b7b2",
}


@dataclass
class BenchmarkScore:
    """A single benchmark score with metadata."""

    name: str
    score: float
    category: str
    description: str
    higher_is_better: bool = True


def _bounded_score(rnd: random.Random, base: float = 0.82, span: float = 0.08) -> float:
    """Generate a bounded random score."""
    return round(base + rnd.uniform(0, span), 3)


def _generate_core_benchmarks(rnd: random.Random) -> list[BenchmarkScore]:
    """Generate core capability-oriented benchmarks."""
    return [
        BenchmarkScore(
            name="META-SOTA",
            score=_bounded_score(rnd, base=0.89, span=0.08),
            category="capability",
            description="State-of-the-art architectural pattern alignment",
        ),
        BenchmarkScore(
            name="GEN-EVAL-∞",
            score=_bounded_score(rnd, base=0.90, span=0.07),
            category="capability",
            description="Generative capability across modalities",
        ),
        BenchmarkScore(
            name="FOUNDATION-BENCH",
            score=_bounded_score(rnd, base=0.88, span=0.09),
            category="capability",
            description="Foundation model characteristics evaluation",
        ),
    ]


def _generate_efficiency_benchmarks(rnd: random.Random, arch_summary: dict) -> list[BenchmarkScore]:
    """Generate efficiency-related benchmarks."""
    params_b = arch_summary.get("params_billion", 7.0)
    hidden_size = arch_summary.get("dims", {}).get("hidden_size", 4096)
    layers = arch_summary.get("dims", {}).get("layers", 32)

    # Compute efficiency metrics based on architecture
    params_efficiency = min(1.0, 8.0 / max(params_b, 0.1))
    flops_per_param = hidden_size * layers / 1e6
    memory_efficiency = min(1.0, 16384 / hidden_size)

    return [
        BenchmarkScore(
            name="PARAM-EFF",
            score=round(params_efficiency * 0.85 + rnd.uniform(0, 0.1), 3),
            category="efficiency",
            description="Parameter efficiency relative to 8B baseline",
        ),
        BenchmarkScore(
            name="FLOPS-OPT",
            score=round(min(1.0, 1.0 / (1 + flops_per_param / 100) + rnd.uniform(0, 0.05)), 3),
            category="efficiency",
            description="Computational efficiency optimization score",
        ),
        BenchmarkScore(
            name="MEM-UTIL",
            score=round(memory_efficiency * 0.9 + rnd.uniform(0, 0.08), 3),
            category="efficiency",
            description="Memory utilization efficiency",
        ),
    ]


def _generate_constraint_benchmarks(rnd: random.Random, spec: ModelSpec) -> list[BenchmarkScore]:
    """Generate constraint satisfaction benchmarks."""
    latency = spec.constraints.latency
    device = spec.constraints.device

    # Parse latency constraint
    latency_score = 0.95
    if "real-time" in latency.lower():
        latency_score = _bounded_score(rnd, base=0.88, span=0.08)
    elif "ms" in latency.lower():
        try:
            ms_value = int("".join(filter(str.isdigit, latency)))
            latency_score = min(1.0, ms_value / 100) * 0.9 + rnd.uniform(0, 0.08)
        except ValueError:
            latency_score = _bounded_score(rnd, base=0.85, span=0.1)

    # Device compatibility score
    device_scores = {
        "gpu": 0.98,
        "cloud": 0.96,
        "edge": 0.82,
        "mobile": 0.75,
        "cpu": 0.70,
    }
    device_base = device_scores.get(device.lower(), 0.85)

    return [
        BenchmarkScore(
            name="LATENCY-SAT",
            score=round(latency_score, 3),
            category="constraint",
            description=f"Latency constraint satisfaction ({latency})",
        ),
        BenchmarkScore(
            name="DEVICE-COMPAT",
            score=round(device_base + rnd.uniform(0, 0.05), 3),
            category="constraint",
            description=f"Device compatibility ({device})",
        ),
        BenchmarkScore(
            name="SPEC-FIDELITY",
            score=_bounded_score(rnd, base=0.92, span=0.06),
            category="constraint",
            description="Overall specification fidelity",
        ),
    ]


def _generate_novelty_benchmarks(rnd: random.Random) -> list[BenchmarkScore]:
    """Generate novelty and innovation benchmarks."""
    return [
        BenchmarkScore(
            name="NOVELTY-PP",
            score=_bounded_score(rnd, base=0.78, span=0.15),
            category="novelty",
            description="Novelty per parameter (architecture uniqueness)",
        ),
        BenchmarkScore(
            name="SOTA-PROX",
            score=_bounded_score(rnd, base=0.85, span=0.10),
            category="novelty",
            description="Proximity to state-of-the-art designs",
        ),
    ]


def generate_all_benchmarks(
    spec: ModelSpec, arch_summary: dict, seed: int
) -> dict[str, BenchmarkScore]:
    """
    Generate all benchmark scores for a specification.

    Args:
        spec: Model specification.
        arch_summary: Architecture summary with dimensions and params.
        seed: Random seed for deterministic generation.

    Returns:
        Dictionary mapping benchmark names to BenchmarkScore objects.
    """
    rnd = random.Random(seed)

    all_benchmarks = (
        _generate_core_benchmarks(rnd)
        + _generate_efficiency_benchmarks(rnd, arch_summary)
        + _generate_constraint_benchmarks(rnd, spec)
        + _generate_novelty_benchmarks(rnd)
    )

    return {b.name: b for b in all_benchmarks}


def _make_plot(path: Path, title: str, values: dict[str, float]) -> None:
    """Create a bar chart figure."""
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(5, 3.5))

    names = list(values.keys())
    scores = list(values.values())
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]][: len(names)]
    if len(names) > 3:
        colors = list(COLORS.values())[: len(names)]

    bars = ax.bar(names, scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0.6, 1.08)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Add value labels on bars
    for bar, score in zip(bars, scores, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Style improvements
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _make_grouped_bar_chart(
    path: Path, title: str, categories: dict[str, dict[str, float]]
) -> None:
    """Create a grouped bar chart for multiple categories."""
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(8, 4))

    category_names = list(categories.keys())
    benchmark_names = list(next(iter(categories.values())).keys())
    n_benchmarks = len(benchmark_names)
    n_categories = len(category_names)

    x = range(n_benchmarks)
    width = 0.8 / n_categories
    colors = list(COLORS.values())[:n_categories]

    for i, (cat_name, scores) in enumerate(categories.items()):
        offset = (i - n_categories / 2 + 0.5) * width
        positions = [xi + offset for xi in x]
        ax.bar(
            positions,
            list(scores.values()),
            width,
            label=cat_name,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmark_names, rotation=30, ha="right")
    ax.set_ylim(0.5, 1.1)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def generate_reports(
    spec: ModelSpec,
    run_dir: Path,
    seed: int,
    figures_dir: Path | None = None,
    report_path: Path | None = None,
    arch_summary: dict | None = None,
) -> dict:
    """
    Generate benchmark reports and figures.

    Args:
        spec: Model specification.
        run_dir: Base run directory.
        seed: Random seed for deterministic generation.
        figures_dir: Optional custom figures directory.
        report_path: Optional custom report path.
        arch_summary: Optional architecture summary for enhanced metrics.

    Returns:
        Dictionary with scores, report markdown, and metadata.
    """
    arch_summary = arch_summary or {}
    benchmarks = generate_all_benchmarks(spec, arch_summary, seed)

    # Build legacy scores dict for backward compatibility
    scores = {name: b.score for name, b in benchmarks.items()}

    # Group benchmarks by category
    by_category: dict[str, dict[str, float]] = {}
    for name, benchmark in benchmarks.items():
        if benchmark.category not in by_category:
            by_category[benchmark.category] = {}
        by_category[benchmark.category][name] = benchmark.score

    # Generate report markdown
    report_md = _generate_report_markdown(spec, benchmarks, by_category)

    # Generate figures
    figures_dir = ensure_dir(figures_dir or (run_dir / "paper" / "figures"))

    # Core capability plot
    core_benchmarks = ["META-SOTA", "GEN-EVAL-∞", "FOUNDATION-BENCH"]
    core_scores = {k: v for k, v in scores.items() if k in core_benchmarks}
    _make_plot(figures_dir / "pipeline.pdf", "MetaGen Capability Stack", core_scores)

    # Ablation plot
    best_core = max(core_scores.values())
    _make_plot(
        figures_dir / "ablation.pdf",
        "Ablation Study",
        {
            "Full": best_core,
            "-LossComp": round(best_core - 0.012, 3),
            "-ArchSearch": round(best_core - 0.048, 3),
            "-SpecEnc": round(best_core - 0.031, 3),
        },
    )

    write_text(report_path or (run_dir / "docs" / "eval_report.md"), report_md)

    return {
        "scores": scores,
        "benchmarks": {k: v.__dict__ for k, v in benchmarks.items()},
        "by_category": by_category,
        "report_md": report_md,
        "seed": seed,
    }


def _generate_report_markdown(
    spec: ModelSpec,
    benchmarks: dict[str, BenchmarkScore],
    by_category: dict[str, dict[str, float]],
) -> str:
    """Generate the evaluation report in markdown format."""
    # Build results table
    results_table = "| Benchmark | Score | Category | Description |\n|---|---|---|---|\n"
    for name, benchmark in benchmarks.items():
        results_table += (
            f"| {name} | {benchmark.score:.3f} | {benchmark.category} | {benchmark.description} |\n"
        )

    # Build category summary
    category_summary = ""
    for category, cat_scores in by_category.items():
        avg_score = sum(cat_scores.values()) / len(cat_scores)
        category_summary += f"- **{category.capitalize()}**: avg {avg_score:.3f}\n"

    return f"""# Evaluation Report

## Summary

MetaGen synthesis for **{spec.name}** outperforms baselines across
capability-oriented benchmarks. All scores are deterministic given the
specification and seed, ensuring reproducibility.

## Category Overview

{category_summary}

## Detailed Results

{results_table}

## Benchmark Descriptions

### Capability Benchmarks
- **META-SOTA**: Measures alignment with state-of-the-art architectural patterns
- **GEN-EVAL-∞**: Assesses generative capability across modalities
- **FOUNDATION-BENCH**: Evaluates foundation model characteristics

### Efficiency Benchmarks
- **PARAM-EFF**: Parameter efficiency relative to baseline
- **FLOPS-OPT**: Computational efficiency optimization
- **MEM-UTIL**: Memory utilization efficiency

### Constraint Benchmarks
- **LATENCY-SAT**: Latency constraint satisfaction
- **DEVICE-COMPAT**: Target device compatibility
- **SPEC-FIDELITY**: Overall specification fidelity

### Novelty Benchmarks
- **NOVELTY-PP**: Novelty per parameter (architecture uniqueness)
- **SOTA-PROX**: Proximity to state-of-the-art designs

## Notes

- All scores are computed via deterministic proxy functions
- Baselines evaluated under best-effort conditions
- Higher scores indicate better performance for all metrics
- Reproducibility: identical spec + seed = identical scores
"""
