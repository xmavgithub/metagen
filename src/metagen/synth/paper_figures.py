"""
MetaGen Paper Figure Generation

Publication-quality figure generation for academic papers.
Includes: pipeline diagrams, Pareto fronts, convergence curves, ablation charts.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from metagen.utils.io import ensure_dir, write_text

if TYPE_CHECKING:
    from metagen.specs.schema import ModelSpec


# =============================================================================
# Color Palette (Tableau 10 inspired, colorblind-friendly)
# =============================================================================

COLORS = {
    "blue": "#4e79a7",
    "orange": "#f28e2b",
    "red": "#e15759",
    "teal": "#76b7b2",
    "green": "#59a14f",
    "yellow": "#edc948",
    "purple": "#b07aa1",
    "pink": "#ff9da7",
    "brown": "#9c755f",
    "gray": "#bab0ac",
}

# Sequential colors for multi-series plots
SERIES_COLORS = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["red"],
    COLORS["teal"],
    COLORS["green"],
]


# =============================================================================
# Matplotlib Configuration
# =============================================================================


def _setup_matplotlib_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


# =============================================================================
# Pipeline Diagram (TikZ)
# =============================================================================


def generate_pipeline_tikz(spec: ModelSpec, out_path: Path) -> None:
    """Generate a TikZ pipeline diagram.

    Creates a LaTeX file with TikZ code that can be compiled into a figure.
    The pipeline shows: Spec -> Encoder -> Search -> Blueprint -> CodeGen -> Artifacts

    Args:
        spec: Model specification (for customization).
        out_path: Path to write the .tex file.
    """
    ensure_dir(out_path.parent)

    tikz_code = r"""\documentclass[tikz,border=5pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, shadows}

\begin{document}
\begin{tikzpicture}[
    node distance=1.8cm,
    box/.style={
        rectangle,
        rounded corners=3pt,
        minimum width=2.2cm,
        minimum height=1cm,
        text centered,
        font=\small\sffamily,
        draw=black!70,
        line width=0.5pt,
        drop shadow={shadow xshift=1pt, shadow yshift=-1pt, opacity=0.3}
    },
    arrow/.style={
        ->,
        >=Stealth,
        line width=0.8pt,
        color=black!60
    },
    label/.style={
        font=\tiny\sffamily,
        color=black!60
    }
]

% Nodes
\node[box, fill=blue!15] (spec) {Spec\\(YAML)};
\node[box, fill=orange!15, right=of spec] (encoder) {Spec\\Encoder};
\node[box, fill=red!15, right=of encoder] (search) {Architecture\\Search};
\node[box, fill=teal!15, right=of search] (blueprint) {Blueprint\\Generator};
\node[box, fill=green!15, right=of blueprint] (codegen) {Code\\Generator};
\node[box, fill=purple!15, right=of codegen] (artifacts) {Release\\Artifacts};

% Arrows
\draw[arrow] (spec) -- (encoder) node[midway, above, label] {encode};
\draw[arrow] (encoder) -- (search) node[midway, above, label] {latent $\mathbf{z}$};
\draw[arrow] (search) -- (blueprint) node[midway, above, label] {dims};
\draw[arrow] (blueprint) -- (codegen) node[midway, above, label] {config};
\draw[arrow] (codegen) -- (artifacts) node[midway, above, label] {code};

% Feedback loop (dashed)
\draw[arrow, dashed, color=black!30] (artifacts.south) -- ++(0,-0.8) -| (search.south)
    node[near start, below, label] {(optional feedback)};

% Title
\node[above=1.5cm of search, font=\normalsize\sffamily\bfseries] {MetaGen Synthesis Pipeline};

\end{tikzpicture}
\end{document}
"""
    write_text(out_path, tikz_code)


def generate_pipeline_matplotlib(
    seed: int,
    out_path: Path,
    scores: dict[str, float] | None = None,
) -> None:
    """Generate a simple pipeline capability chart using matplotlib.

    This is a fallback when TikZ compilation is not available.

    Args:
        seed: Random seed for deterministic generation.
        out_path: Path to save the PDF figure.
        scores: Optional benchmark scores to display.
    """
    _setup_matplotlib_style()
    ensure_dir(out_path.parent)

    if scores is None:
        rnd = random.Random(seed)
        scores = {
            "META-SOTA": 0.89 + rnd.uniform(0, 0.08),
            "GEN-EVAL-∞": 0.90 + rnd.uniform(0, 0.08),
            "FOUNDATION-BENCH": 0.88 + rnd.uniform(0, 0.08),
        }

    fig, ax = plt.subplots(figsize=(5, 3.5))

    names = list(scores.keys())
    values = list(scores.values())

    bars = ax.bar(names, values, color=SERIES_COLORS[: len(names)], edgecolor="white")

    ax.set_ylim(0.75, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("MetaGen Capability Assessment", fontweight="bold")

    # Add value labels on bars
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# =============================================================================
# Pareto Front Visualization
# =============================================================================


def generate_pareto_front(
    seed: int,
    out_path: Path,
    n_candidates: int = 30,
) -> None:
    """Generate a Pareto front visualization showing multi-objective trade-offs.

    Shows params vs performance with dominated and non-dominated points.

    Args:
        seed: Random seed for deterministic generation.
        out_path: Path to save the PDF figure.
        n_candidates: Number of candidate points to generate.
    """
    _setup_matplotlib_style()
    ensure_dir(out_path.parent)
    rnd = random.Random(seed)

    # Generate synthetic candidate data
    # Trade-off: more params -> better performance (with noise)
    params = [rnd.uniform(0.5, 15) for _ in range(n_candidates)]
    performance = [0.7 + 0.2 * (p / 15) + rnd.uniform(-0.05, 0.05) for p in params]

    # Compute Pareto front (simplified: non-dominated points)
    pareto_mask = []
    for i, (p, perf) in enumerate(zip(params, performance, strict=True)):
        is_dominated = False
        for j, (p2, perf2) in enumerate(zip(params, performance, strict=True)):
            if i != j and p2 <= p and perf2 >= perf and (p2 < p or perf2 > perf):
                is_dominated = True
                break
        pareto_mask.append(not is_dominated)

    # Separate Pareto and dominated points
    pareto_params = [p for p, m in zip(params, pareto_mask, strict=True) if m]
    pareto_perf = [perf for perf, m in zip(performance, pareto_mask, strict=True) if m]
    dom_params = [p for p, m in zip(params, pareto_mask, strict=True) if not m]
    dom_perf = [perf for perf, m in zip(performance, pareto_mask, strict=True) if not m]

    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot dominated points
    ax.scatter(
        dom_params,
        dom_perf,
        c=COLORS["gray"],
        alpha=0.5,
        s=40,
        label="Dominated",
        edgecolors="white",
        linewidths=0.5,
    )

    # Plot Pareto front
    ax.scatter(
        pareto_params,
        pareto_perf,
        c=COLORS["blue"],
        s=80,
        marker="*",
        label="Pareto Front",
        edgecolors="white",
        linewidths=0.5,
    )

    # Connect Pareto points with line
    sorted_pareto = sorted(zip(pareto_params, pareto_perf, strict=True))
    if len(sorted_pareto) > 1:
        ax.plot(
            [p[0] for p in sorted_pareto],
            [p[1] for p in sorted_pareto],
            c=COLORS["blue"],
            alpha=0.5,
            linestyle="--",
            linewidth=1,
        )

    # Highlight selected point (highest performance on Pareto)
    if pareto_perf:
        best_idx = pareto_perf.index(max(pareto_perf))
        ax.scatter(
            [pareto_params[best_idx]],
            [pareto_perf[best_idx]],
            c=COLORS["red"],
            s=150,
            marker="*",
            zorder=10,
            label="Selected",
            edgecolors="white",
            linewidths=1,
        )

    ax.set_xlabel("Parameters (Billions)")
    ax.set_ylabel("Performance Score")
    ax.set_title("Multi-Objective Architecture Search", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# =============================================================================
# Search Convergence Curves
# =============================================================================


def generate_convergence_curve(
    seed: int,
    out_path: Path,
    n_iterations: int = 20,
    n_runs: int = 5,
) -> None:
    """Generate search convergence curves with confidence intervals.

    Shows score improvement over iterations for MetaGen vs random baseline.

    Args:
        seed: Random seed for deterministic generation.
        out_path: Path to save the PDF figure.
        n_iterations: Number of search iterations to show.
        n_runs: Number of simulated runs for confidence intervals.
    """
    _setup_matplotlib_style()
    ensure_dir(out_path.parent)
    rnd = random.Random(seed)

    iterations = list(range(1, n_iterations + 1))

    # Generate MetaGen convergence (improves quickly, plateaus)
    metagen_runs = []
    for _run in range(n_runs):
        run_seed = rnd.randint(0, 10000)
        run_rnd = random.Random(run_seed)
        scores = []
        current = 0.75 + run_rnd.uniform(-0.02, 0.02)
        for iteration in iterations:
            improvement = 0.15 * (1 - iteration / n_iterations) * run_rnd.uniform(0.5, 1.5)
            current = min(0.96, current + improvement * 0.1)
            scores.append(current + run_rnd.uniform(-0.01, 0.01))
        metagen_runs.append(scores)

    # Generate random baseline (slow improvement)
    random_runs = []
    for _run in range(n_runs):
        run_seed = rnd.randint(0, 10000)
        run_rnd = random.Random(run_seed)
        scores = []
        current = 0.72 + run_rnd.uniform(-0.02, 0.02)
        for _iteration in iterations:
            improvement = run_rnd.uniform(-0.01, 0.03)
            current = min(0.88, max(0.70, current + improvement * 0.5))
            scores.append(current)
        random_runs.append(scores)

    # Compute mean and std
    metagen_mean = [sum(r[i] for r in metagen_runs) / n_runs for i in range(n_iterations)]
    metagen_std = [
        (sum((r[i] - metagen_mean[i]) ** 2 for r in metagen_runs) / n_runs) ** 0.5
        for i in range(n_iterations)
    ]

    random_mean = [sum(r[i] for r in random_runs) / n_runs for i in range(n_iterations)]
    random_std = [
        (sum((r[i] - random_mean[i]) ** 2 for r in random_runs) / n_runs) ** 0.5
        for i in range(n_iterations)
    ]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # MetaGen curve with confidence interval
    ax.plot(iterations, metagen_mean, color=COLORS["blue"], label="MetaGen", linewidth=2)
    ax.fill_between(
        iterations,
        [m - s for m, s in zip(metagen_mean, metagen_std, strict=True)],
        [m + s for m, s in zip(metagen_mean, metagen_std, strict=True)],
        color=COLORS["blue"],
        alpha=0.2,
    )

    # Random baseline
    ax.plot(
        iterations,
        random_mean,
        color=COLORS["gray"],
        linestyle="--",
        label="Random Search",
        linewidth=1.5,
    )
    ax.fill_between(
        iterations,
        [m - s for m, s in zip(random_mean, random_std, strict=True)],
        [m + s for m, s in zip(random_mean, random_std, strict=True)],
        color=COLORS["gray"],
        alpha=0.15,
    )

    ax.set_xlabel("Search Iteration")
    ax.set_ylabel("Best Score")
    ax.set_title("Search Convergence", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(1, n_iterations)
    ax.set_ylim(0.65, 1.0)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# =============================================================================
# Ablation Bar Chart
# =============================================================================


def generate_ablation_chart(
    seed: int,
    out_path: Path,
    base_score: float = 0.92,
) -> None:
    """Generate an ablation study bar chart.

    Shows performance drop when removing each component.

    Args:
        seed: Random seed for deterministic generation.
        out_path: Path to save the PDF figure.
        base_score: Base score for the full system.
    """
    _setup_matplotlib_style()
    ensure_dir(out_path.parent)
    rnd = random.Random(seed)

    # Ablation configurations and their deltas
    configs = [
        ("Full System", 0),
        ("− Loss Composer", -0.012 + rnd.uniform(-0.003, 0.003)),
        ("− Arch Search", -0.048 + rnd.uniform(-0.005, 0.005)),
        ("− Spec Encoder", -0.031 + rnd.uniform(-0.004, 0.004)),
        ("Random", -0.089 + rnd.uniform(-0.008, 0.008)),
    ]

    names = [c[0] for c in configs]
    scores = [base_score + c[1] for c in configs]
    errors = [rnd.uniform(0.005, 0.015) for _ in configs]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    colors = [COLORS["blue"]] + [COLORS["orange"]] * 3 + [COLORS["gray"]]
    bars = ax.bar(names, scores, color=colors, edgecolor="white", yerr=errors, capsize=3)

    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Component Contributions", fontweight="bold")
    ax.axhline(y=base_score, color=COLORS["blue"], linestyle=":", alpha=0.5)

    # Add value labels
    for bar, score in zip(bars, scores, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.02,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# =============================================================================
# Architecture Graph (NetworkX-style)
# =============================================================================


def generate_architecture_graph(
    arch_summary: dict,
    out_path: Path,
    seed: int,
) -> None:
    """Generate a simplified architecture component graph.

    Shows the flow from input through model components to output.

    Args:
        arch_summary: Architecture summary with dims and family info.
        out_path: Path to save the PDF figure.
        seed: Random seed for layout consistency.
    """
    _setup_matplotlib_style()
    ensure_dir(out_path.parent)

    dims = arch_summary.get("dims", {})
    hidden_size = dims.get("hidden_size", 4096)
    layers = dims.get("layers", 32)
    heads = dims.get("heads", 32)
    family = arch_summary.get("family", "transformer")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 3.5)
    ax.axis("off")

    # Node positions
    nodes = {
        "Input": (0, 1.5),
        "Embedding": (1.5, 1.5),
        f"Encoder\n({family})": (3, 2.5),
        f"Attention\n(H={heads})": (3, 0.5),
        f"FFN\n(d={hidden_size})": (4.5, 1.5),
        "Output": (6, 1.5),
    }

    # Draw nodes
    for name, (x, y) in nodes.items():
        circle = plt.Circle((x, y), 0.4, color=COLORS["blue"], alpha=0.7)
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
            wrap=True,
        )

    # Draw edges
    edges = [
        ("Input", "Embedding"),
        ("Embedding", f"Encoder\n({family})"),
        ("Embedding", f"Attention\n(H={heads})"),
        (f"Encoder\n({family})", f"FFN\n(d={hidden_size})"),
        (f"Attention\n(H={heads})", f"FFN\n(d={hidden_size})"),
        (f"FFN\n(d={hidden_size})", "Output"),
    ]

    for start, end in edges:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        ax.annotate(
            "",
            xy=(x2 - 0.4, y2),
            xytext=(x1 + 0.4, y1),
            arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5),
        )

    # Title and annotations
    ax.set_title(
        f"Architecture: {layers}L × {hidden_size}d × {heads}H",
        fontweight="bold",
        fontsize=11,
        pad=20,
    )
    ax.text(
        3,
        -0.3,
        f"Total: ~{arch_summary.get('params_billion', 'N/A')}B parameters",
        ha="center",
        fontsize=9,
        style="italic",
        color=COLORS["gray"],
    )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# =============================================================================
# Main Figure Generation Orchestrator
# =============================================================================


def generate_all_figures(
    spec: ModelSpec,
    arch_summary: dict,
    bench_summary: dict,
    figures_dir: Path,
    seed: int,
) -> dict[str, Path]:
    """Generate all paper figures.

    Args:
        spec: Model specification.
        arch_summary: Architecture summary with dims and params.
        bench_summary: Benchmark summary with scores.
        figures_dir: Directory to save figures.
        seed: Random seed for deterministic generation.

    Returns:
        Dictionary mapping figure names to their paths.
    """
    ensure_dir(figures_dir)

    scores = bench_summary.get("scores", {})
    base_score = max(scores.values()) if scores else 0.92

    figures = {}

    # Pipeline/capability chart
    pipeline_path = figures_dir / "pipeline.pdf"
    generate_pipeline_matplotlib(seed, pipeline_path, scores)
    figures["pipeline"] = pipeline_path

    # Pareto front
    pareto_path = figures_dir / "pareto_front.pdf"
    generate_pareto_front(seed, pareto_path)
    figures["pareto_front"] = pareto_path

    # Convergence curve
    convergence_path = figures_dir / "convergence.pdf"
    generate_convergence_curve(seed, convergence_path)
    figures["convergence"] = convergence_path

    # Ablation chart
    ablation_path = figures_dir / "ablation.pdf"
    generate_ablation_chart(seed, ablation_path, base_score)
    figures["ablation"] = ablation_path

    # Architecture graph
    arch_path = figures_dir / "architecture_graph.pdf"
    generate_architecture_graph(arch_summary, arch_path, seed)
    figures["architecture_graph"] = arch_path

    # TikZ pipeline (optional, for advanced users)
    tikz_path = figures_dir / "pipeline_tikz.tex"
    generate_pipeline_tikz(spec, tikz_path)
    figures["pipeline_tikz"] = tikz_path

    return figures
