#!/usr/bin/env python3
"""Generate example figures for MetaGen documentation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output directory
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def generate_pareto_front():
    """Generate example Pareto front visualization."""
    np.random.seed(42)

    # Generate candidates
    n_candidates = 50
    params = np.random.uniform(1, 15, n_candidates)
    latency = 50 + params * 30 + np.random.normal(0, 20, n_candidates)
    score = 0.5 + 0.3 * np.log(params) + np.random.normal(0, 0.05, n_candidates)

    # Find Pareto front (simplified)
    pareto_mask = np.zeros(n_candidates, dtype=bool)
    for i in range(n_candidates):
        dominated = False
        for j in range(n_candidates):
            if i != j:
                if params[j] <= params[i] and score[j] >= score[i]:
                    if params[j] < params[i] or score[j] > score[i]:
                        dominated = True
                        break
        if not dominated:
            pareto_mask[i] = True

    # Sort Pareto front by params
    pareto_idx = np.where(pareto_mask)[0]
    pareto_idx = pareto_idx[np.argsort(params[pareto_idx])]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all candidates
    ax.scatter(params[~pareto_mask], score[~pareto_mask],
               alpha=0.4, s=50, c='steelblue', label='Candidates')

    # Plot Pareto front
    ax.scatter(params[pareto_mask], score[pareto_mask],
               s=120, c='crimson', marker='*', label='Pareto Front', zorder=5)

    # Connect Pareto points
    ax.plot(params[pareto_idx], score[pareto_idx],
            'r--', alpha=0.5, linewidth=2)

    ax.set_xlabel('Parameters (Billions)', fontsize=12)
    ax.set_ylabel('Performance Score', fontsize=12)
    ax.set_title('AutoML Pareto Front: Parameters vs Performance', fontsize=14)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pareto_front_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {FIGURES_DIR / 'pareto_front_example.png'}")


def generate_convergence():
    """Generate search convergence visualization."""
    np.random.seed(42)

    generations = np.arange(1, 21)

    # Random search (flat)
    random_mean = 0.7 + 0.15 * (1 - np.exp(-generations / 5)) + np.random.normal(0, 0.02, 20)
    random_std = 0.1 * np.exp(-generations / 10)

    # Evolutionary (converges faster)
    evo_mean = 0.7 + 0.25 * (1 - np.exp(-generations / 3)) + np.random.normal(0, 0.01, 20)
    evo_std = 0.15 * np.exp(-generations / 5)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Random search
    ax.fill_between(generations, random_mean - random_std, random_mean + random_std,
                    alpha=0.2, color='steelblue')
    ax.plot(generations, random_mean, 'o-', color='steelblue',
            label='Random Search', linewidth=2, markersize=5)

    # Evolutionary
    ax.fill_between(generations, evo_mean - evo_std, evo_mean + evo_std,
                    alpha=0.2, color='crimson')
    ax.plot(generations, evo_mean, 's-', color='crimson',
            label='Evolutionary', linewidth=2, markersize=5)

    ax.set_xlabel('Generation / Iteration', fontsize=12)
    ax.set_ylabel('Best Score', fontsize=12)
    ax.set_title('AutoML Search Convergence', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_xlim(1, 20)
    ax.set_ylim(0.6, 1.0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'convergence_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {FIGURES_DIR / 'convergence_example.png'}")


def generate_ablation():
    """Generate ablation study bar chart."""
    components = [
        'Full Model',
        '- Multi-head Attn',
        '- FFN Expansion',
        '- Layer Norm',
        '- Positional Enc',
        '- Residual Conn'
    ]

    scores = [0.932, 0.891, 0.876, 0.823, 0.867, 0.845]
    colors = ['forestgreen'] + ['steelblue'] * 5

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(components, scores, color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10)

    ax.set_xlabel('Performance Score', fontsize=12)
    ax.set_title('Ablation Study: Component Contributions', fontsize=14)
    ax.set_xlim(0.7, 1.0)
    ax.axvline(x=scores[0], color='forestgreen', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ablation_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {FIGURES_DIR / 'ablation_example.png'}")


def generate_benchmark_radar():
    """Generate radar chart of benchmark scores."""
    categories = ['META-SOTA', 'GEN-EVAL-∞', 'FOUNDATION\nBENCH',
                  'PARAM-EFF', 'LATENCY-SAT', 'SPEC-FIDELITY']

    # MetaGen scores
    metagen_scores = [0.92, 0.94, 0.91, 0.93, 0.96, 0.93]
    # Baseline scores
    baseline_scores = [0.85, 0.82, 0.88, 0.78, 0.80, 0.75]

    # Number of categories
    N = len(categories)

    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    metagen_scores += metagen_scores[:1]
    baseline_scores += baseline_scores[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw the chart
    ax.plot(angles, metagen_scores, 'o-', linewidth=2,
            label='MetaGen', color='crimson')
    ax.fill(angles, metagen_scores, alpha=0.25, color='crimson')

    ax.plot(angles, baseline_scores, 's-', linewidth=2,
            label='Baseline', color='steelblue')
    ax.fill(angles, baseline_scores, alpha=0.25, color='steelblue')

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    # Set y-axis
    ax.set_ylim(0.6, 1.0)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Benchmark Comparison', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'benchmark_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {FIGURES_DIR / 'benchmark_radar.png'}")


def generate_modality_overview():
    """Generate modality support overview."""
    modalities = ['Text', 'Image', 'Audio', 'Video', 'Multi-\nmodal']
    architectures = [1, 4, 1, 2, 2]  # Number of supported architectures
    example_specs = [3, 3, 2, 2, 1]  # Number of example specs

    x = np.arange(len(modalities))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, architectures, width,
                   label='Architecture Families', color='steelblue')
    bars2 = ax.bar(x + width/2, example_specs, width,
                   label='Example Specs', color='forestgreen')

    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('MetaGen Modality Support', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, fontsize=11)
    ax.legend()
    ax.set_ylim(0, 5)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'modality_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {FIGURES_DIR / 'modality_overview.png'}")


if __name__ == '__main__':
    print("Generating MetaGen documentation figures...")
    generate_pareto_front()
    generate_convergence()
    generate_ablation()
    generate_benchmark_radar()
    generate_modality_overview()
    print(f"\nAll figures saved to: {FIGURES_DIR}")
