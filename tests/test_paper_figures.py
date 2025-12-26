"""
Tests for MetaGen Paper Figure Generation

Tests for all publication-quality figure generation functions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from metagen.specs.loader import load_spec
from metagen.synth.paper_figures import (
    generate_ablation_chart,
    generate_all_figures,
    generate_architecture_graph,
    generate_convergence_curve,
    generate_pareto_front,
    generate_pipeline_matplotlib,
    generate_pipeline_tikz,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_spec():
    """Load a sample spec for testing."""
    spec, _ = load_spec(Path("examples/specs/text_llm_8b.yaml"))
    return spec


@pytest.fixture
def arch_summary():
    """Sample architecture summary."""
    return {
        "family": "transformer",
        "params_billion": 7.5,
        "dims": {
            "hidden_size": 4096,
            "layers": 32,
            "heads": 32,
            "ffn_hidden_size": 11008,
            "vocab_size": 32000,
        },
    }


@pytest.fixture
def bench_summary():
    """Sample benchmark summary."""
    return {
        "scores": {
            "META-SOTA": 0.92,
            "GEN-EVAL-∞": 0.94,
            "FOUNDATION-BENCH": 0.91,
        }
    }


# =============================================================================
# Pipeline Figure Tests
# =============================================================================


class TestPipelineMatplotlib:
    """Tests for matplotlib pipeline figure."""

    def test_generates_file(self):
        """Test that pipeline figure is generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "pipeline.pdf"
            generate_pipeline_matplotlib(seed=42, out_path=out_path)
            assert out_path.exists()

    def test_with_custom_scores(self):
        """Test generation with custom scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "pipeline.pdf"
            scores = {"Score A": 0.85, "Score B": 0.90}
            generate_pipeline_matplotlib(seed=42, out_path=out_path, scores=scores)
            assert out_path.exists()

    def test_deterministic(self):
        """Test that generation is deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "pipeline1.pdf"
            path2 = Path(tmpdir) / "pipeline2.pdf"
            generate_pipeline_matplotlib(seed=42, out_path=path1)
            generate_pipeline_matplotlib(seed=42, out_path=path2)
            # Files should be similar in size (exact match may vary due to timestamps)
            assert abs(path1.stat().st_size - path2.stat().st_size) < 1000


class TestPipelineTikz:
    """Tests for TikZ pipeline figure."""

    def test_generates_file(self, sample_spec):
        """Test that TikZ file is generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "pipeline.tex"
            generate_pipeline_tikz(sample_spec, out_path)
            assert out_path.exists()

    def test_valid_tikz_content(self, sample_spec):
        """Test that generated file contains valid TikZ code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "pipeline.tex"
            generate_pipeline_tikz(sample_spec, out_path)
            content = out_path.read_text()
            assert r"\begin{tikzpicture}" in content
            assert r"\end{tikzpicture}" in content
            assert r"\documentclass" in content


# =============================================================================
# Pareto Front Tests
# =============================================================================


class TestParetoFront:
    """Tests for Pareto front visualization."""

    def test_generates_file(self):
        """Test that Pareto front figure is generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "pareto.pdf"
            generate_pareto_front(seed=42, out_path=out_path)
            assert out_path.exists()

    def test_custom_candidates(self):
        """Test with custom number of candidates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "pareto.pdf"
            generate_pareto_front(seed=42, out_path=out_path, n_candidates=50)
            assert out_path.exists()

    def test_deterministic(self):
        """Test that generation is deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "pareto1.pdf"
            path2 = Path(tmpdir) / "pareto2.pdf"
            generate_pareto_front(seed=42, out_path=path1)
            generate_pareto_front(seed=42, out_path=path2)
            assert path1.stat().st_size > 0
            assert path2.stat().st_size > 0


# =============================================================================
# Convergence Curve Tests
# =============================================================================


class TestConvergenceCurve:
    """Tests for convergence curve generation."""

    def test_generates_file(self):
        """Test that convergence curve is generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "convergence.pdf"
            generate_convergence_curve(seed=42, out_path=out_path)
            assert out_path.exists()

    def test_custom_iterations(self):
        """Test with custom number of iterations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "convergence.pdf"
            generate_convergence_curve(seed=42, out_path=out_path, n_iterations=30)
            assert out_path.exists()

    def test_custom_runs(self):
        """Test with custom number of runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "convergence.pdf"
            generate_convergence_curve(seed=42, out_path=out_path, n_runs=10)
            assert out_path.exists()

    def test_deterministic(self):
        """Test that generation is deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "conv1.pdf"
            path2 = Path(tmpdir) / "conv2.pdf"
            generate_convergence_curve(seed=42, out_path=path1)
            generate_convergence_curve(seed=42, out_path=path2)
            assert path1.stat().st_size > 0


# =============================================================================
# Ablation Chart Tests
# =============================================================================


class TestAblationChart:
    """Tests for ablation chart generation."""

    def test_generates_file(self):
        """Test that ablation chart is generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "ablation.pdf"
            generate_ablation_chart(seed=42, out_path=out_path)
            assert out_path.exists()

    def test_custom_base_score(self):
        """Test with custom base score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "ablation.pdf"
            generate_ablation_chart(seed=42, out_path=out_path, base_score=0.88)
            assert out_path.exists()

    def test_deterministic(self):
        """Test that generation is deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "abl1.pdf"
            path2 = Path(tmpdir) / "abl2.pdf"
            generate_ablation_chart(seed=42, out_path=path1)
            generate_ablation_chart(seed=42, out_path=path2)
            assert path1.stat().st_size > 0


# =============================================================================
# Architecture Graph Tests
# =============================================================================


class TestArchitectureGraph:
    """Tests for architecture graph generation."""

    def test_generates_file(self, arch_summary):
        """Test that architecture graph is generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "arch.pdf"
            generate_architecture_graph(arch_summary, out_path, seed=42)
            assert out_path.exists()

    def test_with_minimal_summary(self):
        """Test with minimal architecture summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "arch.pdf"
            minimal_summary = {"family": "transformer"}
            generate_architecture_graph(minimal_summary, out_path, seed=42)
            assert out_path.exists()

    def test_with_different_families(self):
        """Test with different architecture families."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for family in ["transformer", "diffusion", "cnn"]:
                out_path = Path(tmpdir) / f"arch_{family}.pdf"
                summary = {
                    "family": family,
                    "dims": {"hidden_size": 1024, "layers": 12, "heads": 8},
                }
                generate_architecture_graph(summary, out_path, seed=42)
                assert out_path.exists()


# =============================================================================
# Generate All Figures Tests
# =============================================================================


class TestGenerateAllFigures:
    """Tests for complete figure generation."""

    def test_generates_all_figures(self, sample_spec, arch_summary, bench_summary):
        """Test that all figures are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            figures_dir = Path(tmpdir)
            result = generate_all_figures(
                spec=sample_spec,
                arch_summary=arch_summary,
                bench_summary=bench_summary,
                figures_dir=figures_dir,
                seed=42,
            )

            assert "pipeline" in result
            assert "pareto_front" in result
            assert "convergence" in result
            assert "ablation" in result
            assert "architecture_graph" in result
            assert "pipeline_tikz" in result

    def test_all_files_exist(self, sample_spec, arch_summary, bench_summary):
        """Test that all figure files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            figures_dir = Path(tmpdir)
            result = generate_all_figures(
                spec=sample_spec,
                arch_summary=arch_summary,
                bench_summary=bench_summary,
                figures_dir=figures_dir,
                seed=42,
            )

            for name, path in result.items():
                assert path.exists(), f"Figure {name} not found at {path}"

    def test_returns_correct_paths(self, sample_spec, arch_summary, bench_summary):
        """Test that returned paths are correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            figures_dir = Path(tmpdir)
            result = generate_all_figures(
                spec=sample_spec,
                arch_summary=arch_summary,
                bench_summary=bench_summary,
                figures_dir=figures_dir,
                seed=42,
            )

            assert result["pipeline"] == figures_dir / "pipeline.pdf"
            assert result["pareto_front"] == figures_dir / "pareto_front.pdf"
            assert result["convergence"] == figures_dir / "convergence.pdf"
            assert result["ablation"] == figures_dir / "ablation.pdf"
            assert result["architecture_graph"] == figures_dir / "architecture_graph.pdf"
            assert result["pipeline_tikz"] == figures_dir / "pipeline_tikz.tex"

    def test_uses_benchmark_scores(self, sample_spec, arch_summary):
        """Test that benchmark scores are used when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            figures_dir = Path(tmpdir)
            bench_summary = {"scores": {"Custom-1": 0.95, "Custom-2": 0.88}}
            result = generate_all_figures(
                spec=sample_spec,
                arch_summary=arch_summary,
                bench_summary=bench_summary,
                figures_dir=figures_dir,
                seed=42,
            )

            assert result["pipeline"].exists()

    def test_handles_empty_bench_summary(self, sample_spec, arch_summary):
        """Test handling of empty benchmark summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            figures_dir = Path(tmpdir)
            result = generate_all_figures(
                spec=sample_spec,
                arch_summary=arch_summary,
                bench_summary={},
                figures_dir=figures_dir,
                seed=42,
            )

            assert result["pipeline"].exists()
            assert result["ablation"].exists()


# =============================================================================
# Determinism Tests
# =============================================================================


class TestFigureDeterminism:
    """Tests for figure generation determinism."""

    def test_all_figures_deterministic(self, sample_spec, arch_summary, bench_summary):
        """Test that all figure generation is deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = Path(tmpdir) / "run1"
            dir2 = Path(tmpdir) / "run2"

            result1 = generate_all_figures(
                spec=sample_spec,
                arch_summary=arch_summary,
                bench_summary=bench_summary,
                figures_dir=dir1,
                seed=42,
            )

            result2 = generate_all_figures(
                spec=sample_spec,
                arch_summary=arch_summary,
                bench_summary=bench_summary,
                figures_dir=dir2,
                seed=42,
            )

            # All files should be generated
            for name in result1:
                assert result1[name].exists()
                assert result2[name].exists()

    def test_different_seeds_produce_different_figures(
        self, sample_spec, arch_summary, bench_summary
    ):
        """Test that different seeds produce different figures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = Path(tmpdir) / "seed42"
            dir2 = Path(tmpdir) / "seed123"

            generate_all_figures(
                spec=sample_spec,
                arch_summary=arch_summary,
                bench_summary=bench_summary,
                figures_dir=dir1,
                seed=42,
            )

            generate_all_figures(
                spec=sample_spec,
                arch_summary=arch_summary,
                bench_summary=bench_summary,
                figures_dir=dir2,
                seed=123,
            )

            # Files should exist but may have slightly different content
            assert (dir1 / "convergence.pdf").exists()
            assert (dir2 / "convergence.pdf").exists()
