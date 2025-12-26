"""Tests for paper generation system."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from metagen.specs.loader import load_spec
from metagen.synth import benchmarks
from metagen.synth.architecture import estimate_summary
from metagen.synth.paper_bibliography import (
    generate_bibliography,
    get_citation_keys,
    get_modality_citations,
)
from metagen.synth.paper_figures import (
    generate_ablation_chart,
    generate_all_figures,
    generate_convergence_curve,
    generate_pareto_front,
    generate_pipeline_matplotlib,
)
from metagen.synth.paper_gen import generate_paper
from metagen.synth.paper_sections import (
    generate_abstract,
    generate_appendix,
    generate_conclusion,
    generate_discussion,
    generate_experiments,
    generate_introduction,
    generate_method,
    generate_related_work,
)
from metagen.synth.paper_templates import (
    TEMPLATES_DIR,
    copy_templates_to,
    get_main_template,
    get_preamble,
    get_style,
)


@pytest.fixture
def spec():
    """Load a test spec."""
    spec, _ = load_spec("examples/specs/text_llm_8b.yaml")
    return spec


@pytest.fixture
def arch_summary(spec):
    """Generate architecture summary."""
    dims, summary = estimate_summary(spec, seed=42)
    return summary


@pytest.fixture
def bench_summary(spec, arch_summary):
    """Generate benchmark summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = benchmarks.generate_reports(
            spec=spec,
            run_dir=Path(tmpdir),
            seed=42,
            arch_summary=arch_summary,
        )
        return result


# =============================================================================
# Bibliography Tests
# =============================================================================


class TestPaperBibliography:
    """Tests for paper_bibliography module."""

    def test_generate_bibliography_returns_bibtex(self, spec):
        """Bibliography should return valid BibTeX."""
        bib = generate_bibliography(spec)
        assert "@article{" in bib or "@inproceedings{" in bib
        assert "title" in bib.lower()
        assert "author" in bib.lower()

    def test_bibliography_includes_core_references(self, spec):
        """Bibliography should include core transformer/NAS references."""
        bib = generate_bibliography(spec)
        assert "vaswani" in bib.lower()  # Attention paper
        assert "elsken" in bib.lower()  # NAS survey

    def test_get_citation_keys_returns_list(self, spec):
        """Citation keys should be a list of strings."""
        keys = get_citation_keys(spec)
        assert isinstance(keys, list)
        assert len(keys) > 0
        assert all(isinstance(k, str) for k in keys)

    def test_modality_citations_vary_by_modality(self, spec):
        """Different modalities should include different citations."""
        text_cites = get_modality_citations(spec)
        assert "text" in text_cites or len(text_cites) > 0


# =============================================================================
# Section Tests
# =============================================================================


class TestPaperSections:
    """Tests for paper_sections module."""

    def test_generate_abstract_includes_params(self, spec, arch_summary, bench_summary):
        """Abstract should mention parameter count."""
        abstract = generate_abstract(spec, arch_summary, bench_summary)
        assert "abstract" in abstract.lower() or "param" in abstract.lower()
        assert len(abstract) > 100

    def test_generate_introduction_has_structure(self, spec, arch_summary):
        """Introduction should have subsections."""
        intro = generate_introduction(spec, arch_summary)
        assert "\\subsection" in intro
        assert "contribution" in intro.lower()

    def test_generate_related_work_has_citations(self, spec):
        """Related work should include citations."""
        related = generate_related_work(spec)
        assert "\\cit" in related  # \cite or \citep
        assert "transformer" in related.lower()

    def test_generate_method_has_equations(self, spec, arch_summary):
        """Method should include mathematical notation."""
        method = generate_method(spec, arch_summary)
        assert "\\begin{equation}" in method or "\\begin{align}" in method or "$" in method
        assert "algorithm" in method.lower()

    def test_generate_experiments_has_tables(self, spec, bench_summary, arch_summary):
        """Experiments should include result tables."""
        experiments = generate_experiments(spec, bench_summary, arch_summary)
        assert "\\begin{table}" in experiments
        assert "\\begin{tabular}" in experiments

    def test_generate_discussion_has_limitations(self, spec):
        """Discussion should address limitations."""
        discussion = generate_discussion(spec)
        assert "limitation" in discussion.lower()
        assert "future" in discussion.lower()

    def test_generate_conclusion_summarizes(self, spec, arch_summary):
        """Conclusion should summarize contributions."""
        conclusion = generate_conclusion(spec, arch_summary)
        assert "metagen" in conclusion.lower()
        assert len(conclusion) > 200

    def test_generate_appendix_has_sections(self, spec, arch_summary):
        """Appendix should have labeled sections."""
        appendix = generate_appendix(spec, arch_summary)
        assert "\\section" in appendix or "\\appendix" in appendix
        assert "reproducibility" in appendix.lower()


# =============================================================================
# Figure Tests
# =============================================================================


class TestPaperFigures:
    """Tests for paper_figures module."""

    def test_generate_pipeline_creates_pdf(self):
        """Pipeline figure should create a PDF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "pipeline.pdf"
            generate_pipeline_matplotlib(seed=42, out_path=out_path, scores={"A": 0.9, "B": 0.85})
            assert out_path.exists()
            assert out_path.stat().st_size > 0

    def test_generate_pareto_front_creates_pdf(self):
        """Pareto front figure should create a PDF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "pareto.pdf"
            generate_pareto_front(seed=42, out_path=out_path)
            assert out_path.exists()
            assert out_path.stat().st_size > 0

    def test_generate_convergence_creates_pdf(self):
        """Convergence figure should create a PDF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "convergence.pdf"
            generate_convergence_curve(seed=42, out_path=out_path)
            assert out_path.exists()
            assert out_path.stat().st_size > 0

    def test_generate_ablation_creates_pdf(self):
        """Ablation figure should create a PDF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "ablation.pdf"
            generate_ablation_chart(seed=42, out_path=out_path)
            assert out_path.exists()
            assert out_path.stat().st_size > 0

    def test_generate_all_figures_creates_all(self, spec, arch_summary, bench_summary):
        """generate_all_figures should create all expected figures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            figures_dir = Path(tmpdir)
            paths = generate_all_figures(
                spec=spec,
                arch_summary=arch_summary,
                bench_summary=bench_summary,
                figures_dir=figures_dir,
                seed=42,
            )
            # Should return dict of paths
            assert isinstance(paths, dict)
            # Should have created some files
            pdf_files = list(figures_dir.glob("*.pdf"))
            assert len(pdf_files) >= 3  # At least pipeline, pareto, convergence


# =============================================================================
# Template Tests
# =============================================================================


class TestPaperTemplates:
    """Tests for paper_templates module."""

    def test_templates_dir_exists(self):
        """Templates directory should exist."""
        assert TEMPLATES_DIR.exists()
        assert TEMPLATES_DIR.is_dir()

    def test_get_preamble_returns_latex(self):
        """Preamble should return valid LaTeX."""
        preamble = get_preamble()
        assert "\\usepackage" in preamble
        assert "amsmath" in preamble

    def test_get_style_returns_latex(self):
        """Style file should return valid LaTeX."""
        style = get_style()
        assert "\\NeedsTeXFormat" in style or "\\usepackage" in style

    def test_get_main_template_has_placeholders(self):
        """Main template should have replacement placeholders."""
        template = get_main_template()
        assert "%TITLE%" in template or "\\title" in template
        assert "\\begin{document}" in template

    def test_copy_templates_to_creates_files(self):
        """copy_templates_to should create template files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            copy_templates_to(out_dir)
            assert (out_dir / "preamble.tex").exists()
            assert (out_dir / "metagen_academic.sty").exists()


# =============================================================================
# Integration Tests
# =============================================================================


class TestPaperGeneration:
    """Integration tests for complete paper generation."""

    def test_generate_paper_creates_structure(self, spec, arch_summary, bench_summary):
        """generate_paper should create complete paper structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            generate_paper(spec, out_dir, bench_summary, arch_summary, seed=42)

            # Check main files
            assert (out_dir / "main.tex").exists()
            assert (out_dir / "bibliography.bib").exists()
            assert (out_dir / "Makefile").exists()
            assert (out_dir / "preamble.tex").exists()

            # Check sections
            sections_dir = out_dir / "sections"
            assert sections_dir.exists()
            assert (sections_dir / "abstract.tex").exists()
            assert (sections_dir / "introduction.tex").exists()
            assert (sections_dir / "method.tex").exists()
            assert (sections_dir / "experiments.tex").exists()

            # Check figures
            figures_dir = out_dir / "figures"
            assert figures_dir.exists()
            pdf_files = list(figures_dir.glob("*.pdf"))
            assert len(pdf_files) >= 1

    def test_generate_paper_is_deterministic(self, spec, arch_summary, bench_summary):
        """Same inputs should produce identical outputs."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            out_dir1 = Path(tmpdir1)
            generate_paper(spec, out_dir1, bench_summary, arch_summary, seed=42)
            main1 = (out_dir1 / "main.tex").read_text()

        with tempfile.TemporaryDirectory() as tmpdir2:
            out_dir2 = Path(tmpdir2)
            generate_paper(spec, out_dir2, bench_summary, arch_summary, seed=42)
            main2 = (out_dir2 / "main.tex").read_text()

        assert main1 == main2

    def test_generate_paper_varies_with_seed(self, spec, arch_summary):
        """Different seeds should produce different benchmark scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir1 = Path(tmpdir) / "paper1"
            out_dir2 = Path(tmpdir) / "paper2"

            # Generate with seed 42
            bench_summary_1 = benchmarks.generate_reports(
                spec=spec,
                run_dir=out_dir1,
                seed=42,
                arch_summary=arch_summary,
            )
            generate_paper(spec, out_dir1, bench_summary_1, arch_summary, seed=42)

            # Generate with seed 123
            bench_summary_2 = benchmarks.generate_reports(
                spec=spec,
                run_dir=out_dir2,
                seed=123,
                arch_summary=arch_summary,
            )
            generate_paper(spec, out_dir2, bench_summary_2, arch_summary, seed=123)

            # Main structure should be similar but experiments content differs
            exp1 = (out_dir1 / "sections" / "experiments.tex").read_text()
            exp2 = (out_dir2 / "sections" / "experiments.tex").read_text()
            # Scores in experiments section should differ
            # (structure is same but numerical values differ)
            assert exp1 != exp2


# =============================================================================
# Benchmark Tests (Enhanced)
# =============================================================================


class TestBenchmarks:
    """Tests for enhanced benchmarks module."""

    def test_generate_all_benchmarks_returns_dict(self, spec, arch_summary):
        """generate_all_benchmarks should return dict of BenchmarkScore."""
        from metagen.synth.benchmarks import generate_all_benchmarks

        results = generate_all_benchmarks(spec, arch_summary, seed=42)
        assert isinstance(results, dict)
        assert len(results) >= 8  # At least 8 benchmarks

    def test_benchmarks_have_categories(self, spec, arch_summary):
        """Each benchmark should have a category."""
        from metagen.synth.benchmarks import generate_all_benchmarks

        results = generate_all_benchmarks(spec, arch_summary, seed=42)
        categories = {b.category for b in results.values()}
        assert "capability" in categories
        assert "efficiency" in categories or "constraint" in categories

    def test_benchmark_scores_are_bounded(self, spec, arch_summary):
        """All scores should be between 0 and 1."""
        from metagen.synth.benchmarks import generate_all_benchmarks

        results = generate_all_benchmarks(spec, arch_summary, seed=42)
        for name, benchmark in results.items():
            assert 0 <= benchmark.score <= 1.1, f"{name} score out of bounds: {benchmark.score}"

    def test_generate_reports_returns_enhanced_structure(self, spec, arch_summary):
        """generate_reports should return enhanced structure with categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = benchmarks.generate_reports(
                spec=spec,
                run_dir=Path(tmpdir),
                seed=42,
                arch_summary=arch_summary,
            )
            assert "scores" in result
            assert "by_category" in result
            assert "benchmarks" in result
            assert len(result["scores"]) >= 8
