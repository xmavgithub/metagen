"""
Tests for MetaGen Supplementary Materials Generation

Tests for spec reference, architecture examples, hyperparameters,
and reproducibility checklist generation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from metagen.specs.loader import load_spec
from metagen.synth.supplementary_gen import (
    generate_architecture_examples,
    generate_hyperparameters,
    generate_reproducibility_checklist,
    generate_spec_reference,
    generate_supplementary_latex,
    generate_supplementary_markdown,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_spec():
    """Load a sample spec for testing."""
    spec, _ = load_spec(Path("examples/specs/text/text_llm_8b.yaml"))
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
        "components": {
            "encoder": "TransformerEncoder",
            "decoder": "AutoregressiveDecoder",
        },
    }


# =============================================================================
# Spec Reference Tests
# =============================================================================


class TestSpecReference:
    """Tests for spec reference generation."""

    def test_generates_content(self, sample_spec):
        """Test that spec reference generates content."""
        content = generate_spec_reference(sample_spec)
        assert len(content) > 0

    def test_contains_section_header(self, sample_spec):
        """Test that content contains section header."""
        content = generate_spec_reference(sample_spec)
        assert "Spec Language Reference" in content

    def test_contains_modality_info(self, sample_spec):
        """Test that content includes modality information."""
        content = generate_spec_reference(sample_spec)
        # Should mention supported modalities
        assert "text" in content.lower()

    def test_contains_architecture_families(self, sample_spec):
        """Test that content lists architecture families."""
        content = generate_spec_reference(sample_spec)
        assert "transformer" in content.lower()

    def test_contains_constraint_formats(self, sample_spec):
        """Test that content explains constraint formats."""
        content = generate_spec_reference(sample_spec)
        assert "gpu" in content.lower() or "device" in content.lower()

    def test_uses_spec_values(self, sample_spec):
        """Test that generated content uses actual spec values."""
        content = generate_spec_reference(sample_spec)
        assert sample_spec.architecture.family in content


# =============================================================================
# Architecture Examples Tests
# =============================================================================


class TestArchitectureExamples:
    """Tests for architecture examples generation."""

    def test_generates_content(self, sample_spec, arch_summary):
        """Test that architecture examples generate content."""
        content = generate_architecture_examples(sample_spec, arch_summary)
        assert len(content) > 0

    def test_contains_section_header(self, sample_spec, arch_summary):
        """Test that content contains section header."""
        content = generate_architecture_examples(sample_spec, arch_summary)
        assert "Generated Architecture" in content

    def test_contains_blueprint(self, sample_spec, arch_summary):
        """Test that content includes blueprint configuration."""
        content = generate_architecture_examples(sample_spec, arch_summary)
        assert "blueprint" in content.lower()
        assert "dims" in content or "hidden_size" in content

    def test_contains_param_breakdown(self, sample_spec, arch_summary):
        """Test that content includes parameter breakdown."""
        content = generate_architecture_examples(sample_spec, arch_summary)
        assert "Parameter" in content
        assert "7.5" in content or "7.50" in content

    def test_contains_component_list(self, sample_spec, arch_summary):
        """Test that content lists components."""
        content = generate_architecture_examples(sample_spec, arch_summary)
        assert "Embedding" in content or "embedding" in content
        assert "FFN" in content or "Feed-Forward" in content


# =============================================================================
# Hyperparameters Tests
# =============================================================================


class TestHyperparameters:
    """Tests for hyperparameter documentation generation."""

    def test_generates_content(self, sample_spec, arch_summary):
        """Test that hyperparameters generate content."""
        content = generate_hyperparameters(sample_spec, arch_summary)
        assert len(content) > 0

    def test_contains_section_header(self, sample_spec, arch_summary):
        """Test that content contains section header."""
        content = generate_hyperparameters(sample_spec, arch_summary)
        assert "Hyperparameter" in content

    def test_contains_architecture_hyperparams(self, sample_spec, arch_summary):
        """Test that content includes architecture hyperparameters."""
        content = generate_hyperparameters(sample_spec, arch_summary)
        assert "Hidden" in content or "hidden" in content
        assert "layers" in content.lower() or "layer" in content.lower()
        assert "4096" in content  # hidden size

    def test_contains_training_hyperparams(self, sample_spec, arch_summary):
        """Test that content includes training hyperparameters."""
        content = generate_hyperparameters(sample_spec, arch_summary)
        assert "Optimizer" in content or "optimizer" in content
        assert "Learning rate" in content or "learning_rate" in content

    def test_contains_search_hyperparams(self, sample_spec, arch_summary):
        """Test that content includes search hyperparameters."""
        content = generate_hyperparameters(sample_spec, arch_summary)
        assert "Search" in content or "search" in content
        assert "budget" in content.lower() or "candidates" in content.lower()


# =============================================================================
# Reproducibility Checklist Tests
# =============================================================================


class TestReproducibilityChecklist:
    """Tests for reproducibility checklist generation."""

    def test_generates_content(self):
        """Test that checklist generates content."""
        content = generate_reproducibility_checklist()
        assert len(content) > 0

    def test_contains_section_header(self):
        """Test that content contains section header."""
        content = generate_reproducibility_checklist()
        assert "Reproducibility" in content

    def test_contains_checklist_items(self):
        """Test that content contains checklist items."""
        content = generate_reproducibility_checklist()
        assert "checkmark" in content or "itemize" in content

    def test_contains_specification_section(self):
        """Test that content has specification reproducibility section."""
        content = generate_reproducibility_checklist()
        assert "Specification" in content

    def test_contains_training_section(self):
        """Test that content has training reproducibility section."""
        content = generate_reproducibility_checklist()
        assert "Training" in content

    def test_contains_code_availability(self):
        """Test that content mentions code availability."""
        content = generate_reproducibility_checklist()
        assert "Code" in content or "github" in content.lower()

    def test_contains_compute_requirements(self):
        """Test that content includes computational requirements."""
        content = generate_reproducibility_checklist()
        assert "Computational" in content or "compute" in content.lower()


# =============================================================================
# Complete Supplementary LaTeX Tests
# =============================================================================


class TestSupplementaryLatex:
    """Tests for complete supplementary LaTeX generation."""

    def test_generates_file(self, sample_spec, arch_summary):
        """Test that LaTeX file is generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = generate_supplementary_latex(sample_spec, arch_summary, Path(tmpdir))
            assert out_path.exists()
            assert out_path.name == "supplementary.tex"

    def test_file_has_content(self, sample_spec, arch_summary):
        """Test that generated file has content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = generate_supplementary_latex(sample_spec, arch_summary, Path(tmpdir))
            content = out_path.read_text()
            assert len(content) > 1000  # Should be substantial

    def test_contains_all_sections(self, sample_spec, arch_summary):
        """Test that file contains all appendix sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = generate_supplementary_latex(sample_spec, arch_summary, Path(tmpdir))
            content = out_path.read_text()
            assert "Spec Language Reference" in content
            assert "Generated Architecture" in content
            assert "Hyperparameter" in content
            assert "Reproducibility" in content

    def test_is_valid_latex(self, sample_spec, arch_summary):
        """Test that content is valid LaTeX structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = generate_supplementary_latex(sample_spec, arch_summary, Path(tmpdir))
            content = out_path.read_text()
            # Should have section commands
            assert r"\section" in content or r"\subsection" in content
            # Should have tables
            assert r"\begin{table}" in content


# =============================================================================
# Complete Supplementary Markdown Tests
# =============================================================================


class TestSupplementaryMarkdown:
    """Tests for complete supplementary Markdown generation."""

    def test_generates_file(self, sample_spec, arch_summary):
        """Test that Markdown file is generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = generate_supplementary_markdown(sample_spec, arch_summary, Path(tmpdir))
            assert out_path.exists()
            assert out_path.name == "supplementary.md"

    def test_file_has_content(self, sample_spec, arch_summary):
        """Test that generated file has content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = generate_supplementary_markdown(sample_spec, arch_summary, Path(tmpdir))
            content = out_path.read_text()
            assert len(content) > 500  # Should be substantial

    def test_contains_all_appendices(self, sample_spec, arch_summary):
        """Test that file contains all appendix sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = generate_supplementary_markdown(sample_spec, arch_summary, Path(tmpdir))
            content = out_path.read_text()
            assert "Appendix A" in content
            assert "Appendix B" in content
            assert "Appendix C" in content
            assert "Appendix D" in content

    def test_is_valid_markdown(self, sample_spec, arch_summary):
        """Test that content is valid Markdown structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = generate_supplementary_markdown(sample_spec, arch_summary, Path(tmpdir))
            content = out_path.read_text()
            # Should have headers
            assert content.startswith("#")
            # Should have tables
            assert "|" in content
            # Should have code blocks
            assert "```" in content

    def test_contains_checklist(self, sample_spec, arch_summary):
        """Test that markdown contains reproducibility checklist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = generate_supplementary_markdown(sample_spec, arch_summary, Path(tmpdir))
            content = out_path.read_text()
            assert "- [x]" in content  # Checked items
            assert "- [ ]" in content  # Unchecked items


# =============================================================================
# Integration Tests
# =============================================================================


class TestSupplementaryIntegration:
    """Integration tests for supplementary generation."""

    def test_deterministic_generation(self, sample_spec, arch_summary):
        """Test that generation is deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = generate_supplementary_latex(sample_spec, arch_summary, Path(tmpdir) / "run1")
            path2 = generate_supplementary_latex(sample_spec, arch_summary, Path(tmpdir) / "run2")

            assert path1.read_text() == path2.read_text()

    def test_both_formats_generated(self, sample_spec, arch_summary):
        """Test that both LaTeX and Markdown can be generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            latex_path = generate_supplementary_latex(sample_spec, arch_summary, Path(tmpdir))
            md_path = generate_supplementary_markdown(sample_spec, arch_summary, Path(tmpdir))

            assert latex_path.exists()
            assert md_path.exists()
            assert latex_path.read_text() != md_path.read_text()
