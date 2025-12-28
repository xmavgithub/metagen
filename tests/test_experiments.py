"""
Tests for MetaGen Experiments Module

Tests baselines, experiment runner, and ablation studies.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from metagen.experiments.ablations import (
    STANDARD_ABLATIONS,
    AblationConfig,
    AblationResult,
    AblationStudy,
    run_ablation,
)
from metagen.experiments.baselines import (
    BaselineMetrics,
    GPT2Baseline,
    MetaGenBaseline,
    RandomSearchBaseline,
    ResNetBaseline,
    SingleObjectiveBaseline,
    UNetBaseline,
    get_baseline,
    list_baselines,
)
from metagen.experiments.runner import (
    ExperimentResult,
    ExperimentRunner,
    ExperimentSuite,
    run_experiment,
)
from metagen.specs.loader import load_spec

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_spec():
    """Load a sample spec for testing."""
    spec, _ = load_spec(Path("examples/specs/text/text_llm_8b.yaml"))
    return spec


@pytest.fixture
def image_spec():
    """Load an image spec for testing."""
    spec, _ = load_spec(Path("examples/specs/image/image_diffusion_sdxl_like.yaml"))
    return spec


# =============================================================================
# Baseline Metrics Tests
# =============================================================================


class TestBaselineMetrics:
    """Tests for BaselineMetrics dataclass."""

    def test_creation(self):
        """Test creating baseline metrics."""
        metrics = BaselineMetrics(
            accuracy=0.85,
            params_million=100.0,
            latency_ms=50.0,
            memory_mb=400.0,
            flops_gflops=200.0,
            training_hours=24.0,
        )
        assert metrics.accuracy == 0.85
        assert metrics.params_million == 100.0

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = BaselineMetrics(
            accuracy=0.85,
            params_million=100.0,
            latency_ms=50.0,
            memory_mb=400.0,
            flops_gflops=200.0,
            training_hours=24.0,
            extra={"custom": "value"},
        )
        d = metrics.to_dict()
        assert d["accuracy"] == 0.85
        assert d["custom"] == "value"

    def test_extra_fields(self):
        """Test extra fields in metrics."""
        metrics = BaselineMetrics(
            accuracy=0.85,
            params_million=100.0,
            latency_ms=50.0,
            memory_mb=400.0,
            flops_gflops=200.0,
            training_hours=24.0,
            extra={"layers": 12, "heads": 8},
        )
        assert metrics.extra["layers"] == 12


# =============================================================================
# Manual Architecture Baseline Tests
# =============================================================================


class TestGPT2Baseline:
    """Tests for GPT-2 baseline."""

    def test_default_size(self):
        """Test default GPT-2 size."""
        baseline = GPT2Baseline()
        assert baseline.size == "small"
        assert baseline.config["params"] == 117

    def test_different_sizes(self):
        """Test different GPT-2 sizes."""
        for size in ["small", "medium", "large", "xl"]:
            baseline = GPT2Baseline(size=size)
            assert baseline.size == size

    def test_invalid_size_fallback(self):
        """Test invalid size falls back to small."""
        baseline = GPT2Baseline(size="invalid")
        assert baseline.size == "small"

    def test_evaluate(self, sample_spec):
        """Test evaluation produces valid metrics."""
        baseline = GPT2Baseline(size="medium")
        metrics = baseline.evaluate(sample_spec, seed=42)

        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.params_million > 0
        assert metrics.latency_ms > 0
        assert metrics.memory_mb > 0

    def test_deterministic_evaluation(self, sample_spec):
        """Test evaluation is deterministic."""
        baseline = GPT2Baseline()
        m1 = baseline.evaluate(sample_spec, seed=42)
        m2 = baseline.evaluate(sample_spec, seed=42)

        assert m1.accuracy == m2.accuracy
        assert m1.params_million == m2.params_million

    def test_different_seeds(self, sample_spec):
        """Test different seeds produce different results."""
        baseline = GPT2Baseline()
        m1 = baseline.evaluate(sample_spec, seed=42)
        m2 = baseline.evaluate(sample_spec, seed=123)

        # Results should differ (statistically)
        assert m1.accuracy != m2.accuracy or m1.latency_ms != m2.latency_ms

    def test_architecture_summary(self, sample_spec):
        """Test architecture summary."""
        baseline = GPT2Baseline(size="large")
        summary = baseline.get_architecture_summary(sample_spec)

        assert summary["name"] == "gpt2"
        assert summary["size"] == "large"
        assert summary["layers"] == 36


class TestResNetBaseline:
    """Tests for ResNet baseline."""

    def test_default_depth(self):
        """Test default ResNet depth."""
        baseline = ResNetBaseline()
        assert baseline.depth == "50"

    def test_different_depths(self):
        """Test different ResNet depths."""
        for depth in ["18", "34", "50", "101", "152"]:
            baseline = ResNetBaseline(depth=depth)
            assert baseline.depth == depth

    def test_evaluate(self, image_spec):
        """Test evaluation produces valid metrics."""
        baseline = ResNetBaseline(depth="50")
        metrics = baseline.evaluate(image_spec, seed=42)

        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.params_million == 25.6  # ResNet-50


class TestUNetBaseline:
    """Tests for U-Net baseline."""

    def test_default_size(self):
        """Test default U-Net size."""
        baseline = UNetBaseline()
        assert baseline.size == "base"

    def test_different_sizes(self):
        """Test different U-Net sizes."""
        for size in ["tiny", "small", "base", "large"]:
            baseline = UNetBaseline(size=size)
            assert baseline.size == size

    def test_evaluate(self, image_spec):
        """Test evaluation produces valid metrics."""
        baseline = UNetBaseline(size="small")
        metrics = baseline.evaluate(image_spec, seed=42)

        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.params_million == 31.4


# =============================================================================
# Search Strategy Baseline Tests
# =============================================================================


class TestRandomSearchBaseline:
    """Tests for random search baseline."""

    def test_default_samples(self):
        """Test default number of samples."""
        baseline = RandomSearchBaseline()
        assert baseline.num_samples == 100

    def test_custom_samples(self):
        """Test custom number of samples."""
        baseline = RandomSearchBaseline(num_samples=500)
        assert baseline.num_samples == 500

    def test_evaluate(self, sample_spec):
        """Test evaluation produces valid metrics."""
        baseline = RandomSearchBaseline(num_samples=200)
        metrics = baseline.evaluate(sample_spec, seed=42)

        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.extra["samples_evaluated"] == 200


class TestSingleObjectiveBaseline:
    """Tests for single-objective NAS baseline."""

    def test_default_objective(self):
        """Test default objective."""
        baseline = SingleObjectiveBaseline()
        assert baseline.objective == "accuracy"

    def test_custom_objective(self):
        """Test custom objective."""
        baseline = SingleObjectiveBaseline(objective="latency")
        assert baseline.objective == "latency"

    def test_evaluate(self, sample_spec):
        """Test evaluation produces valid metrics."""
        baseline = SingleObjectiveBaseline(search_budget=300)
        metrics = baseline.evaluate(sample_spec, seed=42)

        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.extra["search_budget"] == 300


class TestMetaGenBaseline:
    """Tests for MetaGen baseline."""

    def test_default_budget(self):
        """Test default search budget."""
        baseline = MetaGenBaseline()
        assert baseline.search_budget == 200

    def test_evaluate(self, sample_spec):
        """Test evaluation produces valid metrics."""
        baseline = MetaGenBaseline(search_budget=100)
        metrics = baseline.evaluate(sample_spec, seed=42)

        assert 0.0 <= metrics.accuracy <= 1.0
        assert "pareto_solutions" in metrics.extra


# =============================================================================
# Baseline Registry Tests
# =============================================================================


class TestBaselineRegistry:
    """Tests for baseline registry."""

    def test_list_baselines(self):
        """Test listing available baselines."""
        baselines = list_baselines()
        assert "gpt2" in baselines
        assert "resnet" in baselines
        assert "metagen" in baselines

    def test_get_baseline(self):
        """Test getting baseline by name."""
        baseline = get_baseline("gpt2")
        assert isinstance(baseline, GPT2Baseline)

    def test_get_baseline_with_kwargs(self):
        """Test getting baseline with configuration."""
        baseline = get_baseline("gpt2", size="large")
        assert baseline.size == "large"

    def test_get_unknown_baseline(self):
        """Test getting unknown baseline raises error."""
        with pytest.raises(ValueError, match="Unknown baseline"):
            get_baseline("unknown_baseline")


# =============================================================================
# Experiment Result Tests
# =============================================================================


class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_creation(self):
        """Test creating experiment result."""
        metrics = BaselineMetrics(
            accuracy=0.85,
            params_million=100.0,
            latency_ms=50.0,
            memory_mb=400.0,
            flops_gflops=200.0,
            training_hours=24.0,
        )
        result = ExperimentResult(
            spec_name="test-spec",
            baseline_name="gpt2",
            seed=42,
            metrics=metrics,
            architecture_summary={"layers": 12},
        )
        assert result.spec_name == "test-spec"
        assert result.baseline_name == "gpt2"

    def test_to_dict(self):
        """Test converting result to dictionary."""
        metrics = BaselineMetrics(
            accuracy=0.85,
            params_million=100.0,
            latency_ms=50.0,
            memory_mb=400.0,
            flops_gflops=200.0,
            training_hours=24.0,
        )
        result = ExperimentResult(
            spec_name="test-spec",
            baseline_name="gpt2",
            seed=42,
            metrics=metrics,
            architecture_summary={"layers": 12},
        )
        d = result.to_dict()
        assert d["spec_name"] == "test-spec"
        assert d["metrics"]["accuracy"] == 0.85


# =============================================================================
# Experiment Suite Tests
# =============================================================================


class TestExperimentSuite:
    """Tests for ExperimentSuite."""

    def test_add_result(self):
        """Test adding results to suite."""
        suite = ExperimentSuite(name="test-suite")
        metrics = BaselineMetrics(
            accuracy=0.85,
            params_million=100.0,
            latency_ms=50.0,
            memory_mb=400.0,
            flops_gflops=200.0,
            training_hours=24.0,
        )
        result = ExperimentResult(
            spec_name="test-spec",
            baseline_name="gpt2",
            seed=42,
            metrics=metrics,
            architecture_summary={},
        )
        suite.add_result(result)
        assert len(suite.results) == 1

    def test_get_results_by_spec(self):
        """Test filtering results by spec."""
        suite = ExperimentSuite(name="test-suite")
        for spec_name in ["spec1", "spec2", "spec1"]:
            metrics = BaselineMetrics(0.85, 100.0, 50.0, 400.0, 200.0, 24.0)
            result = ExperimentResult(spec_name, "gpt2", 42, metrics, {})
            suite.add_result(result)

        spec1_results = suite.get_results_by_spec("spec1")
        assert len(spec1_results) == 2

    def test_generate_comparison_table(self):
        """Test generating comparison table."""
        suite = ExperimentSuite(name="test-suite")
        metrics = BaselineMetrics(0.85, 100.0, 50.0, 400.0, 200.0, 24.0)
        result = ExperimentResult("test-spec", "gpt2", 42, metrics, {})
        suite.add_result(result)

        table = suite.generate_comparison_table()
        assert "test-spec" in table
        assert "gpt2" in table
        assert "0.8500" in table


# =============================================================================
# Experiment Runner Tests
# =============================================================================


class TestExperimentRunner:
    """Tests for ExperimentRunner."""

    def test_run_single(self, sample_spec):
        """Test running single experiment."""
        runner = ExperimentRunner()
        baseline = GPT2Baseline()
        result = runner.run_single(sample_spec, baseline, seed=42)

        assert result.spec_name == sample_spec.name
        assert result.baseline_name == "gpt2"
        assert result.metrics.accuracy > 0

    def test_run_suite(self):
        """Test running experiment suite."""
        runner = ExperimentRunner()
        suite = runner.run_suite(
            spec_paths=["examples/specs/text/text_llm_8b.yaml"],
            baselines=["gpt2", "random_search"],
            num_runs=2,
            seed=42,
        )

        assert suite.name == "experiment_suite"
        assert len(suite.results) == 4  # 1 spec * 2 baselines * 2 runs

    def test_run_suite_with_output(self):
        """Test running suite with output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(output_dir=Path(tmpdir))
            _suite = runner.run_suite(
                spec_paths=["examples/specs/text/text_llm_8b.yaml"],
                baselines=["gpt2"],
                num_runs=1,
                seed=42,
                suite_name="test_suite",
            )

            assert (Path(tmpdir) / "test_suite.json").exists()
            assert (Path(tmpdir) / "test_suite_report.md").exists()


class TestRunExperiment:
    """Tests for run_experiment convenience function."""

    def test_run_experiment(self):
        """Test running experiment on single spec."""
        suite = run_experiment(
            "examples/specs/text/text_llm_8b.yaml",
            baselines=["gpt2", "metagen"],
            seed=42,
        )
        assert len(suite.results) == 2


# =============================================================================
# Ablation Study Tests
# =============================================================================


class TestAblationConfig:
    """Tests for AblationConfig."""

    def test_creation(self):
        """Test creating ablation config."""
        config = AblationConfig(
            name="test",
            description="Test ablation",
            disabled_components=["component1"],
            score_penalty=0.05,
        )
        assert config.name == "test"
        assert config.score_penalty == 0.05


class TestAblationResult:
    """Tests for AblationResult."""

    def test_creation(self):
        """Test creating ablation result."""
        result = AblationResult(
            config_name="full",
            base_score=0.92,
            ablated_score=0.92,
            delta=0.0,
            relative_drop=0.0,
            disabled_components=[],
        )
        assert result.base_score == 0.92
        assert result.delta == 0.0

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = AblationResult(
            config_name="test",
            base_score=0.92,
            ablated_score=0.85,
            delta=0.07,
            relative_drop=0.076,
            disabled_components=["comp1"],
        )
        d = result.to_dict()
        assert d["config_name"] == "test"
        assert d["delta"] == 0.07


class TestAblationStudy:
    """Tests for AblationStudy."""

    def test_standard_ablations_exist(self):
        """Test standard ablations are defined."""
        assert len(STANDARD_ABLATIONS) > 0
        assert any(a.name == "full" for a in STANDARD_ABLATIONS)

    def test_run_ablation(self, sample_spec):
        """Test running ablation study."""
        study = AblationStudy(seed=42)
        results = study.run(sample_spec, base_score=0.92)

        assert len(results) == len(STANDARD_ABLATIONS)
        assert results[0].config_name == "full"
        assert results[0].ablated_score == 0.92  # Full system has no penalty

    def test_ablation_scores_decrease(self, sample_spec):
        """Test ablated scores are lower than full."""
        study = AblationStudy(seed=42)
        results = study.run(sample_spec, base_score=0.92)

        full_score = results[0].ablated_score
        for result in results[1:]:
            assert result.ablated_score <= full_score + 0.02  # Allow small noise

    def test_deterministic(self, sample_spec):
        """Test ablation study is deterministic."""
        study1 = AblationStudy(seed=42)
        study2 = AblationStudy(seed=42)

        r1 = study1.run(sample_spec, base_score=0.92)
        r2 = study2.run(sample_spec, base_score=0.92)

        for res1, res2 in zip(r1, r2, strict=True):
            assert res1.ablated_score == res2.ablated_score

    def test_generate_table(self, sample_spec):
        """Test generating markdown table."""
        study = AblationStudy(seed=42)
        results = study.run(sample_spec, base_score=0.92)
        table = study.generate_table(results)

        assert "# Ablation Study Results" in table
        assert "full" in table
        assert "Key Findings" in table

    def test_generate_latex_table(self, sample_spec):
        """Test generating LaTeX table."""
        study = AblationStudy(seed=42)
        results = study.run(sample_spec, base_score=0.92)
        table = study.generate_latex_table(results)

        assert "\\begin{table}" in table
        assert "\\toprule" in table
        assert "\\end{table}" in table


class TestRunAblation:
    """Tests for run_ablation convenience function."""

    def test_run_ablation(self, sample_spec):
        """Test running ablation study."""
        results = run_ablation(sample_spec, base_score=0.90, seed=42)
        assert len(results) > 0

    def test_run_ablation_with_output(self, sample_spec):
        """Test running ablation with output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _results = run_ablation(
                sample_spec,
                base_score=0.90,
                seed=42,
                output_dir=Path(tmpdir),
            )

            assert (Path(tmpdir) / "ablation_results.json").exists()
            assert (Path(tmpdir) / "ablation_report.md").exists()
            assert (Path(tmpdir) / "ablation_table.tex").exists()


# =============================================================================
# Integration Tests
# =============================================================================


class TestExperimentsIntegration:
    """Integration tests for experiments module."""

    def test_full_experiment_workflow(self):
        """Test complete experiment workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run experiments
            runner = ExperimentRunner(output_dir=Path(tmpdir))
            suite = runner.run_suite(
                spec_paths=["examples/specs/text/text_llm_8b.yaml"],
                baselines=["gpt2", "random_search", "metagen"],
                num_runs=2,
                seed=42,
            )

            # Check results
            assert len(suite.results) == 6

            # MetaGen should perform well
            metagen_results = suite.get_results_by_baseline("metagen")
            avg_accuracy = sum(r.metrics.accuracy for r in metagen_results) / len(metagen_results)
            assert avg_accuracy > 0.8

    def test_ablation_integration(self, sample_spec):
        """Test ablation study integration."""
        # First run experiment to get base score
        suite = run_experiment(
            "examples/specs/text/text_llm_8b.yaml",
            baselines=["metagen"],
            seed=42,
        )
        base_score = suite.results[0].metrics.accuracy

        # Run ablation study
        results = run_ablation(sample_spec, base_score=base_score, seed=42)

        # Verify ablations show expected patterns
        arch_search_ablation = next(r for r in results if r.config_name == "-arch_search")
        assert arch_search_ablation.delta > 0.05  # Significant impact
