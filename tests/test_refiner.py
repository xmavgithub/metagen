"""
Tests for MetaGen Architecture Refiner

This module tests the ArchitectureRefiner class which iteratively
refines architecture dimensions based on evaluation feedback.

Author: MetaGen Team
"""

from __future__ import annotations

import pytest

from metagen.automl.candidates import CandidateArchitecture
from metagen.automl.refiner import (
    ArchitectureRefiner,
)
from metagen.specs.schema import ModelSpec


class TestRefinerInitialization:
    """Tests for ArchitectureRefiner initialization."""

    def test_default_initialization(self) -> None:
        """Test refiner initializes with defaults."""
        refiner = ArchitectureRefiner()
        assert refiner.seed == 42
        assert refiner.improvement_threshold == 0.001
        assert refiner.patience == 2

    def test_custom_initialization(self) -> None:
        """Test refiner initializes with custom values."""
        refiner = ArchitectureRefiner(seed=123, improvement_threshold=0.01, patience=5)
        assert refiner.seed == 123
        assert refiner.improvement_threshold == 0.01
        assert refiner.patience == 5

    def test_patience_must_be_positive(self) -> None:
        """Test that patience < 1 raises ValueError."""
        with pytest.raises(ValueError, match="patience must be >= 1"):
            ArchitectureRefiner(patience=0)

    def test_patience_negative_raises(self) -> None:
        """Test that negative patience raises ValueError."""
        with pytest.raises(ValueError, match="patience must be >= 1"):
            ArchitectureRefiner(patience=-1)


class TestRefineValidation:
    """Tests for refine method input validation."""

    def test_iterations_must_be_positive(self) -> None:
        """Test that iterations < 1 raises ValueError."""
        spec = ModelSpec(name="test_spec")
        refiner = ArchitectureRefiner()

        with pytest.raises(ValueError, match="iterations must be >= 1"):
            refiner.refine(spec, iterations=0)

    def test_candidates_per_iteration_must_be_positive(self) -> None:
        """Test that candidates_per_iteration < 1 raises ValueError."""
        spec = ModelSpec(name="test_spec")
        refiner = ArchitectureRefiner()

        with pytest.raises(ValueError, match="candidates_per_iteration must be >= 1"):
            refiner.refine(spec, iterations=1, candidates_per_iteration=0)


class TestRefinerIterativeImprovement:
    """Tests for iterative improvement behavior."""

    def test_refiner_iterative_improvement(self) -> None:
        """Test that scores improve or stay same over iterations."""
        spec = ModelSpec(name="refiner_improvement")
        refiner = ArchitectureRefiner(seed=101, improvement_threshold=0.0, patience=10)
        history = refiner.refine(spec, iterations=4, candidates_per_iteration=4)

        scores = [iteration.best.score for iteration in history.iterations]

        assert len(scores) >= 1
        assert scores == sorted(scores)
        assert history.best.score == scores[-1]

    def test_single_iteration(self) -> None:
        """Test refinement with single iteration."""
        spec = ModelSpec(name="single_iteration")
        refiner = ArchitectureRefiner(seed=42)
        history = refiner.refine(spec, iterations=1, candidates_per_iteration=3)

        assert len(history.iterations) == 1
        assert len(history.iterations[0].candidates) == 3

    def test_single_candidate_per_iteration(self) -> None:
        """Test refinement with one candidate per iteration."""
        spec = ModelSpec(name="single_candidate")
        refiner = ArchitectureRefiner(seed=42, patience=5)
        history = refiner.refine(spec, iterations=3, candidates_per_iteration=1)

        # With only 1 candidate (the current best), should converge quickly
        assert len(history.iterations) >= 1
        for iteration in history.iterations:
            assert len(iteration.candidates) == 1


class TestRefinerConvergence:
    """Tests for convergence behavior."""

    def test_convergence_with_low_patience(self) -> None:
        """Test that refiner converges with patience=1."""
        spec = ModelSpec(name="convergence_test")
        refiner = ArchitectureRefiner(seed=42, improvement_threshold=1.0, patience=1)
        history = refiner.refine(spec, iterations=10, candidates_per_iteration=3)

        # With high threshold and low patience, should converge early
        assert history.converged is True
        assert len(history.iterations) < 10

    def test_no_convergence_with_high_patience(self) -> None:
        """Test that refiner doesn't converge with high patience."""
        spec = ModelSpec(name="no_convergence_test")
        refiner = ArchitectureRefiner(seed=42, improvement_threshold=0.0, patience=20)
        history = refiner.refine(spec, iterations=5, candidates_per_iteration=3)

        # With threshold=0 and high patience, shouldn't converge
        assert history.converged is False
        assert len(history.iterations) == 5


class TestRefinerDeterminism:
    """Tests for deterministic behavior."""

    def test_same_seed_produces_same_results(self) -> None:
        """Test that same seed produces identical results."""
        spec = ModelSpec(name="determinism_test")

        refiner1 = ArchitectureRefiner(seed=42)
        history1 = refiner1.refine(spec, iterations=3, candidates_per_iteration=3)

        refiner2 = ArchitectureRefiner(seed=42)
        history2 = refiner2.refine(spec, iterations=3, candidates_per_iteration=3)

        assert history1.best.dims == history2.best.dims
        assert history1.best.score == history2.best.score
        assert len(history1.iterations) == len(history2.iterations)

    def test_different_seeds_produce_different_results(self) -> None:
        """Test that different seeds produce different results."""
        spec = ModelSpec(name="different_seeds")

        refiner1 = ArchitectureRefiner(seed=42)
        history1 = refiner1.refine(spec, iterations=5, candidates_per_iteration=10)

        refiner2 = ArchitectureRefiner(seed=12345)
        history2 = refiner2.refine(spec, iterations=5, candidates_per_iteration=10)

        # Collect all candidates from both runs
        all_dims_1 = [c.dims for it in history1.iterations for c in it.candidates]
        all_dims_2 = [c.dims for it in history2.iterations for c in it.candidates]

        # With different seeds, at least some candidates should differ
        # Compare the sets of unique dimensions
        unique_dims_1 = {tuple(sorted(d.items())) for d in all_dims_1}
        unique_dims_2 = {tuple(sorted(d.items())) for d in all_dims_2}
        assert unique_dims_1 != unique_dims_2

    def test_seed_override_in_refine(self) -> None:
        """Test that seed parameter in refine() overrides instance seed."""
        spec = ModelSpec(name="seed_override")
        refiner = ArchitectureRefiner(seed=42)

        history1 = refiner.refine(spec, iterations=2, candidates_per_iteration=3, seed=100)
        history2 = refiner.refine(spec, iterations=2, candidates_per_iteration=3, seed=100)

        assert history1.best.dims == history2.best.dims
        assert history1.seed == 100
        assert history2.seed == 100


class TestRefinerMutation:
    """Tests for dimension mutation behavior."""

    def test_mutation_respects_constraints(self) -> None:
        """Test that mutations produce valid dimensions."""
        spec = ModelSpec(name="mutation_constraints")
        refiner = ArchitectureRefiner(seed=42)
        history = refiner.refine(spec, iterations=5, candidates_per_iteration=10, mutation_rate=1.0)

        for iteration in history.iterations:
            for candidate in iteration.candidates:
                dims = candidate.dims
                assert dims["hidden_size"] >= 64
                assert dims["hidden_size"] % 64 == 0
                assert dims["layers"] >= 1
                assert dims["heads"] >= 1
                assert dims["hidden_size"] % dims["heads"] == 0


class TestRefinementHistory:
    """Tests for RefinementHistory dataclass."""

    def test_history_contains_all_metadata(self) -> None:
        """Test that history contains all expected metadata."""
        spec = ModelSpec(name="metadata_test")
        refiner = ArchitectureRefiner(seed=42)
        history = refiner.refine(spec, iterations=3, candidates_per_iteration=4)

        assert history.spec_name == "metadata_test"
        assert history.seed == 42
        assert history.candidates_per_iteration == 4
        assert history.max_iterations == 3
        assert isinstance(history.best, CandidateArchitecture)

    def test_history_to_dict(self) -> None:
        """Test serialization to dictionary."""
        spec = ModelSpec(name="serialization_test")
        refiner = ArchitectureRefiner(seed=42)
        history = refiner.refine(spec, iterations=2, candidates_per_iteration=3)

        result = history.to_dict()

        assert result["spec_name"] == "serialization_test"
        assert result["seed"] == 42
        assert result["candidates_per_iteration"] == 3
        assert result["max_iterations"] == 2
        assert isinstance(result["iterations"], list)
        assert isinstance(result["best"], dict)
        assert "converged" in result


class TestRefinementIteration:
    """Tests for RefinementIteration dataclass."""

    def test_iteration_to_dict(self) -> None:
        """Test serialization of iteration to dictionary."""
        spec = ModelSpec(name="iteration_serialization")
        refiner = ArchitectureRefiner(seed=42)
        history = refiner.refine(spec, iterations=1, candidates_per_iteration=3)

        iteration = history.iterations[0]
        result = iteration.to_dict()

        assert result["iteration"] == 1
        assert isinstance(result["candidates"], list)
        assert len(result["candidates"]) == 3
        assert isinstance(result["best"], dict)
        assert "improvement" in result

    def test_iteration_improvement_calculation(self) -> None:
        """Test that improvement is calculated correctly."""
        spec = ModelSpec(name="improvement_calc")
        refiner = ArchitectureRefiner(seed=42, improvement_threshold=0.0, patience=10)
        history = refiner.refine(spec, iterations=3, candidates_per_iteration=5)

        for i, iteration in enumerate(history.iterations):
            if i == 0:
                # First iteration: improvement is relative to initial best
                continue
            # Improvement should be a valid float
            assert isinstance(iteration.improvement, float)


class TestRefinerWithObjectives:
    """Tests for refinement with specific objectives."""

    def test_refine_with_latency_objective(self) -> None:
        """Test refinement with latency objective."""
        spec = ModelSpec(name="latency_objective")
        refiner = ArchitectureRefiner(seed=42)
        history = refiner.refine(
            spec, iterations=2, candidates_per_iteration=3, objectives=["latency"]
        )

        assert len(history.iterations) >= 1
        # With latency objective, latency_ms metric should be present
        for candidate in history.iterations[-1].candidates:
            assert "latency_ms" in candidate.metrics

    def test_refine_with_params_objective(self) -> None:
        """Test refinement with params objective."""
        spec = ModelSpec(name="params_objective")
        refiner = ArchitectureRefiner(seed=42)
        history = refiner.refine(
            spec, iterations=2, candidates_per_iteration=3, objectives=["params"]
        )

        assert len(history.iterations) >= 1
        for candidate in history.iterations[-1].candidates:
            assert "params_billion" in candidate.metrics

    def test_refine_with_multiple_objectives(self) -> None:
        """Test refinement with multiple objectives."""
        spec = ModelSpec(name="multi_objective")
        refiner = ArchitectureRefiner(seed=42)
        history = refiner.refine(
            spec,
            iterations=2,
            candidates_per_iteration=3,
            objectives=["latency", "params", "performance"],
        )

        assert len(history.iterations) >= 1
        for candidate in history.iterations[-1].candidates:
            assert "latency_ms" in candidate.metrics
            assert "params_billion" in candidate.metrics
            assert "performance_proxy" in candidate.metrics


class TestRefinerWithDifferentArchitectures:
    """Tests for refinement with different architecture families."""

    def test_refine_transformer_architecture(self) -> None:
        """Test refinement with transformer architecture."""
        spec = ModelSpec(name="transformer_refine", architecture={"family": "transformer"})
        refiner = ArchitectureRefiner(seed=42)
        history = refiner.refine(spec, iterations=2, candidates_per_iteration=3)

        assert history.best.family == "transformer"

    def test_refine_cnn_architecture(self) -> None:
        """Test refinement with CNN architecture."""
        spec = ModelSpec(name="cnn_refine", architecture={"family": "cnn"})
        refiner = ArchitectureRefiner(seed=42)
        history = refiner.refine(spec, iterations=2, candidates_per_iteration=3)

        assert history.best.family == "cnn"
        # CNN should have adjusted heads
        for candidate in history.iterations[-1].candidates:
            # Heads should be reduced for CNN
            assert candidate.dims["heads"] >= 1


class TestRefinerMutationRate:
    """Tests for mutation rate behavior."""

    def test_zero_mutation_rate(self) -> None:
        """Test refinement with zero mutation rate."""
        spec = ModelSpec(name="zero_mutation")
        refiner = ArchitectureRefiner(seed=42, patience=5)
        history = refiner.refine(spec, iterations=3, candidates_per_iteration=3, mutation_rate=0.0)

        # With zero mutation, all candidates should have same dims
        first_iteration = history.iterations[0]
        base_dims = first_iteration.candidates[0].dims
        for candidate in first_iteration.candidates:
            assert candidate.dims == base_dims

    def test_full_mutation_rate(self) -> None:
        """Test refinement with 100% mutation rate."""
        spec = ModelSpec(name="full_mutation")
        refiner = ArchitectureRefiner(seed=42)
        history = refiner.refine(spec, iterations=3, candidates_per_iteration=5, mutation_rate=1.0)

        # With full mutation, dimensions should vary more
        assert len(history.iterations) >= 1
        # Collect all unique dimension sets
        unique_dims = set()
        for iteration in history.iterations:
            for candidate in iteration.candidates:
                dims_tuple = (
                    candidate.dims["hidden_size"],
                    candidate.dims["layers"],
                    candidate.dims["heads"],
                )
                unique_dims.add(dims_tuple)
        # Should have multiple unique configurations
        assert len(unique_dims) > 1
