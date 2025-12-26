from __future__ import annotations

import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from metagen.automl.candidates import CandidateArchitecture
from metagen.automl.mutation import mutate_dims
from metagen.automl.search_engine import ArchitectureSearchEngine
from metagen.automl.training_utils import maybe_train_prototype
from metagen.specs.schema import ModelSpec
from metagen.synth import architecture

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefinementIteration:
    """One refinement iteration worth of candidates and best score."""

    iteration: int
    candidates: tuple[CandidateArchitecture, ...]
    best: CandidateArchitecture
    improvement: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "iteration": self.iteration,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "best": self.best.to_dict(),
            "improvement": self.improvement,
        }


@dataclass(frozen=True)
class RefinementHistory:
    """Container for iterative refinement results."""

    spec_name: str
    seed: int
    candidates_per_iteration: int
    max_iterations: int
    iterations: tuple[RefinementIteration, ...]
    best: CandidateArchitecture
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "spec_name": self.spec_name,
            "seed": self.seed,
            "candidates_per_iteration": self.candidates_per_iteration,
            "max_iterations": self.max_iterations,
            "iterations": [iteration.to_dict() for iteration in self.iterations],
            "best": self.best.to_dict(),
            "converged": self.converged,
        }


class ArchitectureRefiner:
    """Iteratively refines architectures based on evaluation feedback."""

    def __init__(
        self,
        seed: int = 42,
        improvement_threshold: float = 0.001,
        patience: int = 2,
    ) -> None:
        """
        Initialize the refiner.

        Args:
            seed: Base seed for deterministic refinement.
            improvement_threshold: Minimum score delta to consider as improvement.
            patience: Number of non-improving iterations before convergence.
        """
        if patience < 1:
            raise ValueError("patience must be >= 1")

        self.seed = seed
        self.improvement_threshold = improvement_threshold
        self.patience = patience

    def refine(
        self,
        spec: ModelSpec,
        iterations: int = 5,
        candidates_per_iteration: int = 5,
        *,
        objectives: Iterable[str] | None = None,
        seed: int | None = None,
        mutation_rate: float = 0.3,
        train_prototypes: bool = False,
        prototype_steps: int = 100,
    ) -> RefinementHistory:
        """
        Run iterative refinement over architecture dimensions.

        Args:
            spec: Model specification to guide refinement.
            iterations: Maximum refinement iterations.
            candidates_per_iteration: Number of candidates per iteration.
            objectives: Optional objective hints (latency, params, performance, memory).
            seed: Optional override seed for deterministic refinement.
            mutation_rate: Probability of mutating each dimension per candidate.
            train_prototypes: Whether to train tiny prototypes per candidate.
            prototype_steps: Max steps for prototype training.

        Returns:
            RefinementHistory with iteration-level candidates and the best architecture.
        """
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if candidates_per_iteration < 1:
            raise ValueError("candidates_per_iteration must be >= 1")

        resolved_seed = self.seed if seed is None else seed
        rng = random.Random(resolved_seed)
        base_seed = rng.randint(0, 2**32 - 1)
        base_dims, _summary = architecture.estimate_summary(spec, seed=base_seed)
        family = spec.architecture.family.lower()
        engine = ArchitectureSearchEngine(seed=resolved_seed)

        base_metrics, base_score = engine.evaluate_candidate(base_dims, family, objectives)
        base_metrics, base_score = maybe_train_prototype(
            metrics=base_metrics,
            score=base_score,
            dims=base_dims,
            spec=spec,
            train_prototypes=train_prototypes,
            prototype_steps=prototype_steps,
            rng=rng,
        )
        current_best = CandidateArchitecture(
            dims=base_dims,
            metrics=base_metrics,
            score=base_score,
            family=family,
            seed=rng.randint(0, 2**32 - 1),
        )

        history: list[RefinementIteration] = []
        no_improve = 0
        converged = False

        for iteration in range(1, iterations + 1):
            candidates: list[CandidateArchitecture] = [current_best]
            while len(candidates) < candidates_per_iteration:
                dims = self._mutate_dims(current_best.dims, family, rng, mutation_rate)
                metrics, score = engine.evaluate_candidate(dims, family, objectives)
                metrics, score = maybe_train_prototype(
                    metrics=metrics,
                    score=score,
                    dims=dims,
                    spec=spec,
                    train_prototypes=train_prototypes,
                    prototype_steps=prototype_steps,
                    rng=rng,
                )
                candidates.append(
                    CandidateArchitecture(
                        dims=dims,
                        metrics=metrics,
                        score=score,
                        family=family,
                        seed=rng.randint(0, 2**32 - 1),
                    )
                )

            candidates.sort(key=lambda c: c.score, reverse=True)
            best_iteration = candidates[0]
            improvement = round(best_iteration.score - current_best.score, 4)

            history.append(
                RefinementIteration(
                    iteration=iteration,
                    candidates=tuple(candidates),
                    best=best_iteration,
                    improvement=improvement,
                )
            )

            current_best = best_iteration
            if improvement <= self.improvement_threshold:
                no_improve += 1
            else:
                no_improve = 0
            if no_improve >= self.patience:
                converged = True
                break

        return RefinementHistory(
            spec_name=spec.name,
            seed=resolved_seed,
            candidates_per_iteration=candidates_per_iteration,
            max_iterations=iterations,
            iterations=tuple(history),
            best=current_best,
            converged=converged,
        )

    def _mutate_dims(
        self,
        dims: dict[str, int],
        family: str,
        rng: random.Random,
        mutation_rate: float,
    ) -> dict[str, int]:
        """Mutate dimensions using shared exploitation-mode mutation."""
        return mutate_dims(
            dims, rng, mutation_rate=mutation_rate, family=family, mode="exploitation"
        )
