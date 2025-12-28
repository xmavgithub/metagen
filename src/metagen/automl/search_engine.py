from __future__ import annotations

import logging
import random
from collections.abc import Iterable
from typing import TYPE_CHECKING

from metagen.automl.candidates import CandidateArchitecture, SearchResult
from metagen.automl.mutation import mutate_dims, sample_dims
from metagen.automl.objectives import (
    LatencyObjective,
    ParamsObjective,
    PerformanceObjective,
    compute_pareto_front,
)
from metagen.automl.training_utils import maybe_train_prototype
from metagen.specs.schema import ModelSpec
from metagen.synth import architecture

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from metagen.automl.meta_learner import MetaLearner


class ArchitectureSearchEngine:
    """Searches architecture space for candidate configurations."""

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize the search engine.

        Args:
            seed: Base seed for deterministic search.
        """
        self.seed = seed

    def search(
        self,
        spec: ModelSpec,
        search_budget: int = 10,
        *,
        objectives: Iterable[str] | None = None,
        seed: int | None = None,
        strategy: str = "random",
        generations: int = 3,
        population_size: int | None = None,
        train_prototypes: bool = False,
        prototype_steps: int = 100,
        meta_learner: MetaLearner | None = None,
        transfer_k: int = 0,
    ) -> SearchResult:
        """
        Run a random-search baseline over architecture dimensions.

        Args:
            spec: Model specification to guide search.
            search_budget: Number of candidates to sample.
            objectives: Optional objective hints (latency, params, performance, memory).
            seed: Optional override seed for deterministic search.
            strategy: Search strategy ("random" or "evolution").
            generations: Evolution generations for strategy="evolution".
            population_size: Population size for strategy="evolution".
            train_prototypes: Whether to train tiny prototypes per candidate.
            prototype_steps: Max steps for prototype training.
            meta_learner: Optional MetaLearner for warm-starting search dims.
            transfer_k: Number of transfer candidates to warm-start from history.

        Returns:
            SearchResult with candidates sorted by score (descending).

        Example:
            >>> from metagen.specs.loader import load_spec
            >>> spec, _ = load_spec("examples/specs/text/text_llm_8b.yaml")
            >>> engine = ArchitectureSearchEngine(seed=123)
            >>> result = engine.search(spec, search_budget=5)
            >>> len(result.candidates)
            5
        """
        if search_budget < 1:
            raise ValueError("search_budget must be >= 1")

        if transfer_k < 0:
            raise ValueError("transfer_k must be >= 0")

        rng = random.Random(self.seed if seed is None else seed)
        base_dims = None
        if meta_learner is not None:
            starting_point = meta_learner.predict_good_starting_point(spec)
            if starting_point is not None:
                base_dims = dict(starting_point.dims)
        if base_dims is None:
            base_dims, _summary = architecture.estimate_summary(
                spec,
                seed=rng.randint(0, 2**32 - 1),
            )
        family = spec.architecture.family.lower()
        warm_start_dims: list[dict[str, int]] = []
        if transfer_k > 0 and meta_learner is not None:
            warm_start_dims = meta_learner.suggest_transfer_candidates(spec, limit=transfer_k)

        if strategy not in {"random", "evolution"}:
            raise ValueError("strategy must be 'random' or 'evolution'")

        if strategy == "random":
            candidates = self._random_search(
                spec=spec,
                base_dims=base_dims,
                family=family,
                rng=rng,
                search_budget=search_budget,
                objectives=objectives,
                train_prototypes=train_prototypes,
                prototype_steps=prototype_steps,
                warm_start_dims=warm_start_dims,
            )
        else:
            candidates = self._evolution_search(
                spec=spec,
                base_dims=base_dims,
                family=family,
                rng=rng,
                search_budget=search_budget,
                objectives=objectives,
                generations=generations,
                population_size=population_size,
                train_prototypes=train_prototypes,
                prototype_steps=prototype_steps,
                warm_start_dims=warm_start_dims,
            )

        candidates = self._apply_pareto_ranking(candidates, objectives)
        candidates = self._apply_prototype_ranking(candidates, train_prototypes)
        best = candidates[0]
        return SearchResult(
            spec_name=spec.name,
            search_budget=search_budget,
            seed=self.seed if seed is None else seed,
            candidates=tuple(candidates),
            best=best,
        )

    def evaluate_candidate(
        self,
        dims: dict[str, int],
        family: str,
        objectives: Iterable[str] | None = None,
    ) -> tuple[dict[str, float], float]:
        """
        Score a candidate architecture using heuristic proxies.

        Args:
            dims: Candidate dimensions (hidden_size, layers, heads).
            family: Architecture family.
            objectives: Optional objective hints to weight scoring.

        Returns:
            (metrics, score) where metrics include params/memory/latency proxies.
        """
        params_b = architecture._estimate_params(dims["hidden_size"], dims["layers"], family)
        activation_memory_gb = round(params_b * 0.5, 2)
        kv_cache_gb = round(dims["layers"] * dims["heads"] * 0.01, 2)
        latency_ms = round((dims["layers"] * dims["hidden_size"]) / 256.0, 2)
        performance_proxy = round(
            (dims["hidden_size"] * dims["layers"] * max(1, dims["heads"])) / 1e6, 3
        )

        metrics = {
            "params_billion": params_b,
            "activation_memory_gb": activation_memory_gb,
            "kv_cache_gb": kv_cache_gb,
            "latency_ms": latency_ms,
            "performance_proxy": performance_proxy,
        }

        normalized = self._normalize_objectives(objectives)
        if not normalized:
            score = performance_proxy - (params_b * 0.2) - (latency_ms * 0.02)
        else:
            score = 0.0
            if "performance" in normalized:
                score += performance_proxy
            if "params" in normalized:
                score -= params_b
            if "latency" in normalized:
                score -= latency_ms * 0.1
            if "memory" in normalized:
                score -= activation_memory_gb * 0.5

        return metrics, round(score, 4)

    def _sample_dims(
        self,
        base_dims: dict[str, int],
        family: str,
        rng: random.Random,
    ) -> dict[str, int]:
        """Sample new dimensions using shared sampling strategy."""
        return sample_dims(base_dims, rng, family=family)

    def _random_search(
        self,
        *,
        spec: ModelSpec,
        base_dims: dict[str, int],
        family: str,
        rng: random.Random,
        search_budget: int,
        objectives: Iterable[str] | None,
        train_prototypes: bool,
        prototype_steps: int,
        warm_start_dims: Iterable[dict[str, int]] | None,
    ) -> list[CandidateArchitecture]:
        def dims_key(dims: dict[str, int]) -> tuple[int, int, int] | None:
            try:
                return int(dims["hidden_size"]), int(dims["layers"]), int(dims["heads"])
            except (KeyError, TypeError, ValueError):
                return None

        def build_candidate(dims: dict[str, int]) -> CandidateArchitecture:
            metrics, score = self.evaluate_candidate(dims, family, objectives)
            metrics, score = maybe_train_prototype(
                metrics=metrics,
                score=score,
                dims=dims,
                spec=spec,
                train_prototypes=train_prototypes,
                prototype_steps=prototype_steps,
                rng=rng,
            )
            return CandidateArchitecture(
                dims=dims,
                metrics=metrics,
                score=score,
                family=family,
                seed=rng.randint(0, 2**32 - 1),
            )

        candidates: list[CandidateArchitecture] = []
        seen = set()
        if warm_start_dims:
            for dims in warm_start_dims:
                key = dims_key(dims)
                if key is None or key in seen:
                    continue
                seen.add(key)
                candidates.append(build_candidate(dims))
                if len(candidates) >= search_budget:
                    candidates.sort(key=lambda c: c.score, reverse=True)
                    return candidates

        while len(candidates) < search_budget:
            dims = self._sample_dims(base_dims, family, rng)
            key = dims_key(dims)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(build_candidate(dims))
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _evolution_search(
        self,
        *,
        spec: ModelSpec,
        base_dims: dict[str, int],
        family: str,
        rng: random.Random,
        search_budget: int,
        objectives: Iterable[str] | None,
        generations: int,
        population_size: int | None,
        train_prototypes: bool,
        prototype_steps: int,
        warm_start_dims: Iterable[dict[str, int]] | None,
    ) -> list[CandidateArchitecture]:
        if generations < 1:
            raise ValueError("generations must be >= 1")
        pop_size = population_size or min(10, search_budget)
        pop_size = max(2, pop_size)

        population = self._random_search(
            spec=spec,
            base_dims=base_dims,
            family=family,
            rng=rng,
            search_budget=pop_size,
            objectives=objectives,
            train_prototypes=train_prototypes,
            prototype_steps=prototype_steps,
            warm_start_dims=warm_start_dims,
        )
        all_candidates = list(population)
        evaluated = len(population)

        for _ in range(generations):
            parents = self._select_parents(population, rng)
            children: list[CandidateArchitecture] = []
            while len(children) + evaluated < search_budget and len(children) < pop_size:
                parent_a, parent_b = rng.sample(parents, 2)
                child_dims = self._crossover(parent_a.dims, parent_b.dims, rng)
                child_dims = self._mutate(child_dims, rng, family=family)
                metrics, score = self.evaluate_candidate(child_dims, family, objectives)
                metrics, score = maybe_train_prototype(
                    metrics=metrics,
                    score=score,
                    dims=child_dims,
                    spec=spec,
                    train_prototypes=train_prototypes,
                    prototype_steps=prototype_steps,
                    rng=rng,
                )
                children.append(
                    CandidateArchitecture(
                        dims=child_dims,
                        metrics=metrics,
                        score=score,
                        family=family,
                        seed=rng.randint(0, 2**32 - 1),
                    )
                )

            evaluated += len(children)
            population.extend(children)
            all_candidates.extend(children)
            population.sort(key=lambda c: c.score, reverse=True)
            population = population[:pop_size]

            if evaluated >= search_budget:
                break

        return sorted(all_candidates, key=lambda c: c.score, reverse=True)

    def _select_parents(
        self,
        population: list[CandidateArchitecture],
        rng: random.Random,
    ) -> list[CandidateArchitecture]:
        if not population:
            return []
        population = sorted(population, key=lambda c: c.score, reverse=True)
        top_k = max(2, len(population) // 2)
        parents = population[:top_k]
        rng.shuffle(parents)
        return parents

    def _crossover(
        self,
        parent_a: dict[str, int],
        parent_b: dict[str, int],
        rng: random.Random,
    ) -> dict[str, int]:
        child = {
            "hidden_size": rng.choice([parent_a["hidden_size"], parent_b["hidden_size"]]),
            "layers": rng.choice([parent_a["layers"], parent_b["layers"]]),
            "heads": rng.choice([parent_a["heads"], parent_b["heads"]]),
        }
        return child

    def _mutate(
        self,
        dims: dict[str, int],
        rng: random.Random,
        mutation_rate: float = 0.2,
        family: str = "transformer",
    ) -> dict[str, int]:
        """Mutate dimensions using shared exploration-mode mutation."""
        return mutate_dims(
            dims, rng, mutation_rate=mutation_rate, family=family, mode="exploration"
        )

    def _build_objectives(self, objectives: Iterable[str] | None) -> list:
        normalized = self._normalize_objectives(objectives)
        objective_objs = []
        if "params" in normalized:
            objective_objs.append(ParamsObjective())
        if "latency" in normalized:
            objective_objs.append(LatencyObjective())
        if "performance" in normalized:
            objective_objs.append(PerformanceObjective())
        return objective_objs

    def _apply_pareto_ranking(
        self,
        candidates: list[CandidateArchitecture],
        objectives: Iterable[str] | None,
    ) -> list[CandidateArchitecture]:
        objective_objs = self._build_objectives(objectives)
        if len(objective_objs) < 2:
            return sorted(candidates, key=lambda c: c.score, reverse=True)

        pareto_front = compute_pareto_front(candidates, objective_objs)
        pareto_ids = {id(candidate) for candidate in pareto_front}

        for candidate in candidates:
            candidate.metrics["pareto_front"] = 1.0 if id(candidate) in pareto_ids else 0.0

        return sorted(
            candidates,
            key=lambda c: (
                0 if id(c) in pareto_ids else 1,
                -c.score,
            ),
        )

    def _apply_prototype_ranking(
        self,
        candidates: list[CandidateArchitecture],
        train_prototypes: bool,
    ) -> list[CandidateArchitecture]:
        if not train_prototypes:
            return candidates

        def loss_key(candidate: CandidateArchitecture) -> float:
            loss = candidate.metrics.get("prototype_loss")
            return float(loss) if loss is not None else float("inf")

        return sorted(candidates, key=lambda c: (loss_key(c), -c.score))

    def _normalize_objectives(self, objectives: Iterable[str] | None) -> set[str]:
        if not objectives:
            return set()

        normalized = set()
        for obj in objectives:
            token = obj.lower().strip()
            if token in {"performance", "accuracy", "quality"}:
                normalized.add("performance")
            elif token in {"params", "parameter", "size", "model_size"}:
                normalized.add("params")
            elif token in {"latency", "speed", "throughput"}:
                normalized.add("latency")
            elif token in {"memory", "ram"}:
                normalized.add("memory")

        return normalized
