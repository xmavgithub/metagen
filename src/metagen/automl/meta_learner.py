from __future__ import annotations

import hashlib
import math
from typing import Any

from metagen.automl.history import HistoryDatabase, RunRecord
from metagen.specs.loader import serialize_spec
from metagen.specs.schema import ModelSpec
from metagen.synth import architecture
from metagen.synth.architecture import BlueprintState


class MetaLearner:
    """Learns architecture priors from previous synthesis runs."""

    _family_index = {
        "transformer": 0.0,
        "cnn": 1.0,
        "diffusion": 2.0,
        "hybrid": 3.0,
    }

    def __init__(self, history: HistoryDatabase | None = None, max_history: int = 50) -> None:
        """
        Initialize the meta-learner.

        Args:
            history: Optional HistoryDatabase instance.
            max_history: Max number of historical runs to consider.
        """
        if max_history < 1:
            raise ValueError("max_history must be >= 1")

        self.history = history or HistoryDatabase()
        self.max_history = max_history

    def predict_good_starting_point(self, spec: ModelSpec) -> BlueprintState | None:
        """
        Predict a good starting blueprint for the given spec.

        Args:
            spec: Model specification to match against historical runs.

        Returns:
            BlueprintState for a similar spec, or None if no history exists.
        """
        spec_hash = self._hash_spec(spec)
        target_embedding = self._embed_spec(spec)
        runs = self.history.load_runs(limit=self.max_history)

        best_run = self._nearest_neighbor(runs, target_embedding)
        if best_run is None:
            best_run = self._latest_exact_match(runs, spec_hash)
        if best_run is None:
            return None

        dims = self._extract_dims(best_run)
        if dims is None:
            return None

        seed = int(best_run.blueprint.get("seed", 42))
        return architecture.build_blueprint_from_dims(spec, dims, seed=seed)

    def suggest_transfer_candidates(
        self,
        spec: ModelSpec,
        limit: int = 3,
    ) -> list[dict[str, int]]:
        """
        Return candidate dims from similar specs for warm-starting search.

        Args:
            spec: Model specification to match against history.
            limit: Maximum number of candidate dims to return.

        Returns:
            List of dimension dicts ordered by similarity.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")

        target_embedding = self._embed_spec(spec)
        runs = self.history.load_runs(limit=self.max_history)

        scored: list[tuple[float, dict[str, int]]] = []
        for run in runs:
            embedding = self._extract_embedding(run)
            dims = self._extract_dims(run)
            if embedding is None or dims is None:
                continue
            score = self._cosine_similarity(target_embedding, embedding)
            scored.append((score, dims))

        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return []

        unique: list[dict[str, int]] = []
        seen = set()
        for _score, dims in scored:
            key = (dims.get("hidden_size"), dims.get("layers"), dims.get("heads"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(dims)
            if len(unique) >= limit:
                break
        return unique

    def update_database(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState | dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        """
        Store a run with spec embedding in the history database.

        Args:
            spec: Model specification for this run.
            blueprint: BlueprintState or blueprint dict to store.
            metrics: Metrics payload for the run.
        """
        spec_hash = self._hash_spec(spec)
        embedding = self._embed_spec(spec)
        resolved_metrics = dict(metrics)
        resolved_metrics["spec_embedding"] = embedding

        blueprint_payload = self._normalize_blueprint(blueprint)
        self.history.save_run(spec_hash, blueprint_payload, resolved_metrics)

    def _hash_spec(self, spec: ModelSpec) -> str:
        serialized = serialize_spec(spec)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _embed_spec(self, spec: ModelSpec) -> list[float]:
        family = spec.architecture.family.lower()
        family_id = self._family_index.get(family, -1.0)
        param_budget = self._parse_number(spec.constraints.parameter_budget.max)
        context_window = self._parse_number(spec.constraints.context_window)

        return [
            float(len(spec.modality.inputs)),
            float(len(spec.modality.outputs)),
            float(family_id),
            self._scaled_value(param_budget),
            self._scaled_value(context_window),
            float(len(spec.training.objective)),
        ]

    def _nearest_neighbor(
        self,
        runs: list[RunRecord],
        target_embedding: list[float],
    ) -> RunRecord | None:
        best_run = None
        best_score = -math.inf
        for run in runs:
            embedding = self._extract_embedding(run)
            if embedding is None:
                continue
            if self._extract_dims(run) is None:
                continue
            score = self._cosine_similarity(target_embedding, embedding)
            if score > best_score:
                best_score = score
                best_run = run
        return best_run

    def _latest_exact_match(self, runs: list[RunRecord], spec_hash: str) -> RunRecord | None:
        for run in runs:
            if run.spec_hash != spec_hash:
                continue
            if self._extract_dims(run) is None:
                continue
            return run
        return None

    def _extract_embedding(self, run: RunRecord) -> list[float] | None:
        embedding = run.metrics.get("spec_embedding")
        if not isinstance(embedding, list) or not embedding:
            return None
        try:
            return [float(value) for value in embedding]
        except (TypeError, ValueError):
            return None

    def _extract_dims(self, run: RunRecord) -> dict[str, int] | None:
        dims = run.blueprint.get("dims")
        if not isinstance(dims, dict):
            return None
        try:
            return {key: int(value) for key, value in dims.items()}
        except (TypeError, ValueError):
            return None

    def _normalize_blueprint(self, blueprint: BlueprintState | dict[str, Any]) -> dict[str, Any]:
        if isinstance(blueprint, BlueprintState):
            return {
                "dims": dict(blueprint.dims),
                "seed": blueprint.seed,
                "family": blueprint.family,
            }
        if isinstance(blueprint, dict):
            return dict(blueprint)
        raise TypeError("blueprint must be a BlueprintState or dict")

    def _parse_number(self, value: str) -> float:
        text = value.strip().lower()
        if not text:
            return 0.0

        multipliers = {"k": 1e3, "m": 1e6, "b": 1e9}
        suffix = text[-1]
        if suffix in multipliers:
            try:
                return float(text[:-1]) * multipliers[suffix]
            except ValueError:
                return 0.0

        cleaned = "".join(ch for ch in text if ch.isdigit() or ch == ".")
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def _scaled_value(self, value: float) -> float:
        if value <= 0:
            return 0.0
        return math.log10(value + 1.0)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
