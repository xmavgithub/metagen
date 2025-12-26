from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateArchitecture:
    """Represents a candidate architecture produced during AutoML search."""

    dims: dict[str, int]
    metrics: dict[str, float]
    score: float
    family: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "dims": dict(self.dims),
            "metrics": dict(self.metrics),
            "score": self.score,
            "family": self.family,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class SearchResult:
    """Container for AutoML search results."""

    spec_name: str
    search_budget: int
    seed: int
    candidates: tuple[CandidateArchitecture, ...]
    best: CandidateArchitecture

    def top_k(self, k: int) -> tuple[CandidateArchitecture, ...]:
        """Return top-K candidates by score (descending)."""
        if k <= 0:
            return ()
        return self.candidates[:k]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "spec_name": self.spec_name,
            "search_budget": self.search_budget,
            "seed": self.seed,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "best": self.best.to_dict(),
        }
