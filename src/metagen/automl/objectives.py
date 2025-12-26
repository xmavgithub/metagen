from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from metagen.automl.candidates import CandidateArchitecture


@dataclass(frozen=True)
class Objective(ABC):
    """Base class for multi-objective scoring."""

    name: str
    direction: str  # "min" or "max"

    @abstractmethod
    def value(self, metrics: dict[str, float]) -> float:
        """Extract the objective value from candidate metrics."""
        raise NotImplementedError


class ParamsObjective(Objective):
    """Minimize parameter count (billions)."""

    def __init__(self) -> None:
        super().__init__(name="params", direction="min")

    def value(self, metrics: dict[str, float]) -> float:
        return metrics["params_billion"]


class LatencyObjective(Objective):
    """Minimize latency proxy (ms)."""

    def __init__(self) -> None:
        super().__init__(name="latency", direction="min")

    def value(self, metrics: dict[str, float]) -> float:
        return metrics["latency_ms"]


class PerformanceObjective(Objective):
    """Maximize performance proxy."""

    def __init__(self) -> None:
        super().__init__(name="performance", direction="max")

    def value(self, metrics: dict[str, float]) -> float:
        return metrics["performance_proxy"]


def compute_pareto_front(
    candidates: list[CandidateArchitecture],
    objectives: list[Objective],
) -> tuple[CandidateArchitecture, ...]:
    """
    Compute the Pareto front for the given candidates.

    Args:
        candidates: Candidate architectures with metrics.
        objectives: Objectives to optimize.

    Returns:
        Tuple of candidates on the Pareto front.
    """
    if not candidates or not objectives:
        return tuple(candidates)

    front: list[CandidateArchitecture] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            if _dominates(other.metrics, candidate.metrics, objectives):
                dominated = True
                break
        if not dominated:
            front.append(candidate)

    return tuple(front)


def _dominates(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    objectives: list[Objective],
) -> bool:
    """Return True if metrics_a dominates metrics_b across objectives."""
    better_or_equal = True
    strictly_better = False

    for objective in objectives:
        a_val = objective.value(metrics_a)
        b_val = objective.value(metrics_b)
        if objective.direction == "min":
            if a_val > b_val:
                better_or_equal = False
                break
            if a_val < b_val:
                strictly_better = True
        else:
            if a_val < b_val:
                better_or_equal = False
                break
            if a_val > b_val:
                strictly_better = True

    return better_or_equal and strictly_better
