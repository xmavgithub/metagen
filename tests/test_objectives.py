from metagen.automl.candidates import CandidateArchitecture
from metagen.automl.objectives import (
    LatencyObjective,
    ParamsObjective,
    PerformanceObjective,
    compute_pareto_front,
)


def test_pareto_front_single_dominator() -> None:
    candidates = [
        CandidateArchitecture(
            dims={"hidden_size": 64, "layers": 2, "heads": 2},
            metrics={"params_billion": 2.0, "latency_ms": 5.0, "performance_proxy": 2.0},
            score=0.0,
            family="transformer",
            seed=1,
        ),
        CandidateArchitecture(
            dims={"hidden_size": 64, "layers": 2, "heads": 2},
            metrics={"params_billion": 1.0, "latency_ms": 10.0, "performance_proxy": 1.0},
            score=0.0,
            family="transformer",
            seed=2,
        ),
        CandidateArchitecture(
            dims={"hidden_size": 64, "layers": 2, "heads": 2},
            metrics={"params_billion": 1.0, "latency_ms": 5.0, "performance_proxy": 2.0},
            score=0.0,
            family="transformer",
            seed=3,
        ),
    ]
    objectives = [ParamsObjective(), PerformanceObjective()]
    front = compute_pareto_front(candidates, objectives)
    assert len(front) == 1
    assert front[0].metrics["params_billion"] == 1.0
    assert front[0].metrics["performance_proxy"] == 2.0


def test_pareto_front_multiple_non_dominated() -> None:
    candidates = [
        CandidateArchitecture(
            dims={"hidden_size": 64, "layers": 2, "heads": 2},
            metrics={"params_billion": 1.0, "latency_ms": 10.0, "performance_proxy": 1.0},
            score=0.0,
            family="transformer",
            seed=1,
        ),
        CandidateArchitecture(
            dims={"hidden_size": 64, "layers": 2, "heads": 2},
            metrics={"params_billion": 2.0, "latency_ms": 5.0, "performance_proxy": 2.0},
            score=0.0,
            family="transformer",
            seed=2,
        ),
    ]
    objectives = [ParamsObjective(), PerformanceObjective(), LatencyObjective()]
    front = compute_pareto_front(candidates, objectives)
    assert len(front) == 2
