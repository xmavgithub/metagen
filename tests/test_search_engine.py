import pytest

from metagen.automl.search_engine import ArchitectureSearchEngine
from metagen.specs.schema import ModelSpec


def test_search_returns_candidates() -> None:
    spec = ModelSpec(name="search_test")
    engine = ArchitectureSearchEngine(seed=123)
    result = engine.search(spec, search_budget=5)

    assert len(result.candidates) == 5
    assert result.best in result.candidates
    assert result.candidates[0].score >= result.candidates[-1].score


def test_search_deterministic_seed() -> None:
    spec = ModelSpec(name="deterministic_search")
    engine = ArchitectureSearchEngine(seed=42)

    result_a = engine.search(spec, search_budget=4)
    result_b = engine.search(spec, search_budget=4)

    assert [c.dims for c in result_a.candidates] == [c.dims for c in result_b.candidates]
    assert [c.score for c in result_a.candidates] == [c.score for c in result_b.candidates]


def test_objectives_change_scoring() -> None:
    spec = ModelSpec(name="objective_search")
    engine = ArchitectureSearchEngine(seed=7)

    result_default = engine.search(spec, search_budget=3)
    result_latency = engine.search(spec, search_budget=3, objectives=["latency"])

    assert [c.score for c in result_default.candidates] != [
        c.score for c in result_latency.candidates
    ]


def test_evolution_strategy_runs() -> None:
    spec = ModelSpec(name="evolution_search")
    engine = ArchitectureSearchEngine(seed=11)

    result = engine.search(
        spec,
        search_budget=6,
        strategy="evolution",
        generations=2,
        population_size=3,
    )

    assert 0 < len(result.candidates) <= 6
    assert result.best in result.candidates


def test_pareto_front_flagging() -> None:
    spec = ModelSpec(name="pareto_search")
    engine = ArchitectureSearchEngine(seed=9)
    result = engine.search(
        spec,
        search_budget=4,
        objectives=["params", "performance"],
    )

    pareto_flags = [c.metrics.get("pareto_front") for c in result.candidates]
    assert any(flag == 1.0 for flag in pareto_flags)


def test_prototype_training_integration() -> None:
    pytest.importorskip("torch")

    spec = ModelSpec(
        name="prototype_search",
        modality={"inputs": ["text"], "outputs": ["text"]},
        constraints={"context_window": "32"},
    )
    engine = ArchitectureSearchEngine(seed=13)
    result = engine.search(
        spec,
        search_budget=2,
        train_prototypes=True,
        prototype_steps=2,
    )

    losses = [candidate.metrics.get("prototype_loss") for candidate in result.candidates]
    assert all(loss is not None for loss in losses)
    assert losses == sorted(losses)
