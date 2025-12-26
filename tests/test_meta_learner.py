import random

from metagen.automl.history import HistoryDatabase
from metagen.automl.meta_learner import MetaLearner
from metagen.automl.search_engine import ArchitectureSearchEngine
from metagen.specs.schema import ModelSpec
from metagen.synth import architecture


def _expected_sample_dims(base_dims: dict[str, int], family: str, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    hidden_size = base_dims["hidden_size"]
    layers = base_dims["layers"]

    hidden_factor = rng.choice([0.5, 0.75, 1.0, 1.25, 1.5])
    hidden_size = int(max(64, round(hidden_size * hidden_factor / 64) * 64))

    layer_delta = rng.randint(-4, 4)
    layers = max(1, layers + layer_delta)

    heads = max(1, hidden_size // 64)
    while heads > 1 and hidden_size % heads != 0:
        heads -= 1

    if family == "cnn":
        heads = max(1, heads // 2)

    return {"hidden_size": hidden_size, "layers": layers, "heads": heads}


def test_meta_learner_nearest_neighbor(tmp_path) -> None:
    history = HistoryDatabase(root_dir=tmp_path)
    learner = MetaLearner(history=history)

    spec_a = ModelSpec(name="spec_a", constraints={"parameter_budget": {"max": "200M"}})
    spec_b = ModelSpec(name="spec_b", constraints={"parameter_budget": {"max": "2B"}})

    blueprint_a = architecture.build_blueprint_from_dims(
        spec_a,
        {"hidden_size": 256, "layers": 4, "heads": 4},
        seed=1,
    )
    blueprint_b = architecture.build_blueprint_from_dims(
        spec_b,
        {"hidden_size": 1024, "layers": 16, "heads": 16},
        seed=2,
    )

    learner.update_database(spec_a, blueprint_a, {"score": 1.0})
    learner.update_database(spec_b, blueprint_b, {"score": 1.0})

    predicted = learner.predict_good_starting_point(spec_a)
    assert predicted is not None
    assert predicted.dims == blueprint_a.dims


def test_meta_learner_integration_with_search_engine(tmp_path) -> None:
    history = HistoryDatabase(root_dir=tmp_path)
    learner = MetaLearner(history=history)

    spec = ModelSpec(name="spec_search", constraints={"parameter_budget": {"max": "50M"}})
    base_dims = {"hidden_size": 256, "layers": 6, "heads": 4}
    blueprint = architecture.build_blueprint_from_dims(spec, base_dims, seed=7)
    learner.update_database(spec, blueprint, {"score": 0.9})

    engine = ArchitectureSearchEngine(seed=123)
    result = engine.search(spec, search_budget=1, meta_learner=learner)

    expected = _expected_sample_dims(base_dims, spec.architecture.family.lower(), seed=123)
    assert result.candidates[0].dims == expected
