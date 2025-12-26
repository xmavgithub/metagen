from metagen.automl.history import HistoryDatabase
from metagen.automl.meta_learner import MetaLearner
from metagen.automl.search_engine import ArchitectureSearchEngine
from metagen.specs.schema import ModelSpec
from metagen.synth import architecture


def test_transfer_learning_warm_starts_search(tmp_path) -> None:
    history = HistoryDatabase(root_dir=tmp_path)
    learner = MetaLearner(history=history)

    target_spec = ModelSpec(name="target", constraints={"parameter_budget": {"max": "50M"}})
    other_spec = ModelSpec(name="other", constraints={"parameter_budget": {"max": "5B"}})

    target_blueprint = architecture.build_blueprint_from_dims(
        target_spec,
        {"hidden_size": 256, "layers": 4, "heads": 4},
        seed=1,
    )
    other_blueprint = architecture.build_blueprint_from_dims(
        other_spec,
        {"hidden_size": 1024, "layers": 16, "heads": 16},
        seed=2,
    )

    learner.update_database(target_spec, target_blueprint, {"score": 1.0})
    learner.update_database(other_spec, other_blueprint, {"score": 1.0})

    engine = ArchitectureSearchEngine(seed=123)
    result = engine.search(
        target_spec,
        search_budget=1,
        meta_learner=learner,
        transfer_k=1,
    )

    assert result.candidates[0].dims == target_blueprint.dims
