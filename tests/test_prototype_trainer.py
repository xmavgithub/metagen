import pytest

from metagen.automl.prototype_trainer import PrototypeTrainer
from metagen.specs.schema import ModelSpec
from metagen.synth.architecture import build_blueprint_from_dims


def test_prototype_trainer_runs() -> None:
    pytest.importorskip("torch")

    spec = ModelSpec(
        name="proto_test",
        modality={"inputs": ["text"], "outputs": ["text"]},
        constraints={"context_window": "64"},
    )
    dims = {"hidden_size": 128, "layers": 2, "heads": 2}
    blueprint = build_blueprint_from_dims(spec, dims, seed=123)

    trainer = PrototypeTrainer(max_seq_len_cap=32, max_hidden_size=64, max_layers=2, max_heads=2)
    metrics = trainer.train_prototype(blueprint, spec, budget_steps=2, batch_size=2)

    assert metrics.steps <= 2
    assert metrics.final_loss >= 0
