import pytest

from metagen.automl.synthetic_data import (
    generate_image_noise,
    generate_synthetic_batch,
    generate_text_tokens,
    get_validation_task,
)
from metagen.specs.schema import ModelSpec
from metagen.synth.architecture import build_blueprint_from_dims


def test_validation_task_text() -> None:
    spec = ModelSpec(modality={"inputs": ["text"], "outputs": ["text"]})
    task = get_validation_task(spec)
    assert task.modality == "text"


def test_validation_task_image() -> None:
    spec = ModelSpec(modality={"inputs": ["image"], "outputs": ["image"]})
    task = get_validation_task(spec)
    assert task.modality == "image"


def test_generate_text_tokens_shape() -> None:
    torch = pytest.importorskip("torch")
    batch = generate_text_tokens(batch_size=2, seq_len=8, vocab_size=100)
    assert batch.shape == (2, 8)
    assert batch.dtype == torch.long


def test_generate_image_noise_shape() -> None:
    pytest.importorskip("torch")
    batch = generate_image_noise(batch_size=2, num_channels=3, image_size=16)
    assert batch.shape == (2, 3, 16, 16)


def test_generate_synthetic_batch_text() -> None:
    pytest.importorskip("torch")
    spec = ModelSpec(modality={"inputs": ["text"], "outputs": ["text"]})
    blueprint = build_blueprint_from_dims(
        spec,
        {"hidden_size": 64, "layers": 2, "heads": 2},
        seed=123,
    )
    task, batch = generate_synthetic_batch(spec, blueprint, batch_size=2)
    assert task.modality == "text"
    assert batch.shape[0] == 2
