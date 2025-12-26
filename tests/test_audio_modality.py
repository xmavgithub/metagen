"""
Tests for MetaGen Audio Modality Handler

This module tests the AudioModalityHandler class and related functionality
including sample rate selection, channel configuration, and component generation.

Author: MetaGen Team
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from metagen.specs.schema import ModelSpec
from metagen.synth.architecture import BlueprintState
from metagen.synth.modalities import get_handler
from metagen.synth.modalities.audio import (
    DEFAULT_SAMPLE_RATES,
    AudioModalityHandler,
)
from metagen.synth.modalities.base import ModalityComponents


# Mock classes for testing without real specs
@dataclass
class MockModality:
    """Mock modality for testing."""

    inputs: list[str] = field(default_factory=lambda: ["audio"])
    outputs: list[str] = field(default_factory=lambda: ["audio"])


@dataclass
class MockArchitecture:
    """Mock architecture for testing."""

    family: str = "transformer"


@dataclass
class MockTask:
    """Mock task for testing."""

    type: str = "generation"
    domain: str = "audio"


@dataclass
class MockTraining:
    """Mock training config for testing."""

    objective: list[str] = field(default_factory=lambda: ["reconstruction"])


@dataclass
class MockSpec:
    """Mock spec for testing AudioModalityHandler."""

    name: str = "audio_test"
    description: str = "Test audio model"
    modality: MockModality = field(default_factory=MockModality)
    architecture: MockArchitecture = field(default_factory=MockArchitecture)
    task: MockTask = field(default_factory=MockTask)
    training: MockTraining = field(default_factory=MockTraining)


@dataclass
class MockBlueprintState:
    """Mock BlueprintState for testing."""

    dims: dict = field(default_factory=lambda: {"hidden_size": 512, "layers": 6, "heads": 8})
    sample_rate: int | None = None
    num_channels: int | None = None


class TestAudioModalityHandler:
    """Tests for the AudioModalityHandler class."""

    def test_handler_name(self) -> None:
        """Test that handler name is 'audio'."""
        handler = AudioModalityHandler()
        assert handler.name == "audio"

    def test_handler_supported_families(self) -> None:
        """Test supported architecture families."""
        handler = AudioModalityHandler()
        assert "transformer" in handler.supported_families
        assert "cnn" in handler.supported_families
        assert "rnn" in handler.supported_families
        assert "hybrid" in handler.supported_families

    def test_handler_repr(self) -> None:
        """Test string representation."""
        handler = AudioModalityHandler()
        assert repr(handler) == "AudioModalityHandler(name='audio')"


class TestAugmentBlueprint:
    """Tests for blueprint augmentation."""

    def test_augment_adds_sample_rate(self) -> None:
        """Test that augment_blueprint adds sample_rate."""
        handler = AudioModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.sample_rate in DEFAULT_SAMPLE_RATES

    def test_augment_adds_num_channels(self) -> None:
        """Test that augment_blueprint adds num_channels."""
        handler = AudioModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.num_channels in {1, 2}

    def test_augment_preserves_dims(self) -> None:
        """Test that augmentation preserves original dims."""
        handler = AudioModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.dims == blueprint.dims

    def test_augment_deterministic(self) -> None:
        """Test that augmentation is deterministic given same seed."""
        handler = AudioModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        aug1 = handler.augment_blueprint(spec, blueprint, seed=42)
        aug2 = handler.augment_blueprint(spec, blueprint, seed=42)

        assert aug1.sample_rate == aug2.sample_rate
        assert aug1.num_channels == aug2.num_channels


class TestSampleRateSelection:
    """Tests for sample rate selection logic."""

    def test_music_domain_uses_44100(self) -> None:
        """Test that music domain uses 44.1kHz sample rate."""
        handler = AudioModalityHandler()
        spec = MockSpec(task=MockTask(domain="music"))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.sample_rate == 44100

    def test_speech_domain_uses_16000(self) -> None:
        """Test that speech domain uses 16kHz sample rate."""
        handler = AudioModalityHandler()
        spec = MockSpec(task=MockTask(domain="speech"))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.sample_rate == 16000

    def test_voice_domain_uses_16000(self) -> None:
        """Test that voice domain uses 16kHz sample rate."""
        handler = AudioModalityHandler()
        spec = MockSpec(task=MockTask(domain="voice"))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.sample_rate == 16000

    def test_music_in_name_uses_44100(self) -> None:
        """Test that 'music' in name uses 44.1kHz."""
        handler = AudioModalityHandler()
        spec = MockSpec(name="music_generator")
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.sample_rate == 44100

    def test_speech_in_name_uses_16000(self) -> None:
        """Test that 'speech' in name uses 16kHz."""
        handler = AudioModalityHandler()
        spec = MockSpec(name="speech_synthesizer")
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.sample_rate == 16000


class TestChannelSelection:
    """Tests for audio channel selection."""

    def test_stereo_in_name_uses_2_channels(self) -> None:
        """Test that 'stereo' in name uses 2 channels."""
        handler = AudioModalityHandler()
        spec = MockSpec(name="stereo_audio_model")
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.num_channels == 2

    def test_music_uses_stereo(self) -> None:
        """Test that music models default to stereo."""
        handler = AudioModalityHandler()
        spec = MockSpec(description="A music generation model")
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.num_channels == 2

    def test_general_audio_random_channels(self) -> None:
        """Test that general audio randomly selects mono or stereo."""
        handler = AudioModalityHandler()
        spec = MockSpec(name="generic_audio", description="generic", task=MockTask(domain="audio"))
        blueprint = MockBlueprintState()

        # Run multiple times to verify randomness
        channels_seen = set()
        for seed in range(100):
            augmented = handler.augment_blueprint(spec, blueprint, seed=seed)
            channels_seen.add(augmented.num_channels)

        # Should see both 1 and 2 channels
        assert 1 in channels_seen
        assert 2 in channels_seen


class TestGenerateComponents:
    """Tests for component generation."""

    def test_returns_modality_components(self) -> None:
        """Test that generate_components returns ModalityComponents."""
        handler = AudioModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(sample_rate=44100, num_channels=2)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert isinstance(components, ModalityComponents)

    def test_transformer_encoder_type(self) -> None:
        """Test encoder type for transformer family."""
        handler = AudioModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="transformer"))
        blueprint = MockBlueprintState(sample_rate=44100, num_channels=2)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.encoder_type == "audio_transformer"

    def test_cnn_encoder_type(self) -> None:
        """Test encoder type for CNN family."""
        handler = AudioModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="cnn"))
        blueprint = MockBlueprintState(sample_rate=44100, num_channels=2)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.encoder_type == "audio_cnn"

    def test_data_loader_type(self) -> None:
        """Test that data loader type is audio_dataset."""
        handler = AudioModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(sample_rate=44100, num_channels=2)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.data_loader_type == "audio_dataset"

    def test_contrastive_loss_type(self) -> None:
        """Test loss type for contrastive objectives."""
        handler = AudioModalityHandler()
        spec = MockSpec(training=MockTraining(objective=["contrastive"]))
        blueprint = MockBlueprintState(sample_rate=44100, num_channels=2)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.loss_type == "contrastive"

    def test_reconstruction_loss_type(self) -> None:
        """Test loss type for reconstruction objectives."""
        handler = AudioModalityHandler()
        spec = MockSpec(training=MockTraining(objective=["reconstruction"]))
        blueprint = MockBlueprintState(sample_rate=44100, num_channels=2)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.loss_type == "reconstruction"

    def test_config_contains_audio_params(self) -> None:
        """Test that config contains audio-specific parameters."""
        handler = AudioModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            sample_rate=44100,
            num_channels=2,
            dims={"hidden_size": 512, "layers": 6, "heads": 8},
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.config["sample_rate"] == 44100
        assert components.config["num_channels"] == 2
        assert components.config["hidden_size"] == 512

    def test_template_fragments_include_audio(self) -> None:
        """Test that template fragments include audio components."""
        handler = AudioModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(sample_rate=44100, num_channels=2)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert "embeddings/audio_embedding.py.j2" in components.template_fragments
        assert "encoders/audio_encoder.py.j2" in components.template_fragments
        assert "data/audio_datasets.py.j2" in components.template_fragments


class TestValidateSpec:
    """Tests for spec validation."""

    def test_valid_audio_spec(self) -> None:
        """Test that valid audio spec passes validation."""
        handler = AudioModalityHandler()
        spec = MockSpec()

        # Should not raise
        handler.validate_spec(spec)

    def test_invalid_family_raises(self) -> None:
        """Test that unsupported family raises ValueError."""
        handler = AudioModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="unsupported"))

        with pytest.raises(ValueError) as exc_info:
            handler.validate_spec(spec)

        assert "unsupported" in str(exc_info.value)

    def test_missing_audio_modality_raises(self) -> None:
        """Test that missing audio modality raises ValueError."""
        handler = AudioModalityHandler()
        spec = MockSpec(modality=MockModality(inputs=["text"], outputs=["text"]))

        with pytest.raises(ValueError) as exc_info:
            handler.validate_spec(spec)

        assert "audio" in str(exc_info.value).lower()

    def test_supports_spec_returns_bool(self) -> None:
        """Test that supports_spec returns boolean."""
        handler = AudioModalityHandler()

        valid_spec = MockSpec()
        invalid_spec = MockSpec(architecture=MockArchitecture(family="unsupported"))

        assert handler.supports_spec(valid_spec) is True
        assert handler.supports_spec(invalid_spec) is False


class TestIntegrationWithRegistry:
    """Integration tests with the handler registry."""

    def test_audio_handler_from_registry(self) -> None:
        """Test getting audio handler from registry."""
        spec = ModelSpec(
            name="audio_test",
            modality={"inputs": ["audio"], "outputs": ["audio"]},
            task={"domain": "speech"},
            architecture={"family": "transformer"},
        )
        handler = get_handler(spec)

        assert isinstance(handler, AudioModalityHandler)
        assert handler.name == "audio"

    def test_full_workflow(self) -> None:
        """Test complete audio handler workflow."""
        spec = ModelSpec(
            name="audio_test",
            modality={"inputs": ["audio"], "outputs": ["audio"]},
            task={"domain": "speech"},
            architecture={"family": "transformer"},
        )
        handler = get_handler(spec)

        base = BlueprintState(
            dims={"hidden_size": 256, "layers": 4, "heads": 4},
            family="transformer",
        )
        augmented = handler.augment_blueprint(spec, base, seed=42)

        assert augmented.sample_rate in DEFAULT_SAMPLE_RATES
        assert augmented.num_channels in {1, 2}

        components = handler.generate_components(spec, augmented, seed=42)
        assert components.data_loader_type == "audio_dataset"
        assert any("audio" in fragment for fragment in components.template_fragments)
