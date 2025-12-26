"""
Tests for MetaGen Video Modality Handler

This module tests the VideoModalityHandler class and related functionality
including frame size selection, frame count, and temporal encoder configuration.

Author: MetaGen Team
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from metagen.specs.schema import ModelSpec
from metagen.synth.architecture import BlueprintState
from metagen.synth.modalities import get_handler
from metagen.synth.modalities.base import ModalityComponents
from metagen.synth.modalities.video import (
    DEFAULT_FRAME_COUNTS,
    DEFAULT_FRAME_SIZES,
    VideoModalityHandler,
)


# Mock classes for testing without real specs
@dataclass
class MockModality:
    """Mock modality for testing."""

    inputs: list[str] = field(default_factory=lambda: ["video"])
    outputs: list[str] = field(default_factory=lambda: ["video"])


@dataclass
class MockArchitecture:
    """Mock architecture for testing."""

    family: str = "transformer"


@dataclass
class MockTask:
    """Mock task for testing."""

    type: str = "generation"
    domain: str = "video"


@dataclass
class MockTraining:
    """Mock training config for testing."""

    objective: list[str] = field(default_factory=lambda: ["reconstruction"])


@dataclass
class MockSpec:
    """Mock spec for testing VideoModalityHandler."""

    name: str = "video_test"
    description: str = "Test video model"
    modality: MockModality = field(default_factory=MockModality)
    architecture: MockArchitecture = field(default_factory=MockArchitecture)
    task: MockTask = field(default_factory=MockTask)
    training: MockTraining = field(default_factory=MockTraining)


@dataclass
class MockBlueprintState:
    """Mock BlueprintState for testing."""

    dims: dict = field(default_factory=lambda: {"hidden_size": 512, "layers": 6, "heads": 8})
    image_size: int | None = None
    num_channels: int | None = None
    patch_size: int | None = None
    num_patches: int | None = None
    max_seq_len: int | None = None


class TestVideoModalityHandler:
    """Tests for the VideoModalityHandler class."""

    def test_handler_name(self) -> None:
        """Test that handler name is 'video'."""
        handler = VideoModalityHandler()
        assert handler.name == "video"

    def test_handler_supported_families(self) -> None:
        """Test supported architecture families."""
        handler = VideoModalityHandler()
        assert "transformer" in handler.supported_families
        assert "cnn" in handler.supported_families
        assert "hybrid" in handler.supported_families

    def test_handler_repr(self) -> None:
        """Test string representation."""
        handler = VideoModalityHandler()
        assert repr(handler) == "VideoModalityHandler(name='video')"


class TestAugmentBlueprint:
    """Tests for blueprint augmentation."""

    def test_augment_adds_image_size(self) -> None:
        """Test that augment_blueprint adds image_size (frame size)."""
        handler = VideoModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.image_size in DEFAULT_FRAME_SIZES

    def test_augment_adds_num_channels(self) -> None:
        """Test that augment_blueprint adds num_channels (3 for RGB)."""
        handler = VideoModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.num_channels == 3

    def test_augment_adds_frame_count(self) -> None:
        """Test that augment_blueprint adds frame count via max_seq_len."""
        handler = VideoModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.max_seq_len in DEFAULT_FRAME_COUNTS

    def test_augment_preserves_dims(self) -> None:
        """Test that augmentation preserves original dims."""
        handler = VideoModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.dims == blueprint.dims

    def test_augment_deterministic(self) -> None:
        """Test that augmentation is deterministic given same seed."""
        handler = VideoModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        aug1 = handler.augment_blueprint(spec, blueprint, seed=42)
        aug2 = handler.augment_blueprint(spec, blueprint, seed=42)

        assert aug1.image_size == aug2.image_size
        assert aug1.max_seq_len == aug2.max_seq_len
        assert aug1.patch_size == aug2.patch_size


class TestFrameSizeSelection:
    """Tests for frame size selection logic."""

    def test_realtime_model_uses_224(self) -> None:
        """Test that realtime models use 224 frame size."""
        handler = VideoModalityHandler()
        spec = MockSpec(name="realtime_video_model")
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.image_size == 224

    def test_avatar_model_uses_224(self) -> None:
        """Test that avatar models use 224 frame size."""
        handler = VideoModalityHandler()
        spec = MockSpec(name="avatar_generator")
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.image_size == 224

    def test_general_video_random_frame_size(self) -> None:
        """Test that general video randomly selects frame size."""
        handler = VideoModalityHandler()
        spec = MockSpec(name="generic_video")
        blueprint = MockBlueprintState()

        # Run multiple times to verify randomness
        sizes_seen = set()
        for seed in range(100):
            augmented = handler.augment_blueprint(spec, blueprint, seed=seed)
            sizes_seen.add(augmented.image_size)

        # Should see multiple frame sizes
        assert len(sizes_seen) > 1


class TestFrameCountSelection:
    """Tests for frame count selection."""

    def test_realtime_model_uses_8_frames(self) -> None:
        """Test that realtime models use 8 frames."""
        handler = VideoModalityHandler()
        spec = MockSpec(name="realtime_video")
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.max_seq_len == 8

    def test_avatar_model_uses_8_frames(self) -> None:
        """Test that avatar models use 8 frames."""
        handler = VideoModalityHandler()
        spec = MockSpec(name="avatar_model")
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.max_seq_len == 8


class TestPatchSizeSelection:
    """Tests for patch size selection in video models."""

    def test_transformer_has_patch_size(self) -> None:
        """Test that transformer family has patch_size set."""
        handler = VideoModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="transformer"))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.patch_size is not None
        assert augmented.patch_size > 0

    def test_cnn_has_no_patch_size(self) -> None:
        """Test that CNN family has patch_size=None."""
        handler = VideoModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="cnn"))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.patch_size is None

    def test_patch_size_divides_frame_size(self) -> None:
        """Test that patch_size evenly divides frame size."""
        handler = VideoModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="transformer"))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        if augmented.patch_size:
            assert augmented.image_size % augmented.patch_size == 0


class TestNumPatches:
    """Tests for num_patches computation."""

    def test_transformer_has_num_patches(self) -> None:
        """Test that transformer family has num_patches computed."""
        handler = VideoModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="transformer"))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.num_patches is not None
        expected = (augmented.image_size // augmented.patch_size) ** 2
        assert augmented.num_patches == expected

    def test_cnn_has_no_num_patches(self) -> None:
        """Test that CNN family has num_patches=None."""
        handler = VideoModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="cnn"))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.num_patches is None


class TestGenerateComponents:
    """Tests for component generation."""

    def test_returns_modality_components(self) -> None:
        """Test that generate_components returns ModalityComponents."""
        handler = VideoModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            image_size=224, num_channels=3, patch_size=16, num_patches=196, max_seq_len=16
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert isinstance(components, ModalityComponents)

    def test_temporal_transformer_encoder_type(self) -> None:
        """Test encoder type for transformer family."""
        handler = VideoModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="transformer"))
        blueprint = MockBlueprintState(
            image_size=224, num_channels=3, patch_size=16, num_patches=196, max_seq_len=16
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.encoder_type == "temporal_transformer"

    def test_video_cnn_encoder_type(self) -> None:
        """Test encoder type for CNN family."""
        handler = VideoModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="cnn"))
        blueprint = MockBlueprintState(
            image_size=224, num_channels=3, patch_size=None, num_patches=None, max_seq_len=16
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.encoder_type == "video_cnn"

    def test_data_loader_type(self) -> None:
        """Test that data loader type is video_dataset."""
        handler = VideoModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            image_size=224, num_channels=3, patch_size=16, num_patches=196, max_seq_len=16
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.data_loader_type == "video_dataset"

    def test_diffusion_loss_type(self) -> None:
        """Test loss type for diffusion objectives."""
        handler = VideoModalityHandler()
        spec = MockSpec(training=MockTraining(objective=["diffusion"]))
        blueprint = MockBlueprintState(
            image_size=224, num_channels=3, patch_size=16, num_patches=196, max_seq_len=16
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.loss_type == "diffusion_loss"

    def test_reconstruction_loss_type(self) -> None:
        """Test loss type for reconstruction objectives."""
        handler = VideoModalityHandler()
        spec = MockSpec(training=MockTraining(objective=["reconstruction"]))
        blueprint = MockBlueprintState(
            image_size=224, num_channels=3, patch_size=16, num_patches=196, max_seq_len=16
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.loss_type == "reconstruction"

    def test_config_contains_video_params(self) -> None:
        """Test that config contains video-specific parameters."""
        handler = VideoModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            image_size=224,
            num_channels=3,
            patch_size=16,
            num_patches=196,
            max_seq_len=16,
            dims={"hidden_size": 512, "layers": 6, "heads": 8},
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.config["image_size"] == 224
        assert components.config["num_channels"] == 3
        assert components.config["frames"] == 16
        assert components.config["hidden_size"] == 512

    def test_template_fragments_include_video(self) -> None:
        """Test that template fragments include video components."""
        handler = VideoModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            image_size=224, num_channels=3, patch_size=16, num_patches=196, max_seq_len=16
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert "data/video_datasets.py.j2" in components.template_fragments


class TestValidateSpec:
    """Tests for spec validation."""

    def test_valid_video_spec(self) -> None:
        """Test that valid video spec passes validation."""
        handler = VideoModalityHandler()
        spec = MockSpec()

        # Should not raise
        handler.validate_spec(spec)

    def test_invalid_family_raises(self) -> None:
        """Test that unsupported family raises ValueError."""
        handler = VideoModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="unsupported"))

        with pytest.raises(ValueError) as exc_info:
            handler.validate_spec(spec)

        assert "unsupported" in str(exc_info.value)

    def test_missing_video_modality_raises(self) -> None:
        """Test that missing video modality raises ValueError."""
        handler = VideoModalityHandler()
        spec = MockSpec(modality=MockModality(inputs=["text"], outputs=["text"]))

        with pytest.raises(ValueError) as exc_info:
            handler.validate_spec(spec)

        assert "video" in str(exc_info.value).lower()

    def test_supports_spec_returns_bool(self) -> None:
        """Test that supports_spec returns boolean."""
        handler = VideoModalityHandler()

        valid_spec = MockSpec()
        invalid_spec = MockSpec(architecture=MockArchitecture(family="unsupported"))

        assert handler.supports_spec(valid_spec) is True
        assert handler.supports_spec(invalid_spec) is False


class TestIntegrationWithRegistry:
    """Integration tests with the handler registry."""

    def test_video_handler_from_registry(self) -> None:
        """Test getting video handler from registry."""
        spec = ModelSpec(
            name="video_test",
            modality={"inputs": ["video"], "outputs": ["video"]},
            task={"domain": "video"},
            architecture={"family": "cnn"},
        )
        handler = get_handler(spec)

        assert isinstance(handler, VideoModalityHandler)
        assert handler.name == "video"

    def test_full_workflow(self) -> None:
        """Test complete video handler workflow."""
        spec = ModelSpec(
            name="video_test",
            modality={"inputs": ["video"], "outputs": ["video"]},
            task={"domain": "video"},
            architecture={"family": "cnn"},
        )
        handler = get_handler(spec)

        base = BlueprintState(
            dims={"hidden_size": 256, "layers": 4, "heads": 4},
            family="cnn",
        )
        augmented = handler.augment_blueprint(spec, base, seed=7)

        assert augmented.image_size is not None
        assert augmented.num_channels == 3
        assert augmented.max_seq_len in DEFAULT_FRAME_COUNTS

        components = handler.generate_components(spec, augmented, seed=7)
        assert components.data_loader_type == "video_dataset"
        assert any("video" in fragment for fragment in components.template_fragments)
