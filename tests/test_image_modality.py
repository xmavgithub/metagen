"""
Tests for MetaGen Image Modality Handler

This module tests the ImageModalityHandler class and related functionality
in metagen.synth.modalities.image, including:
- Blueprint augmentation with image_size, patch_size, num_channels
- Component generation for vision models (ViT, CNN, diffusion)
- Image size parsing and validation
- Spec validation

Author: MetaGen Team
Created: 2025-12-25
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from metagen.synth.modalities.base import ModalityComponents
from metagen.synth.modalities.image import (
    DEFAULT_PATCH_SIZE,
    IMAGE_SIZE_PRESETS,
    RGB_CHANNELS,
    ImageModalityHandler,
)


# Mock classes for testing without real specs
@dataclass
class MockModality:
    """Mock modality for testing."""

    inputs: list[str] = field(default_factory=lambda: ["image"])
    outputs: list[str] = field(default_factory=lambda: ["image"])


@dataclass
class MockArchitecture:
    """Mock architecture for testing."""

    family: str = "transformer"


@dataclass
class MockParameterBudget:
    """Mock parameter budget for testing."""

    max: str = "100M"


@dataclass
class MockConstraints:
    """Mock constraints for testing."""

    image_size: int | str = 224
    patch_size: int | None = None
    num_channels: int | None = None
    parameter_budget: MockParameterBudget | None = None


@dataclass
class MockTask:
    """Mock task for testing."""

    type: str = "classification"
    domain: str = "image"


@dataclass
class MockTraining:
    """Mock training config for testing."""

    objective: list[str] = field(default_factory=lambda: ["classification"])


@dataclass
class MockSpec:
    """Mock spec for testing ImageModalityHandler."""

    modality: MockModality = field(default_factory=MockModality)
    architecture: MockArchitecture = field(default_factory=MockArchitecture)
    constraints: MockConstraints = field(default_factory=MockConstraints)
    task: MockTask = field(default_factory=MockTask)
    training: MockTraining = field(default_factory=MockTraining)


@dataclass
class MockBlueprintState:
    """Mock BlueprintState for testing."""

    dims: dict = field(default_factory=lambda: {"hidden_size": 768, "layers": 12, "heads": 12})
    image_size: int | None = None
    num_channels: int | None = None
    patch_size: int | None = None
    num_patches: int | None = None


class TestImageModalityHandler:
    """Tests for the ImageModalityHandler class."""

    def test_handler_name(self) -> None:
        """Test that handler name is 'image'."""
        handler = ImageModalityHandler()
        assert handler.name == "image"

    def test_handler_supported_families(self) -> None:
        """Test supported architecture families."""
        handler = ImageModalityHandler()
        assert "transformer" in handler.supported_families
        assert "cnn" in handler.supported_families
        assert "diffusion" in handler.supported_families
        assert "hybrid" in handler.supported_families

    def test_handler_repr(self) -> None:
        """Test string representation."""
        handler = ImageModalityHandler()
        assert repr(handler) == "ImageModalityHandler(name='image')"


class TestAugmentBlueprint:
    """Tests for blueprint augmentation."""

    def test_augment_adds_image_size(self) -> None:
        """Test that augment_blueprint adds image_size."""
        handler = ImageModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.image_size == 224

    def test_augment_adds_num_channels(self) -> None:
        """Test that augment_blueprint adds num_channels."""
        handler = ImageModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.num_channels == RGB_CHANNELS

    def test_augment_adds_patch_size(self) -> None:
        """Test that augment_blueprint adds patch_size."""
        handler = ImageModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.patch_size == DEFAULT_PATCH_SIZE

    def test_augment_computes_num_patches(self) -> None:
        """Test that num_patches is computed correctly."""
        handler = ImageModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        # 224 / 16 = 14, 14 * 14 = 196
        assert augmented.num_patches == 196

    def test_augment_preserves_dims(self) -> None:
        """Test that augmentation preserves original dims."""
        handler = ImageModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.dims == blueprint.dims

    def test_augment_deterministic(self) -> None:
        """Test that augmentation is deterministic given same seed."""
        handler = ImageModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        aug1 = handler.augment_blueprint(spec, blueprint, seed=42)
        aug2 = handler.augment_blueprint(spec, blueprint, seed=42)

        assert aug1.image_size == aug2.image_size
        assert aug1.patch_size == aug2.patch_size
        assert aug1.num_patches == aug2.num_patches


class TestImageSizeParsing:
    """Tests for image size determination."""

    def test_parse_numeric_size(self) -> None:
        """Test parsing numeric image size."""
        handler = ImageModalityHandler()
        spec = MockSpec(constraints=MockConstraints(image_size=384))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.image_size == 384

    def test_parse_preset_imagenet(self) -> None:
        """Test parsing 'imagenet' preset."""
        handler = ImageModalityHandler()
        spec = MockSpec(constraints=MockConstraints(image_size="imagenet"))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.image_size == IMAGE_SIZE_PRESETS["imagenet"]

    def test_parse_preset_clip(self) -> None:
        """Test parsing 'clip' preset."""
        handler = ImageModalityHandler()
        spec = MockSpec(constraints=MockConstraints(image_size="clip"))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.image_size == 224

    def test_parse_preset_sd(self) -> None:
        """Test parsing 'sd' (Stable Diffusion) preset."""
        handler = ImageModalityHandler()
        spec = MockSpec(constraints=MockConstraints(image_size="sd"))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.image_size == 512

    def test_default_for_generation_task(self) -> None:
        """Test default size for generation tasks."""
        handler = ImageModalityHandler()

        @dataclass
        class NoSizeConstraints:
            pass

        spec = MockSpec(
            constraints=NoSizeConstraints(),
            task=MockTask(type="generation"),
        )
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.image_size == 512  # Larger for generation


class TestPatchSizeDetermination:
    """Tests for patch size determination."""

    def test_cnn_family_no_patches(self) -> None:
        """Test that CNN family uses patch_size=1 (pixel-level)."""
        handler = ImageModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="cnn"))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.patch_size == 1

    def test_explicit_patch_size(self) -> None:
        """Test using explicitly specified patch size."""
        handler = ImageModalityHandler()
        spec = MockSpec(constraints=MockConstraints(image_size=224, patch_size=14))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.patch_size == 14

    def test_patch_size_divisibility(self) -> None:
        """Test that patch size divides image size evenly."""
        handler = ImageModalityHandler()
        spec = MockSpec(constraints=MockConstraints(image_size=224))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.image_size % augmented.patch_size == 0

    def test_large_model_uses_smaller_patches(self) -> None:
        """Test that larger models may use smaller patch sizes."""
        handler = ImageModalityHandler()
        spec = MockSpec(
            constraints=MockConstraints(
                image_size=224,
                parameter_budget=MockParameterBudget(max="2B"),
            )
        )
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        # Large models typically use 14x14 patches
        assert augmented.patch_size == 14


class TestGenerateComponents:
    """Tests for component generation."""

    def test_returns_modality_components(self) -> None:
        """Test that generate_components returns ModalityComponents."""
        handler = ImageModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            image_size=224, num_channels=3, patch_size=16, num_patches=196
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert isinstance(components, ModalityComponents)

    def test_vit_encoder_for_transformer(self) -> None:
        """Test ViT encoder for transformer family."""
        handler = ImageModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="transformer"))
        blueprint = MockBlueprintState(
            image_size=224, num_channels=3, patch_size=16, num_patches=196
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.encoder_type == "vit_encoder"
        assert components.embedding_type == "patch_embedding"

    def test_resnet_encoder_for_cnn(self) -> None:
        """Test ResNet encoder for CNN family."""
        handler = ImageModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="cnn"))
        blueprint = MockBlueprintState(image_size=224, num_channels=3, patch_size=1, num_patches=1)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.encoder_type == "resnet_encoder"
        assert components.embedding_type == "conv_stem"

    def test_unet_encoder_for_diffusion(self) -> None:
        """Test U-Net encoder for diffusion family."""
        handler = ImageModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="diffusion"))
        blueprint = MockBlueprintState(image_size=512, num_channels=3, patch_size=1, num_patches=1)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.encoder_type == "unet_encoder"

    def test_data_loader_type(self) -> None:
        """Test that data loader type is image_folder."""
        handler = ImageModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            image_size=224, num_channels=3, patch_size=16, num_patches=196
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.data_loader_type == "image_folder"

    def test_diffusion_loss_for_diffusion(self) -> None:
        """Test diffusion loss for diffusion objectives."""
        handler = ImageModalityHandler()
        spec = MockSpec(training=MockTraining(objective=["diffusion"]))
        blueprint = MockBlueprintState(
            image_size=512, num_channels=3, patch_size=16, num_patches=1024
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.loss_type == "diffusion_loss"

    def test_config_contains_image_params(self) -> None:
        """Test that config contains image-specific parameters."""
        handler = ImageModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            image_size=224,
            num_channels=3,
            patch_size=16,
            num_patches=196,
            dims={"hidden_size": 768, "layers": 12, "heads": 12},
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.config["image_size"] == 224
        assert components.config["num_channels"] == 3
        assert components.config["patch_size"] == 16
        assert components.config["num_patches"] == 196
        assert components.config["hidden_size"] == 768

    def test_includes_torchvision_import(self) -> None:
        """Test that torchvision transforms is imported."""
        handler = ImageModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            image_size=224, num_channels=3, patch_size=16, num_patches=196
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert "from torchvision import transforms" in components.additional_imports


class TestValidateSpec:
    """Tests for spec validation."""

    def test_valid_image_spec(self) -> None:
        """Test that valid image spec passes validation."""
        handler = ImageModalityHandler()
        spec = MockSpec()

        # Should not raise
        handler.validate_spec(spec)

    def test_invalid_family_raises(self) -> None:
        """Test that unsupported family raises ValueError."""
        handler = ImageModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="unsupported"))

        with pytest.raises(ValueError) as exc_info:
            handler.validate_spec(spec)

        assert "unsupported" in str(exc_info.value)

    def test_missing_image_modality_raises(self) -> None:
        """Test that missing image modality raises ValueError."""
        handler = ImageModalityHandler()
        spec = MockSpec(modality=MockModality(inputs=["text"], outputs=["text"]))

        with pytest.raises(ValueError) as exc_info:
            handler.validate_spec(spec)

        assert "image" in str(exc_info.value).lower()

    def test_image_in_outputs_only_valid(self) -> None:
        """Test that image in outputs only is valid."""
        handler = ImageModalityHandler()
        spec = MockSpec(modality=MockModality(inputs=["text"], outputs=["image"]))

        # Should not raise - image is in outputs
        handler.validate_spec(spec)

    def test_supports_spec_returns_bool(self) -> None:
        """Test that supports_spec returns boolean."""
        handler = ImageModalityHandler()

        valid_spec = MockSpec()
        invalid_spec = MockSpec(architecture=MockArchitecture(family="unsupported"))

        assert handler.supports_spec(valid_spec) is True
        assert handler.supports_spec(invalid_spec) is False


class TestTemplateFragments:
    """Tests for template fragment generation."""

    def test_vit_includes_patch_embedding(self) -> None:
        """Test that ViT includes patch embedding fragment."""
        handler = ImageModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="transformer"))
        blueprint = MockBlueprintState()

        fragments = handler.get_template_fragments(spec, blueprint)

        assert "embeddings/patch_embedding.py.j2" in fragments

    def test_vit_includes_vit_encoder(self) -> None:
        """Test that ViT includes vit encoder fragment."""
        handler = ImageModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="transformer"))
        blueprint = MockBlueprintState()

        fragments = handler.get_template_fragments(spec, blueprint)

        assert "encoders/vit_encoder.py.j2" in fragments

    def test_diffusion_includes_unet(self) -> None:
        """Test that diffusion includes U-Net fragments."""
        handler = ImageModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="diffusion"))
        blueprint = MockBlueprintState()

        fragments = handler.get_template_fragments(spec, blueprint)

        assert "encoders/unet_encoder.py.j2" in fragments

    def test_includes_image_datasets(self) -> None:
        """Test that image datasets fragment is included."""
        handler = ImageModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        fragments = handler.get_template_fragments(spec, blueprint)

        assert "data/image_datasets.py.j2" in fragments


class TestNumPatches:
    """Tests for num_patches computation."""

    def test_standard_vit_patches(self) -> None:
        """Test standard ViT patch count."""
        handler = ImageModalityHandler()
        spec = MockSpec(constraints=MockConstraints(image_size=224))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        # 224/16 = 14, 14*14 = 196
        assert augmented.num_patches == 196

    def test_larger_image_more_patches(self) -> None:
        """Test that larger images produce more patches."""
        handler = ImageModalityHandler()
        spec = MockSpec(constraints=MockConstraints(image_size=384))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        # 384/16 = 24, 24*24 = 576
        assert augmented.num_patches == 576

    def test_smaller_patch_more_patches(self) -> None:
        """Test that smaller patches produce more patches."""
        handler = ImageModalityHandler()
        spec = MockSpec(constraints=MockConstraints(image_size=224, patch_size=14))
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        # 224/14 = 16, 16*16 = 256
        assert augmented.num_patches == 256


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_seed_same_components(self) -> None:
        """Test that same seed produces identical components."""
        handler = ImageModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            image_size=224, num_channels=3, patch_size=16, num_patches=196
        )

        comp1 = handler.generate_components(spec, blueprint, seed=42)
        comp2 = handler.generate_components(spec, blueprint, seed=42)

        assert comp1.encoder_type == comp2.encoder_type
        assert comp1.embedding_type == comp2.embedding_type
        assert comp1.loss_type == comp2.loss_type
        assert comp1.config == comp2.config
