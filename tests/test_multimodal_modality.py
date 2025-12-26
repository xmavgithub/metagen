"""
Tests for MetaGen MultiModal Modality Handler

This module tests the MultiModalModalityHandler class which handles
text+image fusion models (CLIP-style).

Author: MetaGen Team
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from metagen.specs.schema import ModelSpec
from metagen.synth.architecture import BlueprintState
from metagen.synth.modalities import get_handler
from metagen.synth.modalities.base import ModalityComponents
from metagen.synth.modalities.multimodal import MultiModalModalityHandler


# Mock classes for testing without real specs
@dataclass
class MockModality:
    """Mock modality for testing."""

    inputs: list[str] = field(default_factory=lambda: ["text", "image"])
    outputs: list[str] = field(default_factory=lambda: ["text", "image"])


@dataclass
class MockArchitecture:
    """Mock architecture for testing."""

    family: str = "transformer"


@dataclass
class MockConstraints:
    """Mock constraints for testing."""

    context_window: str = "512"


@dataclass
class MockTraining:
    """Mock training config for testing."""

    objective: list[str] = field(default_factory=lambda: ["contrastive"])


@dataclass
class MockTask:
    """Mock task for testing."""

    type: str = "generation"
    domain: str = ""


@dataclass
class MockSpec:
    """Mock spec for testing MultiModalModalityHandler."""

    name: str = "multimodal_test"
    modality: MockModality = field(default_factory=MockModality)
    architecture: MockArchitecture = field(default_factory=MockArchitecture)
    constraints: MockConstraints = field(default_factory=MockConstraints)
    training: MockTraining = field(default_factory=MockTraining)
    task: MockTask = field(default_factory=MockTask)


@dataclass
class MockBlueprintState:
    """Mock BlueprintState for testing."""

    dims: dict = field(default_factory=lambda: {"hidden_size": 512, "layers": 6, "heads": 8})
    vocab_size: int | None = None
    max_seq_len: int | None = None
    image_size: int | None = None
    num_channels: int | None = None
    patch_size: int | None = None
    num_patches: int | None = None


class TestMultiModalModalityHandler:
    """Tests for the MultiModalModalityHandler class."""

    def test_handler_name(self) -> None:
        """Test that handler name is 'multimodal'."""
        handler = MultiModalModalityHandler()
        assert handler.name == "multimodal"

    def test_handler_supported_families(self) -> None:
        """Test supported architecture families."""
        handler = MultiModalModalityHandler()
        assert "transformer" in handler.supported_families
        assert "hybrid" in handler.supported_families

    def test_handler_repr(self) -> None:
        """Test string representation."""
        handler = MultiModalModalityHandler()
        assert repr(handler) == "MultiModalModalityHandler(name='multimodal')"


class TestAugmentBlueprint:
    """Tests for blueprint augmentation."""

    def test_augment_adds_text_fields(self) -> None:
        """Test that augment_blueprint adds text-specific fields."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.vocab_size is not None
        assert augmented.max_seq_len is not None

    def test_augment_adds_image_fields(self) -> None:
        """Test that augment_blueprint adds image-specific fields."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.image_size is not None
        assert augmented.patch_size is not None
        assert augmented.num_patches is not None

    def test_augment_preserves_dims(self) -> None:
        """Test that augmentation preserves original dims."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.dims == blueprint.dims

    def test_augment_deterministic(self) -> None:
        """Test that augmentation is deterministic given same seed."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        aug1 = handler.augment_blueprint(spec, blueprint, seed=42)
        aug2 = handler.augment_blueprint(spec, blueprint, seed=42)

        assert aug1.vocab_size == aug2.vocab_size
        assert aug1.image_size == aug2.image_size


class TestGenerateComponents:
    """Tests for component generation."""

    def test_returns_modality_components(self) -> None:
        """Test that generate_components returns ModalityComponents."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            vocab_size=50257,
            max_seq_len=512,
            image_size=224,
            num_channels=3,
            patch_size=16,
            num_patches=196,
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert isinstance(components, ModalityComponents)

    def test_multimodal_encoder_type(self) -> None:
        """Test encoder type is multimodal_encoder."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            vocab_size=50257,
            max_seq_len=512,
            image_size=224,
            num_channels=3,
            patch_size=16,
            num_patches=196,
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.encoder_type == "multimodal_encoder"

    def test_multimodal_embedding_type(self) -> None:
        """Test embedding type is multimodal_embedding."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            vocab_size=50257,
            max_seq_len=512,
            image_size=224,
            num_channels=3,
            patch_size=16,
            num_patches=196,
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.embedding_type == "multimodal_embedding"

    def test_multimodal_dataset_type(self) -> None:
        """Test data loader type is multimodal_dataset."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            vocab_size=50257,
            max_seq_len=512,
            image_size=224,
            num_channels=3,
            patch_size=16,
            num_patches=196,
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.data_loader_type == "multimodal_dataset"


class TestLossTypeSelection:
    """Tests for loss type selection - including the fixed bug."""

    def test_contrastive_objective_uses_contrastive_loss(self) -> None:
        """Test that contrastive objective uses contrastive loss."""
        handler = MultiModalModalityHandler()
        spec = MockSpec(training=MockTraining(objective=["contrastive"]))
        blueprint = MockBlueprintState(
            vocab_size=50257,
            max_seq_len=512,
            image_size=224,
            num_channels=3,
            patch_size=16,
            num_patches=196,
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.loss_type == "contrastive"

    def test_classification_objective_uses_cross_entropy(self) -> None:
        """Test that classification objective uses cross_entropy loss.

        This test verifies the bug fix where all objectives incorrectly
        returned 'contrastive' loss.
        """
        handler = MultiModalModalityHandler()
        spec = MockSpec(training=MockTraining(objective=["classification"]))
        blueprint = MockBlueprintState(
            vocab_size=50257,
            max_seq_len=512,
            image_size=224,
            num_channels=3,
            patch_size=16,
            num_patches=196,
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.loss_type == "cross_entropy"

    def test_generation_objective_uses_cross_entropy(self) -> None:
        """Test that generation objective uses cross_entropy loss."""
        handler = MultiModalModalityHandler()
        spec = MockSpec(training=MockTraining(objective=["generation"]))
        blueprint = MockBlueprintState(
            vocab_size=50257,
            max_seq_len=512,
            image_size=224,
            num_channels=3,
            patch_size=16,
            num_patches=196,
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.loss_type == "cross_entropy"

    def test_unknown_objective_defaults_to_contrastive(self) -> None:
        """Test that unknown objectives default to contrastive (CLIP-style)."""
        handler = MultiModalModalityHandler()
        spec = MockSpec(training=MockTraining(objective=["unknown_objective"]))
        blueprint = MockBlueprintState(
            vocab_size=50257,
            max_seq_len=512,
            image_size=224,
            num_channels=3,
            patch_size=16,
            num_patches=196,
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.loss_type == "contrastive"

    def test_multiple_objectives_with_contrastive(self) -> None:
        """Test that contrastive takes priority when mixed with others."""
        handler = MultiModalModalityHandler()
        spec = MockSpec(training=MockTraining(objective=["classification", "contrastive"]))
        blueprint = MockBlueprintState(
            vocab_size=50257,
            max_seq_len=512,
            image_size=224,
            num_channels=3,
            patch_size=16,
            num_patches=196,
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.loss_type == "contrastive"


class TestConfigContents:
    """Tests for config dictionary contents."""

    def test_config_contains_text_params(self) -> None:
        """Test that config contains text-specific parameters."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            vocab_size=50257,
            max_seq_len=512,
            image_size=224,
            num_channels=3,
            patch_size=16,
            num_patches=196,
            dims={"hidden_size": 512, "layers": 6, "heads": 8},
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.config["vocab_size"] == 50257
        assert components.config["max_seq_len"] == 512

    def test_config_contains_image_params(self) -> None:
        """Test that config contains image-specific parameters."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            vocab_size=50257,
            max_seq_len=512,
            image_size=224,
            num_channels=3,
            patch_size=16,
            num_patches=196,
            dims={"hidden_size": 512, "layers": 6, "heads": 8},
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.config["image_size"] == 224
        assert components.config["num_channels"] == 3
        assert components.config["patch_size"] == 16
        assert components.config["num_patches"] == 196

    def test_config_contains_model_params(self) -> None:
        """Test that config contains model dimension parameters."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            vocab_size=50257,
            max_seq_len=512,
            image_size=224,
            num_channels=3,
            patch_size=16,
            num_patches=196,
            dims={"hidden_size": 512, "layers": 6, "heads": 8},
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.config["hidden_size"] == 512
        assert components.config["num_layers"] == 6
        assert components.config["num_heads"] == 8


class TestTemplateFragments:
    """Tests for template fragment generation."""

    def test_includes_token_embedding(self) -> None:
        """Test that fragments include token embedding for text."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        fragments = handler.get_template_fragments(spec, blueprint)

        assert "embeddings/token_embedding.py.j2" in fragments

    def test_includes_patch_embedding(self) -> None:
        """Test that fragments include patch embedding for images."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        fragments = handler.get_template_fragments(spec, blueprint)

        assert "embeddings/patch_embedding.py.j2" in fragments

    def test_includes_cross_attention(self) -> None:
        """Test that fragments include cross-attention fusion."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        fragments = handler.get_template_fragments(spec, blueprint)

        assert "fusion/cross_attention.py.j2" in fragments

    def test_includes_multimodal_datasets(self) -> None:
        """Test that fragments include multimodal datasets."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        fragments = handler.get_template_fragments(spec, blueprint)

        assert "data/multimodal_datasets.py.j2" in fragments


class TestValidateSpec:
    """Tests for spec validation."""

    def test_valid_multimodal_spec(self) -> None:
        """Test that valid multimodal spec passes validation."""
        handler = MultiModalModalityHandler()
        spec = MockSpec()

        # Should not raise
        handler.validate_spec(spec)

    def test_invalid_family_raises(self) -> None:
        """Test that unsupported family raises ValueError."""
        handler = MultiModalModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="cnn"))

        with pytest.raises(ValueError) as exc_info:
            handler.validate_spec(spec)

        assert "cnn" in str(exc_info.value).lower()

    def test_missing_text_modality_raises(self) -> None:
        """Test that missing text modality raises ValueError."""
        handler = MultiModalModalityHandler()
        spec = MockSpec(modality=MockModality(inputs=["image"], outputs=["image"]))

        with pytest.raises(ValueError) as exc_info:
            handler.validate_spec(spec)

        assert "text" in str(exc_info.value).lower() or "image" in str(exc_info.value).lower()

    def test_missing_image_modality_raises(self) -> None:
        """Test that missing image modality raises ValueError."""
        handler = MultiModalModalityHandler()
        spec = MockSpec(modality=MockModality(inputs=["text"], outputs=["text"]))

        with pytest.raises(ValueError) as exc_info:
            handler.validate_spec(spec)

        assert "text" in str(exc_info.value).lower() or "image" in str(exc_info.value).lower()

    def test_supports_spec_returns_bool(self) -> None:
        """Test that supports_spec returns boolean."""
        handler = MultiModalModalityHandler()

        valid_spec = MockSpec()
        invalid_spec = MockSpec(architecture=MockArchitecture(family="cnn"))

        assert handler.supports_spec(valid_spec) is True
        assert handler.supports_spec(invalid_spec) is False


class TestIntegrationWithRegistry:
    """Integration tests with the handler registry."""

    def test_multimodal_handler_from_registry(self) -> None:
        """Test getting multimodal handler from registry."""
        spec = ModelSpec(
            name="multimodal_clip",
            modality={"inputs": ["text", "image"], "outputs": ["text", "image"]},
            training={"objective": ["contrastive"]},
            architecture={"family": "transformer"},
        )
        handler = get_handler(spec)

        assert isinstance(handler, MultiModalModalityHandler)
        assert handler.name == "multimodal"

    def test_full_workflow(self) -> None:
        """Test complete multimodal handler workflow."""
        spec = ModelSpec(
            name="multimodal_clip",
            modality={"inputs": ["text", "image"], "outputs": ["text", "image"]},
            training={"objective": ["contrastive"]},
            architecture={"family": "transformer"},
        )
        handler = get_handler(spec)

        base = BlueprintState(
            dims={"hidden_size": 512, "layers": 6, "heads": 8},
            family="transformer",
        )
        augmented = handler.augment_blueprint(spec, base, seed=11)

        assert augmented.vocab_size is not None
        assert augmented.max_seq_len is not None
        assert augmented.image_size is not None
        assert augmented.patch_size is not None

        components = handler.generate_components(spec, augmented, seed=11)
        assert components.loss_type == "contrastive"
        assert "fusion/cross_attention.py.j2" in components.template_fragments
