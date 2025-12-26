"""
Tests for MetaGen Modality Handler Base Classes

This module tests the abstract base classes and dataclasses in
metagen.synth.modalities.base, including:
- ModalityComponents dataclass
- ModalityHandler abstract base class

Author: MetaGen Team
Created: 2025-12-25
"""

from __future__ import annotations

import pytest

from metagen.synth.modalities.base import ModalityComponents, ModalityHandler


class TestModalityComponents:
    """Tests for the ModalityComponents dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating ModalityComponents with required fields."""
        components = ModalityComponents(encoder_type="transformer_encoder")

        assert components.encoder_type == "transformer_encoder"
        assert components.decoder_type is None
        assert components.embedding_type == "token_embedding"
        assert components.data_loader_type == "synthetic"
        assert components.loss_type == "cross_entropy"
        assert components.template_fragments == ()
        assert components.additional_imports == ()
        assert components.config == {}

    def test_full_creation(self) -> None:
        """Test creating ModalityComponents with all fields."""
        components = ModalityComponents(
            encoder_type="vit_encoder",
            decoder_type="diffusion",
            embedding_type="patch_embedding",
            data_loader_type="image_folder",
            loss_type="diffusion_loss",
            template_fragments=("embeddings/patch.j2", "encoders/vit.j2"),
            additional_imports=("from torchvision import transforms",),
            config={"patch_size": 16, "image_size": 224},
        )

        assert components.encoder_type == "vit_encoder"
        assert components.decoder_type == "diffusion"
        assert components.embedding_type == "patch_embedding"
        assert components.data_loader_type == "image_folder"
        assert components.loss_type == "diffusion_loss"
        assert components.template_fragments == ("embeddings/patch.j2", "encoders/vit.j2")
        assert components.additional_imports == ("from torchvision import transforms",)
        assert components.config == {"patch_size": 16, "image_size": 224}

    def test_immutability(self) -> None:
        """Test that ModalityComponents is frozen (immutable)."""
        components = ModalityComponents(encoder_type="transformer_encoder")

        with pytest.raises(AttributeError):
            components.encoder_type = "cnn_encoder"

    def test_with_config(self) -> None:
        """Test the with_config method for updating config."""
        base = ModalityComponents(
            encoder_type="vit_encoder",
            config={"patch_size": 16},
        )

        updated = base.with_config(image_size=224, num_channels=3)

        # Original should be unchanged
        assert base.config == {"patch_size": 16}

        # Updated should have all values
        assert updated.config == {
            "patch_size": 16,
            "image_size": 224,
            "num_channels": 3,
        }

        # Other fields should be preserved
        assert updated.encoder_type == "vit_encoder"

    def test_with_config_overwrites(self) -> None:
        """Test that with_config overwrites existing keys."""
        base = ModalityComponents(
            encoder_type="vit_encoder",
            config={"patch_size": 16},
        )

        updated = base.with_config(patch_size=32)

        assert updated.config == {"patch_size": 32}

    def test_list_to_tuple_conversion(self) -> None:
        """Test that lists are converted to tuples for immutability."""
        # Create with lists (which should be converted)
        components = ModalityComponents(
            encoder_type="transformer_encoder",
            template_fragments=["frag1.j2", "frag2.j2"],  # type: ignore
            additional_imports=["import torch"],  # type: ignore
        )

        # Should be tuples now
        assert isinstance(components.template_fragments, tuple)
        assert isinstance(components.additional_imports, tuple)
        assert components.template_fragments == ("frag1.j2", "frag2.j2")
        assert components.additional_imports == ("import torch",)


class ConcreteHandler(ModalityHandler):
    """Concrete implementation of ModalityHandler for testing."""

    @property
    def name(self) -> str:
        return "test"

    @property
    def supported_families(self) -> list[str]:
        return ["transformer", "cnn"]

    def augment_blueprint(self, spec, blueprint, seed):
        """Mock implementation that returns blueprint unchanged."""
        return blueprint

    def generate_components(self, spec, blueprint, seed):
        """Mock implementation returning basic components."""
        return ModalityComponents(
            encoder_type="test_encoder",
            embedding_type="test_embedding",
        )

    def get_template_fragments(self, spec, blueprint):
        """Mock implementation returning empty list."""
        return []


class TestModalityHandler:
    """Tests for the ModalityHandler abstract base class."""

    def test_concrete_handler_creation(self) -> None:
        """Test creating a concrete handler implementation."""
        handler = ConcreteHandler()

        assert handler.name == "test"
        assert handler.supported_families == ["transformer", "cnn"]

    def test_repr(self) -> None:
        """Test string representation of handler."""
        handler = ConcreteHandler()

        assert repr(handler) == "ConcreteHandler(name='test')"

    def test_valid_families_constant(self) -> None:
        """Test that VALID_FAMILIES contains expected values."""
        expected = ("transformer", "cnn", "diffusion", "hybrid", "rnn", "mlp")
        assert ModalityHandler.VALID_FAMILIES == expected

    def test_abstract_methods_required(self) -> None:
        """Test that abstract methods must be implemented."""

        # This should raise TypeError because abstract methods not implemented
        class IncompleteHandler(ModalityHandler):
            pass

        with pytest.raises(TypeError) as exc_info:
            IncompleteHandler()  # type: ignore

        # Check that the error mentions abstract methods
        assert "abstract" in str(exc_info.value).lower()

    def test_generate_components_returns_correct_type(self) -> None:
        """Test that generate_components returns ModalityComponents."""
        handler = ConcreteHandler()

        # Create mock spec and blueprint
        class MockSpec:
            class modality:
                inputs = ["text"]
                outputs = ["text"]

            class architecture:
                family = "transformer"

        class MockBlueprint:
            pass

        components = handler.generate_components(MockSpec(), MockBlueprint(), seed=42)

        assert isinstance(components, ModalityComponents)
        assert components.encoder_type == "test_encoder"

    def test_supports_spec_valid(self) -> None:
        """Test supports_spec returns True for valid specs."""
        handler = ConcreteHandler()

        class ValidSpec:
            class modality:
                inputs = ["text"]
                outputs = ["text"]

            class architecture:
                family = "transformer"

        assert handler.supports_spec(ValidSpec()) is True

    def test_supports_spec_invalid_family(self) -> None:
        """Test supports_spec returns False for unsupported family."""
        handler = ConcreteHandler()

        class InvalidSpec:
            class modality:
                inputs = ["text"]
                outputs = ["text"]

            class architecture:
                family = "unsupported_family"

        assert handler.supports_spec(InvalidSpec()) is False

    def test_validate_spec_raises_for_invalid_family(self) -> None:
        """Test validate_spec raises ValueError for unsupported family."""
        handler = ConcreteHandler()

        class InvalidSpec:
            class modality:
                inputs = ["text"]
                outputs = ["text"]

            class architecture:
                family = "rnn"  # Not in ConcreteHandler.supported_families

        with pytest.raises(ValueError) as exc_info:
            handler.validate_spec(InvalidSpec())

        assert "rnn" in str(exc_info.value)
        assert "not supported" in str(exc_info.value)

    def test_validate_spec_passes_for_valid_family(self) -> None:
        """Test validate_spec doesn't raise for supported family."""
        handler = ConcreteHandler()

        class ValidSpec:
            class modality:
                inputs = ["text"]
                outputs = ["text"]

            class architecture:
                family = "transformer"

        # Should not raise
        handler.validate_spec(ValidSpec())


class TestModalityHandlerDeterminism:
    """Tests for deterministic behavior of modality handlers."""

    def test_same_seed_same_components(self) -> None:
        """Test that same seed produces same components."""
        handler = ConcreteHandler()

        class MockSpec:
            class modality:
                inputs = ["text"]
                outputs = ["text"]

            class architecture:
                family = "transformer"

        class MockBlueprint:
            pass

        # Generate components twice with same seed
        components1 = handler.generate_components(MockSpec(), MockBlueprint(), seed=42)
        components2 = handler.generate_components(MockSpec(), MockBlueprint(), seed=42)

        # Should be identical
        assert components1.encoder_type == components2.encoder_type
        assert components1.embedding_type == components2.embedding_type
        assert components1.config == components2.config


class TestModalityComponentsEquality:
    """Tests for ModalityComponents equality and hashing."""

    def test_equality(self) -> None:
        """Test that identical components are equal."""
        c1 = ModalityComponents(
            encoder_type="transformer_encoder",
            decoder_type="autoregressive",
            config={"vocab_size": 50257},
        )
        c2 = ModalityComponents(
            encoder_type="transformer_encoder",
            decoder_type="autoregressive",
            config={"vocab_size": 50257},
        )

        assert c1 == c2

    def test_inequality(self) -> None:
        """Test that different components are not equal."""
        c1 = ModalityComponents(encoder_type="transformer_encoder")
        c2 = ModalityComponents(encoder_type="cnn_encoder")

        assert c1 != c2

    def test_not_hashable_with_config(self) -> None:
        """Test that ModalityComponents with config dict is not hashable.

        Note: The config field is a mutable dict, which makes the dataclass
        unhashable even though it's frozen. This is a known limitation.
        """
        c1 = ModalityComponents(encoder_type="transformer_encoder")

        # Should NOT be hashable because config is a dict
        with pytest.raises(TypeError, match="unhashable"):
            hash(c1)
