"""
Tests for MetaGen Text Modality Handler

This module tests the TextModalityHandler class and related functionality
in metagen.synth.modalities.text, including:
- Blueprint augmentation with vocab_size and max_seq_len
- Component generation for text models
- Context window parsing
- Spec validation

Author: MetaGen Team
Created: 2025-12-25
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from metagen.synth.modalities.base import ModalityComponents
from metagen.synth.modalities.text import (
    DEFAULT_SEQ_LEN,
    DEFAULT_VOCAB_SIZE,
    MAX_SEQ_LEN_INFINITE,
    TextModalityHandler,
)


# Mock classes for testing without real specs
@dataclass
class MockModality:
    """Mock modality for testing."""

    inputs: list[str] = field(default_factory=lambda: ["text"])
    outputs: list[str] = field(default_factory=lambda: ["text"])


@dataclass
class MockArchitecture:
    """Mock architecture for testing."""

    family: str = "transformer"


@dataclass
class MockConstraints:
    """Mock constraints for testing."""

    context_window: str = "2048"


@dataclass
class MockTraining:
    """Mock training config for testing."""

    objective: list[str] = field(default_factory=lambda: ["autoregressive"])


@dataclass
class MockSpec:
    """Mock spec for testing TextModalityHandler."""

    modality: MockModality = field(default_factory=MockModality)
    architecture: MockArchitecture = field(default_factory=MockArchitecture)
    constraints: MockConstraints = field(default_factory=MockConstraints)
    training: MockTraining = field(default_factory=MockTraining)


@dataclass
class MockBlueprintState:
    """Mock BlueprintState for testing."""

    dims: dict = field(default_factory=lambda: {"hidden_size": 4096, "layers": 32, "heads": 32})
    vocab_size: int | None = None
    max_seq_len: int | None = None


class TestTextModalityHandler:
    """Tests for the TextModalityHandler class."""

    def test_handler_name(self) -> None:
        """Test that handler name is 'text'."""
        handler = TextModalityHandler()
        assert handler.name == "text"

    def test_handler_supported_families(self) -> None:
        """Test supported architecture families."""
        handler = TextModalityHandler()
        assert "transformer" in handler.supported_families
        assert "rnn" in handler.supported_families
        assert "hybrid" in handler.supported_families
        assert "mlp" in handler.supported_families

    def test_handler_repr(self) -> None:
        """Test string representation."""
        handler = TextModalityHandler()
        assert repr(handler) == "TextModalityHandler(name='text')"


class TestAugmentBlueprint:
    """Tests for blueprint augmentation."""

    def test_augment_adds_vocab_size(self) -> None:
        """Test that augment_blueprint adds vocab_size."""
        handler = TextModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.vocab_size == DEFAULT_VOCAB_SIZE

    def test_augment_adds_max_seq_len(self) -> None:
        """Test that augment_blueprint adds max_seq_len."""
        handler = TextModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.max_seq_len == 2048  # From MockConstraints

    def test_augment_preserves_dims(self) -> None:
        """Test that augmentation preserves original dims."""
        handler = TextModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.dims == blueprint.dims

    def test_augment_deterministic(self) -> None:
        """Test that augmentation is deterministic given same seed."""
        handler = TextModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        aug1 = handler.augment_blueprint(spec, blueprint, seed=42)
        aug2 = handler.augment_blueprint(spec, blueprint, seed=42)

        assert aug1.vocab_size == aug2.vocab_size
        assert aug1.max_seq_len == aug2.max_seq_len


class TestContextWindowParsing:
    """Tests for context window parsing."""

    def test_parse_numeric(self) -> None:
        """Test parsing plain numeric context window."""
        handler = TextModalityHandler()
        spec = MockSpec(constraints=MockConstraints(context_window="2048"))

        result = handler._parse_context_window(spec)

        assert result == 2048

    def test_parse_with_k_suffix(self) -> None:
        """Test parsing context window with 'k' suffix."""
        handler = TextModalityHandler()
        spec = MockSpec(constraints=MockConstraints(context_window="256k"))

        result = handler._parse_context_window(spec)

        assert result == 262144  # 256 * 1024

    def test_parse_with_m_suffix(self) -> None:
        """Test parsing context window with 'M' suffix."""
        handler = TextModalityHandler()
        spec = MockSpec(constraints=MockConstraints(context_window="1M"))

        result = handler._parse_context_window(spec)

        assert result == 1048576  # 1 * 1024 * 1024

    def test_parse_infinite(self) -> None:
        """Test parsing 'infinite' context window."""
        handler = TextModalityHandler()

        for ctx in ["infinite", "inf", "∞", "INFINITE"]:
            spec = MockSpec(constraints=MockConstraints(context_window=ctx))
            result = handler._parse_context_window(spec)
            assert result == MAX_SEQ_LEN_INFINITE

    def test_parse_with_spaces(self) -> None:
        """Test parsing context window with spaces."""
        handler = TextModalityHandler()
        spec = MockSpec(constraints=MockConstraints(context_window="256 k"))

        result = handler._parse_context_window(spec)

        assert result == 262144

    def test_parse_invalid_returns_default(self) -> None:
        """Test that invalid context window returns default."""
        handler = TextModalityHandler()
        spec = MockSpec(constraints=MockConstraints(context_window="invalid"))

        result = handler._parse_context_window(spec)

        assert result == DEFAULT_SEQ_LEN

    def test_parse_case_insensitive(self) -> None:
        """Test that parsing is case insensitive."""
        handler = TextModalityHandler()

        for ctx in ["256K", "256k", "1m", "1M"]:
            spec = MockSpec(constraints=MockConstraints(context_window=ctx))
            result = handler._parse_context_window(spec)
            assert result > 0


class TestGenerateComponents:
    """Tests for component generation."""

    def test_returns_modality_components(self) -> None:
        """Test that generate_components returns ModalityComponents."""
        handler = TextModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(vocab_size=50257, max_seq_len=2048)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert isinstance(components, ModalityComponents)

    def test_encoder_type_transformer(self) -> None:
        """Test encoder type for transformer family."""
        handler = TextModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="transformer"))
        blueprint = MockBlueprintState(vocab_size=50257, max_seq_len=2048)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.encoder_type == "transformer_encoder"

    def test_encoder_type_rnn(self) -> None:
        """Test encoder type for RNN family."""
        handler = TextModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="rnn"))
        blueprint = MockBlueprintState(vocab_size=50257, max_seq_len=2048)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.encoder_type == "rnn_encoder"

    def test_decoder_type_autoregressive(self) -> None:
        """Test decoder type for autoregressive objective."""
        handler = TextModalityHandler()
        spec = MockSpec(training=MockTraining(objective=["autoregressive"]))
        blueprint = MockBlueprintState(vocab_size=50257, max_seq_len=2048)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.decoder_type == "autoregressive"

    def test_decoder_type_mlm(self) -> None:
        """Test decoder type for masked LM objective."""
        handler = TextModalityHandler()
        spec = MockSpec(training=MockTraining(objective=["mlm"]))
        blueprint = MockBlueprintState(vocab_size=50257, max_seq_len=2048)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.decoder_type == "masked_lm"

    def test_embedding_type(self) -> None:
        """Test that embedding type is token_embedding."""
        handler = TextModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(vocab_size=50257, max_seq_len=2048)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.embedding_type == "token_embedding"

    def test_data_loader_type(self) -> None:
        """Test that data loader type is text_dataset."""
        handler = TextModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(vocab_size=50257, max_seq_len=2048)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.data_loader_type == "text_dataset"

    def test_loss_type_cross_entropy(self) -> None:
        """Test loss type for autoregressive objective."""
        handler = TextModalityHandler()
        spec = MockSpec(training=MockTraining(objective=["autoregressive"]))
        blueprint = MockBlueprintState(vocab_size=50257, max_seq_len=2048)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.loss_type == "cross_entropy"

    def test_loss_type_contrastive(self) -> None:
        """Test loss type for contrastive objective."""
        handler = TextModalityHandler()
        spec = MockSpec(training=MockTraining(objective=["contrastive"]))
        blueprint = MockBlueprintState(vocab_size=50257, max_seq_len=2048)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.loss_type == "contrastive"

    def test_config_contains_text_params(self) -> None:
        """Test that config contains text-specific parameters."""
        handler = TextModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(
            vocab_size=50257,
            max_seq_len=2048,
            dims={"hidden_size": 4096, "layers": 32, "heads": 32},
        )

        components = handler.generate_components(spec, blueprint, seed=42)

        assert components.config["vocab_size"] == 50257
        assert components.config["max_seq_len"] == 2048
        assert components.config["hidden_size"] == 4096
        assert components.config["num_layers"] == 32
        assert components.config["num_heads"] == 32

    def test_template_fragments_includes_embedding(self) -> None:
        """Test that template fragments include token embedding."""
        handler = TextModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(vocab_size=50257, max_seq_len=2048)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert "embeddings/token_embedding.py.j2" in components.template_fragments

    def test_template_fragments_includes_encoder(self) -> None:
        """Test that template fragments include encoder."""
        handler = TextModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(vocab_size=50257, max_seq_len=2048)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert "encoders/transformer_encoder.py.j2" in components.template_fragments

    def test_additional_imports(self) -> None:
        """Test that additional imports are included."""
        handler = TextModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(vocab_size=50257, max_seq_len=2048)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert "import torch" in components.additional_imports
        assert "import torch.nn as nn" in components.additional_imports


class TestValidateSpec:
    """Tests for spec validation."""

    def test_valid_text_spec(self) -> None:
        """Test that valid text spec passes validation."""
        handler = TextModalityHandler()
        spec = MockSpec()

        # Should not raise
        handler.validate_spec(spec)

    def test_invalid_family_raises(self) -> None:
        """Test that unsupported family raises ValueError."""
        handler = TextModalityHandler()
        spec = MockSpec(architecture=MockArchitecture(family="unsupported"))

        with pytest.raises(ValueError) as exc_info:
            handler.validate_spec(spec)

        assert "unsupported" in str(exc_info.value)

    def test_missing_text_modality_raises(self) -> None:
        """Test that missing text modality raises ValueError."""
        handler = TextModalityHandler()
        spec = MockSpec(modality=MockModality(inputs=["image"], outputs=["image"]))

        with pytest.raises(ValueError) as exc_info:
            handler.validate_spec(spec)

        assert "text" in str(exc_info.value).lower()

    def test_text_in_outputs_only_valid(self) -> None:
        """Test that text in outputs only is valid."""
        handler = TextModalityHandler()
        spec = MockSpec(modality=MockModality(inputs=["image"], outputs=["text"]))

        # Should not raise - text is in outputs
        handler.validate_spec(spec)

    def test_supports_spec_returns_bool(self) -> None:
        """Test that supports_spec returns boolean."""
        handler = TextModalityHandler()

        valid_spec = MockSpec()
        invalid_spec = MockSpec(architecture=MockArchitecture(family="unsupported"))

        assert handler.supports_spec(valid_spec) is True
        assert handler.supports_spec(invalid_spec) is False


class TestGetTemplateFragments:
    """Tests for template fragment retrieval."""

    def test_returns_list(self) -> None:
        """Test that get_template_fragments returns a list."""
        handler = TextModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        fragments = handler.get_template_fragments(spec, blueprint)

        assert isinstance(fragments, list)

    def test_includes_basic_fragments(self) -> None:
        """Test that basic fragments are included."""
        handler = TextModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState()

        fragments = handler.get_template_fragments(spec, blueprint)

        assert "embeddings/token_embedding.py.j2" in fragments
        assert "data/text_datasets.py.j2" in fragments
        assert "losses/cross_entropy.py.j2" in fragments

    def test_autoregressive_includes_decoder(self) -> None:
        """Test that autoregressive spec includes decoder fragment."""
        handler = TextModalityHandler()
        spec = MockSpec(training=MockTraining(objective=["autoregressive"]))
        blueprint = MockBlueprintState()

        fragments = handler.get_template_fragments(spec, blueprint)

        assert "decoders/autoregressive_decoder.py.j2" in fragments


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_seed_same_components(self) -> None:
        """Test that same seed produces identical components."""
        handler = TextModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(vocab_size=50257, max_seq_len=2048)

        comp1 = handler.generate_components(spec, blueprint, seed=42)
        comp2 = handler.generate_components(spec, blueprint, seed=42)

        assert comp1.encoder_type == comp2.encoder_type
        assert comp1.decoder_type == comp2.decoder_type
        assert comp1.loss_type == comp2.loss_type
        assert comp1.config == comp2.config

    def test_different_seed_same_output(self) -> None:
        """Test that text handler is deterministic regardless of seed.

        Note: TextModalityHandler currently doesn't use randomness,
        so all seeds should produce the same output.
        """
        handler = TextModalityHandler()
        spec = MockSpec()
        blueprint = MockBlueprintState(vocab_size=50257, max_seq_len=2048)

        comp1 = handler.generate_components(spec, blueprint, seed=1)
        comp2 = handler.generate_components(spec, blueprint, seed=999)

        # For text, output is deterministic regardless of seed
        assert comp1.encoder_type == comp2.encoder_type
        assert comp1.decoder_type == comp2.decoder_type
