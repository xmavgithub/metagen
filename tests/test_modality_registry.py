"""
Tests for MetaGen Modality Handler Registry

This module tests the registry and factory functions in
metagen.synth.modalities.registry, including:
- get_handler() function
- get_handler_by_name() function
- register_handler() decorator
- Registry management functions

Author: MetaGen Team
Created: 2025-12-25
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from metagen.synth.modalities.base import ModalityComponents, ModalityHandler
from metagen.synth.modalities.registry import (
    _HANDLER_CACHE,
    _HANDLER_REGISTRY,
    clear_handler_cache,
    get_handler,
    get_handler_by_name,
    list_registered_modalities,
    register_handler,
)
from metagen.synth.modalities.text import TextModalityHandler


# Mock classes for testing
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
    """Mock spec for testing registry functions."""

    modality: MockModality = field(default_factory=MockModality)
    architecture: MockArchitecture = field(default_factory=MockArchitecture)
    constraints: MockConstraints = field(default_factory=MockConstraints)
    training: MockTraining = field(default_factory=MockTraining)


class TestGetHandler:
    """Tests for get_handler function."""

    def test_returns_text_handler_for_text_spec(self) -> None:
        """Test that get_handler returns TextModalityHandler for text specs."""
        spec = MockSpec()

        handler = get_handler(spec)

        assert isinstance(handler, TextModalityHandler)
        assert handler.name == "text"

    def test_raises_for_unregistered_modality(self) -> None:
        """Test that get_handler raises for unknown modalities."""
        spec = MockSpec(modality=MockModality(inputs=["unknown"]))

        with pytest.raises(ValueError) as exc_info:
            get_handler(spec)

        assert "unknown" in str(exc_info.value)
        assert "No handler registered" in str(exc_info.value)

    def test_raises_for_empty_inputs(self) -> None:
        """Test that get_handler raises for specs with no inputs."""
        spec = MockSpec(modality=MockModality(inputs=[]))

        with pytest.raises(ValueError) as exc_info:
            get_handler(spec)

        assert "no input modalities" in str(exc_info.value).lower()

    def test_caches_handler_instances(self) -> None:
        """Test that get_handler caches and reuses handler instances."""
        spec = MockSpec()

        handler1 = get_handler(spec)
        handler2 = get_handler(spec)

        assert handler1 is handler2

    def test_validates_spec(self) -> None:
        """Test that get_handler validates spec against handler."""
        # Spec with unsupported architecture for text handler
        spec = MockSpec(architecture=MockArchitecture(family="unsupported"))

        with pytest.raises(ValueError) as exc_info:
            get_handler(spec)

        assert "unsupported" in str(exc_info.value)

    def test_prefers_multimodal_handler(self) -> None:
        """Test that multi-modal specs use the multimodal handler when available."""
        spec = MockSpec(modality=MockModality(inputs=["text", "image"]))

        handler = get_handler(spec)

        assert handler.name == "multimodal"


class TestGetHandlerByName:
    """Tests for get_handler_by_name function."""

    def test_returns_text_handler(self) -> None:
        """Test getting text handler by name."""
        handler = get_handler_by_name("text")

        assert isinstance(handler, TextModalityHandler)

    def test_case_insensitive(self) -> None:
        """Test that modality name is case insensitive."""
        handler1 = get_handler_by_name("TEXT")
        handler2 = get_handler_by_name("Text")
        handler3 = get_handler_by_name("text")

        assert handler1 is handler2 is handler3

    def test_raises_for_unknown_modality(self) -> None:
        """Test that unknown modality raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_handler_by_name("unknown")

        assert "unknown" in str(exc_info.value)

    def test_caches_instances(self) -> None:
        """Test that instances are cached."""
        handler1 = get_handler_by_name("text")
        handler2 = get_handler_by_name("text")

        assert handler1 is handler2


class TestListRegisteredModalities:
    """Tests for list_registered_modalities function."""

    def test_returns_sorted_list(self) -> None:
        """Test that function returns sorted list."""
        modalities = list_registered_modalities()

        assert isinstance(modalities, list)
        assert modalities == sorted(modalities)

    def test_includes_text(self) -> None:
        """Test that text is in registered modalities."""
        modalities = list_registered_modalities()

        assert "text" in modalities


class TestClearHandlerCache:
    """Tests for clear_handler_cache function."""

    def test_clears_cache(self) -> None:
        """Test that cache is cleared."""
        # Get a handler to populate cache
        _ = get_handler_by_name("text")
        assert len(_HANDLER_CACHE) > 0

        # Clear cache
        clear_handler_cache()

        assert len(_HANDLER_CACHE) == 0

    def test_new_instances_after_clear(self) -> None:
        """Test that new instances are created after clear."""
        handler1 = get_handler_by_name("text")

        clear_handler_cache()

        handler2 = get_handler_by_name("text")

        # Should be different instances
        assert handler1 is not handler2


class TestRegisterHandler:
    """Tests for register_handler decorator."""

    def test_registers_new_handler(self) -> None:
        """Test registering a new handler."""

        # Create a test handler
        class TestHandler(ModalityHandler):
            @property
            def name(self) -> str:
                return "test_modality"

            @property
            def supported_families(self) -> list[str]:
                return ["transformer"]

            def augment_blueprint(self, spec, blueprint, seed):
                return blueprint

            def generate_components(self, spec, blueprint, seed):
                return ModalityComponents(encoder_type="test")

            def get_template_fragments(self, spec, blueprint):
                return []

        # Register it
        old_registry = dict(_HANDLER_REGISTRY)
        try:
            register_handler("test_modality")(TestHandler)

            assert "test_modality" in _HANDLER_REGISTRY
            assert _HANDLER_REGISTRY["test_modality"] is TestHandler

            # Verify it works
            handler = get_handler_by_name("test_modality")
            assert handler.name == "test_modality"
        finally:
            # Clean up
            _HANDLER_REGISTRY.clear()
            _HANDLER_REGISTRY.update(old_registry)
            clear_handler_cache()

    def test_replaces_existing_handler(self) -> None:
        """Test that registering same modality replaces handler."""
        old_registry = dict(_HANDLER_REGISTRY)
        old_cache = dict(_HANDLER_CACHE)

        class Handler1(ModalityHandler):
            @property
            def name(self) -> str:
                return "replace_test"

            @property
            def supported_families(self) -> list[str]:
                return ["transformer"]

            def augment_blueprint(self, spec, blueprint, seed):
                return blueprint

            def generate_components(self, spec, blueprint, seed):
                return ModalityComponents(encoder_type="handler1")

            def get_template_fragments(self, spec, blueprint):
                return []

        class Handler2(ModalityHandler):
            @property
            def name(self) -> str:
                return "replace_test"

            @property
            def supported_families(self) -> list[str]:
                return ["transformer"]

            def augment_blueprint(self, spec, blueprint, seed):
                return blueprint

            def generate_components(self, spec, blueprint, seed):
                return ModalityComponents(encoder_type="handler2")

            def get_template_fragments(self, spec, blueprint):
                return []

        try:
            register_handler("replace_test")(Handler1)
            register_handler("replace_test")(Handler2)

            assert _HANDLER_REGISTRY["replace_test"] is Handler2
        finally:
            _HANDLER_REGISTRY.clear()
            _HANDLER_REGISTRY.update(old_registry)
            _HANDLER_CACHE.clear()
            _HANDLER_CACHE.update(old_cache)


class TestRegistryIntegration:
    """Integration tests for the registry."""

    def test_full_workflow(self) -> None:
        """Test complete handler workflow via registry."""
        # Get handler via registry
        spec = MockSpec()
        handler = get_handler(spec)

        # Verify handler works
        assert handler.name == "text"

        # Use handler to generate components
        blueprint = type(
            "Blueprint",
            (),
            {
                "dims": {"hidden_size": 4096, "layers": 32, "heads": 32},
                "vocab_size": 50257,
                "max_seq_len": 2048,
            },
        )()

        components = handler.generate_components(spec, blueprint, seed=42)

        assert isinstance(components, ModalityComponents)
        assert components.encoder_type == "transformer_encoder"

    def test_registry_contains_expected_modalities(self) -> None:
        """Test that registry has expected built-in modalities."""
        modalities = list_registered_modalities()

        # Currently only text is registered
        assert "text" in modalities

        # Future: These should be added
        # assert "image" in modalities
        # assert "audio" in modalities
        # assert "video" in modalities
