"""
Tests for MetaGen Task Handler Registry

This module tests the registry and factory functions in
metagen.synth.tasks.registry, including:
- get_task_handler() function
- get_task_handler_by_name() function
- register_task() decorator
- Registry management functions

Author: MetaGen Team
Created: 2025-12-28
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest

from metagen.synth.architecture import BlueprintState
from metagen.synth.tasks.base import TaskComponents, TaskHandler
from metagen.synth.tasks.registry import (
    _TASK_HANDLER_CACHE,
    _TASK_REGISTRY,
    GENERATIVE_TASK_TYPES,
    clear_task_handler_cache,
    get_task_handler,
    get_task_handler_by_name,
    is_generative_task,
    list_registered_task_types,
    register_task,
)


# Mock classes for testing
@dataclass
class MockModality:
    """Mock modality for testing."""

    inputs: list[str] = field(default_factory=lambda: ["image"])
    outputs: list[str] = field(default_factory=lambda: ["label"])


@dataclass
class MockTask:
    """Mock task for testing."""

    type: str = "classification"
    domain: str = "image"
    num_classes: int | None = 1000


@dataclass
class MockArchitecture:
    """Mock architecture for testing."""

    family: str = "transformer"


@dataclass
class MockSpec:
    """Mock spec for testing task registry functions."""

    modality: MockModality = field(default_factory=MockModality)
    task: MockTask = field(default_factory=MockTask)
    architecture: MockArchitecture = field(default_factory=MockArchitecture)


# Test task handler for registration tests
class TestTaskHandler(TaskHandler):
    """Test implementation of TaskHandler for testing."""

    @property
    def name(self) -> str:
        return "test_task"

    @property
    def supported_modalities(self) -> list[str]:
        return ["image", "text"]

    @property
    def output_type(self) -> str:
        return "label"

    def augment_blueprint(self, spec, blueprint, seed):
        return replace(blueprint, num_classes=1000)

    def get_head_architecture(self, spec, blueprint):
        return {
            "type": "test_head",
            "hidden_dim": blueprint.dims["hidden_size"],
            "num_classes": 1000,
        }

    def get_loss_function(self, spec):
        return "cross_entropy"

    def get_metrics(self, spec):
        return ["accuracy", "f1"]


class TestGetTaskHandler:
    """Tests for get_task_handler function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Store original registry state
        self.original_registry = dict(_TASK_REGISTRY)
        self.original_cache = dict(_TASK_HANDLER_CACHE)

        # Register test handler
        _TASK_REGISTRY["test_task"] = TestTaskHandler

    def teardown_method(self):
        """Clean up after tests."""
        _TASK_REGISTRY.clear()
        _TASK_REGISTRY.update(self.original_registry)
        _TASK_HANDLER_CACHE.clear()
        _TASK_HANDLER_CACHE.update(self.original_cache)

    def test_returns_handler_for_registered_task(self) -> None:
        """Test that get_task_handler returns handler for registered tasks."""
        spec = MockSpec(task=MockTask(type="test_task"))

        handler = get_task_handler(spec)

        assert handler is not None
        assert isinstance(handler, TestTaskHandler)
        assert handler.name == "test_task"

    def test_returns_none_for_generation_task(self) -> None:
        """Test that get_task_handler returns None for generative tasks."""
        spec = MockSpec(task=MockTask(type="generation"))

        handler = get_task_handler(spec)

        assert handler is None

    def test_returns_none_for_unregistered_task(self) -> None:
        """Test that get_task_handler returns None for unknown tasks."""
        spec = MockSpec(task=MockTask(type="unregistered_task"))

        handler = get_task_handler(spec)

        assert handler is None

    def test_caches_handler_instances(self) -> None:
        """Test that get_task_handler caches and reuses handler instances."""
        spec = MockSpec(task=MockTask(type="test_task"))

        handler1 = get_task_handler(spec)
        handler2 = get_task_handler(spec)

        assert handler1 is handler2

    def test_validates_spec_modality(self) -> None:
        """Test that get_task_handler validates spec against handler."""
        # Spec with unsupported modality for test handler
        spec = MockSpec(
            modality=MockModality(inputs=["unsupported"]),
            task=MockTask(type="test_task"),
        )

        with pytest.raises(ValueError) as exc_info:
            get_task_handler(spec)

        assert "unsupported" in str(exc_info.value)


class TestGetTaskHandlerByName:
    """Tests for get_task_handler_by_name function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.original_registry = dict(_TASK_REGISTRY)
        self.original_cache = dict(_TASK_HANDLER_CACHE)
        _TASK_REGISTRY["test_task"] = TestTaskHandler

    def teardown_method(self):
        """Clean up after tests."""
        _TASK_REGISTRY.clear()
        _TASK_REGISTRY.update(self.original_registry)
        _TASK_HANDLER_CACHE.clear()
        _TASK_HANDLER_CACHE.update(self.original_cache)

    def test_returns_handler(self) -> None:
        """Test getting handler by name."""
        handler = get_task_handler_by_name("test_task")

        assert handler is not None
        assert isinstance(handler, TestTaskHandler)

    def test_case_insensitive(self) -> None:
        """Test that task type is case insensitive."""
        handler1 = get_task_handler_by_name("TEST_TASK")
        handler2 = get_task_handler_by_name("Test_Task")
        handler3 = get_task_handler_by_name("test_task")

        assert handler1 is handler2 is handler3

    def test_returns_none_for_unknown(self) -> None:
        """Test that unknown task type returns None."""
        handler = get_task_handler_by_name("unknown_task")

        assert handler is None

    def test_returns_none_for_generation(self) -> None:
        """Test that generation returns None."""
        handler = get_task_handler_by_name("generation")

        assert handler is None


class TestIsGenerativeTask:
    """Tests for is_generative_task function."""

    def test_returns_true_for_generation(self) -> None:
        """Test that generation task returns True."""
        spec = MockSpec(task=MockTask(type="generation"))

        assert is_generative_task(spec) is True

    def test_returns_false_for_classification(self) -> None:
        """Test that classification task returns False."""
        spec = MockSpec(task=MockTask(type="classification"))

        assert is_generative_task(spec) is False


class TestListRegisteredTaskTypes:
    """Tests for list_registered_task_types function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.original_registry = dict(_TASK_REGISTRY)
        _TASK_REGISTRY["aaa_task"] = TestTaskHandler
        _TASK_REGISTRY["zzz_task"] = TestTaskHandler

    def teardown_method(self):
        """Clean up after tests."""
        _TASK_REGISTRY.clear()
        _TASK_REGISTRY.update(self.original_registry)

    def test_returns_sorted_list(self) -> None:
        """Test that function returns sorted list."""
        task_types = list_registered_task_types()

        assert isinstance(task_types, list)
        assert task_types == sorted(task_types)

    def test_includes_registered_types(self) -> None:
        """Test that registered types are included."""
        task_types = list_registered_task_types()

        assert "aaa_task" in task_types
        assert "zzz_task" in task_types


class TestClearTaskHandlerCache:
    """Tests for clear_task_handler_cache function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.original_registry = dict(_TASK_REGISTRY)
        self.original_cache = dict(_TASK_HANDLER_CACHE)
        _TASK_REGISTRY["test_task"] = TestTaskHandler

    def teardown_method(self):
        """Clean up after tests."""
        _TASK_REGISTRY.clear()
        _TASK_REGISTRY.update(self.original_registry)
        _TASK_HANDLER_CACHE.clear()
        _TASK_HANDLER_CACHE.update(self.original_cache)

    def test_clears_cache(self) -> None:
        """Test that cache is cleared."""
        # Get a handler to populate cache
        _ = get_task_handler_by_name("test_task")
        assert len(_TASK_HANDLER_CACHE) > 0

        # Clear cache
        clear_task_handler_cache()

        assert len(_TASK_HANDLER_CACHE) == 0

    def test_new_instances_after_clear(self) -> None:
        """Test that new instances are created after clear."""
        handler1 = get_task_handler_by_name("test_task")

        clear_task_handler_cache()

        handler2 = get_task_handler_by_name("test_task")

        # Should be different instances
        assert handler1 is not handler2


class TestRegisterTask:
    """Tests for register_task decorator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.original_registry = dict(_TASK_REGISTRY)
        self.original_cache = dict(_TASK_HANDLER_CACHE)

    def teardown_method(self):
        """Clean up after tests."""
        _TASK_REGISTRY.clear()
        _TASK_REGISTRY.update(self.original_registry)
        _TASK_HANDLER_CACHE.clear()
        _TASK_HANDLER_CACHE.update(self.original_cache)

    def test_registers_new_handler(self) -> None:
        """Test registering a new handler."""

        @register_task("new_test_task")
        class NewTestHandler(TaskHandler):
            @property
            def name(self) -> str:
                return "new_test_task"

            @property
            def supported_modalities(self) -> list[str]:
                return ["image"]

            @property
            def output_type(self) -> str:
                return "label"

            def augment_blueprint(self, spec, blueprint, seed):
                return blueprint

            def get_head_architecture(self, spec, blueprint):
                return {"type": "new_head"}

            def get_loss_function(self, spec):
                return "cross_entropy"

            def get_metrics(self, spec):
                return ["accuracy"]

        assert "new_test_task" in _TASK_REGISTRY
        assert _TASK_REGISTRY["new_test_task"] is NewTestHandler

        handler = get_task_handler_by_name("new_test_task")
        assert handler.name == "new_test_task"

    def test_replaces_existing_handler(self) -> None:
        """Test that registering same task type replaces handler."""

        @register_task("replace_test")
        class Handler1(TaskHandler):
            @property
            def name(self) -> str:
                return "replace_test"

            @property
            def supported_modalities(self) -> list[str]:
                return ["image"]

            @property
            def output_type(self) -> str:
                return "label"

            def augment_blueprint(self, spec, blueprint, seed):
                return blueprint

            def get_head_architecture(self, spec, blueprint):
                return {"type": "handler1"}

            def get_loss_function(self, spec):
                return "mse"

            def get_metrics(self, spec):
                return []

        @register_task("replace_test")
        class Handler2(TaskHandler):
            @property
            def name(self) -> str:
                return "replace_test"

            @property
            def supported_modalities(self) -> list[str]:
                return ["image"]

            @property
            def output_type(self) -> str:
                return "label"

            def augment_blueprint(self, spec, blueprint, seed):
                return blueprint

            def get_head_architecture(self, spec, blueprint):
                return {"type": "handler2"}

            def get_loss_function(self, spec):
                return "cross_entropy"

            def get_metrics(self, spec):
                return []

        assert _TASK_REGISTRY["replace_test"] is Handler2


class TestTaskHandlerBase:
    """Tests for TaskHandler base class methods."""

    def test_validate_spec_raises_for_empty_inputs(self) -> None:
        """Test that validate_spec raises for specs with no inputs."""
        handler = TestTaskHandler()
        spec = MockSpec(modality=MockModality(inputs=[]))

        with pytest.raises(ValueError) as exc_info:
            handler.validate_spec(spec)

        assert "no input modalities" in str(exc_info.value).lower()

    def test_validate_spec_raises_for_unsupported_modality(self) -> None:
        """Test that validate_spec raises for unsupported modalities."""
        handler = TestTaskHandler()
        spec = MockSpec(modality=MockModality(inputs=["unsupported"]))

        with pytest.raises(ValueError) as exc_info:
            handler.validate_spec(spec)

        assert "unsupported" in str(exc_info.value)
        assert "image, text" in str(exc_info.value)

    def test_supports_spec_returns_true(self) -> None:
        """Test that supports_spec returns True for valid specs."""
        handler = TestTaskHandler()
        spec = MockSpec()

        assert handler.supports_spec(spec) is True

    def test_supports_spec_returns_false(self) -> None:
        """Test that supports_spec returns False for invalid specs."""
        handler = TestTaskHandler()
        spec = MockSpec(modality=MockModality(inputs=["unsupported"]))

        assert handler.supports_spec(spec) is False

    def test_generate_components(self) -> None:
        """Test generate_components method."""
        handler = TestTaskHandler()
        spec = MockSpec()
        blueprint = BlueprintState(dims={"hidden_size": 768, "layers": 12, "heads": 12})

        components = handler.generate_components(spec, blueprint, seed=42)

        assert isinstance(components, TaskComponents)
        assert components.head_type == "test_head"
        assert components.loss_type == "cross_entropy"
        assert "accuracy" in components.metrics

    def test_get_template_fragments(self) -> None:
        """Test default get_template_fragments method."""
        handler = TestTaskHandler()
        spec = MockSpec()
        blueprint = BlueprintState(dims={"hidden_size": 768, "layers": 12, "heads": 12})

        fragments = handler.get_template_fragments(spec, blueprint)

        assert "heads/test_task_head.py.j2" in fragments

    def test_repr(self) -> None:
        """Test __repr__ method."""
        handler = TestTaskHandler()

        assert "TestTaskHandler" in repr(handler)
        assert "test_task" in repr(handler)


class TestTaskComponents:
    """Tests for TaskComponents dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        components = TaskComponents(head_type="test")

        assert components.loss_type == "cross_entropy"
        assert components.metrics == ("accuracy",)
        assert components.template_fragments == ()
        assert components.additional_imports == ()
        assert components.config == {}

    def test_with_config(self) -> None:
        """Test with_config creates new instance."""
        components = TaskComponents(head_type="test")
        updated = components.with_config(num_classes=1000, dropout=0.1)

        assert components.config == {}
        assert updated.config == {"num_classes": 1000, "dropout": 0.1}
        assert components is not updated

    def test_list_to_tuple_conversion(self) -> None:
        """Test that lists are converted to tuples."""
        components = TaskComponents(
            head_type="test",
            metrics=["a", "b"],  # type: ignore
            template_fragments=["x.j2"],  # type: ignore
        )

        assert isinstance(components.metrics, tuple)
        assert isinstance(components.template_fragments, tuple)


class TestGenerativeTaskTypes:
    """Tests for GENERATIVE_TASK_TYPES constant."""

    def test_contains_generation(self) -> None:
        """Test that generation is in generative types."""
        assert "generation" in GENERATIVE_TASK_TYPES

    def test_is_immutable(self) -> None:
        """Test that GENERATIVE_TASK_TYPES is a set."""
        assert isinstance(GENERATIVE_TASK_TYPES, set)
