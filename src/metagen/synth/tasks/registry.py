"""
MetaGen Task Handler Registry

This module provides the registry and factory functions for task handlers.
The registry maps task type names to handler classes and provides a convenient
`get_task_handler()` function for obtaining the appropriate handler for a spec.

The registry supports:
- Automatic handler selection based on spec.task.type
- Handler caching for performance
- Custom handler registration for extensibility

Architecture:
    Spec → get_task_handler(spec) → TaskHandler instance (or None)

    The handler is selected based on spec.task.type. If no handler is
    registered for that task type (e.g., "generation"), None is returned
    to fall back to modality-only routing.

Example Usage:
    >>> from metagen.synth.tasks import get_task_handler
    >>> from metagen.specs.loader import load_spec
    >>>
    >>> spec = load_spec("examples/specs/image_classifier.yaml")
    >>> handler = get_task_handler(spec)
    >>> if handler:
    ...     print(handler.name)
    'classification'
    >>>
    >>> # For generative models, returns None (use modality handler)
    >>> gen_spec = load_spec("examples/specs/text_llm_8b.yaml")
    >>> handler = get_task_handler(gen_spec)
    >>> print(handler)
    None

Registering Custom Handlers:
    >>> from metagen.synth.tasks.registry import register_task
    >>>
    >>> @register_task("custom_task")
    ... class CustomTaskHandler(TaskHandler):
    ...     # ... implementation
    ...
    >>> handler = get_task_handler(spec_with_custom_task)

Author: MetaGen Team
Created: 2025-12-28
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from metagen.synth.tasks.base import TaskHandler

if TYPE_CHECKING:
    from metagen.specs.schema import ModelSpec

logger = logging.getLogger(__name__)

# Registry mapping task type names to handler classes
_TASK_REGISTRY: dict[str, type[TaskHandler]] = {}

# Cache for handler instances (one per handler class)
# Handlers are stateless, so we can reuse instances
_TASK_HANDLER_CACHE: dict[type[TaskHandler], TaskHandler] = {}

# Task types that fall back to modality-only routing
# These don't need task handlers (handled by modality handlers)
GENERATIVE_TASK_TYPES = {"generation"}


def register_task(task_type: str):
    """
    Decorator to register a task handler class.

    Use this decorator to add task handlers to the registry.
    The handler will be automatically available via get_task_handler().

    Args:
        task_type: The task type this handler supports.
            Must be lowercase (e.g., "classification", "detection").

    Returns:
        Decorator function that registers the handler class.

    Example:
        >>> @register_task("classification")
        ... class ClassificationTaskHandler(TaskHandler):
        ...     @property
        ...     def name(self) -> str:
        ...         return "classification"
        ...
        ...     # ... other methods
        ...
        >>> handler = get_task_handler(classification_spec)
        >>> print(handler.name)
        'classification'

    Note:
        If a handler for this task type is already registered, it will be
        replaced with the new handler (with a warning logged).
    """

    def decorator(cls: type[TaskHandler]) -> type[TaskHandler]:
        if task_type in _TASK_REGISTRY:
            logger.warning(
                f"Replacing existing handler for task type '{task_type}': "
                f"{_TASK_REGISTRY[task_type].__name__} -> {cls.__name__}"
            )
        _TASK_REGISTRY[task_type] = cls
        logger.debug(f"Registered handler {cls.__name__} for task type '{task_type}'")
        return cls

    return decorator


def get_task_handler(spec: ModelSpec) -> TaskHandler | None:
    """
    Get the appropriate task handler for a spec.

    Examines the spec's task.type to determine which task handler to use.
    Returns None for generative tasks (which use modality handlers only).

    Args:
        spec: Model specification with task information.

    Returns:
        TaskHandler instance for the spec's task type, or None if the
        task type should use modality-only routing.

    Example:
        >>> spec = load_spec("examples/specs/image_classifier.yaml")
        >>> handler = get_task_handler(spec)
        >>> if handler:
        ...     augmented = handler.augment_blueprint(spec, blueprint, 42)
        ...     print(augmented.num_classes)
        1000

        >>> # Generative tasks return None
        >>> gen_spec = load_spec("examples/specs/text_llm_8b.yaml")
        >>> handler = get_task_handler(gen_spec)
        >>> print(handler)
        None

    Note:
        If a handler is found, it validates that it supports the spec's
        modality before returning.
    """
    task_type = spec.task.type.lower()

    # Generative tasks use modality handlers only
    if task_type in GENERATIVE_TASK_TYPES:
        logger.debug(f"Task type '{task_type}' uses modality-only routing")
        return None

    handler_cls = _TASK_REGISTRY.get(task_type)
    if handler_cls is None:
        logger.debug(f"No handler registered for task type '{task_type}'")
        return None

    # Get or create cached instance
    if handler_cls not in _TASK_HANDLER_CACHE:
        _TASK_HANDLER_CACHE[handler_cls] = handler_cls()
        logger.debug(f"Created task handler instance: {handler_cls.__name__}")

    handler = _TASK_HANDLER_CACHE[handler_cls]

    # Validate that handler supports this spec's modality
    handler.validate_spec(spec)

    return handler


def get_task_handler_by_name(task_type: str) -> TaskHandler | None:
    """
    Get a task handler by task type directly.

    Unlike get_task_handler(), this doesn't require a spec and doesn't
    perform spec validation.

    Args:
        task_type: The task type (e.g., "classification", "detection").

    Returns:
        TaskHandler instance for the specified task type, or None if
        no handler is registered.

    Example:
        >>> handler = get_task_handler_by_name("classification")
        >>> if handler:
        ...     print(handler.name)
        'classification'
    """
    task_type = task_type.lower()

    if task_type in GENERATIVE_TASK_TYPES:
        return None

    handler_cls = _TASK_REGISTRY.get(task_type)
    if handler_cls is None:
        return None

    if handler_cls not in _TASK_HANDLER_CACHE:
        _TASK_HANDLER_CACHE[handler_cls] = handler_cls()

    return _TASK_HANDLER_CACHE[handler_cls]


def list_registered_task_types() -> list[str]:
    """
    List all registered task type names.

    Returns:
        Sorted list of registered task type names.

    Example:
        >>> task_types = list_registered_task_types()
        >>> print(task_types)
        ['classification', 'detection', 'regression', ...]
    """
    return sorted(_TASK_REGISTRY.keys())


def is_generative_task(spec: ModelSpec) -> bool:
    """
    Check if a spec represents a generative task.

    Generative tasks use modality handlers only and don't need
    task handlers.

    Args:
        spec: Model specification.

    Returns:
        True if this is a generative task, False otherwise.
    """
    return spec.task.type.lower() in GENERATIVE_TASK_TYPES


def clear_task_handler_cache() -> None:
    """
    Clear the task handler instance cache.

    Useful for testing or when handlers need to be recreated.
    Does not affect the registry itself.

    Example:
        >>> clear_task_handler_cache()
        >>> # Next get_task_handler() call will create new instances
    """
    _TASK_HANDLER_CACHE.clear()
    logger.debug("Task handler cache cleared")


# Note: Task handlers are registered by importing their modules
# See __init__.py for the list of registered handlers
logger.debug(f"Task registry initialized. Registered types: {list_registered_task_types()}")
