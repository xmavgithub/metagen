"""
MetaGen Modality Handler Registry

This module provides the registry and factory functions for modality handlers.
The registry maps modality names to handler classes and provides a convenient
`get_handler()` function for obtaining the appropriate handler for a spec.

The registry supports:
- Automatic handler selection based on spec modality
- Handler caching for performance
- Custom handler registration for extensibility

Architecture:
    Spec → get_handler(spec) → ModalityHandler instance

    The handler is selected based on the primary input modality in the spec.
    For multi-modal specs, the first input modality determines the handler.

Example Usage:
    >>> from metagen.synth.modalities import get_handler
    >>> from metagen.specs.loader import load_spec
    >>>
    >>> spec = load_spec("examples/specs/text/text_llm_8b.yaml")
    >>> handler = get_handler(spec)
    >>> print(handler.name)
    'text'
    >>>
    >>> # Check if handler supports the spec
    >>> handler.validate_spec(spec)  # Raises if incompatible
    >>>
    >>> # Use handler in synthesis pipeline
    >>> components = handler.generate_components(spec, blueprint, seed=42)

Registering Custom Handlers:
    >>> from metagen.synth.modalities.registry import register_handler
    >>>
    >>> @register_handler("custom")
    ... class CustomHandler(ModalityHandler):
    ...     # ... implementation
    ...
    >>> handler = get_handler(spec_with_custom_modality)

See Also:
    - ModalityHandler: Base class for all handlers
    - TextModalityHandler: Text modality implementation
    - ImageModalityHandler: Image modality implementation (coming soon)

Author: MetaGen Team
Created: 2025-12-25
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from metagen.synth.modalities.audio import AudioModalityHandler
from metagen.synth.modalities.base import ModalityHandler
from metagen.synth.modalities.image import ImageModalityHandler
from metagen.synth.modalities.multimodal import MultiModalModalityHandler
from metagen.synth.modalities.text import TextModalityHandler
from metagen.synth.modalities.video import VideoModalityHandler

if TYPE_CHECKING:
    from metagen.specs.schema import ModelSpec

# Module-level logger
logger = logging.getLogger(__name__)

# Registry mapping modality names to handler classes
# This is populated at module load time and can be extended via register_handler()
_HANDLER_REGISTRY: dict[str, type[ModalityHandler]] = {}

# Cache for handler instances (one per handler class)
# Handlers are stateless, so we can reuse instances
_HANDLER_CACHE: dict[type[ModalityHandler], ModalityHandler] = {}


def register_handler(modality_name: str):
    """
    Decorator to register a modality handler class.

    Use this decorator to add custom modality handlers to the registry.
    The handler will be automatically available via get_handler().

    Args:
        modality_name: The modality name this handler supports.
            Must be lowercase (e.g., "text", "image", "audio").

    Returns:
        Decorator function that registers the handler class.

    Example:
        >>> @register_handler("custom")
        ... class CustomHandler(ModalityHandler):
        ...     @property
        ...     def name(self) -> str:
        ...         return "custom"
        ...
        ...     # ... other methods
        ...
        >>> # Now CustomHandler is available
        >>> handler = get_handler(spec_with_custom_modality)
        >>> print(handler.name)
        'custom'

    Note:
        If a handler for this modality is already registered, it will be
        replaced with the new handler (with a warning logged).
    """

    def decorator(cls: type[ModalityHandler]) -> type[ModalityHandler]:
        if modality_name in _HANDLER_REGISTRY:
            logger.warning(
                f"Replacing existing handler for modality '{modality_name}': "
                f"{_HANDLER_REGISTRY[modality_name].__name__} -> {cls.__name__}"
            )
        _HANDLER_REGISTRY[modality_name] = cls
        logger.debug(f"Registered handler {cls.__name__} for modality '{modality_name}'")
        return cls

    return decorator


def get_handler(spec: ModelSpec) -> ModalityHandler:
    """
    Get the appropriate modality handler for a spec.

    Examines the spec's modality.inputs to determine the primary modality,
    then returns a cached handler instance for that modality.

    Args:
        spec: Model specification with modality information.

    Returns:
        ModalityHandler instance appropriate for the spec's modality.

    Raises:
        ValueError: If no handler is registered for the spec's modality.

    Example:
        >>> spec = load_spec("examples/specs/text/text_llm_8b.yaml")
        >>> handler = get_handler(spec)
        >>> print(handler.name)
        'text'
        >>> print(handler.supported_families)
        ['transformer', 'rnn', 'hybrid', 'mlp']

        >>> # Handler validates and augments blueprints
        >>> handler.validate_spec(spec)
        >>> augmented = handler.augment_blueprint(spec, blueprint, seed=42)
        >>> print(augmented.vocab_size)
        50257

    Note:
        For multi-modal specs (e.g., inputs: [text, image]), the registry
        prefers a dedicated "multimodal" handler when registered. Otherwise
        it falls back to the first input modality.
    """
    # Get primary modality from inputs
    if not spec.modality.inputs:
        raise ValueError("Spec has no input modalities defined")

    inputs = [m.lower() for m in spec.modality.inputs]
    outputs = [m.lower() for m in spec.modality.outputs]
    modalities = set(inputs + outputs)

    handler_cls = None
    if "multimodal" in modalities:
        handler_cls = _HANDLER_REGISTRY.get("multimodal")
    if handler_cls is None and {"text", "image"}.issubset(modalities):
        handler_cls = _HANDLER_REGISTRY.get("multimodal")
    if handler_cls is None:
        primary_modality = inputs[0]
        handler_cls = _HANDLER_REGISTRY.get(primary_modality)
    if handler_cls is None:
        available = ", ".join(sorted(_HANDLER_REGISTRY.keys())) or "(none)"
        raise ValueError(
            f"No handler registered for modality '{primary_modality}'. "
            f"Available modalities: {available}"
        )

    # Get or create cached instance
    if handler_cls not in _HANDLER_CACHE:
        _HANDLER_CACHE[handler_cls] = handler_cls()
        logger.debug(f"Created handler instance: {handler_cls.__name__}")

    handler = _HANDLER_CACHE[handler_cls]

    # Validate that handler supports this spec
    handler.validate_spec(spec)

    return handler


def get_handler_by_name(modality_name: str) -> ModalityHandler:
    """
    Get a handler by modality name directly.

    Unlike get_handler(), this doesn't require a spec and doesn't
    perform spec validation.

    Args:
        modality_name: The modality name (e.g., "text", "image").

    Returns:
        ModalityHandler instance for the specified modality.

    Raises:
        ValueError: If no handler is registered for this modality.

    Example:
        >>> handler = get_handler_by_name("text")
        >>> print(handler.name)
        'text'
        >>> print(type(handler))
        <class 'TextModalityHandler'>
    """
    modality_name = modality_name.lower()

    handler_cls = _HANDLER_REGISTRY.get(modality_name)
    if handler_cls is None:
        available = ", ".join(sorted(_HANDLER_REGISTRY.keys())) or "(none)"
        raise ValueError(
            f"No handler registered for modality '{modality_name}'. "
            f"Available modalities: {available}"
        )

    if handler_cls not in _HANDLER_CACHE:
        _HANDLER_CACHE[handler_cls] = handler_cls()

    return _HANDLER_CACHE[handler_cls]


def list_registered_modalities() -> list[str]:
    """
    List all registered modality names.

    Returns:
        Sorted list of registered modality names.

    Example:
        >>> modalities = list_registered_modalities()
        >>> print(modalities)
        ['audio', 'image', 'text', 'video']
    """
    return sorted(_HANDLER_REGISTRY.keys())


def clear_handler_cache() -> None:
    """
    Clear the handler instance cache.

    Useful for testing or when handlers need to be recreated.
    Does not affect the registry itself.

    Example:
        >>> clear_handler_cache()
        >>> # Next get_handler() call will create new instances
    """
    _HANDLER_CACHE.clear()
    logger.debug("Handler cache cleared")


# Register built-in handlers at module load time
# Order matters: register handlers from most to least common
_HANDLER_REGISTRY["text"] = TextModalityHandler
_HANDLER_REGISTRY["image"] = ImageModalityHandler
_HANDLER_REGISTRY["audio"] = AudioModalityHandler
_HANDLER_REGISTRY["video"] = VideoModalityHandler
_HANDLER_REGISTRY["multimodal"] = MultiModalModalityHandler

logger.debug(f"Initialized handler registry with modalities: {list_registered_modalities()}")
