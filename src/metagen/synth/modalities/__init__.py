"""
MetaGen Modality Handlers

This package provides modality-specific handlers for the MetaGen synthesis
pipeline. Each handler encapsulates the logic for generating architecture
components, code templates, and data loaders for a specific input/output
modality.

Supported Modalities:
    - text: Language models (GPT, BERT, etc.)
    - image: Vision models (ViT, CNN, diffusion)
    - audio: Audio/speech models
    - video: Video/temporal models
    - multimodal: Fusion models (text + image)

Quick Start:
    >>> from metagen.synth.modalities import get_handler
    >>> from metagen.specs.loader import load_spec
    >>>
    >>> spec = load_spec("examples/specs/text/text_llm_8b.yaml")
    >>> handler = get_handler(spec)
    >>> print(handler.name)
    'text'

For custom modalities, subclass ModalityHandler:
    >>> from metagen.synth.modalities import ModalityHandler, ModalityComponents
    >>>
    >>> class MyHandler(ModalityHandler):
    ...     name = "custom"
    ...     supported_families = ["transformer"]
    ...     # ... implement abstract methods

Author: MetaGen Team
Created: 2025-12-25
"""

from metagen.synth.modalities.audio import AudioModalityHandler
from metagen.synth.modalities.base import ModalityComponents, ModalityHandler
from metagen.synth.modalities.image import ImageModalityHandler
from metagen.synth.modalities.multimodal import MultiModalModalityHandler
from metagen.synth.modalities.registry import (
    clear_handler_cache,
    get_handler,
    get_handler_by_name,
    list_registered_modalities,
    register_handler,
)
from metagen.synth.modalities.text import TextModalityHandler
from metagen.synth.modalities.video import VideoModalityHandler

__all__ = [
    # Base classes
    "ModalityHandler",
    "ModalityComponents",
    # Handler implementations
    "TextModalityHandler",
    "ImageModalityHandler",
    "AudioModalityHandler",
    "VideoModalityHandler",
    "MultiModalModalityHandler",
    # Factory functions
    "get_handler",
    "get_handler_by_name",
    "register_handler",
    "list_registered_modalities",
    "clear_handler_cache",
]
