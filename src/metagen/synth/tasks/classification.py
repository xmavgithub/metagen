"""
MetaGen Classification Task Handler

This module provides the task handler for classification tasks, including
image classification, text classification, and audio classification.

Supported modalities:
- image: ViT, ResNet, CNN-based classifiers
- text: BERT, RoBERTa, transformer-based classifiers
- audio: AST, audio spectrogram classifiers
- tabular: MLP-based classifiers

Example spec:
    metagen_version: "1.0"
    name: "imagenet_classifier"
    modality:
      inputs: [image]
      outputs: [label]
    task:
      type: classification
      domain: image
      num_classes: 1000
    architecture:
      family: transformer

Author: MetaGen Team
Created: 2025-12-28
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from metagen.synth.tasks.base import TaskHandler
from metagen.synth.tasks.registry import register_task

if TYPE_CHECKING:
    from metagen.specs.schema import ModelSpec
    from metagen.synth.architecture import BlueprintState

logger = logging.getLogger(__name__)

# Default number of classes for common datasets
DEFAULT_NUM_CLASSES = {
    "imagenet": 1000,
    "cifar10": 10,
    "cifar100": 100,
    "mnist": 10,
    "sentiment": 2,
    "generic": 1000,
}


@register_task("classification")
class ClassificationTaskHandler(TaskHandler):
    """
    Task handler for classification tasks.

    Supports image, text, audio, and tabular classification with various
    architecture families (transformer, cnn, mlp).

    The handler:
    - Augments BlueprintState with num_classes
    - Generates classification head architecture
    - Specifies cross-entropy loss and accuracy metrics
    """

    @property
    def name(self) -> str:
        return "classification"

    @property
    def supported_modalities(self) -> list[str]:
        return ["image", "text", "audio", "tabular", "multimodal"]

    @property
    def output_type(self) -> str:
        return "label"

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        """Add classification-specific parameters to blueprint."""
        # Get num_classes from spec or use domain default
        num_classes = spec.task.num_classes
        if num_classes is None:
            domain = (spec.task.domain or "generic").lower()
            num_classes = DEFAULT_NUM_CLASSES.get(domain, DEFAULT_NUM_CLASSES["generic"])
            logger.debug(f"Using default num_classes={num_classes} for domain '{domain}'")

        return replace(blueprint, num_classes=num_classes)

    def get_head_architecture(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> dict[str, Any]:
        """Define classification head architecture."""
        hidden_size = blueprint.dims["hidden_size"]
        num_classes = blueprint.num_classes or 1000

        # Determine head complexity based on num_classes
        if num_classes > 10000:
            # Large vocabulary: use hierarchical softmax or sampled softmax
            head_type = "hierarchical_classification_head"
            intermediate_dim = hidden_size
        elif num_classes > 1000:
            # Medium vocabulary: add intermediate layer
            head_type = "classification_head"
            intermediate_dim = hidden_size // 2
        else:
            # Small vocabulary: simple linear head
            head_type = "classification_head"
            intermediate_dim = None

        return {
            "type": head_type,
            "hidden_dim": hidden_size,
            "intermediate_dim": intermediate_dim,
            "num_classes": num_classes,
            "dropout": 0.1,
            "activation": "gelu",
            "use_layer_norm": True,
            "pooling": "cls_token" if blueprint.family == "transformer" else "global_avg",
        }

    def get_loss_function(self, spec: ModelSpec) -> str:
        """Return classification loss function."""
        num_classes = spec.task.num_classes or 1000

        # Use label smoothing for large class counts
        if num_classes > 100:
            return "cross_entropy_with_label_smoothing"
        return "cross_entropy"

    def get_metrics(self, spec: ModelSpec) -> list[str]:
        """Return classification evaluation metrics."""
        num_classes = spec.task.num_classes or 1000

        metrics = ["accuracy"]

        # Add top-k accuracy for large class counts
        if num_classes > 10:
            metrics.append("top5_accuracy")

        # Add per-class metrics for moderate class counts
        if num_classes <= 100:
            metrics.extend(["f1_macro", "precision_macro", "recall_macro"])

        # Add confusion matrix for small class counts
        if num_classes <= 20:
            metrics.append("confusion_matrix")

        return metrics

    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        """Get classification template fragments."""
        fragments = ["heads/classification_head.py.j2"]

        # Add modality-specific data loader template
        primary_input = spec.modality.inputs[0].lower()
        if primary_input == "image":
            fragments.append("data/image_datasets.py.j2")
        elif primary_input == "text":
            fragments.append("data/text_datasets.py.j2")
        elif primary_input == "audio":
            fragments.append("data/audio_datasets.py.j2")

        # Add loss template
        fragments.append("losses/cross_entropy.py.j2")

        return fragments
