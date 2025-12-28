"""
MetaGen Detection Task Handler

This module provides the task handler for object detection tasks, including
YOLO-style CNN detectors and transformer-based DETR variants.

Supported modalities:
- image: Standard object detection on images
- video: Spatiotemporal object detection on video

Example spec:
    metagen_version: "1.0"
    name: "yolo_detector"
    modality:
      inputs: [image]
      outputs: [bounding_boxes]
    task:
      type: object_detection
      domain: coco
      num_classes: 80
      num_anchors: 9
    architecture:
      family: cnn

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

# Default number of classes for common detection datasets
DEFAULT_NUM_CLASSES = {
    "coco": 80,
    "voc": 20,
    "pascal": 20,
    "openimages": 601,
    "generic": 80,
}

DEFAULT_NUM_ANCHORS = 9


@register_task("object_detection")
class DetectionTaskHandler(TaskHandler):
    """
    Task handler for object detection tasks.

    Supports image and video object detection with CNN or transformer backbones.

    The handler:
    - Augments BlueprintState with num_classes and num_anchors
    - Generates detection head architecture
    - Specifies detection loss and mAP metrics
    """

    @property
    def name(self) -> str:
        return "object_detection"

    @property
    def supported_modalities(self) -> list[str]:
        return ["image", "video", "multimodal"]

    @property
    def output_type(self) -> str:
        return "bounding_boxes"

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        """Add detection-specific parameters to blueprint."""
        num_classes = spec.task.num_classes
        if num_classes is None:
            domain = spec.task.domain.lower()
            num_classes = DEFAULT_NUM_CLASSES.get(domain, DEFAULT_NUM_CLASSES["generic"])
            logger.debug(f"Using default num_classes={num_classes} for domain '{domain}'")

        num_anchors = spec.task.num_anchors or DEFAULT_NUM_ANCHORS

        return replace(blueprint, num_classes=num_classes, num_anchors=num_anchors)

    def get_head_architecture(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> dict[str, Any]:
        """Define detection head architecture."""
        hidden_size = blueprint.dims["hidden_size"]
        num_classes = (
            blueprint.num_classes
            or spec.task.num_classes
            or DEFAULT_NUM_CLASSES.get(spec.task.domain.lower(), DEFAULT_NUM_CLASSES["generic"])
        )
        num_anchors = blueprint.num_anchors or spec.task.num_anchors or DEFAULT_NUM_ANCHORS

        return {
            "type": "detection_head",
            "hidden_dim": hidden_size,
            "num_classes": num_classes,
            "num_anchors": num_anchors,
            "box_dim": 4,
            "dropout": 0.1,
            "feature_pyramid": blueprint.family in {"cnn", "hybrid"},
        }

    def get_loss_function(self, spec: ModelSpec) -> str:
        """Return detection loss function."""
        return "detection_loss"

    def get_metrics(self, spec: ModelSpec) -> list[str]:
        """Return detection evaluation metrics."""
        return ["mAP", "mAP_50", "mAP_75"]

    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        """Get detection template fragments."""
        fragments = [
            "heads/detection_head.py.j2",
            "losses/detection_loss.py.j2",
        ]

        primary_input = spec.modality.inputs[0].lower()
        if primary_input in {"image", "video"}:
            fragments.append("data/detection_datasets.py.j2")

        return fragments
