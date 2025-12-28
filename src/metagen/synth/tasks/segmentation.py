"""
MetaGen Segmentation Task Handlers

This module provides task handlers for segmentation tasks, including
semantic, instance, and panoptic segmentation.

Supported modalities:
- image: Per-pixel segmentation on images
- video: Temporal segmentation on video

Example spec:
    metagen_version: "1.0"
    name: "unet_segmenter"
    modality:
      inputs: [image]
      outputs: [segmentation_mask]
    task:
      type: semantic_segmentation
      domain: cityscapes
      num_classes: 19
      mask_resolution: 512
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

DEFAULT_NUM_CLASSES = {
    "coco": 80,
    "cityscapes": 19,
    "ade20k": 150,
    "voc": 21,
    "pascal": 21,
    "medical": 2,
    "generic": 21,
}

DEFAULT_MASK_RESOLUTION = 128


class BaseSegmentationTaskHandler(TaskHandler):
    """Shared implementation for segmentation task handlers."""

    task_type: str = "semantic_segmentation"
    segmentation_kind: str = "semantic"

    @property
    def name(self) -> str:
        return self.task_type

    @property
    def supported_modalities(self) -> list[str]:
        return ["image", "video", "multimodal"]

    @property
    def output_type(self) -> str:
        return "segmentation_mask"

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        """Add segmentation-specific parameters to blueprint."""
        num_classes = spec.task.num_classes
        if num_classes is None:
            domain = (spec.task.domain or "generic").lower()
            num_classes = DEFAULT_NUM_CLASSES.get(domain, DEFAULT_NUM_CLASSES["generic"])
            logger.debug(f"Using default num_classes={num_classes} for domain '{domain}'")

        mask_resolution = spec.task.mask_resolution
        if mask_resolution is None:
            mask_resolution = blueprint.image_size or DEFAULT_MASK_RESOLUTION

        return replace(
            blueprint,
            num_classes=num_classes,
            mask_resolution=mask_resolution,
        )

    def get_head_architecture(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> dict[str, Any]:
        """Define segmentation head architecture."""
        hidden_size = blueprint.dims["hidden_size"]
        task = getattr(spec, "task", None)
        task_num_classes = getattr(task, "num_classes", None) if task else None
        task_domain = getattr(task, "domain", None) if task else None
        domain = (task_domain or "generic").lower()
        num_classes = (
            blueprint.num_classes
            or task_num_classes
            or DEFAULT_NUM_CLASSES.get(domain, DEFAULT_NUM_CLASSES["generic"])
        )
        mask_resolution = (
            blueprint.mask_resolution
            or spec.task.mask_resolution
            or blueprint.image_size
            or DEFAULT_MASK_RESOLUTION
        )

        return {
            "type": "segmentation_head",
            "hidden_dim": hidden_size,
            "num_classes": num_classes,
            "mask_resolution": mask_resolution,
            "segmentation_type": self.segmentation_kind,
            "dropout": 0.1,
            "upsample_mode": "bilinear",
        }

    def get_loss_function(self, spec: ModelSpec) -> str:
        """Return segmentation loss function."""
        return "dice_loss"

    def get_metrics(self, spec: ModelSpec) -> list[str]:
        """Return segmentation evaluation metrics."""
        if self.segmentation_kind == "instance":
            return ["mAP_mask", "AP_50", "AP_75", "AR_100"]
        if self.segmentation_kind == "panoptic":
            return ["PQ", "SQ", "RQ"]
        return ["mIoU", "pixel_accuracy", "mean_accuracy"]

    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        """Get segmentation template fragments."""
        fragments = [
            "heads/segmentation_head.py.j2",
            "losses/dice_loss.py.j2",
        ]

        primary_input = spec.modality.inputs[0].lower()
        if primary_input in {"image", "video"}:
            fragments.append("data/segmentation_datasets.py.j2")

        return fragments


@register_task("semantic_segmentation")
class SemanticSegmentationTaskHandler(BaseSegmentationTaskHandler):
    """Task handler for semantic segmentation."""

    task_type = "semantic_segmentation"
    segmentation_kind = "semantic"


@register_task("instance_segmentation")
class InstanceSegmentationTaskHandler(BaseSegmentationTaskHandler):
    """Task handler for instance segmentation."""

    task_type = "instance_segmentation"
    segmentation_kind = "instance"


@register_task("panoptic_segmentation")
class PanopticSegmentationTaskHandler(BaseSegmentationTaskHandler):
    """Task handler for panoptic segmentation."""

    task_type = "panoptic_segmentation"
    segmentation_kind = "panoptic"
