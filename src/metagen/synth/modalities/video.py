"""
MetaGen Video Modality Handler

Provides a VideoModalityHandler for temporal/video synthesis. The handler
augments blueprints with frame sizing and selects encoder templates for
3D CNNs or temporal transformers.
"""

from __future__ import annotations

import logging
import random
from dataclasses import replace
from typing import TYPE_CHECKING

from metagen.synth.modalities.base import ModalityComponents, ModalityHandler

if TYPE_CHECKING:
    from metagen.specs.schema import ModelSpec
    from metagen.synth.architecture import BlueprintState

logger = logging.getLogger(__name__)

DEFAULT_FRAME_SIZES = (224, 256, 320)
DEFAULT_FRAME_COUNTS = (8, 16, 32)
PATCH_SIZES = (8, 16, 32)


class VideoModalityHandler(ModalityHandler):
    """Modality handler for video/temporal model synthesis.

    This handler supports video processing architectures including temporal
    transformers and 3D CNNs. It augments blueprints with frame sizing,
    patch configuration for ViT-style models, and temporal sequence length.
    """

    @property
    def name(self) -> str:
        """Return the modality handler name."""
        return "video"

    @property
    def supported_families(self) -> list[str]:
        """Return list of supported architecture families."""
        return ["transformer", "cnn", "hybrid"]

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        """Augment blueprint with video-specific parameters.

        Adds frame size, channel count, frame count (as max_seq_len),
        patch size, and num_patches for transformer-based architectures.

        Args:
            spec: Model specification defining video requirements.
            blueprint: Base blueprint to augment.
            seed: Random seed for deterministic parameter selection.

        Returns:
            Augmented BlueprintState with video-specific fields populated.

        Raises:
            ValueError: If spec does not include video modality.
        """
        self._ensure_video_spec(spec)
        rnd = random.Random(seed)

        image_size = self._select_frame_size(spec, rnd)
        num_channels = 3
        frame_count = self._select_frame_count(spec, rnd)
        family = spec.architecture.family.lower()
        patch_size = self._select_patch_size(image_size, family)
        num_patches = self._compute_num_patches(image_size, patch_size) if patch_size else None

        logger.debug(
            "Video augmentation: image_size=%s, frames=%s, patch_size=%s",
            image_size,
            frame_count,
            patch_size,
        )

        return replace(
            blueprint,
            image_size=image_size,
            num_channels=num_channels,
            patch_size=patch_size,
            num_patches=num_patches,
            max_seq_len=frame_count,
        )

    def generate_components(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> ModalityComponents:
        """Generate video-specific architecture components.

        Creates encoder, embedding, data loader, and loss configurations
        appropriate for video processing. Supports temporal transformers
        for transformer/hybrid families and video CNNs otherwise.

        Args:
            spec: Model specification with training objectives.
            blueprint: Augmented blueprint with video parameters.
            seed: Random seed (unused but kept for interface consistency).

        Returns:
            ModalityComponents with video-specific configuration.
        """
        family = spec.architecture.family.lower()
        objectives = [obj.lower() for obj in spec.training.objective]

        use_transformer = family in {"transformer", "hybrid"}
        encoder_type = "temporal_transformer" if use_transformer else "video_cnn"
        embedding_type = "patch_embedding" if use_transformer else "pixel_embedding"
        loss_type = self._select_loss_type(objectives)
        template_fragments = self._get_template_fragments_list(use_transformer, loss_type)

        additional_imports = (
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
        )

        config = {
            "image_size": blueprint.image_size,
            "num_channels": blueprint.num_channels,
            "patch_size": blueprint.patch_size,
            "num_patches": blueprint.num_patches,
            "frames": blueprint.max_seq_len,
            "hidden_size": blueprint.dims["hidden_size"],
            "num_layers": blueprint.dims["layers"],
            "num_heads": blueprint.dims["heads"],
        }

        return ModalityComponents(
            encoder_type=encoder_type,
            decoder_type=None,
            embedding_type=embedding_type,
            data_loader_type="video_dataset",
            loss_type=loss_type,
            template_fragments=tuple(template_fragments),
            additional_imports=additional_imports,
            config=config,
        )

    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        """Get Jinja2 template fragments for video code generation.

        Args:
            spec: Model specification with architecture family.
            blueprint: Blueprint state (unused but part of interface).

        Returns:
            List of template fragment paths for video model generation.
        """
        family = spec.architecture.family.lower()
        objectives = [obj.lower() for obj in spec.training.objective]
        loss_type = self._select_loss_type(objectives)
        use_transformer = family in {"transformer", "hybrid"}
        return self._get_template_fragments_list(use_transformer, loss_type)

    def validate_spec(self, spec: ModelSpec) -> None:
        """Validate that spec is suitable for video processing.

        Args:
            spec: Model specification to validate.

        Raises:
            ValueError: If spec doesn't include video modality or has
                unsupported architecture family.
        """
        super().validate_spec(spec)
        modalities = {m.lower() for m in spec.modality.inputs + spec.modality.outputs}
        if "video" not in modalities:
            raise ValueError("VideoModalityHandler requires video in spec modalities")

    def _ensure_video_spec(self, spec: ModelSpec) -> None:
        """Verify spec includes video modality.

        Args:
            spec: Model specification to check.

        Raises:
            ValueError: If video is not in inputs or outputs.
        """
        modalities = {m.lower() for m in spec.modality.inputs + spec.modality.outputs}
        if "video" not in modalities:
            raise ValueError("Spec does not include video modality")

    def _select_frame_size(self, spec: ModelSpec, rnd: random.Random) -> int:
        """Select video frame size based on spec name hints.

        Real-time or avatar models use smaller 224px frames for efficiency.
        Other models randomly select from DEFAULT_FRAME_SIZES.

        Args:
            spec: Model specification with name hints.
            rnd: Seeded random instance.

        Returns:
            Frame size in pixels (224, 256, or 320).
        """
        name = spec.name.lower()
        if "realtime" in name or "avatar" in name:
            return 224
        return rnd.choice(DEFAULT_FRAME_SIZES)

    def _select_frame_count(self, spec: ModelSpec, rnd: random.Random) -> int:
        """Select number of frames per video clip.

        Real-time or avatar models use fewer frames (8) for lower latency.
        Other models randomly select from DEFAULT_FRAME_COUNTS.

        Args:
            spec: Model specification with name hints.
            rnd: Seeded random instance.

        Returns:
            Number of frames (8, 16, or 32).
        """
        name = spec.name.lower()
        if "realtime" in name or "avatar" in name:
            return 8
        return rnd.choice(DEFAULT_FRAME_COUNTS)

    def _select_patch_size(self, image_size: int, family: str) -> int | None:
        """Select patch size for ViT-style temporal transformers.

        CNNs don't use patches and return None. Transformers select
        the first patch size that evenly divides the frame size.

        Args:
            image_size: Frame size in pixels.
            family: Architecture family (transformer, cnn, hybrid).

        Returns:
            Patch size or None for non-transformer architectures.
        """
        if family not in {"transformer", "hybrid"}:
            return None
        for candidate in PATCH_SIZES:
            if image_size % candidate == 0:
                return candidate
        return PATCH_SIZES[1]

    def _compute_num_patches(self, image_size: int, patch_size: int | None) -> int | None:
        """Compute number of patches per frame.

        Args:
            image_size: Frame size in pixels.
            patch_size: Patch size or None.

        Returns:
            Number of patches (image_size / patch_size)^2, or None if no patches.
        """
        if not patch_size:
            return None
        return (image_size // patch_size) ** 2

    def _select_loss_type(self, objectives: list[str]) -> str:
        """Select loss function based on training objectives.

        Diffusion objectives use diffusion_loss, others use reconstruction.

        Args:
            objectives: List of training objective strings.

        Returns:
            Loss type string ('diffusion_loss' or 'reconstruction').
        """
        if "diffusion" in objectives:
            return "diffusion_loss"
        return "reconstruction"

    def _get_template_fragments_list(self, use_transformer: bool, loss_type: str) -> list[str]:
        """Build list of Jinja2 template fragments for code generation.

        Constructs template list based on architecture type:
        - Transformers: patch embedding + video transformer encoder
        - CNNs: pixel embedding + video CNN encoder

        Args:
            use_transformer: True for transformer/hybrid, False for CNN.
            loss_type: Loss function type for fragment selection.

        Returns:
            Ordered list of template fragment paths.
        """
        fragments = [
            "data/video_datasets.py.j2",
        ]
        if use_transformer:
            fragments.insert(0, "embeddings/patch_embedding.py.j2")
            fragments.insert(1, "encoders/video_transformer.py.j2")
        else:
            fragments.insert(0, "embeddings/pixel_embedding.py.j2")
            fragments.insert(1, "encoders/video_cnn.py.j2")

        loss_fragment_map = {
            "diffusion_loss": "losses/diffusion_loss.py.j2",
            "reconstruction": "losses/reconstruction.py.j2",
        }
        fragments.append(loss_fragment_map[loss_type])
        return fragments
