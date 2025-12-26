"""
MetaGen Multi-Modal Fusion Handler

Implements a handler for multi-modal specs (text + image) with
cross-attention fragments and CLIP-style contrastive objectives.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from metagen.synth.modalities.base import ModalityComponents, ModalityHandler
from metagen.synth.modalities.image import ImageModalityHandler
from metagen.synth.modalities.text import TextModalityHandler

if TYPE_CHECKING:
    from metagen.specs.schema import ModelSpec
    from metagen.synth.architecture import BlueprintState

logger = logging.getLogger(__name__)


class MultiModalModalityHandler(ModalityHandler):
    """Handler for multi-modal fusion (text + image)."""

    @property
    def name(self) -> str:
        return "multimodal"

    @property
    def supported_families(self) -> list[str]:
        return ["transformer", "hybrid"]

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        self._ensure_multimodal_spec(spec)

        text_handler = TextModalityHandler()
        image_handler = ImageModalityHandler()

        augmented = text_handler.augment_blueprint(spec, blueprint, seed)
        augmented = image_handler.augment_blueprint(spec, augmented, seed + 1)

        return augmented

    def generate_components(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> ModalityComponents:
        objectives = [obj.lower() for obj in spec.training.objective]
        # Select loss type based on training objective
        if "contrastive" in objectives:
            loss_type = "contrastive"
        elif "classification" in objectives:
            loss_type = "cross_entropy"
        elif "generation" in objectives:
            loss_type = "cross_entropy"
        else:
            loss_type = "contrastive"  # Default for multimodal (CLIP-style)

        template_fragments = self._get_template_fragments_list()

        additional_imports = (
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
        )

        config = {
            "vocab_size": blueprint.vocab_size,
            "max_seq_len": blueprint.max_seq_len,
            "image_size": blueprint.image_size,
            "num_channels": blueprint.num_channels,
            "patch_size": blueprint.patch_size,
            "num_patches": blueprint.num_patches,
            "hidden_size": blueprint.dims["hidden_size"],
            "num_layers": blueprint.dims["layers"],
            "num_heads": blueprint.dims["heads"],
        }

        return ModalityComponents(
            encoder_type="multimodal_encoder",
            decoder_type=None,
            embedding_type="multimodal_embedding",
            data_loader_type="multimodal_dataset",
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
        return self._get_template_fragments_list()

    def validate_spec(self, spec: ModelSpec) -> None:
        super().validate_spec(spec)
        self._ensure_multimodal_spec(spec)

    def _ensure_multimodal_spec(self, spec: ModelSpec) -> None:
        modalities = {m.lower() for m in spec.modality.inputs + spec.modality.outputs}
        if not {"text", "image"}.issubset(modalities):
            raise ValueError("MultiModalModalityHandler expects text and image modalities")

    def _get_template_fragments_list(self) -> list[str]:
        return [
            "embeddings/token_embedding.py.j2",
            "embeddings/patch_embedding.py.j2",
            "encoders/transformer_encoder.py.j2",
            "fusion/cross_attention.py.j2",
            "data/multimodal_datasets.py.j2",
            "losses/contrastive.py.j2",
        ]
