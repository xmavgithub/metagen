"""
MetaGen Audio Modality Handler

This module provides the AudioModalityHandler for audio/speech synthesis.
It adds audio-specific parameters (sample rate, channels) and selects
template fragments for audio encoders and datasets.
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

DEFAULT_SAMPLE_RATES = (16000, 22050, 32000, 44100)


class AudioModalityHandler(ModalityHandler):
    """Modality handler for audio/speech model synthesis."""

    @property
    def name(self) -> str:
        return "audio"

    @property
    def supported_families(self) -> list[str]:
        return ["transformer", "cnn", "rnn", "hybrid"]

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        self._ensure_audio_spec(spec)
        rnd = random.Random(seed)

        sample_rate = self._select_sample_rate(spec, rnd)
        num_channels = self._select_num_channels(spec, rnd)

        logger.debug(
            "Audio augmentation: sample_rate=%s, num_channels=%s",
            sample_rate,
            num_channels,
        )

        return replace(
            blueprint,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

    def generate_components(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> ModalityComponents:
        family = spec.architecture.family.lower()
        objectives = [obj.lower() for obj in spec.training.objective]

        encoder_type = "audio_transformer" if family in {"transformer", "hybrid"} else "audio_cnn"
        loss_type = self._select_loss_type(objectives)
        template_fragments = self._get_template_fragments_list(loss_type)

        additional_imports = (
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
        )

        config = {
            "sample_rate": blueprint.sample_rate,
            "num_channels": blueprint.num_channels,
            "hidden_size": blueprint.dims["hidden_size"],
            "num_layers": blueprint.dims["layers"],
            "num_heads": blueprint.dims["heads"],
        }

        return ModalityComponents(
            encoder_type=encoder_type,
            decoder_type=None,
            embedding_type="audio_embedding",
            data_loader_type="audio_dataset",
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
        objectives = [obj.lower() for obj in spec.training.objective]
        loss_type = self._select_loss_type(objectives)
        return self._get_template_fragments_list(loss_type)

    def validate_spec(self, spec: ModelSpec) -> None:
        super().validate_spec(spec)
        modalities = {m.lower() for m in spec.modality.inputs + spec.modality.outputs}
        if "audio" not in modalities:
            raise ValueError("AudioModalityHandler requires audio in spec modalities")

    def _ensure_audio_spec(self, spec: ModelSpec) -> None:
        modalities = {m.lower() for m in spec.modality.inputs + spec.modality.outputs}
        if "audio" not in modalities:
            raise ValueError("Spec does not include audio modality")

    def _select_sample_rate(self, spec: ModelSpec, rnd: random.Random) -> int:
        """Select sample rate based on spec domain (music vs speech)."""
        domain = getattr(spec.task, "domain", "").lower() if hasattr(spec, "task") else ""
        name = getattr(spec, "name", "").lower()
        if "music" in domain or "music" in name:
            return 44100
        if "speech" in domain or "voice" in domain or "speech" in name:
            return 16000
        return rnd.choice(DEFAULT_SAMPLE_RATES)

    def _select_num_channels(self, spec: ModelSpec, rnd: random.Random) -> int:
        """Select number of audio channels (1=mono, 2=stereo)."""
        name = getattr(spec, "name", "")
        description = getattr(spec, "description", "")
        domain = getattr(spec.task, "domain", "") if hasattr(spec, "task") else ""
        text = f"{name} {description} {domain}".lower()
        if "stereo" in text or "music" in text:
            return 2
        # 50% mono, 50% stereo for general audio
        return rnd.choice([1, 2])

    def _select_loss_type(self, objectives: list[str]) -> str:
        if "contrastive" in objectives:
            return "contrastive"
        return "reconstruction"

    def _get_template_fragments_list(self, loss_type: str) -> list[str]:
        fragments = [
            "embeddings/audio_embedding.py.j2",
            "encoders/audio_encoder.py.j2",
            "data/audio_datasets.py.j2",
        ]
        loss_fragment_map = {
            "contrastive": "losses/contrastive.py.j2",
            "reconstruction": "losses/reconstruction.py.j2",
        }
        fragments.append(loss_fragment_map[loss_type])
        return fragments
