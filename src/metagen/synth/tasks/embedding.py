"""
MetaGen Embedding Task Handler

This module provides the task handler for embedding/representation learning tasks,
including contrastive learning, metric learning, and dense retrieval.

Supported modalities:
- text: Sentence transformers, dense retrievers
- image: Visual embeddings, CLIP-style models
- audio: Audio embeddings, speaker verification
- multimodal: Cross-modal embeddings

Example spec:
    metagen_version: "1.0"
    name: "sentence_embedder"
    modality:
      inputs: [text]
      outputs: [embedding]
    task:
      type: embedding
      domain: text
      embedding_dim: 768
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

# Common embedding dimensions
COMMON_EMBEDDING_DIMS = {
    "small": 256,
    "base": 768,
    "large": 1024,
    "xlarge": 2048,
}


@register_task("embedding")
class EmbeddingTaskHandler(TaskHandler):
    """
    Task handler for embedding/representation learning tasks.

    Supports text, image, audio, and multimodal embeddings with
    contrastive learning objectives.

    The handler:
    - Augments BlueprintState with embedding_dim
    - Generates embedding head with L2 normalization
    - Specifies contrastive/triplet loss and retrieval metrics
    """

    @property
    def name(self) -> str:
        return "embedding"

    @property
    def supported_modalities(self) -> list[str]:
        return ["text", "image", "audio", "multimodal"]

    @property
    def output_type(self) -> str:
        return "embedding"

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        """Add embedding-specific parameters to blueprint."""
        embedding_dim = spec.task.embedding_dim
        if embedding_dim is None:
            # Default to hidden size or common dimension
            embedding_dim = min(blueprint.dims["hidden_size"], 768)
            logger.debug(f"Using default embedding_dim={embedding_dim}")

        return replace(blueprint, embedding_dim=embedding_dim)

    def get_head_architecture(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> dict[str, Any]:
        """Define embedding head architecture."""
        hidden_size = blueprint.dims["hidden_size"]
        embedding_dim = blueprint.embedding_dim or 768

        # Determine if we need a projection layer
        needs_projection = hidden_size != embedding_dim

        return {
            "type": "embedding_head",
            "hidden_dim": hidden_size,
            "embedding_dim": embedding_dim,
            "needs_projection": needs_projection,
            "normalize": True,  # L2 normalize embeddings
            "dropout": 0.0,  # Typically no dropout for embeddings
            "pooling": "mean" if blueprint.family == "transformer" else "global_avg",
            "temperature": 0.07,  # For contrastive loss
        }

    def get_loss_function(self, spec: ModelSpec) -> str:
        """Return embedding loss function."""
        # InfoNCE/contrastive loss is standard for embeddings
        return "contrastive_loss"

    def get_metrics(self, spec: ModelSpec) -> list[str]:
        """Return embedding evaluation metrics."""
        return [
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "mrr",  # Mean Reciprocal Rank
            "ndcg",  # Normalized Discounted Cumulative Gain
            "cosine_similarity",
        ]

    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        """Get embedding template fragments."""
        fragments = ["heads/embedding_head.py.j2"]

        # Add contrastive loss template
        fragments.append("losses/contrastive_loss.py.j2")

        # Add modality-specific data loader
        primary_input = spec.modality.inputs[0].lower()
        if primary_input == "text":
            fragments.append("data/text_pairs_dataset.py.j2")
        elif primary_input == "image":
            fragments.append("data/image_pairs_dataset.py.j2")

        return fragments
