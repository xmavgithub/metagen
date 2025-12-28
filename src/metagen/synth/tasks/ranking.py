"""
MetaGen Ranking Task Handler

This module provides the task handler for learning-to-rank tasks,
including document ranking, recommendation ranking, and search result ranking.

Supported modalities:
- text: Document ranking, query-document matching
- tabular: Feature-based ranking (e.g., ad ranking)
- multimodal: Cross-modal ranking

Example spec:
    metagen_version: "1.0"
    name: "document_ranker"
    modality:
      inputs: [text]
      outputs: [regression]
    task:
      type: ranking
      domain: search
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


@register_task("ranking")
class RankingTaskHandler(TaskHandler):
    """
    Task handler for learning-to-rank tasks.

    Supports text, tabular, and multimodal ranking with pairwise
    and listwise loss functions.

    The handler:
    - Generates ranking head with score output
    - Specifies pairwise/listwise loss functions
    - Provides ranking-specific metrics (NDCG, MRR, MAP)
    """

    @property
    def name(self) -> str:
        return "ranking"

    @property
    def supported_modalities(self) -> list[str]:
        return ["text", "tabular", "multimodal"]

    @property
    def output_type(self) -> str:
        return "regression"  # Ranking produces relevance scores

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        """Add ranking-specific parameters to blueprint."""
        # Ranking outputs a single score per item
        return replace(blueprint, num_outputs=1)

    def get_head_architecture(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> dict[str, Any]:
        """Define ranking head architecture."""
        hidden_size = blueprint.dims["hidden_size"]

        return {
            "type": "ranking_head",
            "hidden_dim": hidden_size,
            "intermediate_dim": hidden_size // 2,
            "num_outputs": 1,  # Single relevance score
            "dropout": 0.1,
            "activation": "relu",
            "output_activation": None,  # Linear for unbounded scores
            "pooling": "cls_token" if blueprint.family == "transformer" else "global_avg",
            "loss_type": "pairwise",  # or "listwise"
        }

    def get_loss_function(self, spec: ModelSpec) -> str:
        """Return ranking loss function."""
        # Pairwise ranking loss (RankNet-style)
        return "pairwise_ranking_loss"

    def get_metrics(self, spec: ModelSpec) -> list[str]:
        """Return ranking evaluation metrics."""
        return [
            "ndcg_at_5",
            "ndcg_at_10",
            "mrr",  # Mean Reciprocal Rank
            "map",  # Mean Average Precision
            "precision_at_1",
            "precision_at_5",
        ]

    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        """Get ranking template fragments."""
        fragments = ["heads/ranking_head.py.j2"]

        # Add ranking loss template
        fragments.append("losses/pairwise_ranking_loss.py.j2")

        # Add data loader for pairs/lists
        primary_input = spec.modality.inputs[0].lower()
        if primary_input == "text":
            fragments.append("data/ranking_pairs_dataset.py.j2")

        return fragments
