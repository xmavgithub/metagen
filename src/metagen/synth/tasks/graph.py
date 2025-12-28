"""
MetaGen Graph Task Handlers

This module provides task handlers for graph neural network tasks:
node classification, link prediction, graph classification, and recommendation.

Supported modalities:
- graph: Graph/network structured data
- tabular: Structured feature data paired with graph topology
- multimodal: Graph + other modalities

Example spec:
    metagen_version: "1.0"
    name: "molecule_classifier"
    modality:
      inputs: [graph]
      outputs: [label]
    task:
      type: graph_classification
      domain: chemistry
      num_classes: 2
      node_features: 64
      edge_features: 16
    architecture:
      family: gnn

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

DEFAULT_NODE_FEATURES = 128
DEFAULT_EDGE_FEATURES = 32
DEFAULT_NUM_CLASSES = {
    "chemistry": 2,
    "social": 5,
    "generic": 2,
}


def _resolve_graph_features(spec: ModelSpec) -> tuple[int, int]:
    node_features = spec.task.node_features or DEFAULT_NODE_FEATURES
    edge_features = spec.task.edge_features or DEFAULT_EDGE_FEATURES
    return node_features, edge_features


@register_task("node_classification")
class NodeClassificationTaskHandler(TaskHandler):
    """Task handler for node classification."""

    @property
    def name(self) -> str:
        return "node_classification"

    @property
    def supported_modalities(self) -> list[str]:
        return ["graph", "tabular", "multimodal"]

    @property
    def output_type(self) -> str:
        return "label"

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        node_features, edge_features = _resolve_graph_features(spec)
        num_classes = spec.task.num_classes or DEFAULT_NUM_CLASSES.get(
            spec.task.domain.lower(), DEFAULT_NUM_CLASSES["generic"]
        )
        return replace(
            blueprint,
            node_features=node_features,
            edge_features=edge_features,
            num_classes=num_classes,
        )

    def get_head_architecture(self, spec: ModelSpec, blueprint: BlueprintState) -> dict[str, Any]:
        return {
            "type": "node_classification_head",
            "hidden_dim": blueprint.dims["hidden_size"],
            "num_classes": blueprint.num_classes or spec.task.num_classes or 2,
            "dropout": 0.1,
        }

    def get_loss_function(self, spec: ModelSpec) -> str:
        return "cross_entropy"

    def get_metrics(self, spec: ModelSpec) -> list[str]:
        return ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        return [
            "heads/node_classification_head.py.j2",
            "losses/cross_entropy.py.j2",
            "data/graph_datasets.py.j2",
        ]


@register_task("graph_classification")
class GraphClassificationTaskHandler(TaskHandler):
    """Task handler for graph-level classification."""

    @property
    def name(self) -> str:
        return "graph_classification"

    @property
    def supported_modalities(self) -> list[str]:
        return ["graph", "tabular", "multimodal"]

    @property
    def output_type(self) -> str:
        return "label"

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        node_features, edge_features = _resolve_graph_features(spec)
        num_classes = spec.task.num_classes or DEFAULT_NUM_CLASSES.get(
            spec.task.domain.lower(), DEFAULT_NUM_CLASSES["generic"]
        )
        return replace(
            blueprint,
            node_features=node_features,
            edge_features=edge_features,
            num_classes=num_classes,
        )

    def get_head_architecture(self, spec: ModelSpec, blueprint: BlueprintState) -> dict[str, Any]:
        return {
            "type": "graph_classification_head",
            "hidden_dim": blueprint.dims["hidden_size"],
            "num_classes": blueprint.num_classes or spec.task.num_classes or 2,
            "pooling": "mean",
            "dropout": 0.1,
        }

    def get_loss_function(self, spec: ModelSpec) -> str:
        return "cross_entropy"

    def get_metrics(self, spec: ModelSpec) -> list[str]:
        return ["accuracy", "f1_macro", "roc_auc"]

    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        return [
            "heads/graph_classification_head.py.j2",
            "losses/cross_entropy.py.j2",
            "data/graph_datasets.py.j2",
        ]


@register_task("link_prediction")
class LinkPredictionTaskHandler(TaskHandler):
    """Task handler for link prediction."""

    @property
    def name(self) -> str:
        return "link_prediction"

    @property
    def supported_modalities(self) -> list[str]:
        return ["graph", "tabular", "multimodal"]

    @property
    def output_type(self) -> str:
        return "graph"

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        node_features, edge_features = _resolve_graph_features(spec)
        num_outputs = spec.task.num_outputs or 1
        return replace(
            blueprint,
            node_features=node_features,
            edge_features=edge_features,
            num_outputs=num_outputs,
        )

    def get_head_architecture(self, spec: ModelSpec, blueprint: BlueprintState) -> dict[str, Any]:
        return {
            "type": "link_prediction_head",
            "hidden_dim": blueprint.dims["hidden_size"],
            "output_dim": blueprint.num_outputs or spec.task.num_outputs or 1,
        }

    def get_loss_function(self, spec: ModelSpec) -> str:
        return "bce_with_logits"

    def get_metrics(self, spec: ModelSpec) -> list[str]:
        return ["roc_auc", "pr_auc", "hits_at_10"]

    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        return [
            "heads/link_prediction_head.py.j2",
            "losses/bce_loss.py.j2",
            "data/graph_datasets.py.j2",
        ]


@register_task("recommendation")
class RecommendationTaskHandler(TaskHandler):
    """Task handler for recommendation tasks."""

    @property
    def name(self) -> str:
        return "recommendation"

    @property
    def supported_modalities(self) -> list[str]:
        return ["graph", "tabular", "multimodal"]

    @property
    def output_type(self) -> str:
        return "regression"

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        node_features, edge_features = _resolve_graph_features(spec)
        num_outputs = spec.task.num_outputs or 1
        return replace(
            blueprint,
            node_features=node_features,
            edge_features=edge_features,
            num_outputs=num_outputs,
        )

    def get_head_architecture(self, spec: ModelSpec, blueprint: BlueprintState) -> dict[str, Any]:
        return {
            "type": "recommendation_head",
            "hidden_dim": blueprint.dims["hidden_size"],
            "output_dim": blueprint.num_outputs or spec.task.num_outputs or 1,
            "dropout": 0.1,
        }

    def get_loss_function(self, spec: ModelSpec) -> str:
        return "mse"

    def get_metrics(self, spec: ModelSpec) -> list[str]:
        return ["rmse", "mae", "ndcg_at_10"]

    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        return [
            "heads/recommendation_head.py.j2",
            "losses/mse_loss.py.j2",
            "data/graph_datasets.py.j2",
        ]
