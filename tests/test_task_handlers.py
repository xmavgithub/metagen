"""
Tests for MetaGen Phase 2 Task Handlers

This module tests the task handler implementations:
- ClassificationTaskHandler
- RegressionTaskHandler
- EmbeddingTaskHandler
- RankingTaskHandler

Author: MetaGen Team
Created: 2025-12-28
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from metagen.synth.architecture import BlueprintState
from metagen.synth.tasks import (
    ClassificationTaskHandler,
    EmbeddingTaskHandler,
    RankingTaskHandler,
    RegressionTaskHandler,
    get_task_handler,
    list_registered_task_types,
)
from metagen.synth.tasks.base import TaskComponents


# Mock classes for testing
@dataclass
class MockModality:
    """Mock modality for testing."""

    inputs: list[str] = field(default_factory=lambda: ["image"])
    outputs: list[str] = field(default_factory=lambda: ["label"])


@dataclass
class MockTask:
    """Mock task for testing."""

    type: str = "classification"
    domain: str = "imagenet"
    num_classes: int | None = 1000
    num_outputs: int | None = None
    embedding_dim: int | None = None
    horizon: int | None = None
    lookback: int | None = None
    action_space: str | None = None
    num_actions: int | None = None
    action_dim: int | None = None
    num_anchors: int | None = None
    mask_resolution: int | None = None
    node_features: int | None = None
    edge_features: int | None = None


@dataclass
class MockArchitecture:
    """Mock architecture for testing."""

    family: str = "transformer"


@dataclass
class MockSpec:
    """Mock spec for testing task handlers."""

    modality: MockModality = field(default_factory=MockModality)
    task: MockTask = field(default_factory=MockTask)
    architecture: MockArchitecture = field(default_factory=MockArchitecture)


def make_blueprint(hidden_size: int = 768, layers: int = 12, heads: int = 12) -> BlueprintState:
    """Create a test BlueprintState."""
    return BlueprintState(
        dims={"hidden_size": hidden_size, "layers": layers, "heads": heads},
        family="transformer",
    )


class TestHandlerRegistration:
    """Tests for handler registration."""

    def test_classification_registered(self) -> None:
        """Test that classification handler is registered."""
        assert "classification" in list_registered_task_types()

    def test_regression_registered(self) -> None:
        """Test that regression handler is registered."""
        assert "regression" in list_registered_task_types()

    def test_embedding_registered(self) -> None:
        """Test that embedding handler is registered."""
        assert "embedding" in list_registered_task_types()

    def test_ranking_registered(self) -> None:
        """Test that ranking handler is registered."""
        assert "ranking" in list_registered_task_types()


class TestClassificationTaskHandler:
    """Tests for ClassificationTaskHandler."""

    def test_name(self) -> None:
        """Test handler name property."""
        handler = ClassificationTaskHandler()
        assert handler.name == "classification"

    def test_supported_modalities(self) -> None:
        """Test supported modalities."""
        handler = ClassificationTaskHandler()
        assert "image" in handler.supported_modalities
        assert "text" in handler.supported_modalities
        assert "audio" in handler.supported_modalities

    def test_output_type(self) -> None:
        """Test output type."""
        handler = ClassificationTaskHandler()
        assert handler.output_type == "label"

    def test_augment_blueprint_with_num_classes(self) -> None:
        """Test augment_blueprint with explicit num_classes."""
        handler = ClassificationTaskHandler()
        spec = MockSpec(task=MockTask(num_classes=100))
        blueprint = make_blueprint()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.num_classes == 100

    def test_augment_blueprint_default_imagenet(self) -> None:
        """Test augment_blueprint uses imagenet default."""
        handler = ClassificationTaskHandler()
        spec = MockSpec(task=MockTask(num_classes=None, domain="imagenet"))
        blueprint = make_blueprint()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.num_classes == 1000

    def test_augment_blueprint_default_cifar(self) -> None:
        """Test augment_blueprint uses cifar default."""
        handler = ClassificationTaskHandler()
        spec = MockSpec(task=MockTask(num_classes=None, domain="cifar10"))
        blueprint = make_blueprint()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.num_classes == 10

    def test_get_head_architecture(self) -> None:
        """Test get_head_architecture returns correct structure."""
        handler = ClassificationTaskHandler()
        spec = MockSpec(task=MockTask(num_classes=1000))
        blueprint = make_blueprint()
        blueprint = handler.augment_blueprint(spec, blueprint, seed=42)

        head = handler.get_head_architecture(spec, blueprint)

        assert head["type"] == "classification_head"
        assert head["num_classes"] == 1000
        assert head["hidden_dim"] == 768
        assert "dropout" in head
        assert "pooling" in head

    def test_get_loss_function(self) -> None:
        """Test get_loss_function for classification."""
        handler = ClassificationTaskHandler()
        spec = MockSpec(task=MockTask(num_classes=1000))

        loss = handler.get_loss_function(spec)

        assert "cross_entropy" in loss

    def test_get_metrics(self) -> None:
        """Test get_metrics for classification."""
        handler = ClassificationTaskHandler()
        spec = MockSpec(task=MockTask(num_classes=1000))

        metrics = handler.get_metrics(spec)

        assert "accuracy" in metrics
        assert "top5_accuracy" in metrics

    def test_get_metrics_small_classes(self) -> None:
        """Test get_metrics includes f1 for small class counts."""
        handler = ClassificationTaskHandler()
        spec = MockSpec(task=MockTask(num_classes=10))

        metrics = handler.get_metrics(spec)

        assert "accuracy" in metrics
        assert "f1_macro" in metrics

    def test_get_task_handler_returns_classification(self) -> None:
        """Test that get_task_handler returns ClassificationTaskHandler."""
        spec = MockSpec()

        handler = get_task_handler(spec)

        assert isinstance(handler, ClassificationTaskHandler)


class TestRegressionTaskHandler:
    """Tests for RegressionTaskHandler."""

    def test_name(self) -> None:
        """Test handler name property."""
        handler = RegressionTaskHandler()
        assert handler.name == "regression"

    def test_supported_modalities(self) -> None:
        """Test supported modalities."""
        handler = RegressionTaskHandler()
        assert "tabular" in handler.supported_modalities
        assert "image" in handler.supported_modalities
        assert "text" in handler.supported_modalities

    def test_output_type(self) -> None:
        """Test output type."""
        handler = RegressionTaskHandler()
        assert handler.output_type == "regression"

    def test_augment_blueprint_with_num_outputs(self) -> None:
        """Test augment_blueprint with explicit num_outputs."""
        handler = RegressionTaskHandler()
        spec = MockSpec(
            modality=MockModality(inputs=["tabular"], outputs=["regression"]),
            task=MockTask(type="regression", num_outputs=5),
        )
        blueprint = make_blueprint()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.num_outputs == 5

    def test_augment_blueprint_default_single_output(self) -> None:
        """Test augment_blueprint defaults to 1 output."""
        handler = RegressionTaskHandler()
        spec = MockSpec(
            modality=MockModality(inputs=["tabular"], outputs=["regression"]),
            task=MockTask(type="regression", num_outputs=None),
        )
        blueprint = make_blueprint()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.num_outputs == 1

    def test_get_head_architecture(self) -> None:
        """Test get_head_architecture returns correct structure."""
        handler = RegressionTaskHandler()
        spec = MockSpec(
            modality=MockModality(inputs=["tabular"], outputs=["regression"]),
            task=MockTask(type="regression", num_outputs=1),
        )
        blueprint = make_blueprint()
        blueprint = handler.augment_blueprint(spec, blueprint, seed=42)

        head = handler.get_head_architecture(spec, blueprint)

        assert head["type"] == "regression_head"
        assert head["num_outputs"] == 1
        assert "use_batch_norm" in head

    def test_get_loss_function(self) -> None:
        """Test get_loss_function for regression."""
        handler = RegressionTaskHandler()
        spec = MockSpec(task=MockTask(type="regression"))

        loss = handler.get_loss_function(spec)

        assert loss == "mse"

    def test_get_metrics(self) -> None:
        """Test get_metrics for regression."""
        handler = RegressionTaskHandler()
        spec = MockSpec(task=MockTask(type="regression"))

        metrics = handler.get_metrics(spec)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2_score" in metrics


class TestEmbeddingTaskHandler:
    """Tests for EmbeddingTaskHandler."""

    def test_name(self) -> None:
        """Test handler name property."""
        handler = EmbeddingTaskHandler()
        assert handler.name == "embedding"

    def test_supported_modalities(self) -> None:
        """Test supported modalities."""
        handler = EmbeddingTaskHandler()
        assert "text" in handler.supported_modalities
        assert "image" in handler.supported_modalities
        assert "audio" in handler.supported_modalities

    def test_output_type(self) -> None:
        """Test output type."""
        handler = EmbeddingTaskHandler()
        assert handler.output_type == "embedding"

    def test_augment_blueprint_with_embedding_dim(self) -> None:
        """Test augment_blueprint with explicit embedding_dim."""
        handler = EmbeddingTaskHandler()
        spec = MockSpec(
            modality=MockModality(inputs=["text"], outputs=["embedding"]),
            task=MockTask(type="embedding", embedding_dim=512),
        )
        blueprint = make_blueprint()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.embedding_dim == 512

    def test_augment_blueprint_default_embedding_dim(self) -> None:
        """Test augment_blueprint uses default embedding_dim."""
        handler = EmbeddingTaskHandler()
        spec = MockSpec(
            modality=MockModality(inputs=["text"], outputs=["embedding"]),
            task=MockTask(type="embedding", embedding_dim=None),
        )
        blueprint = make_blueprint(hidden_size=768)

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        # Should default to min(hidden_size, 768)
        assert augmented.embedding_dim == 768

    def test_get_head_architecture(self) -> None:
        """Test get_head_architecture returns correct structure."""
        handler = EmbeddingTaskHandler()
        spec = MockSpec(
            modality=MockModality(inputs=["text"], outputs=["embedding"]),
            task=MockTask(type="embedding", embedding_dim=768),
        )
        blueprint = make_blueprint()
        blueprint = handler.augment_blueprint(spec, blueprint, seed=42)

        head = handler.get_head_architecture(spec, blueprint)

        assert head["type"] == "embedding_head"
        assert head["embedding_dim"] == 768
        assert head["normalize"] is True
        assert "temperature" in head

    def test_get_loss_function(self) -> None:
        """Test get_loss_function for embedding."""
        handler = EmbeddingTaskHandler()
        spec = MockSpec(task=MockTask(type="embedding"))

        loss = handler.get_loss_function(spec)

        assert loss == "contrastive_loss"

    def test_get_metrics(self) -> None:
        """Test get_metrics for embedding."""
        handler = EmbeddingTaskHandler()
        spec = MockSpec(task=MockTask(type="embedding"))

        metrics = handler.get_metrics(spec)

        assert "recall_at_1" in metrics
        assert "mrr" in metrics


class TestRankingTaskHandler:
    """Tests for RankingTaskHandler."""

    def test_name(self) -> None:
        """Test handler name property."""
        handler = RankingTaskHandler()
        assert handler.name == "ranking"

    def test_supported_modalities(self) -> None:
        """Test supported modalities."""
        handler = RankingTaskHandler()
        assert "text" in handler.supported_modalities
        assert "tabular" in handler.supported_modalities

    def test_output_type(self) -> None:
        """Test output type is regression (scores)."""
        handler = RankingTaskHandler()
        assert handler.output_type == "regression"

    def test_augment_blueprint(self) -> None:
        """Test augment_blueprint sets single output."""
        handler = RankingTaskHandler()
        spec = MockSpec(
            modality=MockModality(inputs=["text"], outputs=["regression"]),
            task=MockTask(type="ranking"),
        )
        blueprint = make_blueprint()

        augmented = handler.augment_blueprint(spec, blueprint, seed=42)

        assert augmented.num_outputs == 1

    def test_get_head_architecture(self) -> None:
        """Test get_head_architecture returns correct structure."""
        handler = RankingTaskHandler()
        spec = MockSpec(
            modality=MockModality(inputs=["text"], outputs=["regression"]),
            task=MockTask(type="ranking"),
        )
        blueprint = make_blueprint()

        head = handler.get_head_architecture(spec, blueprint)

        assert head["type"] == "ranking_head"
        assert head["num_outputs"] == 1
        assert "loss_type" in head

    def test_get_loss_function(self) -> None:
        """Test get_loss_function for ranking."""
        handler = RankingTaskHandler()
        spec = MockSpec(task=MockTask(type="ranking"))

        loss = handler.get_loss_function(spec)

        assert "ranking" in loss.lower()

    def test_get_metrics(self) -> None:
        """Test get_metrics for ranking."""
        handler = RankingTaskHandler()
        spec = MockSpec(task=MockTask(type="ranking"))

        metrics = handler.get_metrics(spec)

        assert "ndcg_at_10" in metrics or "ndcg_at_5" in metrics
        assert "mrr" in metrics


class TestGenerateComponents:
    """Tests for generate_components method."""

    def test_classification_generates_components(self) -> None:
        """Test generate_components for classification."""
        handler = ClassificationTaskHandler()
        spec = MockSpec()
        blueprint = make_blueprint()
        blueprint = handler.augment_blueprint(spec, blueprint, seed=42)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert isinstance(components, TaskComponents)
        assert components.head_type == "classification_head"
        assert "cross_entropy" in components.loss_type
        assert "accuracy" in components.metrics

    def test_regression_generates_components(self) -> None:
        """Test generate_components for regression."""
        handler = RegressionTaskHandler()
        spec = MockSpec(
            modality=MockModality(inputs=["tabular"], outputs=["regression"]),
            task=MockTask(type="regression"),
        )
        blueprint = make_blueprint()
        blueprint = handler.augment_blueprint(spec, blueprint, seed=42)

        components = handler.generate_components(spec, blueprint, seed=42)

        assert isinstance(components, TaskComponents)
        assert components.head_type == "regression_head"
        assert components.loss_type == "mse"


class TestValidation:
    """Tests for spec validation."""

    def test_classification_rejects_unsupported_modality(self) -> None:
        """Test that classification rejects unsupported modalities."""
        handler = ClassificationTaskHandler()
        spec = MockSpec(modality=MockModality(inputs=["graph"]))

        with pytest.raises(ValueError):
            handler.validate_spec(spec)

    def test_regression_accepts_tabular(self) -> None:
        """Test that regression accepts tabular input."""
        handler = RegressionTaskHandler()
        spec = MockSpec(
            modality=MockModality(inputs=["tabular"]),
            task=MockTask(type="regression"),
        )

        # Should not raise
        handler.validate_spec(spec)

    def test_embedding_rejects_tabular(self) -> None:
        """Test that embedding rejects tabular input."""
        handler = EmbeddingTaskHandler()
        spec = MockSpec(
            modality=MockModality(inputs=["tabular"]),
            task=MockTask(type="embedding"),
        )

        with pytest.raises(ValueError):
            handler.validate_spec(spec)
