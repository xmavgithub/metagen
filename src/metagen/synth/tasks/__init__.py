"""
MetaGen Task Handlers

This package provides task-specific handlers for the MetaGen synthesis
pipeline. Task handlers complement modality handlers by adding task-specific
logic for classification, detection, segmentation, RL, and other task types.

Supported Task Types:
    Phase 2 (Classification & Regression):
    - classification: Image/text/audio classification
    - regression: Numeric prediction
    - embedding: Vector representation learning
    - ranking: Learning to rank

    Phase 3 (Detection & Segmentation):
    - object_detection: Bounding box prediction
    - instance_segmentation: Per-instance masks
    - semantic_segmentation: Per-pixel classification
    - panoptic_segmentation: Combined instance + semantic

    Phase 4 (Sequence & Time Series):
    - sequence_labeling: NER, POS tagging
    - time_series_forecast: Temporal prediction
    - anomaly_detection: Outlier detection
    - speech_recognition: Audio to text

    Phase 5 (Reinforcement Learning):
    - policy_gradient: PPO, A3C, REINFORCE
    - value_based: DQN, Rainbow
    - actor_critic: SAC, TD3
    - model_based: World models, MuZero

    Phase 6 (Graph Neural Networks):
    - node_classification: GNN node labels
    - link_prediction: Edge prediction
    - graph_classification: Whole-graph labels
    - recommendation: User-item matching

Quick Start:
    >>> from metagen.synth.tasks import get_task_handler
    >>> from metagen.specs.loader import load_spec
    >>>
    >>> spec = load_spec("examples/specs/image/image_classifier_resnet.yaml")
    >>> handler = get_task_handler(spec)
    >>> if handler:
    ...     print(handler.name)
    'classification'

For custom tasks, subclass TaskHandler:
    >>> from metagen.synth.tasks import TaskHandler, TaskComponents
    >>>
    >>> class MyHandler(TaskHandler):
    ...     name = "custom"
    ...     supported_modalities = ["image", "text"]
    ...     output_type = "label"
    ...     # ... implement abstract methods

Author: MetaGen Team
Created: 2025-12-28
"""

from metagen.synth.tasks.base import TaskComponents, TaskHandler

# Phase 2: Classification & Regression handlers
# Import to register handlers with the registry
from metagen.synth.tasks.classification import ClassificationTaskHandler
from metagen.synth.tasks.detection import DetectionTaskHandler
from metagen.synth.tasks.embedding import EmbeddingTaskHandler
from metagen.synth.tasks.ranking import RankingTaskHandler
from metagen.synth.tasks.registry import (
    clear_task_handler_cache,
    get_task_handler,
    get_task_handler_by_name,
    is_generative_task,
    list_registered_task_types,
    register_task,
)
from metagen.synth.tasks.regression import RegressionTaskHandler
from metagen.synth.tasks.segmentation import (
    InstanceSegmentationTaskHandler,
    PanopticSegmentationTaskHandler,
    SemanticSegmentationTaskHandler,
)
from metagen.synth.tasks.time_series import (
    AnomalyDetectionTaskHandler,
    TimeSeriesForecastTaskHandler,
)

__all__ = [
    # Base classes
    "TaskHandler",
    "TaskComponents",
    # Factory functions
    "get_task_handler",
    "get_task_handler_by_name",
    "register_task",
    "list_registered_task_types",
    "is_generative_task",
    "clear_task_handler_cache",
    # Phase 2 handlers
    "ClassificationTaskHandler",
    "RegressionTaskHandler",
    "EmbeddingTaskHandler",
    "RankingTaskHandler",
    # Phase 3 handlers
    "DetectionTaskHandler",
    "SemanticSegmentationTaskHandler",
    "InstanceSegmentationTaskHandler",
    "PanopticSegmentationTaskHandler",
    # Phase 4 handlers
    "TimeSeriesForecastTaskHandler",
    "AnomalyDetectionTaskHandler",
]
