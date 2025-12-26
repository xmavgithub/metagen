"""AutoML components for MetaGen."""

from metagen.automl.candidates import CandidateArchitecture, SearchResult
from metagen.automl.history import HistoryDatabase, RunRecord
from metagen.automl.meta_learner import MetaLearner
from metagen.automl.mutation import mutate_dims, sample_dims
from metagen.automl.objectives import LatencyObjective, ParamsObjective, PerformanceObjective
from metagen.automl.prototype_trainer import PrototypeTrainer, TrainingMetrics
from metagen.automl.refiner import ArchitectureRefiner, RefinementHistory, RefinementIteration
from metagen.automl.search_engine import ArchitectureSearchEngine
from metagen.automl.synthetic_data import (
    ValidationTask,
    generate_generic_features,
    generate_image_noise,
    generate_synthetic_batch,
    generate_text_tokens,
    get_validation_task,
)
from metagen.automl.training_utils import maybe_train_prototype

__all__ = [
    "ArchitectureSearchEngine",
    "CandidateArchitecture",
    "HistoryDatabase",
    "LatencyObjective",
    "MetaLearner",
    "ParamsObjective",
    "PerformanceObjective",
    "PrototypeTrainer",
    "RefinementHistory",
    "RefinementIteration",
    "ArchitectureRefiner",
    "RunRecord",
    "SearchResult",
    "TrainingMetrics",
    "ValidationTask",
    "generate_generic_features",
    "generate_image_noise",
    "generate_synthetic_batch",
    "generate_text_tokens",
    "get_validation_task",
    "maybe_train_prototype",
    "mutate_dims",
    "sample_dims",
]
