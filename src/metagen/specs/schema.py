from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError, model_validator

# Base modality types (existing)
SUPPORTED_INPUTS = {
    # Existing generative modalities
    "text",
    "image",
    "audio",
    "video",
    "3d",
    "multimodal",
    # New input types for extended model support
    "time_series",  # Temporal/sequential data
    "graph",  # Graph/network data
    "tabular",  # Structured/tabular data
    "point_cloud",  # 3D point clouds
}

SUPPORTED_OUTPUTS = {
    # Existing generative modalities
    "text",
    "image",
    "audio",
    "video",
    "3d",
    "multimodal",
    # New output types for extended model support
    "label",  # Classification output
    "bounding_boxes",  # Detection output
    "segmentation_mask",  # Segmentation output
    "embedding",  # Vector representation
    "action",  # RL action output
    "time_series",  # Temporal prediction
    "graph",  # Graph output
    "regression",  # Numeric prediction
}

UNSUPPORTED_MODALITIES = {"taste", "smell", "vibes"}

# Valid task types for task-based routing
VALID_TASK_TYPES = {
    # Existing
    "generation",
    # Phase 1: Classification & Regression
    "classification",
    "regression",
    "embedding",
    "ranking",
    # Phase 2: Detection & Segmentation
    "object_detection",
    "instance_segmentation",
    "semantic_segmentation",
    "panoptic_segmentation",
    # Phase 3: Sequence & Time Series
    "sequence_labeling",
    "time_series_forecast",
    "anomaly_detection",
    "speech_recognition",
    # Phase 4: Reinforcement Learning
    "policy_gradient",
    "value_based",
    "actor_critic",
    "model_based",
    # Phase 5: Graph Neural Networks
    "node_classification",
    "link_prediction",
    "graph_classification",
    "recommendation",
}


class ParameterBudget(BaseModel):
    max: str = Field(
        "20B",
        description="Maximum parameter budget (string allows expressive values)",
    )


class Constraints(BaseModel):
    latency: str = Field("near-real-time", description="Latency requirement")
    device: str = Field("consumer_gpu", description="Deployment device class")
    parameter_budget: ParameterBudget = Field(default_factory=ParameterBudget)
    memory_budget: str = Field("12GB", description="Approx memory budget")
    context_window: str = Field("128k", description="Context window length")
    throughput: str = Field("30fps", description="Throughput requirement")


class TrainingDataGovernance(BaseModel):
    pii: str = Field("we tried", description="PII handling statement")
    copyright: str = Field("mostly", description="Copyright handling statement")


class TrainingData(BaseModel):
    sources: list[str] = Field(default_factory=lambda: ["synthetic", "licensed"])
    size: str = Field("unknown but large", description="Dataset size statement")
    governance: TrainingDataGovernance = Field(default_factory=TrainingDataGovernance)


class TrainingCompute(BaseModel):
    hardware: str = Field("8xH100", description="Compute hardware")
    duration: str = Field("3 days", description="Training duration")


class TrainingAlignment(BaseModel):
    method: list[str] = Field(default_factory=lambda: ["rlhf", "rlaif"])
    policy: str = Field("helpful-harmless-ish", description="Alignment policy")


class Training(BaseModel):
    objective: list[str] = Field(default_factory=lambda: ["diffusion", "autoregressive"])
    data: TrainingData = Field(default_factory=TrainingData)
    compute: TrainingCompute = Field(default_factory=TrainingCompute)
    alignment: TrainingAlignment = Field(default_factory=TrainingAlignment)


class ArchitectureComponent(BaseModel):
    name: str
    type: str


class Architecture(BaseModel):
    family: str = Field("transformer", description="Architecture family")
    components: list[ArchitectureComponent] = Field(
        default_factory=lambda: [
            ArchitectureComponent(name="SpecEncoder", type="transformer_encoder"),
            ArchitectureComponent(name="ModelLatent", type="hypernetwork_latent"),
            ArchitectureComponent(name="ArchitectureSynth", type="graph_generator"),
            ArchitectureComponent(name="LossComposer", type="objective_mixer"),
            ArchitectureComponent(name="PaperHead", type="latex_decoder"),
        ]
    )


class Outputs(BaseModel):
    artifacts: list[str] = Field(
        default_factory=lambda: [
            "pytorch_skeleton",
            "training_recipe",
            "benchmark_report",
            "paper",
            "model_card",
        ]
    )


class Evaluation(BaseModel):
    benchmarks: list[str] = Field(
        default_factory=lambda: ["META-SOTA", "GEN-EVAL-∞", "FOUNDATION-BENCH"]
    )
    baselines: list[str] = Field(default_factory=lambda: ["GPT-4", "Gemini", "Llama", "SDXL"])
    metrics: list[str] = Field(
        default_factory=lambda: ["Spec-Fidelity@1", "SOTA-Proximity", "Novelty-Per-Parameter"]
    )


class Reproducibility(BaseModel):
    seed: int = 42
    determinism: str = Field("aspirational", description="Determinism statement")


class Modality(BaseModel):
    inputs: list[str] = Field(default_factory=lambda: ["text"])
    outputs: list[str] = Field(default_factory=lambda: ["text"])


class Task(BaseModel):
    """Task configuration for model synthesis."""

    type: str = Field("generation", description="Task type (generation, classification, etc.)")
    domain: str = Field("generic", description="Application domain (generic, image, finance, etc.)")

    # Classification/Regression fields
    num_classes: int | None = Field(None, description="Number of output classes for classification")
    num_outputs: int | None = Field(None, description="Number of outputs for regression")

    # Time Series fields
    horizon: int | None = Field(None, description="Prediction horizon for time series forecasting")
    lookback: int | None = Field(None, description="Number of historical steps to consider")

    # RL fields
    action_space: str | None = Field(
        None, description="Action space type: 'discrete' or 'continuous'"
    )
    num_actions: int | None = Field(None, description="Number of discrete actions")
    action_dim: int | None = Field(None, description="Dimension of continuous action space")

    # Detection/Segmentation fields
    num_anchors: int | None = Field(None, description="Number of anchor boxes for detection")
    mask_resolution: int | None = Field(None, description="Output mask resolution for segmentation")

    # Graph fields
    node_features: int | None = Field(None, description="Number of node features for GNN")
    edge_features: int | None = Field(None, description="Number of edge features for GNN")

    # Embedding fields
    embedding_dim: int | None = Field(None, description="Output embedding dimension")

    model_config = {"extra": "forbid"}


class ModelSpec(BaseModel):
    metagen_version: str = Field("1.0")
    name: str = Field("metagen_default")
    description: str = Field("A universal foundation model synthesizer.")
    modality: Modality = Field(default_factory=Modality)
    task: Task = Field(default_factory=Task)
    constraints: Constraints = Field(default_factory=Constraints)
    training: Training = Field(default_factory=Training)
    architecture: Architecture = Field(default_factory=Architecture)
    outputs: Outputs = Field(default_factory=Outputs)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    reproducibility: Reproducibility = Field(default_factory=Reproducibility)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def check_modalities(self) -> ModelSpec:
        warnings = []
        unsupported = set(self.modality.outputs) & UNSUPPORTED_MODALITIES
        if unsupported:
            warnings.append(f"Unsupported modalities requested: {', '.join(sorted(unsupported))}")
        context = (self.constraints.context_window or "").lower()
        if "∞" in context or context in {"infinite", "inf"}:
            warnings.append("Infinite context requested; approximating with 1M tokens.")
        if warnings:
            # Actionable warnings stored on instance for downstream logging
            object.__setattr__(self, "_warnings", warnings)
        return self

    @property
    def warnings(self) -> list[str]:
        return getattr(self, "_warnings", [])


def validate_spec(data: dict) -> ModelSpec:
    """Validate a specification dict and return a ModelSpec."""
    try:
        return ModelSpec(**data)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def json_schema() -> dict:
    """Return JSON schema for ModelSpec."""
    return ModelSpec.model_json_schema()
