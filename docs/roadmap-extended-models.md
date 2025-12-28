# MetaGen Extended Model Types Roadmap

## Overview

This document outlines the plan to extend MetaGen beyond generative models to support a wide variety of AI model types including classification, detection, segmentation, reinforcement learning, and more.

## Current Architecture

MetaGen currently uses a **handler-based registry pattern**:

```
Spec (YAML) → ModalityHandler → BlueprintState → CodeGen → Outputs
```

**Existing handlers:**
- `TextModalityHandler` - LLMs, text generation
- `ImageModalityHandler` - ViT, diffusion models
- `AudioModalityHandler` - Audio generation
- `VideoModalityHandler` - Video synthesis
- `MultiModalModalityHandler` - CLIP-style models

## Proposed Extension: Task-Based Handlers

### New Concept: TaskHandler

Instead of only modality-based routing, we add **task-based routing**:

```python
# Current: modality → handler
get_handler(spec)  # routes by spec.modality.inputs/outputs

# Extended: task + modality → handler
get_task_handler(spec)  # routes by spec.task.type + spec.modality
```

### Task Types to Support

#### Phase 1: Classification & Regression (Low Complexity)

| Task Type | Description | Architecture Families |
|-----------|-------------|----------------------|
| `classification` | Image/text classification | transformer, cnn, mlp |
| `regression` | Numeric prediction | mlp, transformer |
| `embedding` | Vector representation | transformer, cnn |
| `ranking` | Learning to rank | transformer, mlp |

**Example Spec:**
```yaml
metagen_version: "1.0"
name: "imagenet_classifier"
modality:
  inputs: [image]
  outputs: [label]  # New output type
task:
  type: classification
  domain: image
  num_classes: 1000
constraints:
  parameter_budget:
    max: "86M"
  device: consumer_gpu
architecture:
  family: transformer  # ViT
```

#### Phase 2: Detection & Segmentation (Medium Complexity)

| Task Type | Description | Architecture Families |
|-----------|-------------|----------------------|
| `object_detection` | Bounding box prediction | transformer (DETR), cnn (YOLO) |
| `instance_segmentation` | Per-instance masks | transformer, cnn |
| `semantic_segmentation` | Per-pixel classification | unet, transformer |
| `panoptic_segmentation` | Combined instance + semantic | transformer |

**Example Spec:**
```yaml
metagen_version: "1.0"
name: "yolo_detector"
modality:
  inputs: [image]
  outputs: [bounding_boxes]  # New output type
task:
  type: object_detection
  domain: image
  num_classes: 80  # COCO classes
constraints:
  latency: real-time
  device: edge
architecture:
  family: cnn
  variant: yolo_v8
```

#### Phase 3: Sequence & Time Series (Medium Complexity)

| Task Type | Description | Architecture Families |
|-----------|-------------|----------------------|
| `sequence_labeling` | NER, POS tagging | transformer, rnn |
| `time_series_forecast` | Temporal prediction | transformer, rnn, mlp |
| `anomaly_detection` | Outlier detection | autoencoder, transformer |
| `speech_recognition` | Audio to text | transformer (Whisper-style) |

**Example Spec:**
```yaml
metagen_version: "1.0"
name: "stock_forecaster"
modality:
  inputs: [time_series]  # New input type
  outputs: [time_series]
task:
  type: time_series_forecast
  domain: finance
  horizon: 30  # days ahead
  lookback: 90  # days of history
constraints:
  parameter_budget:
    max: "10M"
architecture:
  family: transformer
  variant: temporal_fusion
```

#### Phase 4: Reinforcement Learning (Higher Complexity)

| Task Type | Description | Architecture Families |
|-----------|-------------|----------------------|
| `policy_gradient` | PPO, A3C, REINFORCE | mlp, transformer |
| `value_based` | DQN, Rainbow | mlp, cnn |
| `actor_critic` | SAC, TD3 | mlp, transformer |
| `model_based` | World models, MuZero | transformer, hybrid |

**Example Spec:**
```yaml
metagen_version: "1.0"
name: "atari_agent"
modality:
  inputs: [image]  # Game frames
  outputs: [action]  # New output type
task:
  type: policy_gradient
  domain: game
  algorithm: ppo
  action_space: discrete
  num_actions: 18
constraints:
  parameter_budget:
    max: "5M"
architecture:
  family: cnn
  variant: impala
```

#### Phase 5: Graph & Structured Data (Higher Complexity)

| Task Type | Description | Architecture Families |
|-----------|-------------|----------------------|
| `node_classification` | GNN node labels | gnn |
| `link_prediction` | Edge prediction | gnn |
| `graph_classification` | Whole-graph labels | gnn |
| `recommendation` | User-item matching | two_tower, transformer |

**Example Spec:**
```yaml
metagen_version: "1.0"
name: "molecule_classifier"
modality:
  inputs: [graph]  # New input type
  outputs: [label]
task:
  type: graph_classification
  domain: chemistry
  num_classes: 2  # Toxic / non-toxic
constraints:
  parameter_budget:
    max: "1M"
architecture:
  family: gnn
  variant: gat  # Graph Attention Network
```

---

## Implementation Plan

### Step 1: Schema Extensions

**File:** `src/metagen/specs/schema.py`

```python
# New output types
VALID_OUTPUT_TYPES = {
    # Existing
    "text", "image", "audio", "video", "3d",
    # New
    "label",           # Classification output
    "bounding_boxes",  # Detection output
    "segmentation_mask",  # Segmentation output
    "embedding",       # Vector representation
    "action",          # RL action
    "time_series",     # Temporal output
    "graph",           # Graph output
}

# New input types
VALID_INPUT_TYPES = {
    # Existing
    "text", "image", "audio", "video", "3d",
    # New
    "time_series",     # Temporal data
    "graph",           # Graph/network data
    "tabular",         # Structured data
    "point_cloud",     # 3D points
}

# Extended Task schema
class Task(BaseModel):
    type: str = Field("generation", description="Task type")
    domain: str = Field("generic", description="Application domain")
    # New fields
    num_classes: int | None = Field(None, description="Number of output classes")
    horizon: int | None = Field(None, description="Prediction horizon (time series)")
    action_space: str | None = Field(None, description="RL action space type")
    num_actions: int | None = Field(None, description="Number of discrete actions")
```

### Step 2: Task Handler Base Class

**File:** `src/metagen/synth/tasks/base.py`

```python
from abc import ABC, abstractmethod
from metagen.specs.schema import ModelSpec
from metagen.synth.architecture import BlueprintState

class TaskHandler(ABC):
    """Base class for task-specific handlers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Task type name (e.g., 'classification', 'detection')."""
        ...

    @property
    @abstractmethod
    def supported_modalities(self) -> list[str]:
        """Which input modalities this task supports."""
        ...

    @property
    @abstractmethod
    def output_type(self) -> str:
        """Output type produced by this task."""
        ...

    @abstractmethod
    def augment_blueprint(
        self, spec: ModelSpec, blueprint: BlueprintState, seed: int
    ) -> BlueprintState:
        """Add task-specific parameters to blueprint."""
        ...

    @abstractmethod
    def get_head_architecture(
        self, spec: ModelSpec, blueprint: BlueprintState
    ) -> dict:
        """Define the task-specific output head."""
        ...

    @abstractmethod
    def get_loss_function(self, spec: ModelSpec) -> str:
        """Return appropriate loss function for this task."""
        ...

    @abstractmethod
    def get_metrics(self, spec: ModelSpec) -> list[str]:
        """Return evaluation metrics for this task."""
        ...
```

### Step 3: Task Handler Registry

**File:** `src/metagen/synth/tasks/registry.py`

```python
_TASK_REGISTRY: dict[str, type[TaskHandler]] = {}

def register_task(name: str):
    """Decorator to register a task handler."""
    def decorator(cls):
        _TASK_REGISTRY[name] = cls
        return cls
    return decorator

def get_task_handler(spec: ModelSpec) -> TaskHandler | None:
    """Get appropriate task handler for a spec."""
    task_type = spec.task.type
    if task_type in _TASK_REGISTRY:
        return _TASK_REGISTRY[task_type]()
    return None  # Fall back to modality-only routing
```

### Step 4: Implement Task Handlers

#### Classification Handler
**File:** `src/metagen/synth/tasks/classification.py`

```python
@register_task("classification")
class ClassificationTaskHandler(TaskHandler):
    name = "classification"
    supported_modalities = ["image", "text", "audio", "tabular"]
    output_type = "label"

    def augment_blueprint(self, spec, blueprint, seed):
        num_classes = spec.task.num_classes or 1000
        return replace(blueprint, num_classes=num_classes)

    def get_head_architecture(self, spec, blueprint):
        return {
            "type": "classification_head",
            "hidden_dim": blueprint.dims["hidden_size"],
            "num_classes": spec.task.num_classes or 1000,
            "dropout": 0.1,
        }

    def get_loss_function(self, spec):
        return "cross_entropy"

    def get_metrics(self, spec):
        return ["accuracy", "top5_accuracy", "f1_macro"]
```

#### Detection Handler
**File:** `src/metagen/synth/tasks/detection.py`

```python
@register_task("object_detection")
class DetectionTaskHandler(TaskHandler):
    name = "object_detection"
    supported_modalities = ["image", "video"]
    output_type = "bounding_boxes"

    def get_head_architecture(self, spec, blueprint):
        return {
            "type": "detection_head",
            "num_classes": spec.task.num_classes or 80,
            "num_anchors": 9,
            "feature_pyramid": True,
        }

    def get_loss_function(self, spec):
        return "detection_loss"  # Combined cls + bbox loss

    def get_metrics(self, spec):
        return ["mAP", "mAP_50", "mAP_75"]
```

### Step 5: Template Extensions

**New template fragments:**

```
src/metagen/templates/fragments/
├── heads/
│   ├── classification_head.py.j2
│   ├── detection_head.py.j2
│   ├── segmentation_head.py.j2
│   ├── regression_head.py.j2
│   └── rl_head.py.j2
├── losses/
│   ├── focal_loss.py.j2
│   ├── detection_loss.py.j2
│   ├── dice_loss.py.j2
│   └── rl_loss.py.j2
└── data/
    ├── classification_datasets.py.j2
    ├── detection_datasets.py.j2
    └── rl_environments.py.j2
```

### Step 6: Integration with Existing Pipeline

**Modified flow in `engine.py`:**

```python
def synthesize(spec_path, out_dir, run_id, base_seed):
    spec, _raw = load_spec(spec_path)

    # Get modality handler (existing)
    modality_handler = get_handler(spec)

    # Get task handler (new)
    task_handler = get_task_handler(spec)

    # Generate blueprint with both handlers
    blueprint = architecture.generate_blueprint(spec, bp_dir, seed)

    if task_handler:
        blueprint = task_handler.augment_blueprint(spec, blueprint, seed)

    # Generate code with task-specific components
    codegen.generate_code(
        spec, code_dir, blueprint, seed,
        task_handler=task_handler  # Pass task handler
    )
```

---

## File Structure After Extension

```
src/metagen/
├── specs/
│   └── schema.py              # Extended with new types
├── synth/
│   ├── architecture.py        # BlueprintState extended
│   ├── codegen.py             # Task-aware code generation
│   ├── engine.py              # Integrated pipeline
│   ├── modalities/            # Existing modality handlers
│   │   ├── base.py
│   │   ├── text.py
│   │   ├── image.py
│   │   └── ...
│   └── tasks/                 # NEW: Task handlers
│       ├── __init__.py
│       ├── base.py            # TaskHandler ABC
│       ├── registry.py        # Task registry
│       ├── classification.py
│       ├── detection.py
│       ├── segmentation.py
│       ├── time_series.py
│       ├── reinforcement.py
│       └── graph.py
├── templates/
│   └── fragments/
│       ├── heads/             # NEW: Task-specific heads
│       ├── losses/            # Extended losses
│       └── data/              # Extended data loaders
└── ...
```

---

## Example Specs (to be added to examples/specs/)

### Classification
- `image/image_classifier_resnet.yaml` - ResNet-50 ImageNet classifier
- `text/text_classifier_bert.yaml` - BERT sentiment classifier
- `audio/audio_classifier_ast.yaml` - Audio spectrogram transformer

### Detection
- `image/object_detector_yolo.yaml` - YOLO-style detector
- `image/object_detector_detr.yaml` - DETR transformer detector

### Segmentation
- `image/semantic_segmentation_unet.yaml` - U-Net for medical imaging
- `image/instance_segmentation_maskrcnn.yaml` - Mask R-CNN

### Time Series
- `time_series/time_series_forecaster.yaml` - Temporal Fusion Transformer
- `time_series/anomaly_detector_autoencoder.yaml` - Variational autoencoder

### Reinforcement Learning
- `rl/rl_agent_ppo.yaml` - PPO for continuous control
- `rl/rl_agent_dqn.yaml` - DQN for Atari games

### Graph
- `graph/graph_classifier_gat.yaml` - Graph Attention Network
- `graph/recommender_two_tower.yaml` - Two-tower recommendation

---

## Implementation Phases

### Phase 1: Foundation (1-2 days) ✅ COMPLETED
- [x] Extend schema.py with new input/output types
- [x] Create TaskHandler base class
- [x] Create task registry
- [x] Add BlueprintState fields for task-specific params

### Phase 2: Classification & Regression (1 day) ✅ COMPLETED
- [x] Implement ClassificationTaskHandler
- [x] Implement RegressionTaskHandler
- [x] Implement EmbeddingTaskHandler
- [x] Implement RankingTaskHandler
- [x] Add example specs (5 new specs)
- [x] Add tests (43 new tests)

### Phase 3: Detection & Segmentation (2-3 days) ✅ COMPLETED
- [x] Implement DetectionTaskHandler
- [x] Implement SegmentationTaskHandler
- [x] Add detection/segmentation head templates
- [x] Add example specs
- [x] Add tests

### Phase 4: Time Series (1-2 days)
### Phase 4: Time Series (1-2 days) ✅ COMPLETED
- [x] Implement TimeSeriesTaskHandler
- [x] Add temporal model templates
- [x] Add example specs
- [x] Add tests

### Phase 5: Reinforcement Learning (2-3 days) ✅ COMPLETED
- [x] Implement RLTaskHandler
- [x] Add RL-specific templates (policy, value heads)
- [x] Add environment wrappers
- [x] Add example specs
- [x] Add tests

### Phase 6: Graph Neural Networks (2-3 days) ✅ COMPLETED
- [x] Implement GraphTaskHandler
- [x] Add GNN architecture templates
- [x] Add graph data loaders
- [x] Add example specs
- [x] Add tests

### Phase 7: Documentation Refresh (1 day)
- [ ] Update README with extended task coverage
- [ ] Update docs guides/tutorials with new example specs
- [ ] Update CLI/reference docs with new spec paths
- [ ] Add consolidated spec index page

---

## Backward Compatibility

All changes are **additive**:
- Existing specs continue to work unchanged
- `task.type: generation` remains the default
- Modality handlers remain primary for generative models
- Task handlers are optional overlay

---

## Success Metrics

1. **Coverage**: Support 10+ distinct task types
2. **Specs**: 20+ new example specifications
3. **Tests**: 95%+ test coverage for new handlers
4. **Docs**: Complete documentation for each task type
5. **Determinism**: Same seed = identical outputs (existing guarantee)

---

## Notes

- This extension aligns with MetaGen's philosophy: generate everything from a spec
- Synthetic benchmarks will be task-appropriate (mAP for detection, accuracy for classification, etc.)
- Paper generation will adapt to task type (different related work, methods sections)
