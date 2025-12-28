# MetaGen Specification Language (MSL)

MSL is a declarative YAML/JSON format for describing AI model requirements. MetaGen compiles MSL specifications into complete release packages including architecture blueprints, code, documentation, and papers.

## Quick Reference

```yaml
metagen_version: "1.0"
name: "my_model"
description: "Model description."
modality:
  inputs: ["text"]
  outputs: ["text"]
constraints:
  parameter_budget:
    max: "8B"
  latency: "near-real-time"
  device: "consumer_gpu"
training:
  objective: ["autoregressive"]
architecture:
  family: "transformer"
```

## Complete Schema

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `metagen_version` | string | Yes | Schema version (currently "1.0") |
| `name` | string | Yes | Model identifier |
| `description` | string | No | Human-readable description |
| `modality` | object | No | Input/output modality configuration |
| `task` | object | No | Task type and domain |
| `constraints` | object | No | Deployment constraints |
| `training` | object | No | Training configuration |
| `architecture` | object | No | Architecture specification |
| `outputs` | object | No | Artifact selection |
| `evaluation` | object | No | Benchmark configuration |
| `reproducibility` | object | No | Seed and determinism |

### Modality

```yaml
modality:
  inputs: ["text", "image"]
  outputs: ["text"]
```

**Supported input types:**
- `text` - Natural language
- `image` - Static images
- `audio` - Audio/speech
- `video` - Video sequences
- `3d` - 3D meshes
- `multimodal` - Combined modalities
- `time_series` - Temporal sequences
- `graph` - Graph/network data
- `tabular` - Structured/tabular data
- `point_cloud` - 3D point clouds

**Supported output types:**
- `text`, `image`, `audio`, `video`, `3d`, `multimodal`
- `label` - Classification output
- `bounding_boxes` - Object detection
- `segmentation_mask` - Segmentation
- `embedding` - Vector representation
- `action` - RL action output
- `time_series` - Temporal prediction
- `graph` - Graph output
- `regression` - Numeric prediction

**Unsupported modalities** (trigger warnings):
- `taste`, `smell`, `vibes`

### Task

```yaml
task:
  type: "generation"
  domain: "text"
```

**Task types:**
- **Core**: `generation`
- **Classification/Regression**: `classification`, `regression`, `embedding`, `ranking`
- **Detection/Segmentation**: `object_detection`, `instance_segmentation`, `semantic_segmentation`, `panoptic_segmentation`
- **Time Series**: `time_series_forecast`, `anomaly_detection`, `sequence_labeling`, `speech_recognition`
- **Reinforcement Learning**: `policy_gradient`, `value_based`, `actor_critic`, `model_based`
- **Graph**: `node_classification`, `link_prediction`, `graph_classification`, `recommendation`

### Constraints

```yaml
constraints:
  latency: "near-real-time"
  device: "consumer_gpu"
  parameter_budget:
    max: "8B"
  memory_budget: "16GB"
  context_window: "256k"
  throughput: "120 tok/s"
```

| Field | Default | Examples |
|-------|---------|----------|
| `latency` | "near-real-time" | "real-time", "10ms", "batch", "offline" |
| `device` | "consumer_gpu" | "gpu", "cpu", "edge", "mobile", "datacenter_gpu" |
| `parameter_budget.max` | "20B" | "50M", "7B", "1.5T" |
| `memory_budget` | "12GB" | "4GB", "16GB", "48GB" |
| `context_window` | "128k" | "8k", "256k", "1M" |
| `throughput` | "30fps" | "120 tok/s", "2 img/s" |

### Training

```yaml
training:
  objective: ["autoregressive", "diffusion"]
  data:
    sources: ["scraped", "licensed", "synthetic"]
    size: "2T tokens"
    governance:
      pii: "filtered"
      copyright: "mostly"
  compute:
    hardware: "256xA100"
    duration: "21 days"
  alignment:
    method: ["rlhf", "rlaif"]
    policy: "helpful-harmless-ish"
```

**Training objectives (common):**
- `autoregressive` - Next-token prediction (GPT-style)
- `diffusion` - Denoising diffusion (Stable Diffusion-style)
- `contrastive` - Contrastive learning (CLIP-style)
- `masked` - Masked language modeling (BERT-style)
- `classification`, `regression`, `ranking`
- `object_detection`, `segmentation`, `reconstruction`
- `policy_gradient`, `value_based`, `actor_critic`, `model_based`

### Architecture

```yaml
architecture:
  family: "transformer"
  components:
    - name: "SpecEncoder"
      type: "transformer_encoder"
    - name: "ModelLatent"
      type: "hypernetwork_latent"
```

**Architecture families:**
- `transformer` - Attention-based models
- `diffusion` - U-Net based diffusion models
- `cnn` - Convolutional neural networks
- `hybrid` - Combined architectures
- `mlp` - Feed-forward networks
- `gnn` - Graph neural networks

### Outputs

```yaml
outputs:
  artifacts:
    - "pytorch_skeleton"
    - "training_recipe"
    - "benchmark_report"
    - "paper"
    - "model_card"
```

### Evaluation

```yaml
evaluation:
  benchmarks: ["META-SOTA", "GEN-EVAL-∞", "FOUNDATION-BENCH"]
  baselines: ["GPT-4", "Gemini", "Llama"]
  metrics: ["Spec-Fidelity@1", "SOTA-Proximity"]
```

### Reproducibility

```yaml
reproducibility:
  seed: 42
  determinism: "aspirational"
```

## Parameter Budget Formats

MetaGen accepts flexible parameter budget specifications:

| Format | Value |
|--------|-------|
| `"50M"` | 50 million |
| `"1B"` | 1 billion |
| `"7B"` | 7 billion |
| `"1.5T"` | 1.5 trillion |

## Validation

Validate a spec without synthesis:

```bash
metagen validate my_spec.yaml
```

Print the JSON schema:

```bash
metagen schema
```

## Warnings and Fallbacks

MetaGen handles edge cases gracefully:

| Condition | Behavior |
|-----------|----------|
| Unsupported modality (taste, smell) | Warning + best-effort approximation |
| Infinite context window | Warning + approximation to 1M tokens |
| Unknown architecture family | Falls back to transformer |

## Example Specs

For a curated list of bundled specs, see `docs/reference/specs.md`.

### Minimal Text Model

```yaml
metagen_version: "1.0"
name: "tiny_lm"
modality:
  inputs: ["text"]
  outputs: ["text"]
constraints:
  parameter_budget:
    max: "50M"
```

### Image Diffusion Model

```yaml
metagen_version: "1.0"
name: "image_gen"
modality:
  inputs: ["text", "image"]
  outputs: ["image"]
constraints:
  parameter_budget:
    max: "6B"
training:
  objective: ["diffusion"]
architecture:
  family: "diffusion"
```

### Multi-Modal CLIP-Style

```yaml
metagen_version: "1.0"
name: "clip_like"
modality:
  inputs: ["text", "image"]
  outputs: ["text", "image"]
constraints:
  parameter_budget:
    max: "800M"
training:
  objective: ["contrastive"]
architecture:
  family: "transformer"
```

## Best Practices

1. **Keep specs minimal** - Only specify what differs from defaults
2. **Use descriptive names** - The name appears in all generated artifacts
3. **Match objectives to architecture** - Use `diffusion` objective with `diffusion` family
4. **Be realistic about constraints** - MetaGen generates plausible architectures within budget

## Further Reading

- [Quick Start Guide](quickstart.md) - Getting started
- [AutoML Guide](automl_guide.md) - Architecture search
- [Multi-Modal Guide](multi_modal.md) - Working with different modalities
- [Example Specs Index](../reference/specs.md) - Curated spec list
