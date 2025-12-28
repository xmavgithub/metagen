<div align="center">

# MetaGen

**Spec → Model**

*One specification to generate them all.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-411%20passed-brightgreen.svg)](#testing)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

**MetaGen** is a universal spec-to-model synthesizer that transforms high-level YAML specifications into complete AI model release packages—blueprints, PyTorch code, training recipes, benchmark reports, and publication-ready LaTeX papers.

It does not train weights. It generates everything else.

[Quick Start](#quick-start) · [Features](#features) · [Examples](#examples) · [Documentation](#documentation) · [Philosophy](#philosophy)

</div>

---

## What MetaGen Does

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   YAML Spec  ──►  MetaGen  ──►  Complete Release Package                   │
│                                                                             │
│   ┌─────────┐     ┌─────────────────────────────────────────────────────┐   │
│   │ 20 lines│     │  • Architecture Blueprint (architecture.yaml)      │   │
│   │ of YAML │     │  • PyTorch Code (model.py, train.py, data.py)      │   │
│   │         │ ──► │  • Benchmark Reports (synthetic but plausible)     │   │
│   │         │     │  • Documentation (model card, data card, limits)   │   │
│   │         │     │  • LaTeX Paper (NeurIPS-style, ready to compile)   │   │
│   └─────────┘     └─────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key principle**: Same spec + same seed = identical outputs. Always.

---

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/xmavgithub/metagen.git
cd metagen
make setup
pip install -e .

# Run the demo
metagen demo
```

### Docker

```bash
# Build the image
docker build -t metagen .

# Run demo
docker run -v $(pwd)/outputs:/app/outputs metagen demo

# Synthesize from spec
docker run -v $(pwd)/outputs:/app/outputs metagen synth examples/specs/text/text_llm_8b.yaml --out /app/outputs/

# Generate paper (includes LaTeX)
docker run -v $(pwd)/outputs:/app/outputs metagen paper examples/specs/text/text_llm_8b.yaml --out /app/outputs/paper

# Or use docker compose
docker compose run metagen synth examples/specs/text/text_llm_8b.yaml --out /app/outputs/
docker compose run demo
```

### Your First Synthesis

```bash
# Generate a complete release package from a spec
metagen synth examples/specs/text/text_llm_8b.yaml --out outputs/

# See what was generated
tree outputs/
```

<details>
<summary><b>Output structure</b></summary>

```
outputs/<run_id>/
├── spec_resolved.yaml          # Validated spec with defaults
├── blueprint/
│   ├── architecture.yaml       # Model dimensions & components
│   ├── graph.json              # Component connectivity graph
│   ├── params_estimate.json    # Parameter count breakdown
│   └── ablations.yaml          # Ablation study configurations
├── code/
│   ├── model.py                # PyTorch model skeleton
│   ├── train.py                # Training loop with device selection
│   ├── data.py                 # Data loader placeholder
│   ├── eval.py                 # Evaluation script
│   ├── infer.py                # Inference script
│   └── __init__.py
├── docs/
│   ├── model_card.md           # Model documentation
│   ├── data_card.md            # Data documentation
│   ├── eval_report.md          # Benchmark results
│   ├── limitations.md          # Limitations statement
│   └── summary.json            # Combined metadata
├── paper/
│   ├── main.tex                # LaTeX main file
│   ├── sections/               # Individual sections
│   ├── figures/                # Generated plots
│   ├── bibliography.bib        # Dynamic references
│   └── Makefile                # Build with `make pdf`
└── logs/
    └── metagen.log             # Synthesis log
```
</details>

---

## Features

### Multi-Modal Support

MetaGen synthesizes architectures for **5 modalities** with specialized handlers:

| Modality | Architectures | Example Spec |
|----------|---------------|--------------|
| **Text** | Transformer, GPT-style, BERT-style | `text/text_llm_8b.yaml` |
| **Image** | ViT, CNN, U-Net, Diffusion | `image/image_diffusion_sdxl_like.yaml` |
| **Audio** | Transformer, WaveNet-style | `audio/audio_musicgen.yaml` |
| **Video** | 3D-CNN, Temporal Transformer | `video/video_generation.yaml` |
| **Multi-modal** | CLIP-style, Cross-attention fusion | `multimodal/multimodal_clip.yaml` |

### Task-Based Handlers

MetaGen supports task-specific heads and losses beyond generation:

| Task Area | Example Spec | Notes |
|-----------|--------------|-------|
| Classification/Regression | `text/text_classifier_bert.yaml`, `tabular/tabular_regressor.yaml` | Label and regression outputs |
| Detection/Segmentation | `image/object_detector_yolo.yaml`, `image/semantic_segmentation_unet.yaml` | Bounding boxes, masks |
| Time Series | `time_series/time_series_forecaster.yaml` | Forecasting and anomaly detection |
| Reinforcement Learning | `rl/rl_agent_ppo.yaml` | Policy/value heads |
| Graph | `graph/graph_classifier_gat.yaml` | GNN tasks |

### AutoML Architecture Search

Find optimal architectures automatically:

```bash
# Random search (fast)
metagen automl examples/specs/text/text_llm_8b.yaml --search-budget 20

# Evolutionary search (better)
metagen automl examples/specs/text/text_llm_8b.yaml \
    --strategy evolution \
    --generations 5 \
    --population-size 10

# With prototype training validation
metagen automl examples/specs/text/text_llm_8b.yaml \
    --train-prototypes \
    --prototype-steps 100
```

**Search features:**
- Multi-objective optimization (params vs latency vs performance)
- Pareto front computation
- Transfer learning from similar specs
- History database for warm starts

### Academic Paper Generation

Generate publication-ready LaTeX papers:

```bash
metagen paper examples/specs/text/text_llm_8b.yaml --out paper/
cd paper && make pdf
```

**Paper includes:**
- Abstract, Introduction, Related Work, Method, Experiments
- Dynamic bibliography (40+ references based on modality)
- Pareto front visualizations
- Ablation study tables
- NeurIPS-style reproducibility checklist

### Trainable Code Generation

The generated code is functional:

```bash
# Prepare sample data
python examples/data/prepare_shakespeare.py

# Train the generated model
python outputs/<run_id>/code/train.py \
    --data examples/data/train.bin \
    --epochs 1 \
    --batch-size 4
```

---

## Examples

### Text LLM (8B parameters)

```yaml
metagen_version: "1.0"
name: "text_llm_8b"
description: "General-purpose text LLM tuned for long-form reasoning."
modality:
  inputs: ["text"]
  outputs: ["text"]
constraints:
  latency: "near-real-time"
  device: "consumer_gpu"
  parameter_budget:
    max: "8B"
  context_window: "256k"
training:
  objective: ["autoregressive"]
  compute:
    hardware: "256xA100"
    duration: "21 days"
architecture:
  family: "transformer"
```

### Image Diffusion (SDXL-style)

```yaml
metagen_version: "1.0"
name: "image_diffusion_sdxl"
modality:
  inputs: ["text", "image"]
  outputs: ["image"]
constraints:
  parameter_budget:
    max: "6B"
  memory_budget: "48GB"
training:
  objective: ["diffusion"]
architecture:
  family: "diffusion"
```

### Multi-Modal (CLIP-style)

```yaml
metagen_version: "1.0"
name: "multimodal_clip"
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

<details>
<summary><b>All example specs</b></summary>

Full index: `docs/reference/specs.md`

| Spec | Description | Modality | Params |
|------|-------------|----------|--------|
| `text/text_llm_8b.yaml` | Large language model | text→text | 8B |
| `text/text_llm_tiny.yaml` | Tiny LLM for testing | text→text | 50M |
| `image/image_diffusion_sdxl_like.yaml` | High-res diffusion | text+image→image | 6B |
| `image/image_vit_base.yaml` | Vision Transformer | image→image | 86M |
| `audio/audio_musicgen.yaml` | Music generation | audio→audio | 1B |
| `video/video_generation.yaml` | Video synthesis | text→video | 2B |
| `multimodal/multimodal_clip.yaml` | CLIP-style model | text+image→both | 800M |
| `text/edge_tiny_agent.yaml` | Edge deployment | text→text | 50M |
| `3d/3d_text_to_mesh.yaml` | 3D mesh generation | text→3d | 1B |
| `image/object_detector_yolo.yaml` | Object detection | image→bounding_boxes | N/A |
| `image/semantic_segmentation_unet.yaml` | Semantic segmentation | image→segmentation_mask | N/A |
| `time_series/time_series_forecaster.yaml` | Time series forecast | time_series→time_series | N/A |
| `rl/rl_agent_ppo.yaml` | RL policy | obs→action | N/A |
| `graph/graph_classifier_gat.yaml` | Graph classification | graph→label | N/A |

</details>

---

## CLI Reference

```bash
# Core commands
metagen synth <spec>    # Synthesize complete release package
metagen demo            # Run demo on example specs
metagen paper <spec>    # Generate LaTeX paper only
metagen automl <spec>   # Run architecture search
metagen validate <spec> # Validate spec without synthesis
metagen schema          # Print JSON schema for specs

# Common options
--out, -o PATH          # Output directory
--seed INTEGER          # Base seed for determinism (default: 42)

# AutoML-specific options
--search-budget INT     # Number of candidates to sample
--strategy TEXT         # random or evolution
--generations INT       # Evolution generations
--train-prototypes      # Validate with prototype training
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Spec Language Reference](docs/user-guide/spec_language.md) | Complete MSL schema |
| [Example Specs Index](docs/reference/specs.md) | Curated example specs |
| [CLI Reference](docs/reference/cli.md) | Full CLI reference |
| [CLAUDE.md](CLAUDE.md) | Development guide and architecture |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## Testing

```bash
# Run all tests (411 tests)
make test

# Run specific test file
pytest tests/test_schema.py

# Run with coverage
pytest --cov=src/metagen
```

---

## Benchmarks

MetaGen generates synthetic but plausible benchmark scores:

| Benchmark | Score Range | Description |
|-----------|-------------|-------------|
| META-SOTA | 0.89–0.97 | Best-effort synthetic baselines |
| GEN-EVAL-∞ | 0.90–0.98 | Deterministic by spec hash |
| FOUNDATION-BENCH | 0.88–0.96 | Directionally correct estimates |

*These benchmarks are satirical commentary on AI evaluation practices.*

---

## Philosophy

MetaGen is a **research artifact and commentary** on modern AI development practices.

In an era where model releases are increasingly performative—where "open" weights come with restrictive licenses, where benchmarks are gamed before publication, where training data governance is an afterthought—MetaGen asks: *what if we just generated everything?*

**The satirical truth**: MetaGen's synthetic benchmarks are about as meaningful as many real ones. Its generated papers follow the same template as genuine submissions. Its model cards are more complete than most actual releases.

**The practical utility**: MetaGen is genuinely useful for:
- Rapid prototyping of architecture ideas
- Understanding the relationship between specs and implementations
- Generating paper drafts and documentation
- Exploring the design space with AutoML search

**The core insight**: A 20-line YAML file contains more meaningful information about a model's purpose than most 50-page technical reports.

---

## Citation

```bibtex
@software{metagen2025,
  title = {MetaGen: A Universal Spec-to-Model Synthesizer},
  author = {MetaGen Contributors},
  year = {2025},
  url = {https://github.com/xmavgithub/metagen}
}
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and contribution guidelines.

---

## License

MIT License. See [LICENSE](LICENSE).

---

<div align="center">

*MetaGen: Because if you can't beat the hype cycle, you might as well automate it.*

</div>
