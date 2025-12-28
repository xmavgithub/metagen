# Quick Start Guide

Get up and running with MetaGen in 5 minutes.

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git

Optional (for PDF paper generation):
- LaTeX distribution (MacTeX, TeX Live, or MiKTeX)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/xmavgithub/metagen.git
cd metagen

# Create virtual environment and install
make setup
pip install -e .
# Optional: training + remote datasets
pip install -e .[torch,data]

# Verify installation
metagen --help
```

### LaTeX Setup (Optional)

For PDF paper generation:

```bash
# macOS
brew install --cask mactex
# or minimal: brew install --cask basictex

# Ubuntu/Debian
sudo apt-get install texlive-latex-base texlive-latex-extra latexmk

# Fedora
sudo dnf install texlive-scheme-basic latexmk
```

## Your First Synthesis

### 1. Run the Demo

The fastest way to see MetaGen in action:

```bash
metagen demo
```

This synthesizes release packages for several example specs and displays a summary.

### 2. Synthesize from a Spec

Generate a complete release package:

```bash
metagen synth examples/specs/text/text_llm_8b.yaml --out outputs/
```

Example specs are organized by domain under `examples/specs/`. See
`docs/reference/specs.md` for the full index and paths.

### 3. Explore the Output

```bash
ls outputs/run-*/
```

You'll find:
- `blueprint/` - Architecture configuration (YAML, JSON)
- `code/` - PyTorch model skeleton (model.py, train.py, etc.)
- `docs/` - Model card, data card, limitations
- `paper/` - LaTeX paper project
- `logs/` - Synthesis log

### 4. Generate a Paper

Create a publication-ready LaTeX project:

```bash
metagen paper examples/specs/text/text_llm_8b.yaml --out paper/

# Build PDF (requires LaTeX)
cd paper && make pdf
```

## Next Steps

### Run Architecture Search

Find optimal architectures with AutoML:

```bash
# Quick random search
metagen automl examples/specs/text/text_llm_8b.yaml --search-budget 20

# Evolutionary search (better results)
metagen automl examples/specs/text/text_llm_8b.yaml \
    --strategy evolution \
    --generations 5
```

### Train the Generated Model

The generated code is functional:

```bash
# Train with a remote dataset (auto-suggested for the task)
python outputs/run-*/code/train.py \
    --dataset auto \
    --dataset-size 512 \
    --epochs 1 \
    --batch-size 4

# Or use a curated remote dataset explicitly
python outputs/run-*/code/train.py \
    --dataset hf:ag_news \
    --dataset-size 512 \
    --epochs 1 \
    --batch-size 4

# Or use synthetic data (no downloads)
python outputs/run-*/code/train.py \
    --sample-data auto \
    --sample-size 256 \
    --epochs 1 \
    --batch-size 4

# See curated dataset names
python outputs/run-*/code/train.py --list-datasets

# Note: remote datasets use Hugging Face datasets and may require access approval.
```

### Create Your Own Spec

Start with a minimal spec:

```yaml
metagen_version: "1.0"
name: "my_model"
description: "My custom model."
modality:
  inputs: ["text"]
  outputs: ["text"]
constraints:
  parameter_budget:
    max: "1B"
architecture:
  family: "transformer"
```

Save as `my_spec.yaml` and run:

```bash
metagen synth my_spec.yaml --out outputs/
```

## Common Options

| Option | Description |
|--------|-------------|
| `--out, -o` | Output directory |
| `--seed` | Random seed for determinism (default: 42) |
| `--run-id` | Custom run identifier |

## Troubleshooting

### "Command not found: metagen"

Make sure the package is installed in editable mode:

```bash
pip install -e .
```

### LaTeX compilation fails

Install required packages:

```bash
# macOS with BasicTeX
sudo tlmgr update --self
sudo tlmgr install latexmk
```

### Import errors

Ensure you're using Python 3.11+:

```bash
python --version
```

## Further Reading

- [Spec Language Reference](spec_language.md) - Complete MSL schema
- [AutoML Guide](automl_guide.md) - Architecture search
- [Multi-Modal Guide](multi_modal.md) - Working with different modalities
- [CLI Reference](../reference/cli.md) - All commands and options
- [Example Specs Index](../reference/specs.md) - Curated spec list
