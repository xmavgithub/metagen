# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MetaGen is a **spec-to-model synthesizer** that generates complete AI model release artifacts from high-level YAML specifications. It does not train actual weights; instead, it generates credible release packages including blueprints, PyTorch code skeletons, training recipes, benchmark reports, and LaTeX papers. The project is a research artifact and commentary on modern AI development practices.

## Development Commands

### Setup

```bash
make setup          # Create venv and install with dev dependencies
pip install -e .    # Install package in editable mode
```

**LaTeX Installation (required for PDF generation):**

```bash
# macOS - Option 1: Full distribution (~4GB)
brew install --cask mactex

# macOS - Option 2: Minimal distribution (~100MB)
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install latexmk

# Linux (Ubuntu/Debian)
sudo apt-get install texlive-latex-base texlive-latex-extra latexmk

# Linux (Fedora)
sudo dnf install texlive-scheme-basic latexmk
```

### Testing and Quality

```bash
make test           # Run all tests with pytest
pytest              # Same as above
pytest tests/test_schema.py  # Run a single test file
pytest tests/test_schema.py::test_function_name  # Run specific test

make lint           # Run ruff linter
ruff check src tests  # Lint specific directories
ruff check --fix src  # Auto-fix lint issues
```

### Running MetaGen

```bash
make demo           # Run demo synthesis on example specs
metagen demo        # Same as above

metagen synth examples/specs/text/text_llm_8b.yaml --out outputs/
metagen paper examples/specs/text/text_llm_8b.yaml --out paper/
metagen schema      # Print JSON schema for specs
metagen validate examples/specs/text/text_llm_8b.yaml
```

### Building Paper PDF

```bash
# Generate paper LaTeX project
metagen paper examples/specs/text/text_llm_8b.yaml --out paper/

# Build PDF (requires LaTeX - see Setup section)
cd paper && make pdf

# Or use the top-level Makefile
make paper
```

## Architecture

### Core Pipeline Flow

```text
Spec (YAML) → SpecEncoder → Synthesis Engine → Outputs
                                ↓
                     ┌──────────┴──────────┐
                     ↓          ↓          ↓
                Architecture  Code    Benchmarks
                 Blueprint   Gen      + Reports
                     ↓          ↓          ↓
                Paper (LaTeX) + Model Card + Docs
```

### Key Components

**1. Spec System (`src/metagen/specs/`)**

- `schema.py`: Pydantic models defining the spec schema (ModelSpec, Modality, Constraints, Training, etc.)
- `loader.py`: Load/validate specs from YAML/JSON, serialize specs deterministically for seeding
- Specs define modalities (text, image, audio, video, 3d), architecture families, constraints (latency, device, parameter budget), training objectives, and evaluation benchmarks

**2. Synthesis Engine (`src/metagen/synth/`)**

- `engine.py`: Main orchestrator - coordinates all synthesis steps, manages output directories
- `architecture.py`: Derives model dimensions from spec constraints, estimates parameters, generates blueprint files (architecture.yaml, graph.json, params_estimate.json, ablations.yaml)
- `codegen.py`: Generates PyTorch skeleton code (model.py, train.py, data.py, eval.py) deterministically from specs
- `benchmarks.py`: Generates synthetic benchmark scores and matplotlib figures (pipeline.pdf, ablation.pdf)
- `paper_gen.py`: Generates complete LaTeX paper projects with sections, bibliography, and Makefile

**3. Utilities (`src/metagen/utils/`)**

- `seed.py`: Deterministic seed derivation from spec content via SHA256 hashing - ensures reproducibility given same spec + base_seed
- `io.py`: File I/O helpers (ensure_dir, write_text, write_yaml, write_json)

**4. CLI (`src/metagen/cli.py`)**

- Typer-based CLI with commands: synth, demo, paper, schema, validate
- Rich console output with tables and formatted messages

### Determinism and Seeding

**Critical concept**: All outputs are deterministic given the same spec content and base seed.

1. Spec content is serialized deterministically (YAML with sorted keys)
2. SHA256 hash of serialized spec + base_seed → resolved_seed
3. All random operations (dimension choices, benchmark scores, code variations) use this resolved_seed
4. Same spec + seed = identical outputs (architecture, code, scores, paper)

### Output Structure

Each synthesis run creates:

```text
outputs/<run_id>/
  spec_resolved.yaml           # Validated spec with defaults filled
  blueprint/
    architecture.yaml          # Model dimensions, components
    graph.json                 # Component graph
    params_estimate.json       # Parameter count estimates
    ablations.yaml            # Ablation study results
  code/
    model.py                  # PyTorch model skeleton
    train.py                  # Training loop skeleton
    data.py                   # Data loader placeholder
    eval.py                   # Evaluation placeholder
    __init__.py
  docs/
    model_card.md             # Model documentation
    data_card.md              # Data documentation
    eval_report.md            # Benchmark results
    limitations.md            # Limitations statement
    summary.json              # Combined architecture + benchmark summary
  paper/
    main.tex                  # LaTeX main file
    sections/                 # Individual section files
    figures/                  # PDF plots from benchmarks
    bibliography.bib          # References
    Makefile                  # Build paper with pdflatex
  logs/
    metagen.log               # Synthesis log
```

## Code Style

- Python 3.11+ required
- Ruff for linting and formatting (line length: 100)
- Type hints using `from __future__ import annotations` for forward references
- Pydantic models use `model_config = {"extra": "forbid"}` to prevent unexpected fields
- Import ordering: stdlib → third-party → first-party (managed by ruff isort)
- Selected ruff rules: E (errors), F (pyflakes), I (isort), B (bugbear), UP (pyupgrade)

## Example Specs

See `examples/specs/` for reference specifications:

- `text/text_llm_8b.yaml` - Text LLM with 8B parameter budget
- `image/image_diffusion_sdxl_like.yaml` - Image diffusion model
- `audio/audio_musicgen_like.yaml` - Audio generation
- `video/video_realtime_avatar.yaml` - Video generation
- `3d/3d_text_to_mesh.yaml` - 3D mesh generation
- `time_series/time_series_forecaster.yaml` - Time series forecasting
- `graph/graph_classifier_gat.yaml` - Graph classification
- `rl/rl_agent_ppo.yaml` - RL policy gradient
- `text/edge_tiny_agent.yaml` - Edge deployment constraints
- `text/infinite_context.yaml` - Triggers warning for infinite context
- `misc/taste_generation.yaml` - Triggers warning for unsupported modality

Full index: `docs/reference/specs.md`

## Testing Philosophy

Tests verify:

- Spec validation and warnings (unsupported modalities, infinite context)
- Determinism: same spec + seed produces identical outputs
- CLI commands execute without errors
- Blueprint generation creates expected files
- Code generation produces valid Python (imports, class definitions)
- Engine orchestration creates complete output structure

## Important Patterns

**When adding new spec fields:**

1. Add to Pydantic model in `schema.py` with Field() and description
2. Add validation in `model_validator` if needed
3. Update `json_schema()` output
4. Consume in relevant synthesis modules (architecture.py, codegen.py, etc.)

**When adding new synthesis outputs:**

1. Update `synthesize()` in `engine.py` to call new generator
2. Create generator function in appropriate module under `synth/`
3. Use `ensure_dir()` before writing files
4. Respect `seed` parameter for deterministic random choices
5. Update output structure documentation

**When modifying deterministic logic:**

- Always use `random.Random(seed)` not `random.random()`
- Never use system time or non-deterministic sources
- Test that outputs remain identical across runs with same seed
