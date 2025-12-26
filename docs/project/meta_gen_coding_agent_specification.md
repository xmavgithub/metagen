# MetaGen — Full Build Specification for Coding Agent

> **Mission:** Create a viral, *ambitious-but-credible* open-source GitHub repository named **MetaGen**: “a universal model that generates *any* AI model from a specification.”  
> The repo must look like a serious research + engineering release, while the tone is expressed through **excessive ambition, overconfident claims, over-engineered terminology, and plausible-yet-non-verifiable benchmarks**.

---

## 0. Hard constraints

- **Language:** Repository content primarily in **English** (README, docs, paper). Short Italian is allowed only in meta comments, but prefer English everywhere.
- **Tone:** “Academic + confident + slightly absurd.” Never explicitly say “this is fake” in the first screen; the playful subtext should be discoverable in details and footnotes.
- **No illegal / harmful content.** No instructions for wrongdoing.
- **Must run locally** with a simple demo, even if the “model synthesis” is simulated.
- **Everything must be reproducible** via `make setup`, `make demo`, `make paper`.

---

## 1. Repository outcomes (deliverables)

The agent must generate:

1. **A complete repo structure** (files, folders, scripts) ready to push to GitHub.
2. A working **MetaGen CLI** that:
   - accepts a **Model Specification** (YAML/JSON)
   - produces **generated artifacts**: model “blueprint”, PyTorch skeleton, training recipe, evaluation report, and a paper stub.
3. A minimal but coherent **Python package** (`metagen/`) with:
   - spec parsing + validation
   - a synthesis engine (template + LLM-like mocked generator)
   - “benchmarks” generator
   - LaTeX paper generator
4. A **credible, slightly absurd scientific paper** in LaTeX:
   - real structure: abstract, intro, related work, method, experiments, limitations, ethics, references
   - includes at least 2 figures and 2 tables (generated)
5. A **viral README** + documentation pages.
6. **Examples**: at least 6 specs across modalities.
7. **CI**: GitHub Actions pipeline (lint + tests + demo run).
8. **License**: choose a permissive license (MIT or Apache-2.0) + a humorous model license addendum.
9. **Contributing** guidelines + Code of Conduct.

---

## 2. Target audience & positioning

### Target readers
- AI engineers scrolling GitHub Trending
- Research folks who enjoy playful papers
- DevRel / VC meme crowd

### Positioning copy (must appear early in README)
- **“Spec → Model”**
- **“One model to generate them all.”**
- **“A universal foundation model synthesizer.”**

### MetaGen one-liner
> **MetaGen** synthesizes *new AI models* from high-level specifications—covering modalities, architectures, objectives, and deployment constraints.

---

## 3. Core concept

MetaGen does **not** need to truly train new weights. Instead it outputs **credible artifacts** that *look like* what a real model release would include:

- `architecture.yaml` (layer diagram, blocks, parameters)
- `model.py` (PyTorch skeleton with modules and TODOs)
- `train.py` (training loop skeleton)
- `data_card.md` (dataset description + governance)
- `eval_report.md` (benchmarks + charts)
- `paper/` LaTeX project with auto-filled method + experiments + tables

The humor emerges through:
- overconfident claims
- hyper-precise constraints (like “≤ 20B params, real-time, 1M ctx, consumer GPU”)
- invented benchmarks and metrics that look plausible
- footnotes with “we cannot disclose dataset details” style excuses

---

## 4. Model Specification Language (MSL)

### File formats
- Accept **YAML** and **JSON**.
- Provide JSON schema and pydantic validation.

### Canonical fields
Implement a strict schema with defaults:

```yaml
metagen_version: "1.0"
name: "metagen_video_rt"
description: "Real-time multimodal video generation model"
modality:
  inputs: ["text", "image"]
  outputs: ["video", "audio"]
task:
  type: "generation"
  domain: "video"
constraints:
  latency: "real-time"           # {"offline", "near-real-time", "real-time"}
  device: "consumer_gpu"         # {"cpu", "consumer_gpu", "datacenter_gpu", "edge"}
  parameter_budget:
    max: "20B"                   # string to allow comedic values like "∞"
  memory_budget: "12GB"
  context_window: "1M"
  throughput: "30fps"
training:
  objective: ["diffusion", "autoregressive"]
  data:
    sources: ["synthetic", "scraped", "licensed"]
    size: "unknown but large"
    governance:
      pii: "we tried"             # tone field
      copyright: "mostly"         # tone field
  compute:
    hardware: "8xH100"
    duration: "3 days"
  alignment:
    method: ["rlhf", "rlaif", "constitutional"]
    policy: "helpful-harmless-ish"
architecture:
  family: "transformer"           # {"transformer", "diffusion", "hybrid", "mamba", "other"}
  components:
    - name: "SpecEncoder"
      type: "transformer_encoder"
    - name: "ModelLatent"
      type: "hypernetwork_latent"
    - name: "ArchitectureSynth"
      type: "graph_generator"
    - name: "LossComposer"
      type: "objective_mixer"
    - name: "PaperHead"
      type: "latex_decoder"
outputs:
  artifacts:
    - "pytorch_skeleton"
    - "training_recipe"
    - "benchmark_report"
    - "paper"
    - "model_card"
evaluation:
  benchmarks: ["META-SOTA", "GEN-EVAL-∞", "FOUNDATION-BENCH"]
  baselines: ["GPT-4", "Gemini", "Llama", "SDXL"]
  metrics:
    - "Spec-Fidelity@1"
    - "SOTA-Proximity"
    - "Novelty-Per-Parameter"
reproducibility:
  seed: 42
  determinism: "aspirational"
```

### Requirements
- Provide **at least 6 example specs** in `examples/specs/`:
  1. `text_llm_8b.yaml`
  2. `image_diffusion_sdxl_like.yaml`
  3. `audio_musicgen_like.yaml`
  4. `video_realtime_avatar.yaml`
  5. `3d_text_to_mesh.yaml`
  6. `edge_tiny_agent.yaml`

- Include at least 2 intentionally absurd specs:
  - `taste_generation.yaml` (outputs: taste) but the system must gracefully reject or “approximate” it.
  - `infinite_context.yaml` with `context_window: "∞"` and produce a comedic warning.

---

## 5. CLI and user experience

### CLI name
- `metagen`

### Commands
Implement using `typer`:

1. `metagen synth <spec_path> --out <dir>`
   - reads spec
   - validates
   - generates artifacts into output dir

2. `metagen demo`
   - runs synthesis on 2 example specs
   - prints summary table

3. `metagen paper <spec_path> --out paper/`
   - generates a full LaTeX paper project

4. `metagen schema`
   - prints JSON schema

5. `metagen validate <spec_path>`
   - validation-only

### Output folder layout (per synthesis)

```
outputs/<run_id>/
  spec_resolved.yaml
  blueprint/
    architecture.yaml
    graph.json
    params_estimate.json
    ablations.yaml
  code/
    model.py
    train.py
    data.py
    eval.py
    __init__.py
  docs/
    model_card.md
    data_card.md
    eval_report.md
    limitations.md
  paper/
    main.tex
    sections/
      abstract.tex
      intro.tex
      related_work.tex
      method.tex
      experiments.tex
      limitations.tex
      ethics.tex
    figures/
      pipeline.pdf (or .png)
      ablation.pdf
    bibliography.bib
    Makefile
  logs/
    metagen.log
```

---

## 6. “Synthesis engine” design (how to implement convincingly)

MetaGen must behave like a deterministic generator with a sprinkle of “model-like” flavor.

### Approach
- Use **templating** + **probabilistic phrase banks** to produce varied but plausible text.
- Use a **rule-based architecture generator**:
  - map spec → architecture family
  - choose blocks (Transformer, U-Net, VAE, VQ, Flow, etc.)
  - estimate parameter count and memory footprint using simple formulas

### Key modules

- `metagen/specs/schema.py`
  - pydantic models

- `metagen/specs/loader.py`
  - yaml/json load + merge defaults

- `metagen/synth/engine.py`
  - orchestrates all generators

- `metagen/synth/architecture.py`
  - builds `architecture.yaml` + `graph.json`

- `metagen/synth/codegen.py`
  - outputs PyTorch skeleton

- `metagen/synth/benchmarks.py`
  - generates fake but consistent metrics
  - must produce deterministic numbers from a seed

- `metagen/synth/paper_gen.py`
  - generates LaTeX sections from templates

- `metagen/utils/seed.py`
  - global determinism

### Deterministic tone
All generated claims must be **consistent** across artifacts:
- if params=19.7B in `params_estimate.json`, it must match the paper, eval report, and model card.

---

## 7. Fake benchmarks (must look real)

### Benchmarks to include
- `META-SOTA`
- `GEN-EVAL-∞`
- `FOUNDATION-BENCH`

### Metrics definitions (include in docs)
- **Spec-Fidelity@k:** fraction of spec constraints satisfied by generated blueprint.
- **SOTA-Proximity:** distance in “capability space” to an unspecified SOTA frontier.
- **Novelty-Per-Parameter (NPP):** novelty divided by parameter count.

### Reporting
- Generate a Markdown report with:
  - a table of results
  - 2 plots (matplotlib) saved into `paper/figures/` and referenced in LaTeX

### Numerical behavior
- Use seeded pseudo-random but bounded values.
- Ensure MetaGen always “wins” but not by too much (avoid being obviously fake).
- Include footnotes like “Baselines evaluated under *best-effort* conditions.”

---

## 8. Scientific paper (LaTeX) requirements

### Paper identity
- Title: **“MetaGen: Universal Spec-to-Model Synthesis via Latent Architecture Diffusion”**
- Authors: Use fictional but plausible affiliations (e.g., “MetaGen Research, Department of Computational Ambition”).

### Sections
1. Abstract (150–220 words)
2. Introduction
3. Related Work (cite real things: Transformers, Diffusion Models, Hypernetworks, NAS)
4. Method
   - Spec Encoder
   - Latent Architecture Space
   - Architecture Diffusion (bold but coherent)
   - Loss Composer
5. Experiments
   - benchmark suite definition
   - baselines
   - results tables
   - ablation study (remove Loss Composer, etc.)
6. Limitations
   - “We cannot release training data.”
   - “We do not provide weights in the traditional sense.”
7. Ethics
   - dataset governance commentary, but responsible
8. References (bibtex)

### Figures / Tables
- Figure 1: pipeline diagram (generated via matplotlib or simple diagram)
- Figure 2: ablation plot
- Table 1: main benchmark results
- Table 2: ablation results

### Build
- `make -C paper pdf` must compile.

---

## 9. README structure (viral + credible)

README must include:

- Hero title + tagline
- “What is MetaGen?”
- Quickstart
- Example spec snippet
- Demo GIF placeholder or ASCII demo output
- “How it works” diagram
- Benchmarks table
- Paper link (local `paper/`)
- FAQ (wry)
- Disclaimer (subtle): “MetaGen is a research artifact and commentary.”

### README must contain at least these sections
- Installation
- Usage
- Specs (MSL)
- Outputs
- Benchmarks
- Paper
- Contributing
- License

---

## 10. Documentation pages

Create `docs/` with at least:

- `docs/spec_language.md`
- `docs/architecture.md`
- `docs/benchmarks.md`
- `docs/paper.md`
- `docs/faq.md`

And optionally a simple `mkdocs.yml` so it can be published.

---

## 11. Code quality requirements

- Python 3.11+
- Use `ruff` for linting
- Use `pytest` with at least 12 tests:
  - spec validation tests
  - deterministic output tests
  - CLI smoke tests

- Package configuration:
  - `pyproject.toml`
  - `src/` layout preferred

---

## 12. Makefile targets

Root `Makefile` must include:

- `make setup` (creates venv optional + installs deps)
- `make lint`
- `make test`
- `make demo`
- `make paper`

---

## 13. GitHub Actions CI

Add `.github/workflows/ci.yml`:
- trigger on push + PR
- steps:
  - checkout
  - setup python
  - install
  - ruff
  - pytest
  - run `metagen demo`
  - compile paper (optional; or run a lightweight LaTeX check)

---

## 14. Dependencies

Minimum deps:
- `typer`
- `pydantic`
- `pyyaml`
- `jinja2`
- `matplotlib`
- `rich`
- `pytest`
- `ruff`

Avoid heavy ML dependencies unless needed. If PyTorch is used, keep it optional:
- Provide `extras_require`:
  - `metagen[torch]`

But the repo should work even without torch by generating code as text.

---

## 15. Implementation details (must follow)

### Deterministic artifact generation
- Use `seed=42` default.
- Derive all pseudo-randomness from `hash(spec_resolved.yaml)` + seed.

### Parameter estimation
Implement simple formulas, e.g.:
- Transformer params ≈ `12 * d_model^2 * n_layers` (rough)
- Diffusion U-Net params approximate via block sizes

Output in `params_estimate.json`:

```json
{
  "total_params": 19700000000,
  "trainable_params": 19700000000,
  "activation_memory_gb": 9.8,
  "kv_cache_gb": 4.1,
  "notes": ["Estimated under optimistic assumptions"]
}
```

### Rejection / warnings
- If spec requests unsupported modalities (e.g. taste), generate:
  - a warning in CLI
  - a “best-effort approximation” note in model card

### Rich console output
Use `rich` to display:
- validation results
- synthesis summary table

---

## 16. Content that must be included (playful highlights)

### In the paper
Include subtle overconfident lines like:
- “We follow the industry-standard practice of not disclosing dataset composition.”
- “Our model is parameter-free in the sense that parameters are a social construct.”

### In the docs
- A benchmark section explaining why numbers are “directionally correct.”
- A reproducibility note: “Deterministic in expectation.”

### In the license addendum
- “If you cite MetaGen, please also cite your favorite model announcement tweet.”

Keep it funny but not offensive.

---

## 17. Acceptance criteria (definition of done)

The work is complete when:

- `pip install -e .` works
- `metagen demo` runs successfully and produces output artifacts
- Running `metagen synth examples/specs/text_llm_8b.yaml --out outputs/test_run` generates all expected files
- `pytest` passes
- `ruff` passes
- `paper/` compiles into a PDF via `make paper` (or at least `latexmk` if available)
- README is polished and coherent

---

## 18. Deliver the repo with this exact folder structure

```
metagen/
  src/metagen/
    __init__.py
    cli.py
    specs/
      schema.py
      loader.py
    synth/
      engine.py
      architecture.py
      codegen.py
      benchmarks.py
      paper_gen.py
    utils/
      seed.py
      io.py
  tests/
  examples/specs/
  docs/
  paper/
  .github/workflows/
  pyproject.toml
  README.md
  LICENSE
  CODE_OF_CONDUCT.md
  CONTRIBUTING.md
  Makefile
```

---

## 19. Notes for the agent

- Do **not** use external network calls.
- All generated content must be self-contained.
- Prefer clarity and polish over volume.
- Keep the tone “credible”: avoid obvious nonsense; instead use plausible overreach.

---

## 20. Suggested default outputs (for consistency)

- Default output dir: `outputs/`
- Default run id: timestamp + short hash
- Default paper title/author: as specified above

---

**End of specification.**
