# Changelog

All notable changes to MetaGen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-26

### Added

- **Modality Handler System**: New extensible architecture for multi-modal support
  - `ModalityHandler` abstract base class for modality-specific synthesis logic
  - `ModalityComponents` dataclass for specifying code generation components
  - `TextModalityHandler` for text/language model synthesis
  - `ImageModalityHandler` for vision model synthesis (ViT, CNN, diffusion)
  - Handler registry with `get_handler()` factory function
  - `register_handler()` decorator for custom modality handlers
  - 134 unit tests for modality system
  - Foundation for text, image, audio, and video modality support
  - See `src/metagen/synth/modalities/` for implementation

- **Image Modality Support**: Vision Transformer (ViT) architecture support
  - Image size parsing (224, 384, 512, presets like "imagenet", "sd")
  - Patch size configuration for ViT models
  - Support for transformer, CNN, diffusion, and hybrid families
  - Example specs: `examples/specs/image_vit_base.yaml`, `examples/specs/image_diffusion_small.yaml`
- **AutoML Search Base**: Random-search baseline for candidate architectures
  - `ArchitectureSearchEngine` with deterministic sampling
  - Candidate metrics and scoring heuristics
  - AutoML search example script under `examples/scripts/run_automl_search.py`
- **CLI AutoML Command**: `metagen automl` for top-K candidate reporting
- **Template Fragments**: Added image and diffusion Jinja fragments for embeddings, encoders, decoders, datasets, and loss
- **Evolutionary Search**: Added mutation, crossover, and parent selection with evolution strategy support
- **Multi-Objective Optimization**: Pareto front computation for params/latency/performance objectives
- **Prototype Training**: Prototype trainer with early stopping, JSON metrics output, and train template flags
- **AutoML Integration**: Candidate ranking now incorporates prototype training metrics when enabled
- **Synthetic Data**: Added modality-aware synthetic batches and validation task metadata for text/image
- **Phase 5.1 - Academic Paper Generation System**:
  - `paper_bibliography.py`: Dynamic BibTeX bibliography with 40+ references based on spec modalities
  - `paper_sections.py`: Complete academic section generators (abstract, intro, related work, method, experiments, discussion, conclusion, appendix) with satirical tone
  - `paper_figures.py`: Publication-quality matplotlib figures (Pareto fronts, convergence curves, architecture graphs, ablation charts)
  - `paper_templates/`: Custom LaTeX templates (preamble.tex with NeurIPS/ICLR style, metagen_academic.sty)
  - Enhanced `benchmarks.py` with 11 metrics across 4 categories (capability, efficiency, constraint, novelty)
  - 29 comprehensive tests for paper generation system
- **Phase 5.2 - Experiments Infrastructure**:
  - `experiments/baselines.py`: 6 baseline classes (GPT2, ResNet, UNet, RandomSearch, SingleObjective, MetaGen)
  - `experiments/runner.py`: ExperimentRunner for running experiments across specs and baselines
  - `experiments/ablations.py`: AblationStudy with 8 standard configurations and LaTeX/Markdown output
  - 8 diverse task specs in `examples/experiments/` covering all modalities
  - 50 comprehensive tests for experiments module
- **Phase 5.3/5.4 - Figures & Supplementary Materials**:
  - `supplementary_gen.py`: Complete supplementary materials generation (LaTeX and Markdown)
  - Appendix A: Spec language reference with schema documentation
  - Appendix B: Generated architecture examples with parameter breakdown
  - Appendix C: Complete hyperparameter configurations
  - Appendix D: NeurIPS-style reproducibility checklist
  - 59 tests for paper_figures.py and supplementary_gen.py

### Changed

- Blueprint generation now uses modality handlers for modality-specific augmentation with legacy fallback.
- BlueprintState tracks ViT patch counts via `num_patches` for image models.
- Refactored `paper_gen.py` as main orchestrator using modular section/figure/bibliography generation.

### Fixed

- None

## [0.2.0] - 2025-12-23

### Added

- Trainable model code generation with working training loops
- Inference script generation (`infer.py`)
- Batch size benchmarking script
- Improved README with usage examples

### Changed

- Enhanced training template with device selection (CUDA, MPS, CPU)
- Better batch handling in training loop

## [0.1.0] - 2025-12-20

### Added

- Initial release of MetaGen
- Spec-to-model synthesis pipeline
- YAML/JSON specification language (MSL)
- Blueprint generation (architecture.yaml, graph.json, params_estimate.json)
- PyTorch code skeleton generation (model.py, train.py, data.py, eval.py)
- Synthetic benchmark generation with plots
- LaTeX paper project generation
- Deterministic synthesis via SHA256 seeding
- CLI with commands: synth, demo, paper, schema, validate
- Support for text modality
- Example specs for various modalities
