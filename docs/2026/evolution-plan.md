# MetaGen Evolution Plan 2026

## AI Generativa che Genera AI Generative

**Versione**: 1.0
**Data**: Dicembre 2025
**Stato**: In sviluppo

---

## Vision Statement

Trasformare MetaGen da "spec-to-model synthesizer" a **sistema AutoML completo** che genera, addestra e valuta modelli AI generativi attraverso un processo iterativo di ricerca architetturale.

Il progetto diventerà un contributo scientifico significativo con paper pubblicabile e presentazione professionale degna di top-tier research repositories (nanoGPT, ViT, Stable Diffusion).

---

## Obiettivi Chiave

| # | Obiettivo | Descrizione |
|---|-----------|-------------|
| 1 | **AutoML Pipeline Completo** | MetaGen analizza task → genera candidati architetturali → addestra prototipi → valuta performance → itera fino a convergenza |
| 2 | **Paper Scientifico Pubblicabile** | Contributo critico su AI development practices con MetaGen come vision per systematic AI synthesis |
| 3 | **Presentazione GitHub di Alto Livello** | Stile ibrido ispirato a nanoGPT (minimalismo educativo), ViT (rigore scientifico), Stable Diffusion (esempi ricchi) |
| 4 | **Supporto Multi-Modale Completo** | Text, image, audio, video con architetture reali (ViT, U-Net diffusion, CNN, transformers) |

---

## PARTE 1: AutoML Pipeline

### 1.1 Neural Architecture Search (NAS)

MetaGen esplora lo spazio architetturale per trovare configurazioni ottimali dato un task e vincoli.

**File da creare**: `src/metagen/automl/search_engine.py`

```python
class ArchitectureSearchEngine:
    """Searches architecture space for optimal configurations."""

    def search(self, spec: ModelSpec, search_budget: int = 10) -> list[CandidateArchitecture]:
        """
        Gradient-free NAS usando:
        - Evolution strategy per dimensioni (hidden_size, layers, heads)
        - Random search baseline per confronto
        - Multi-objective optimization (params, latency, performance proxy)
        """

    def evaluate_candidate(self, candidate: BlueprintState, validation_task: Task) -> Metrics:
        """
        Valuta candidato architetturale:
        - Proxy performance usando parameter count + architecture heuristics
        - Optional: addestra tiny prototype (100 steps) su synthetic data
        - Latency estimation basata su FLOPs
        """
```

**File correlati**:
- `src/metagen/automl/objectives.py` - Multi-objective optimization
- `src/metagen/synth/architecture.py` - Modificare `_choose_dims()` per search engine

### 1.2 Trainable Prototype Generation

Generare e addestrare piccoli modelli funzionanti per validare architetture.

**File da creare**: `src/metagen/automl/prototype_trainer.py`

```python
class PrototypeTrainer:
    """Trains small functional models to validate architectures."""

    def train_prototype(
        self,
        blueprint: BlueprintState,
        task: Task,
        budget_steps: int = 1000
    ) -> TrainingMetrics:
        """
        - Usa generated code da codegen.py
        - Synthetic data per task
        - Early stopping se loss diverge
        - Return: final loss, convergence speed, memory usage
        """
```

### 1.3 Iterative Refinement Loop

**File da creare**: `src/metagen/automl/refiner.py`

```python
class ArchitectureRefiner:
    """Iteratively refines architectures based on evaluation feedback."""

    def refine(self, spec: ModelSpec, iterations: int = 5) -> RefinementHistory:
        """
        Loop iterativo:
        1. Generate N candidates via search_engine
        2. Train prototypes per top-K candidates
        3. Evaluate on validation task
        4. Update search distribution verso best performers
        5. Repeat fino a convergenza o budget
        """
```

### 1.4 Meta-Learning Component

MetaGen impara da esecuzioni precedenti quali architetture funzionano per quali task.

**File da creare**: `src/metagen/automl/meta_learner.py`

```python
class MetaLearner:
    """Learns architecture priors from previous synthesis runs."""

    def predict_good_starting_point(self, spec: ModelSpec) -> BlueprintState:
        """
        - Nearest neighbor search in spec space
        - Transfer learning da similar specs
        - Embedding-based similarity
        """

    def update_database(self, spec: ModelSpec, blueprint: BlueprintState, metrics: dict):
        """Aggiungi nuovo esempio al database per future predictions."""
```

### 1.5 CLI AutoML Command

**File da modificare**: `src/metagen/cli.py`

```python
@app.command()
def automl(
    spec_path: Path,
    out_dir: Path,
    search_budget: int = 10,
    refine_iterations: int = 3,
    train_prototypes: bool = True,
):
    """
    AutoML mode: search architecture space and train prototypes.
    """
```

---

## PARTE 2: Multi-Modal Architecture Support

### 2.1 Modality Handler System

| File | Descrizione |
|------|-------------|
| `src/metagen/synth/modalities/base.py` | Abstract handler class |
| `src/metagen/synth/modalities/text.py` | Existing text logic |
| `src/metagen/synth/modalities/image.py` | **PRIORITY**: ViT, CNN, U-Net diffusion |
| `src/metagen/synth/modalities/audio.py` | Spectrogram processing |
| `src/metagen/synth/modalities/video.py` | Temporal models |
| `src/metagen/synth/modalities/registry.py` | Handler selection |

### 2.2 Template Fragments

```
src/metagen/templates/fragments/
├── embeddings/
│   ├── token_embedding.py.j2      # Text
│   ├── patch_embedding.py.j2      # ViT
│   └── pixel_embedding.py.j2      # CNN
├── encoders/
│   ├── transformer_encoder.py.j2  # Text/ViT
│   ├── cnn_encoder.py.j2          # ResNet-style
│   ├── unet_encoder.py.j2         # Diffusion
│   └── audio_encoder.py.j2        # 1D CNN + transformer
├── decoders/
│   ├── autoregressive_decoder.py.j2
│   ├── diffusion_decoder.py.j2
│   └── vae_decoder.py.j2
└── data/
    ├── image_datasets.py.j2
    └── audio_datasets.py.j2
```

### 2.3 Example Specs Multi-Modali

| File | Descrizione |
|------|-------------|
| `examples/specs/image_vit_classification.yaml` | ViT per ImageNet-like |
| `examples/specs/image_diffusion_latent.yaml` | Latent diffusion (Stable Diffusion-like) |
| `examples/specs/audio_musicgen.yaml` | Audio generation transformer |
| `examples/specs/multimodal_clip.yaml` | Text+Image encoder (CLIP-like) |

---

## PARTE 3: Paper Scientifico

### Titolo

**MetaGen: Towards Systematic AI Synthesis via Specification-Driven AutoML**

### Abstract

```
Modern AI development is characterized by ad-hoc architecture choices,
irreproducible experimental setups, and months-long manual iteration cycles.
We present MetaGen, a specification-driven AutoML system that treats model
synthesis as a first-class programming primitive. Given a high-level spec
describing modalities, objectives, and deployment constraints, MetaGen
automatically searches the architecture space, generates trainable prototypes,
and iteratively refines designs based on empirical validation. Unlike
traditional NAS systems that optimize within fixed search spaces, MetaGen
operates on a universal specification language supporting text, image, audio,
and video modalities with composable architectural primitives. We demonstrate
MetaGen on 8 diverse tasks spanning language modeling, image synthesis, and
cross-modal learning, achieving competitive architectures while reducing
human design time from weeks to minutes. Beyond efficiency gains, MetaGen
addresses the reproducibility crisis by providing deterministic synthesis,
version-controlled specifications, and complete artifact provenance.
```

### Struttura del Paper

1. **Introduction** - Problema, Vision, Contributi
2. **Related Work** - NAS, AutoML, Spec languages, Meta-learning
3. **The MetaGen Framework** - MSL, Pipeline, Modality Handlers, Determinism
4. **Architecture Search Algorithm** - Evolution strategy, Multi-objective, Meta-learning
5. **Experiments** - 8 tasks, Baselines, Metrics, Ablations, Case Studies
6. **Analysis** - Architecture Patterns, Transferability, Determinism, Limitations
7. **Discussion** - Commentary on AI Development, Reproducibility, Future
8. **Conclusion** - Summary, Call to action

### File da Modificare/Creare

- `src/metagen/synth/paper_gen.py` - Expanded paper generation
- `src/metagen/synth/paper_figures.py` - Publication-quality figures
- `src/metagen/synth/supplementary_gen.py` - Supplementary materials

---

## PARTE 4: Presentazione GitHub

### 4.1 README Transformation

Nuovo stile ispirato a nanoGPT + ViT + Stable Diffusion:
- Gallery con esempi visivi
- Badges (arXiv, PyPI, CI, coverage)
- Quick start in 3 comandi
- Architecture diagram
- Citation block
- Community links

### 4.2 Documentation Structure

```
docs/
├── user-guide/
│   ├── quickstart.md
│   ├── spec_language.md
│   ├── automl_guide.md
│   └── multi_modal.md
├── reference/
│   ├── architecture.md
│   ├── api.md
│   └── cli.md
├── research/
│   ├── paper.pdf
│   └── experiments.md
├── tutorials/
│   ├── 01_first_synthesis.md
│   ├── 02_custom_spec.md
│   └── 03_architecture_search.md
└── development/
    ├── contributing.md
    └── testing_strategy.md
```

### 4.3 Visual Assets

- `assets/logo.png` - MetaGen logo
- `assets/architecture_diagram.png` - System architecture
- `assets/search_convergence.png` - Example search results
- `assets/demo.gif` - Animated CLI demo

### 4.4 Community Files

- `CITATION.cff` - Citation File Format
- `SECURITY.md` - Security policy
- `.github/workflows/ci.yml` - GitHub Actions
- `.github/ISSUE_TEMPLATE/` - Issue templates
- `.github/PULL_REQUEST_TEMPLATE.md` - PR template

---

## PARTE 5: Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

| Task | Descrizione |
|------|-------------|
| Modality Handlers | Creare `src/metagen/synth/modalities/` structure |
| Text Handler | Implementare `TextModalityHandler` (wrap existing logic) |
| Image Handler | Implementare `ImageModalityHandler` con ViT support |
| Search Engine Base | Creare `src/metagen/automl/search_engine.py` |
| CLI AutoML | Aggiungere `metagen automl` command |

**Deliverable**: `metagen automl spec.yaml` genera multiple candidate architectures

### Phase 2: Prototype Training (Week 3)

| Task | Descrizione |
|------|-------------|
| Prototype Trainer | `src/metagen/automl/prototype_trainer.py` |
| Template Modifications | `train.py.j2`: add `--prototype-mode` flag |
| Integration | search_engine usa prototype_trainer per evaluation |

**Deliverable**: AutoML valida architetture via training reale

### Phase 3: Refinement & Meta-Learning (Week 4)

| Task | Descrizione |
|------|-------------|
| Architecture Refiner | `src/metagen/automl/refiner.py` |
| Meta-Learner | `src/metagen/automl/meta_learner.py` + SQLite database |
| Multi-Objective | `src/metagen/automl/objectives.py` - Pareto front |

**Deliverable**: MetaGen impara da runs precedenti e itera intelligentemente

### Phase 4: Multi-Modal Completeness (Week 5-6)

| Task | Descrizione |
|------|-------------|
| Audio Modality | `src/metagen/synth/modalities/audio.py` |
| Video Modality | `src/metagen/synth/modalities/video.py` |
| Multi-Modal Fusion | Cross-attention fragments, CLIP-like contrastive learning |

**Deliverable**: MetaGen supporta tutte le modalità con architetture reali

### Phase 5: Paper & Experiments (Week 7-8)

| Task | Descrizione |
|------|-------------|
| Experimental Suite | 8 task specs, Run AutoML su tutti task |
| Baselines | Manual architectures, Random search, Single-objective NAS |
| Paper Generation | Modificare `paper_gen.py`, Figure generation |
| Reproducibility | Scripts per replicare esperimenti |

**Deliverable**: Paper completo con esperimenti rigorosi

### Phase 6: Presentation & Documentation (Week 9-10)

| Task | Descrizione |
|------|-------------|
| README Transformation | Nuovo README con gallery, badges, citation |
| Documentation | User guide, API reference, Tutorials |
| Visual Assets | Logo, Architecture diagrams |
| Community Setup | CI/CD, GitHub templates |

**Deliverable**: Repository presentabile a livello research professionale

### Phase 7: Polish & Release (Week 11-12)

| Task | Descrizione |
|------|-------------|
| Testing | Coverage >80%, Integration tests |
| Performance | Profile synthesis, Cache optimization |
| Examples & Gallery | 12+ example specs, Colab notebooks |
| Release | PyPI package, Versioning, Changelog |

**Deliverable**: MetaGen 1.0 release-ready

---

## Metodologia di Sviluppo

### Principi Fondamentali

1. **Piccoli step incrementali**: Ogni feature viene sviluppata in modo atomico e testabile
2. **Un branch per ogni step**: Ogni incremento ha il suo branch dedicato su GitHub
3. **Codice completamente commentato**: Ogni funzione, classe e modulo deve avere docstring e commenti inline
4. **Esempi abbondanti**: Ogni nuova funzionalità include almeno 2-3 esempi di utilizzo

### Branch Strategy

```
main
  └── feature/phase1-modality-handlers
        ├── feature/phase1.1-base-handler-class
        ├── feature/phase1.2-text-handler-refactor
        ├── feature/phase1.3-image-handler-vit
        └── feature/phase1.4-image-handler-diffusion
  └── feature/phase2-automl-search
        ├── feature/phase2.1-search-engine-base
        ├── feature/phase2.2-evolution-strategy
        ├── feature/phase2.3-multi-objective
        └── feature/phase2.4-prototype-trainer
  └── feature/phase3-meta-learning
        ├── feature/phase3.1-history-database
        ├── feature/phase3.2-meta-learner
        └── feature/phase3.3-transfer-learning
  └── feature/phase4-paper-scientific
        ├── feature/phase4.1-paper-structure
        ├── feature/phase4.2-experiments
        └── feature/phase4.3-figures-generation
  └── feature/phase5-github-presentation
        ├── feature/phase5.1-readme-transform
        ├── feature/phase5.2-documentation
        └── feature/phase5.3-examples-gallery
```

### Workflow per Ogni Step

```bash
# 1. Creare branch dal parent
git checkout feature/phase1-modality-handlers
git checkout -b feature/phase1.1-base-handler-class

# 2. Sviluppare con commit frequenti e descrittivi
git commit -m "feat(modalities): add ModalityHandler ABC with docstrings"
git commit -m "feat(modalities): add ModalityComponents dataclass"
git commit -m "docs(modalities): add usage examples in docstrings"
git commit -m "test(modalities): add unit tests for base handler"

# 3. Push e PR
git push -u origin feature/phase1.1-base-handler-class
gh pr create --base feature/phase1-modality-handlers

# 4. Dopo review e merge, continuare con prossimo step
```

### Requisiti per Ogni Step

Ogni step incrementale DEVE includere:

1. **Codice con documentazione completa**
   - Docstring per ogni modulo, classe, funzione
   - Commenti inline per logica non ovvia
   - Type hints su tutti i parametri e return values

2. **Test unitari**
   - Coverage minimo 80% per nuovo codice
   - Test per casi normali e edge cases
   - Test di determinismo (stesso seed = stesso output)

3. **Esempi di utilizzo**
   - Almeno 2 esempi nel docstring di ogni funzione pubblica
   - Esempio end-to-end in `examples/` per ogni feature
   - Notebook Jupyter se appropriato

4. **Aggiornamento documentazione**
   - Update CHANGELOG.md
   - Update relevant docs/ files
   - Update README se necessario

### Struttura Esempi

```
examples/
├── specs/                          # Spec YAML examples
│   ├── text_llm_8b.yaml           # Existing
│   ├── image_vit_base.yaml        # New: ViT example
│   ├── image_diffusion_small.yaml # New: Diffusion example
│   └── automl_search_demo.yaml    # New: AutoML example
├── notebooks/                      # Jupyter notebooks
│   ├── 01_quickstart.ipynb        # Basic synthesis
│   ├── 02_automl_search.ipynb     # Architecture search
│   ├── 03_multi_modal.ipynb       # Multi-modal models
│   └── 04_custom_handler.ipynb    # Extending MetaGen
├── scripts/                        # Standalone scripts
│   ├── run_automl_search.py       # AutoML CLI example
│   ├── compare_architectures.py   # Compare multiple specs
│   └── visualize_search.py        # Plot search results
└── outputs/                        # Example outputs
    ├── shakespeare/               # Existing
    ├── vit_imagenet/             # ViT synthesis output
    └── automl_results/           # Search results example
```

---

## Critical Files Summary

### AutoML Core (da creare)

1. `src/metagen/automl/search_engine.py` - Architecture search
2. `src/metagen/automl/prototype_trainer.py` - Training validation
3. `src/metagen/automl/refiner.py` - Iterative refinement
4. `src/metagen/automl/meta_learner.py` - Transfer learning
5. `src/metagen/automl/objectives.py` - Multi-objective optimization

### Multi-Modal (da creare)

6. `src/metagen/synth/modalities/base.py` - Handler abstraction
7. `src/metagen/synth/modalities/image.py` - **PRIORITY** image support
8. `src/metagen/synth/modalities/audio.py` - Audio support
9. `src/metagen/synth/modalities/video.py` - Video support
10. `src/metagen/templates/fragments/` - **35+ fragment files**

### Paper & Docs (da creare)

11. `src/metagen/synth/paper_figures.py` - Publication figures
12. `src/metagen/synth/supplementary_gen.py` - Supplementary materials
13. `docs/research/paper.tex` - Enhanced paper source
14. `docs/` - **Complete documentation tree** (20+ files)

### File da modificare

15. `src/metagen/synth/architecture.py` - Modality-aware dimensions, FLOPs estimation
16. `src/metagen/synth/codegen.py` - Fragment composition system
17. `src/metagen/synth/paper_gen.py` - Expanded paper generation
18. `src/metagen/cli.py` - Add `automl` command
19. `src/metagen/templates/train.py.j2` - Prototype mode, metrics output
20. `README.md` - Complete transformation

---

## Success Criteria

### AutoML Functionality

- [ ] `metagen automl` command works end-to-end
- [ ] Architecture search converges in <30 minutes per task
- [ ] Prototype training validates top candidates
- [ ] Meta-learning improves search over multiple runs
- [ ] Multi-objective optimization produces Pareto fronts

### Multi-Modal Support

- [ ] Text modality: full support (existing)
- [ ] Image modality: ViT, CNN, U-Net diffusion working
- [ ] Audio modality: spectrogram + transformer working
- [ ] Video modality: temporal encoding working
- [ ] Multi-modal: CLIP-like models working

### Paper Quality

- [ ] 8+ page paper with all sections
- [ ] Rigorous experiments with baselines and ablations
- [ ] Publication-quality figures
- [ ] Supplementary materials
- [ ] Reproducibility package

### GitHub Presentation

- [ ] README rivals top research repositories
- [ ] Visual gallery with examples
- [ ] Complete documentation
- [ ] 80%+ test coverage
- [ ] CI/CD automated
- [ ] Community files complete

---

## Risk Mitigation

1. **Scope creep**: Focus su Phase 1-3 prima di multi-modal complete
2. **Training instability**: Prototype training usa conservative hyperparams e early stopping
3. **Search inefficiency**: Start con small search budgets (N=10), scale dopo validation
4. **Paper rigor**: Collect comprehensive metrics da subito, iterate on paper structure
5. **Backward compatibility**: Text modality deve continuare a funzionare identicamente

---

## Vision Statement Finale

Al completamento di questo piano, MetaGen sarà:

- **Un sistema AutoML completo** che genera, addestra e valuta modelli AI generativi attraverso ricerca architetturale automatica

- **Un contributo scientifico significativo** con paper pubblicabile che critica e propone soluzioni per pratiche di sviluppo AI moderne

- **Un progetto GitHub di ricerca di alto livello** con presentazione professionale, documentazione completa e community engagement

- **Una vision per il futuro dell'AI development**: systematic, reproducible, specification-driven synthesis come paradigma per la field

**Meta-goal**: MetaGen diventa l'esempio di riferimento per "come fare ricerca AI riproducibile" e uno strumento effettivamente usato dalla community per prototipazione rapida.
