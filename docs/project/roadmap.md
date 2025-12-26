# MetaGen Roadmap - Trainable Models

**Navigation**: [📚 Docs Home](../README.md) | [🏗️ Architecture v2](../development/architecture_v2.md) | [📋 Trainable Models Spec](../development/trainable_models_spec.md) | [🧪 Testing Strategy](../development/testing_strategy.md)

---

**Obiettivo**: Trasformare MetaGen in un sistema ibrido che genera modelli realmente trainabili mantenendo gli elementi satirici.

**Stato**: 🚧 In Sviluppo

---

## Status Overview

| WU | Nome | Stato | Completamento | Note |
|----|------|-------|---------------|------|
| WU-1 | Documentation | ✅ | 100% | COMPLETED |
| WU-2 | Blueprint Sync | ⏳ | 0% | Waiting |
| WU-3 | Templates | ⏳ | 0% | Waiting |
| WU-4 | Model Complete | ⏳ | 0% | Waiting |
| WU-5 | Data Loaders | ⏳ | 0% | Waiting |
| WU-6 | Training Loop | ⏳ | 0% | Waiting |
| WU-7 | Real Evaluation | ⏳ | 0% | Waiting |
| WU-8 | Mode System & CLI | ⏳ | 0% | Waiting |

**Legenda**:
- ✅ Completato
- 🚧 In corso
- ⏳ In attesa
- ❌ Bloccato
- ⚠️ Problemi

---

## Detailed Progress

### WU-1: Documentation & Infrastructure ✅ COMPLETED

**Obiettivo**: Creare documentazione e infrastruttura per il progetto

**Task completati**:

- [x] Created [roadmap.md](../project/roadmap.md) (this file)
- [x] Created [trainable_models_spec.md](../development/trainable_models_spec.md)
- [x] Created [architecture_v2.md](../development/architecture_v2.md)
- [x] Created [testing_strategy.md](../development/testing_strategy.md)
- [x] Added pytest markers to `pyproject.toml`
- [x] Organized docs folder structure

**Data completamento**: 2025-12-23

**Note**: Fondamenta documentali stabilite

---

### WU-2: Blueprint Sync ⏳ WAITING

**Obiettivo**: Sincronizzare dimensioni tra blueprint e codice generato

**Task**:
- [ ] Create `BlueprintState` dataclass in `architecture.py`
- [ ] Extend `_choose_dims()` to compute all parameters
- [ ] Modify `generate_blueprint()` to return `BlueprintState`
- [ ] Update `engine.py` to pass blueprint to codegen
- [ ] Remove random choices in `codegen.py`
- [ ] Add tests for dimension consistency

**File da modificare**:
- `src/metagen/synth/architecture.py`
- `src/metagen/synth/codegen.py`
- `src/metagen/synth/engine.py`
- `tests/test_architecture.py`
- `tests/test_codegen.py`

**Blockers**: Nessuno

---

### WU-3: Templates ⏳ WAITING

**Obiettivo**: Sistema di templating Jinja2 funzionante

**Task**:
- [ ] Create `src/metagen/synth/templates/` folder
- [ ] Implement `template_loader.py`
- [ ] Create first template: `transformer_causal.py.jinja`
- [ ] Update `codegen.py` to use templates
- [ ] Add template rendering tests

**File da creare**:
- `src/metagen/synth/templates/template_loader.py`
- `src/metagen/synth/templates/transformer_causal.py.jinja`
- `tests/test_templates.py`

**Dipendenze**: WU-2 (requires BlueprintState)

---

### WU-4: Model Complete ⏳ WAITING

**Obiettivo**: Modello transformer causale completo

**Task**:
- [ ] Extend template with token embeddings
- [ ] Add position embeddings
- [ ] Implement causal transformer blocks
- [ ] Add LM head (output projection)
- [ ] Test import and instantiation
- [ ] Test forward pass shapes

**File da modificare**:
- `src/metagen/synth/templates/transformer_causal.py.jinja`
- `src/metagen/synth/architecture.py` (add vocab_size, max_seq_len)
- `tests/test_trainable_text.py` (new)

**Dipendenze**: WU-3 (requires templates)

---

### WU-5: Data Loaders ⏳ WAITING

**Obiettivo**: Data loader con tensori PyTorch validi

**Task**:
- [ ] Create `src/metagen/data/mock_data.py`
- [ ] Implement `generate_mock_text()` returning tensors
- [ ] Create dataloader template
- [ ] Generate functional `data.py`
- [ ] Test data loader iteration

**File da creare**:
- `src/metagen/data/mock_data.py`
- `src/metagen/synth/templates/dataloader_text.py.jinja`
- `tests/test_dataloaders.py`

**Dipendenze**: WU-4 (requires model structure)

---

### WU-6: Training Loop ⏳ WAITING

**Obiettivo**: Training loop funzionante end-to-end

**Task**:
- [ ] Create train loop template
- [ ] Implement optimizer setup
- [ ] Add loss computation (cross-entropy)
- [ ] Implement backward pass
- [ ] Add gradient clipping
- [ ] Test 1-step training

**File da creare**:
- `src/metagen/synth/templates/train_loop.py.jinja`

**File da modificare**:
- `tests/test_trainable_text.py` (add training test)

**Dipendenze**: WU-5 (requires data loaders)

---

### WU-7: Real Evaluation ⏳ WAITING

**Obiettivo**: Calcolo perplexity reale

**Task**:
- [ ] Create eval template
- [ ] Implement perplexity computation
- [ ] Add train/val split to data loaders
- [ ] Generate `eval.py` with real metrics
- [ ] Keep satirical metrics for paper
- [ ] Test evaluation

**File da creare**:
- `src/metagen/synth/templates/eval_metrics.py.jinja`

**File da modificare**:
- `src/metagen/synth/templates/dataloader_text.py.jinja`
- `tests/test_trainable_text.py` (add eval test)

**Dipendenze**: WU-6 (requires training infrastructure)

---

### WU-8: Mode System & CLI ⏳ WAITING

**Obiettivo**: Sistema con modalità mock/trainable

**Task**:
- [ ] Add `synth_mode` parameter to `engine.synthesize()`
- [ ] Implement mode="mock" vs mode="trainable" logic
- [ ] Update CLI with `--mode` flag
- [ ] Add `Training.data_mode` to schema
- [ ] Test both modes
- [ ] Verify backward compatibility

**File da modificare**:
- `src/metagen/synth/engine.py`
- `src/metagen/specs/schema.py`
- `src/metagen/cli.py`
- `tests/test_cli.py`

**Dipendenze**: WU-7 (requires all components)

---

## Milestones

### 🎯 Milestone 1: Consistency (after WU-2)
**Target**: Dimensioni consistenti tra blueprint e codice
**Status**: ⏳ Not Started
**Deliverable**: `architecture.yaml` e `model.py` con stesse dimensioni

### 🎯 Milestone 2: Importability (after WU-4)
**Target**: Modello importabile e instanziabile
**Status**: ⏳ Not Started
**Deliverable**: `from model import MetaGenModel` funziona

### 🎯 Milestone 3: Trainability (after WU-6)
**Target**: Modello trainabile end-to-end
**Status**: ⏳ Not Started
**Deliverable**: 1-step training completa senza errori

### 🎯 Milestone 4: Hybrid System (after WU-8)
**Target**: Sistema completo con mock e trainable
**Status**: ⏳ Not Started
**Deliverable**: CLI con `--mode` flag funzionante

---

## Timeline Estimata

| Periodo | WU | Obiettivo |
|---------|----|-----------|
| Week 1 | WU-1, WU-2 | Documentation + Blueprint Sync |
| Week 2 | WU-3, WU-4 | Templates + Complete Model |
| Week 3 | WU-5, WU-6 | Data Loaders + Training |
| Week 4 | WU-7, WU-8 | Evaluation + Mode System |

**Note**: Timeline puramente indicativa, dipende da sessioni di lavoro

---

## Rischi e Mitigazioni

| Rischio | Probabilità | Impatto | Mitigazione |
|---------|-------------|---------|-------------|
| Test esistenti si rompono | Media | Alto | Backward compatibility via mode="mock" |
| Template rendering complesso | Bassa | Medio | Start simple, iterate |
| Training non converge | Bassa | Basso | Test solo 1-step, non convergenza |
| PyTorch dependencies | Bassa | Medio | Already in pyproject.toml |

---

## Success Criteria

Per considerare il progetto completo, devono essere soddisfatti:

- [x] Tutti i documenti creati ✅
- [ ] Tutte le 8 WU complete
- [ ] `make test` passa al 100%
- [ ] `make lint` senza errori
- [ ] Text LLM trainabile funzionante
- [ ] Mode system mock/trainable operativo
- [ ] Backward compatibility mantenuta
- [ ] Documentation aggiornata

---

## Future Work (Post WU-8)

Dopo il completamento delle 8 work units, considereremo:

1. **Real Datasets**: Integrazione dataset reali (WikiText, etc.)
2. **Image Diffusion**: Modelli di diffusione per immagini
3. **Multimodal**: Modelli image-text
4. **Audio/Video**: Modalità temporali
5. **Edge Models**: Ottimizzazioni per device consumer
6. **Checkpoint Management**: Save/load trained models
7. **Distributed Training**: Multi-GPU support
8. **Hyperparameter Tuning**: Auto-tuning via Optuna

---

## Notes

- **Codice funzionante**: Ogni WU produce codice eseguibile e testato
- **Incrementalità**: Ogni step aggiunge valore utilizzabile
- **Testing**: Test-driven approach, ogni modifica ha test
- **Documentation**: Aggiornare questo file dopo ogni WU

---

**Ultimo aggiornamento**: 2025-12-23
**Prossima WU**: WU-2 (Blueprint Sync)
**Owner**: Mauro
**Repo**: `/Users/mauro/Developer/metagen`
