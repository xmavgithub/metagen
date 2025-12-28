# Contesto Codebase MetaGen (per tutte le fasi)

Questo file contiene la mappa rapida della codebase e le regole generali valide per tutte le fasi.

## Regole generali

- Python 3.11+, ruff line length 100.
- Import order: stdlib, third-party, first-party.
- Usare `from __future__ import annotations` nei file Python nuovi.
- Determinismo: usare seed espliciti per dati sintetici o generatori.
- Preferire modifiche ai template (non agli output generati).
- Non modificare `outputs/` o `paper/` nel repo.
- Test minimi: `ruff check src/metagen tests`.
- Eseguire test aggiuntivi indicati nella fase.
- Non fare rebase o amend se non richiesto.

## Struttura principale

- `src/metagen/cli.py`:
  - Comandi CLI: `synth`, `demo`, `paper`, `validate`, `schema`, `automl`.
  - Qui si aggiungono nuovi comandi CLI.

- `src/metagen/specs/`:
  - `schema.py`: schema e VALID_TASK_TYPES.
  - `loader.py`: carica e valida le spec.

- `src/metagen/synth/`:
  - `engine.py`: orchestrazione synth.
  - `architecture.py`: blueprint e stime dimensioni.
  - `codegen.py`: rendering template Jinja.
  - `tasks/`: task handlers (classification, detection, etc.).
  - `modalities/`: modality handlers (text, image, audio, video, multimodal).
  - `paper_gen.py`: generazione paper.

- `src/metagen/templates/`:
  - `model.py.j2`, `train.py.j2`, `data.py.j2`, `eval.py.j2`, `infer.py.j2`.
  - `fragments/`: encoder, head, data loader, loss, embeddings.

- `examples/specs/`:
  - Spec esempio per task e modalita.

- `docs/`:
  - Guide, CLI reference, spec language.

## Template e flusso di generazione

1) `engine.synthesize` -> blueprint + task components.
2) `codegen.generate_code` rende:
   - `model.py.j2`, `train.py.j2`, `data.py.j2`, `eval.py.j2`, `infer.py.j2`.
3) `model.py.j2` include frammenti in base a `task_components.template_fragments`.

## Punti chiave dove intervenire

### Dati e dataset

- `src/metagen/templates/data.py.j2`:
  - Catalogo HF (REMOTE_DATASETS), dataset mapping, `load_data()`.
  - Qui si aggiungono dataset e mapping campi.

- `src/metagen/templates/fragments/data/*`:
  - Dataset specifici per task (detection, segmentation, time_series, ecc.).

### Modelli / Encoder / Head

- `src/metagen/templates/fragments/encoders/*`:
  - Encoder per modalita (vit, resnet, audio, video, unet).

- `src/metagen/templates/fragments/heads/*`:
  - Head per task (classification, detection, segmentation, ranking, ecc.).

- `src/metagen/templates/model.py.j2`:
  - Assemblaggio del modello con task head.

### Training / Eval / Inferenza

- `src/metagen/templates/train.py.j2`:
  - CLI e training loop, loss handling.

- `src/metagen/templates/eval.py.j2`:
  - Eval generico (da estendere per metriche task).

- `src/metagen/templates/infer.py.j2`:
  - Inferenza generica (spesso da specializzare per task).

### Task Handlers

- `src/metagen/synth/tasks/`:
  - Decidono head type, loss, template fragments.
  - Se aggiungi un task, registra con `@register_task`.

### Modality Handlers

- `src/metagen/synth/modalities/`:
  - Selezionano encoder/embeddings/data fragments per modalita.
  - Image handler: `src/metagen/synth/modalities/image.py`.

## Test utili

- `tests/test_schema.py`: validazione schema.
- `tests/test_sample_data.py`: sample data e training script.

## File di output

- `outputs/` e `paper/` non vanno modificati in repo.

## Consigli pratici

- Aggiornare docs quando si toccano task/modality.
