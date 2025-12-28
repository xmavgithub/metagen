# Fase 0 - Allineamento e Stabilita (Istruzioni Agent)

## Scopo

Allineare docs, schema e demo, introdurre un dataset inspector, migliorare validazioni.

Prima di iniziare: leggi `agent_instructions/CONTEXT.md`.

## Deliverable

- Comando CLI: `metagen inspect-dataset <spec|dataset>`.
- Docs allineate ai task implementati.
- Demo con almeno 2 spec non generative (vision + time series).

## Versioning specifico

- Branch: `feature/phase-0-align`.
- 1-2 commit massimo: core + docs/test.
- Messaggi commit: `phase0: ...`.

## Dove intervenire (file principali)

- `src/metagen/cli.py`: aggiungere comando `inspect-dataset` e aggiornare `demo`.
- `src/metagen/specs/schema.py`: verificare task types e warnings.
- `src/metagen/specs/loader.py`: se serve validazione extra per inspect.
- `src/metagen/utils/io.py`: helper per file locali.
- `docs/user-guide/spec_language.md`: allineamento task/documentazione.
- `docs/reference/cli.md`: documentare `inspect-dataset`.
- `README.md`: aggiornare snippet CLI se necessario.
- `tests/test_schema.py`: test schema.
- (Nuovo) `tests/test_inspect_dataset.py`: test comando inspect.

## Task Operativi

1) **CLI dataset inspector**
- Aggiungere comando `inspect-dataset` in `src/metagen/cli.py`.
- Implementare helper (es. `src/metagen/data/inspect.py`).
- Supporto iniziale:
  - HF dataset (via `datasets.load_dataset` se disponibile).
  - Local CSV/Parquet/JSONL.
  - Directory immagini (ImageFolder style).
- Output: schema campi, tipo, sample n=3.
- Errori chiari se campi mancanti.

2) **Docs**
- Aggiornare `docs/user-guide/spec_language.md`.
- Aggiornare `docs/reference/cli.md` con `inspect-dataset`.
- Aggiornare README se necessario.

3) **Demo**
- In `metagen demo`, includere almeno:
  - 1 spec vision (classification).
  - 1 spec time series.

## Test richiesti

- `pytest tests/test_schema.py`.
- Aggiungere almeno 1 test per `inspect-dataset` (unit o snapshot).

## Criteri di accettazione

- `metagen inspect-dataset` funziona con dataset HF e CSV locale.
- Demo genera output per spec non generative.
- Docs coerenti con schema.

## Non fare

- Non cambiare behavior di synth/paper se non richiesto.
