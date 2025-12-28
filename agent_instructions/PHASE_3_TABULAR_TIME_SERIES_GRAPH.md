# Fase 3 - Tabular / Time Series / Graph (Istruzioni Agent)

## Scopo

Rendere tabular, time series e graph usabili end-to-end con inferenza utente e metriche minime.

Prima di iniziare: leggi `agent_instructions/CONTEXT.md`.

## Deliverable

- `infer_tabular.py` (CSV/Parquet).
- `infer_timeseries.py` (CSV/NPY con lookback/horizon).
- `infer_graph.py` (edge list + node features).
- Metriche minime: accuracy/f1, RMSE/MAE, ROC AUC/PR AUC.

## Versioning specifico

- Branch: `feature/phase-3-tabular-ts-graph`.
- 1-2 commit: core + tests/docs.

## Dove intervenire (file principali)

- `src/metagen/templates/data.py.j2`: mapping dataset tabular/time_series/graph e loader.
- `src/metagen/templates/fragments/data/time_series_datasets.py.j2`: windowing, split, dataset HF.
- `src/metagen/templates/fragments/data/graph_datasets.py.j2`: edge list, node features, split.
- `src/metagen/templates/fragments/data/`: aggiungere dataset tabular (es. `tabular_classification_dataset.py.j2`, `tabular_regression_dataset.py.j2`) se mancanti.
- `src/metagen/templates/fragments/encoders/rnn_encoder.py.j2`: baseline time series.
- `src/metagen/templates/fragments/heads/`:
  - `time_series_head.py.j2`
  - `graph_classification_head.py.j2`
  - `node_classification_head.py.j2`
  - `link_prediction_head.py.j2`
- `src/metagen/synth/tasks/`: `classification.py`, `regression.py`, `time_series.py`, `graph.py`.
- `src/metagen/templates/train.py.j2` / `infer.py.j2`: script inferenza tabular/time series/graph.
- `examples/specs/tabular/`, `examples/specs/time_series/`, `examples/specs/graph/`.

## Task Operativi

1) **Tabular**
- Preprocessing: normalize, handle missing values.
- Supporto CSV/Parquet in `data.py.j2`.
- `infer_tabular.py` con input file.

2) **Time Series**
- Windowing robusto (lookback/horizon) + padding.
- `infer_timeseries.py` con input file e output forecast.

3) **Graph**
- Supporto edge list locale + node features.
- Link prediction: output e metriche minime.

## Test richiesti

- Test smoke: synth + train per tabular e time series.
- Test inferenza per tabular/time series con file locale.

## Criteri di accettazione

- Tabular/time series: pipeline completa con input utente.
- Graph: link prediction con metriche minime.

## Non fare

- Non modificare pipeline vision/text/audio.
