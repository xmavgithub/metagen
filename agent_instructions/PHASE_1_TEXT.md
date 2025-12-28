# Fase 1 - Text Utile (Istruzioni Agent)

## Scopo

Rendere i task text realmente utili per esperimenti: tokenizer reale, training coerente, inferenza utente e metriche minime.

Prima di iniziare: leggi `agent_instructions/CONTEXT.md`.

## Deliverable

- Tokenizer HF opzionale in training/inferenza.
- CLI inferenza text (`infer_text.py`) con top-k e mapping label.
- Eval rapido: accuracy + f1_macro.

## Versioning specifico

- Branch: `feature/phase-1-text`.
- 2 commit max: core + tests/docs.
- Messaggi: `phase1: ...`.

## Dove intervenire (file principali)

- `src/metagen/templates/data.py.j2`: tokenizzazione, mapping campi text/label, dataset HF.
- `src/metagen/templates/train.py.j2`: flag `--tokenizer`, uso tokenizer nel training.
- `src/metagen/templates/infer.py.j2`: inferenza text con top-k e label mapping.
- `src/metagen/templates/fragments/data/text_datasets.py.j2`: sample dataset text.
- `src/metagen/synth/tasks/classification.py`: se servono metriche o head specifici.
- `tests/test_sample_data.py`: test smoke training/inferenza text.

## Task Operativi

1) **Tokenizer HF**
- Aggiungere supporto `--tokenizer` in `train.py.j2` e `infer_text.py.j2`.
- Fallback a tokenizer semplice se `transformers` non installato.
- Cache tokenizer e allineamento `max_seq_len`.

2) **Data pipeline text**
- Aggiornare `data.py.j2` per usare tokenizer se disponibile.
- Usare campi dataset (text_field/label_field) quando presenti.

3) **Inferenza text**
- Script generato con `--text`, `--top-k`, `--checkpoint`, `--device`.
- Mapping label per dataset noti (es. ag_news).

4) **Eval rapido**
- Aggiungere funzione di eval per classification text (accuracy + f1_macro).
- Se `torchmetrics` disponibile, usarlo; altrimenti implementare semplice.

## Test richiesti

- Aggiungere test minimo su generation `infer_text.py` (es. shape/top-k).
- Aggiungere test su tokenizzazione deterministica.

## Criteri di accettazione

- Text classification: input utente produce predizioni plausibili.
- Tokenizer HF usabile con flag e fallback stabile.
- Eval rapido funziona su sample dataset.

## Non fare

- Non modificare pipeline vision/audio/video.
- Non introdurre dipendenze non necessarie.
