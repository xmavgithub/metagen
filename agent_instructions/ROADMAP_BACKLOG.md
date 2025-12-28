# MetaGen Roadmap Operativa (Backlog)

Questa roadmap e un backlog operativo derivato dalla gap analysis. Ogni epic ha deliverable, dipendenze e criteri di accettazione. Le fasi sono pensate per rilasci incrementali utili.

## Legenda

- **Livelli**: Lite (default) e Full (opzionale).
- **Focus**: Vision + Text prima di tutto.
- **Dataset**: HF + locali, progressivi.

---

# Fase 0 - Allineamento e Stabilita (1-2 settimane)

## Epic A - Dataset Inspector

**Obiettivo**: ispezionare dataset (HF e locali) per schema e sample.

- A1: CLI `metagen inspect-dataset <spec|dataset>`
  - Output: schema campi, tipi, sample rows.
  - Supporto iniziale: HF + CSV/Parquet + JSONL.
- A2: Validator campi per task mode (classification, detection, ranking).
- A3: Mapping automatico campi comuni (text/label/image/boxes).

**Dipendenze**: nessuna.

**Criteri di accettazione**:
- Il comando stampa schema e almeno un sample per HF e locale.
- Errori chiari su campi mancanti.

## Epic B - Docs e Demo

**Obiettivo**: allineamento docs e demo non generativi.

- B1: Docs aggiornate (task types reali, esempi non generativi).
- B2: `metagen demo` include almeno 2 spec non generative (vision + time series).

**Dipendenze**: nessuna.

---

# Fase 1 - Text Utile (2-3 settimane)

## Epic C - Tokenizer HF (Lite/Full)

**Obiettivo**: tokenizzazione reale per text task.

- C1: Integrazione HF tokenizer (`--tokenizer` / campo spec).
- C2: Cache tokenizer + compatibilita con `max_seq_len`.
- C3: Fallback a tokenizer semplice se HF non disponibile.

**Dipendenze**: Epic A (dataset inspector) per mapping campi.

## Epic D - Text CLI + Eval

**Obiettivo**: inferenza utente e metriche minime per text.

- D1: `infer_text.py` generato (top-k + label mapping).
- D2: Eval rapido con accuracy + f1_macro.
- D3: Ranking pairwise con MRR + nDCG@10.

**Dipendenze**: Epic C.

---

# Fase 2 - Vision Utile (4-6 settimane)

## Epic E - Vision Lite (default)

**Obiettivo**: pipeline minima ma utile su immagini reali.

- E1: ResNet-lite encoder + preprocessing (classification).
- E2: YOLO-lite detection (box + NMS).
- E3: UNet-lite segmentation (mask + IoU).
- E4: `infer_image.py` con overlay box/mask.

**Dipendenze**: Epic A (dataset mapping), Epic B (docs demo aggiornate).

## Epic F - Vision Full (opzionale)

**Obiettivo**: pipeline piu fedele ai paper.

- F1: ViT patch embedding.
- F2: DETR-lite encoder/decoder.
- F3: UNet full (deep).

**Dipendenze**: Epic E.

---

# Fase 3 - Tabular/Time Series/Graph (2-4 settimane)

## Epic G - Tabular

- G1: Preprocessing tabular (normalize, missing values).
- G2: `infer_tabular.py` (CSV/Parquet).
- G3: Eval accuracy/f1 o RMSE/MAE.

## Epic H - Time Series

- H1: Windowing robusto (lookback/horizon).
- H2: `infer_timeseries.py` (CSV/NPY).
- H3: Eval RMSE + MAE.

## Epic I - Graph

- I1: Supporto edge list locale + node features.
- I2: `infer_graph.py`.
- I3: Eval ROC AUC + PR AUC.

---

# Fase 4 - Audio/Video/Multimodale (4-8 settimane)

## Epic J - Audio

- J1: Mel spectrogram preprocessing.
- J2: Backbone CNN o AST-lite.
- J3: `infer_audio.py` + eval.

## Epic K - Video

- K1: Frame sampling.
- K2: Backbone 3D CNN o ViT semplice.
- K3: `infer_video.py` + eval.

## Epic L - Multimodale

- L1: CLIP-lite encoder.
- L2: Loss contrastiva.
- L3: `infer_multimodal.py`.

---

# Metriche Minime (confermate)

- Classification: accuracy + f1_macro.
- Detection: mAP50 + mAP.
- Segmentation: mean IoU.
- Text ranking: MRR + nDCG@10.
- Time series: RMSE + MAE.
- Graph link prediction: ROC AUC + PR AUC.
- RL: episodic return.

---

# Definition of Done (Utile per Esperimenti)

Per ogni task in examples/specs:
1) `metagen synth` + training su dataset reale
2) inferenza su input utente
3) metriche minime valide e output interpretabili

---

Se vuoi, posso convertire questo backlog in ticket con stime e priorita (P0/P1/P2).
