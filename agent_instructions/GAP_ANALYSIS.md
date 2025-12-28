# Analisi Gap MetaGen (Utilita Pratica) - Roadmap Dettagliata

Scopo: rendere MetaGen utile per esperimenti reali (non solo scaffolding), restando spec-to-blueprint.

## Input e Vincoli (confermati)

- Priorita: Vision, Text (poi gli altri).
- Qualita attesa: risultati ragionevoli.
- Hardware e tempo: variabili; l'utente decide in base alle risorse (CPU/MPS/CUDA).
- Dipendenze: tutte quelle necessarie.
- Dataset: sia Hugging Face sia locali/privati.
- Focus: training pipeline completa + inferenza utente (CLI).
- Filosofia: resta spec-to-blueprint, ma codice realmente usabile per esperimenti.
- Livelli: due livelli Lite/Full.

## Decisioni Chiave

- **Due livelli**: Lite (default) e Full (opzionale).
- **Vision Lite**: ResNet (stabile e veloce).
- **Vision Full**: ViT (piu fedele ai paper).
- **Detection Lite**: YOLO-lite (veloce, pragmatico).
- **Detection Full**: DETR-lite (piu coerente con output transformer).
- **Metriche minime**: definite sotto e confermate.
- **Dataset locali**: supporto progressivo, ma copertura completa a regime.

## Stato Attuale (Sintesi)

- Sintesi deterministica e affidabile.
- Task handlers completi per molte tipologie.
- Catalogo dataset HF + sample data sintetici.
- Flussi CLI (synth/demo/paper/validate) stabili.

Limite principale: per molte modalita manca l'encoder specifico, quindi training/inferenza su input reali non sono utili.

## Gap Principali (riassunto)

- Vision: mancano encoder visivi, preprocessing e post-processing.
- Text: tokenizer reale assente.
- Ranking: data flow pairwise fragile.
- Audio/Video/Multimodale: preprocessing e encoder assenti.
- Evaluation: metriche task-specific mancano.
- Dataset locali: supporto non uniforme.

## Metriche Minime (confermate)

- Classification (vision/text/tabular): accuracy + f1_macro.
- Detection: mAP50 (quick eval) + mAP (full).
- Segmentation: mean IoU.
- Text ranking: MRR + nDCG@10.
- Time series: RMSE + MAE.
- Graph link prediction: ROC AUC + PR AUC.
- RL: episodic return (logging).

## Supporto Dataset Locali (progressivo)

Fase 0-1:
- CSV/Parquet per tabular.
- Cartelle immagini con classi (ImageFolder).
- JSONL per testo (campi `text`, `label`).

Fase 2:
- Audio folder + CSV labels.
- Video folder + CSV labels.

Fase 3:
- Graph edge list (CSV) + node features (CSV/NPY).

---

# Roadmap Dettagliata

## Visione di insieme

- **Fase 0**: Allineamento, dataset inspector, demo non generativi.
- **Fase 1**: Text utile (tokenizer reale, inferenza CLI, eval base).
- **Fase 2**: Vision utile (encoder + detection/segmentation reali).
- **Fase 3**: Tabular/Time Series/Graph utili.
- **Fase 4**: Audio/Video/Multimodale utili.

Ogni fase produce output funzionanti end-to-end per almeno una spec reale.

## Workstream trasversali

- **WS-DATA**: dataset mapping HF + locali + inspect CLI.
- **WS-MODEL**: encoder/backbone per modalita.
- **WS-EVAL**: metriche minime e quick eval.
- **WS-CLI**: inferenza utente e UX.
- **WS-DOCS**: allineamento docs/demo.

## Fase 0 - Allineamento e Stabilita (1-2 settimane)

**Obiettivi**
- Docs allineate con task implementati.
- Dataset catalog corretto e verificato.
- CLI per inspect dataset (schema + sample).
- Demo include almeno 2 spec non generative.

**Task principali**
- WS-DOCS: allineare `spec_language.md` a `VALID_TASK_TYPES`.
- WS-DATA: validatore dataset e stampa schema campione.
- WS-CLI: `metagen inspect-dataset <spec|dataset>`.
- WS-DOCS: aggiornare `metagen demo` con spec non generative.

**Deliverable**
- CLI inspect dataset.
- Demo aggiornata.
- Docs aggiornate.

**Criteri di accettazione**
- `metagen inspect-dataset` mostra schema e sample per HF e local.
- `metagen demo` include almeno 1 spec vision e 1 time series.

## Fase 1 - Text Utile (2-3 settimane)

**Obiettivi**
- Tokenizer reale per text task (HF opzionale).
- Inferenza CLI su input utente.
- Metriche base per classification/embedding/ranking.

**Task principali**
- WS-MODEL: integrazione tokenizer HF (con fallback semplice).
- WS-DATA: mapping label -> class names per dataset comuni.
- WS-CLI: `infer_text.py` generato con `--text` e `--top-k`.
- WS-EVAL: accuracy + f1_macro su sample ridotti.

**Deliverable**
- Tokenizer flag in train/infer.
- Eval rapido text.

**Criteri di accettazione**
- Text classification: input utente produce predizioni plausibili.
- Ranking: dataset pairwise funziona con ms_marco/natural_questions.

## Fase 2 - Vision Utile (4-6 settimane)

**Obiettivi**
- Encoder visivo reale e preprocessing standard.
- Detection pipeline funzionante su immagini utente.
- Segmentation con output e metriche minime.

**Task principali (Lite)**
- WS-MODEL: ResNet-lite backbone + head classification.
- WS-MODEL: YOLO-lite head con box decoder.
- WS-MODEL: UNet-lite per segmentation.
- WS-DATA: ImageFolder + HF image dataset mapping.
- WS-CLI: `infer_image.py` con overlay box/mask.

**Task principali (Full)**
- WS-MODEL: ViT + patch embedding.
- WS-MODEL: DETR-lite (encoder + decoder + post-processing).
- WS-MODEL: UNet full (piu profondo).

**Deliverable**
- Vision Lite e Full selezionabili da spec/CLI (`--profile lite|full`).
- Inference CLI su immagine utente con output visivo.

**Criteri di accettazione**
- Detection: su immagini reali, box visibili e non casuali.
- Segmentation: output mask con IoU > baseline random.

## Fase 3 - Tabular/Time Series/Graph (2-4 settimane)

**Obiettivi**
- Inferenza CLI user-friendly.
- Metriche minime e dataset mapping stabile.

**Task principali**
- WS-CLI: `infer_tabular.py` (CSV/Parquet).
- WS-CLI: `infer_timeseries.py` (CSV/NPY, lookback/horizon).
- WS-CLI: `infer_graph.py` (edge list + node features).
- WS-EVAL: RMSE/MAE, accuracy/ROC.

**Criteri di accettazione**
- Tabular e time series: training + inferenza con input locale.
- Graph: link prediction con metriche minime.

## Fase 4 - Audio/Video/Multimodale (4-8 settimane)

**Obiettivi**
- Preprocessing + encoder base.
- Demo e inferenza utente.

**Task principali**
- Audio: mel spectrogram + CNN.
- Video: frame sampling + backbone semplice.
- Multimodale: CLIP-lite con loss contrastiva.

**Criteri di accettazione**
- Audio/Video: training e inferenza su dataset HF e locale.

---

# Backlog Operativo (epics principali)

## Epic A - Dataset Inspector
- A1: `metagen inspect-dataset` (HF + locale) con schema e sample.
- A2: mapping dataset fields + default task_mode.

## Epic B - Text Realistico
- B1: HF tokenizer integration (train/infer).
- B2: label mapping e top-k output.
- B3: eval rapido (accuracy/f1).

## Epic C - Vision Lite
- C1: ResNet-lite + preprocessing immagini.
- C2: YOLO-lite detection head + decoder.
- C3: UNet-lite segmentation + IoU.
- C4: inferenza con overlay box/mask.

## Epic D - Vision Full
- D1: ViT patch embedding.
- D2: DETR-lite encoder/decoder.
- D3: UNet full.

## Epic E - CLI Inferenza
- E1: CLI text/tabular/time series/graph.
- E2: CLI vision con overlay.

## Epic F - Evaluation Framework
- F1: quick eval per task.
- F2: metriche standard per report.

---

# Definition of Done (Utile per Esperimenti)

Per ogni task in examples/specs:
1) `metagen synth` + training su dataset reale
2) inferenza su input utente
3) metriche minime valide e output interpretabili

---

Se vuoi, trasformo questa roadmap in ticketing dettagliato con stime per task e owner.
