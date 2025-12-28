# Fase 4 - Audio / Video / Multimodale (Istruzioni Agent)

## Scopo

Abilitare audio, video e multimodale con preprocessing e encoder base per esperimenti reali.

Prima di iniziare: leggi `agent_instructions/CONTEXT.md`.

## Deliverable

- Audio: mel spectrogram + backbone CNN, inferenza CLI.
- Video: frame sampling + backbone semplice, inferenza CLI.
- Multimodale: CLIP-lite con loss contrastiva.
- Metriche minime per classification.

## Versioning specifico

- Branch: `feature/phase-4-av-mm`.
- 1-2 commit: core + tests/docs.

## Dove intervenire (file principali)

- `src/metagen/synth/modalities/audio.py`, `src/metagen/synth/modalities/video.py`, `src/metagen/synth/modalities/multimodal.py`: scelta encoder/data fragments.
- `src/metagen/templates/fragments/encoders/`:
  - `audio_encoder.py.j2`
  - `video_cnn.py.j2`
  - `video_transformer.py.j2`
- `src/metagen/templates/fragments/data/`:
  - `audio_datasets.py.j2`
  - `video_datasets.py.j2`
  - `multimodal_datasets.py.j2`
- `src/metagen/templates/fragments/heads/`:
  - `classification_head.py.j2`
  - `embedding_head.py.j2` (multimodale)
- `src/metagen/templates/fragments/losses/contrastive.py.j2`: CLIP-lite.
- `src/metagen/templates/train.py.j2` / `infer.py.j2`: script inferenza audio/video/multimodale.
- `examples/specs/audio/`, `examples/specs/video/`, `examples/specs/multimodal/`.

## Task Operativi

1) **Audio**
- Preprocessing mel spectrogram (torchaudio).
- Backbone CNN semplice.
- `infer_audio.py` con file input.

2) **Video**
- Frame sampling da video file.
- Backbone 3D CNN o ViT semplice.
- `infer_video.py`.

3) **Multimodale**
- Encoder testo + encoder immagine.
- Loss contrastiva (CLIP-lite).
- Inferenza per similarita immagine-testo.

## Test richiesti

- Smoke test: synth + train con dataset HF ridotto per audio/video.
- Inferenza su input locale (audio/video).

## Criteri di accettazione

- Audio/video: training e inferenza funzionanti su input reale.
- Multimodale: similarita consistente su sample.

## Non fare

- Non toccare le fasi precedenti se non necessario.
