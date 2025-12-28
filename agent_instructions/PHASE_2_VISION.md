# Fase 2 - Vision Utile (Istruzioni Agent)

## Scopo

Rendere i task vision realmente utili (classification/detection/segmentation) con encoder visivo e preprocessing.

Prima di iniziare: leggi `agent_instructions/CONTEXT.md`.

## Deliverable

- Vision Lite (default): ResNet-lite + YOLO-lite + UNet-lite.
- Vision Full (opzionale): ViT + DETR-lite + UNet full.
- Inferenza immagine con overlay box/mask.
- Metriche minime: accuracy/f1, mAP50/mAP, mean IoU.

## Versioning specifico

- Branch: `feature/phase-2-vision`.
- Commit separati: Lite e Full se possibile.
- Messaggi: `phase2: ...`.

## Dove intervenire (file principali)

- `src/metagen/synth/modalities/image.py`: selezione encoder/embeddings/data fragments.
- `src/metagen/templates/fragments/encoders/`:
  - `resnet_encoder.py.j2` (Lite)
  - `vit_encoder.py.j2` (Full)
  - `unet_encoder.py.j2` (segmentation)
- `src/metagen/templates/fragments/embeddings/`:
  - `patch_embedding.py.j2` (ViT)
  - `pixel_embedding.py.j2` (CNN)
- `src/metagen/templates/fragments/heads/`:
  - `detection_head.py.j2`
  - `segmentation_head.py.j2`
  - `classification_head.py.j2`
- `src/metagen/templates/fragments/losses/`:
  - `detection_loss.py.j2`
  - `dice_loss.py.j2`
- `src/metagen/templates/fragments/data/`:
  - `image_datasets.py.j2`
  - `detection_datasets.py.j2`
  - `segmentation_datasets.py.j2`
- `src/metagen/templates/train.py.j2` / `infer.py.j2`: CLI e inferenza.
- `src/metagen/synth/tasks/detection.py` / `segmentation.py`: task defaults.
- `examples/specs/image/`: spec vision di riferimento.
## Task Operativi

1) **Encoder e preprocessing**
- Template per ResNet-lite (torchvision) per classification.
- Preprocessing immagini (resize, normalize) nel data loader.

2) **Detection Lite (YOLO)**
- Head YOLO-lite: pred_boxes + pred_logits.
- Post-processing: score threshold + NMS.
- Dataset mapping per detection (HF cppe-5, coco).

3) **Segmentation Lite (UNet-lite)**
- UNet-lite template + loss + IoU.
- Dataset mapping (oxford_iiit_pet).

4) **Vision Full (optional)**
- ViT + DETR-lite + UNet full.

5) **Inferenza utente**
- Generare `infer_image.py` con overlay box/mask.
- Supporto `--image`, `--output`, `--score-threshold`.

## Test richiesti

- Test smoke: synth + train per image classification e detection con dataset HF ridotti.
- Test inferenza su immagine (shape output).

## Criteri di accettazione

- Detection produce box non casuali su immagini coerenti (PPE).
- Segmentation produce mask con IoU sopra baseline random.
- Classification produce accuracy > random su dataset piccolo.

## Non fare

- Non cambiare i task text/audio/video in questa fase.
