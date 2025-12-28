# Example Specs Index

MetaGen ships curated example specs under `examples/specs/`, organized by domain and task.
Use these as starting points for synthesis, papers, and AutoML search.

## Quick Use

```bash
metagen synth examples/specs/text/text_llm_8b.yaml --out outputs/
metagen synth examples/specs/image/object_detector_yolo.yaml --out outputs/
```

## Text

| Spec | Description |
| --- | --- |
| `text/text_llm_8b.yaml` | Large language model, transformer |
| `text/text_llm_tiny.yaml` | Tiny LLM for tests |
| `text/text_classifier_bert.yaml` | BERT-style classifier |
| `text/sentence_embedder.yaml` | Text embedding model |
| `text/document_ranker.yaml` | Ranking model |
| `text/edge_tiny_agent.yaml` | Edge deployment constraints |
| `text/infinite_context.yaml` | Infinite context warning case |

## Image

| Spec | Description |
| --- | --- |
| `image/image_vit_base.yaml` | Vision Transformer |
| `image/image_classifier_resnet.yaml` | CNN classifier |
| `image/image_diffusion_small.yaml` | Small diffusion |
| `image/image_diffusion_sdxl_like.yaml` | SDXL-style diffusion |
| `image/object_detector_yolo.yaml` | YOLO-style detector |
| `image/object_detector_detr.yaml` | DETR-style detector |
| `image/semantic_segmentation_unet.yaml` | U-Net segmentation |
| `image/instance_segmentation_maskrcnn.yaml` | Mask R-CNN |

## Audio

| Spec | Description |
| --- | --- |
| `audio/audio_musicgen.yaml` | Music generation |
| `audio/audio_musicgen_like.yaml` | Extended audio config |

## Video

| Spec | Description |
| --- | --- |
| `video/video_generation.yaml` | Text-to-video |
| `video/video_realtime_avatar.yaml` | Real-time avatar generation |

## Multimodal

| Spec | Description |
| --- | --- |
| `multimodal/multimodal_clip.yaml` | CLIP-style contrastive |

## 3D

| Spec | Description |
| --- | --- |
| `3d/3d_text_to_mesh.yaml` | Text-to-3D mesh |

## Tabular

| Spec | Description |
| --- | --- |
| `tabular/tabular_regressor.yaml` | Tabular regression |

## Time Series

| Spec | Description |
| --- | --- |
| `time_series/time_series_forecaster.yaml` | Forecasting model |
| `time_series/anomaly_detector_autoencoder.yaml` | Anomaly detector |

## Graph

| Spec | Description |
| --- | --- |
| `graph/graph_classifier_gat.yaml` | GAT classifier |
| `graph/recommender_two_tower.yaml` | Two-tower recommendation |

## Reinforcement Learning

| Spec | Description |
| --- | --- |
| `rl/rl_agent_ppo.yaml` | PPO policy gradient |
| `rl/rl_agent_dqn.yaml` | DQN value-based |

## Misc

| Spec | Description |
| --- | --- |
| `misc/taste_generation.yaml` | Unsupported modality warning |
