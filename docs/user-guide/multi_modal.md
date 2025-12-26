# Multi-Modal Guide

MetaGen supports 5 modalities with specialized handlers that generate appropriate architectures, code, and documentation for each type.

## Supported Modalities

| Modality | Inputs | Outputs | Example Use Cases |
|----------|--------|---------|-------------------|
| **Text** | text | text | LLMs, chatbots, translation |
| **Image** | text, image | image | ViT, diffusion, classification |
| **Audio** | audio, text | audio | Music generation, TTS, ASR |
| **Video** | video, text | video | Video generation, understanding |
| **Multi-modal** | text+image | text+image | CLIP, VLMs, retrieval |

## Text Modality

The default modality for language models.

### Example Spec

```yaml
metagen_version: "1.0"
name: "text_llm"
modality:
  inputs: ["text"]
  outputs: ["text"]
constraints:
  parameter_budget:
    max: "8B"
  context_window: "256k"
training:
  objective: ["autoregressive"]
architecture:
  family: "transformer"
```

### Generated Architecture

- **Embedding**: Token embeddings with positional encoding
- **Encoder**: Transformer layers with multi-head attention
- **Decoder**: Autoregressive decoder with causal masking
- **Output**: Linear projection to vocabulary

### Architecture Families

- `transformer` - Standard GPT-style architecture

## Image Modality

For vision models including classification, generation, and diffusion.

### Vision Transformer (ViT)

```yaml
metagen_version: "1.0"
name: "image_vit"
modality:
  inputs: ["image"]
  outputs: ["image"]
constraints:
  parameter_budget:
    max: "86M"
training:
  objective: ["classification"]
architecture:
  family: "transformer"
```

### Diffusion Model

```yaml
metagen_version: "1.0"
name: "image_diffusion"
modality:
  inputs: ["text", "image"]
  outputs: ["image"]
constraints:
  parameter_budget:
    max: "6B"
training:
  objective: ["diffusion"]
architecture:
  family: "diffusion"
```

### Architecture Families

- `transformer` - Vision Transformer (ViT)
- `cnn` - Convolutional networks (ResNet-style)
- `diffusion` - U-Net based diffusion
- `hybrid` - Combined CNN + Transformer

### Image-Specific Parameters

The image handler automatically configures:

| Parameter | Description | Values |
|-----------|-------------|--------|
| `image_size` | Input resolution | 224, 384, 512, 1024 |
| `patch_size` | ViT patch size | 14, 16, 32 |
| `num_channels` | Color channels | 3 (RGB) |

### Size Presets

Use shorthand presets in your spec:

| Preset | Resolution |
|--------|------------|
| `imagenet` | 224x224 |
| `clip` | 224x224 |
| `sd` | 512x512 |
| `sd_xl` | 1024x1024 |

## Audio Modality

For audio generation, speech, and music models.

### Example Spec

```yaml
metagen_version: "1.0"
name: "audio_musicgen"
modality:
  inputs: ["audio"]
  outputs: ["audio"]
constraints:
  parameter_budget:
    max: "1B"
  context_window: "32k"
training:
  objective: ["autoregressive"]
architecture:
  family: "transformer"
```

### Generated Architecture

- **Encoder**: Audio spectrogram encoder
- **Backbone**: Transformer with temporal attention
- **Decoder**: WaveNet-style or autoregressive decoder

### Audio-Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `sample_rate` | Audio sample rate | 16000 Hz |
| `n_mels` | Mel spectrogram bins | 80 |
| `hop_length` | Spectrogram hop | 256 |

## Video Modality

For video generation and understanding.

### Example Spec

```yaml
metagen_version: "1.0"
name: "video_gen"
modality:
  inputs: ["text"]
  outputs: ["video"]
constraints:
  parameter_budget:
    max: "2B"
training:
  objective: ["diffusion"]
architecture:
  family: "diffusion"
```

### Generated Architecture

- **Encoder**: 3D CNN or frame-wise transformer
- **Temporal**: Temporal attention across frames
- **Decoder**: Video diffusion or autoregressive

### Video-Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_frames` | Frames per clip | 16 |
| `frame_rate` | Frames per second | 8 |
| `resolution` | Frame resolution | 256x256 |

## Multi-Modal

For models that process multiple modalities together.

### CLIP-Style Contrastive

```yaml
metagen_version: "1.0"
name: "multimodal_clip"
modality:
  inputs: ["text", "image"]
  outputs: ["text", "image"]
constraints:
  parameter_budget:
    max: "800M"
training:
  objective: ["contrastive"]
architecture:
  family: "transformer"
```

### Vision-Language Model

```yaml
metagen_version: "1.0"
name: "vlm"
modality:
  inputs: ["text", "image"]
  outputs: ["text"]
constraints:
  parameter_budget:
    max: "7B"
training:
  objective: ["autoregressive"]
architecture:
  family: "transformer"
```

### Generated Architecture

- **Text Encoder**: Transformer text encoder
- **Image Encoder**: ViT or CNN image encoder
- **Fusion**: Cross-attention or late fusion
- **Projection**: Shared embedding space

### Fusion Strategies

MetaGen generates appropriate fusion based on the task:

| Strategy | Use Case |
|----------|----------|
| Early fusion | Joint encoding from start |
| Cross-attention | Query across modalities |
| Late fusion | Combine at final layers |
| Contrastive | Align separate encoders |

## Modality Handler System

MetaGen uses specialized handlers for each modality:

```python
from metagen.synth.modalities import get_handler

# Get handler for a spec
handler = get_handler(spec)

# Augment blueprint with modality-specific fields
handler.augment_blueprint(blueprint)

# Generate modality-specific components
components = handler.generate_components()
```

### Available Handlers

| Handler | Modality | Families |
|---------|----------|----------|
| `TextModalityHandler` | text | transformer |
| `ImageModalityHandler` | image | transformer, cnn, diffusion, hybrid |
| `AudioModalityHandler` | audio | transformer |
| `VideoModalityHandler` | video | transformer, diffusion |
| `MultimodalHandler` | multi | transformer, hybrid |

## Unsupported Modalities

Some modalities trigger warnings:

```yaml
modality:
  outputs: ["taste"]  # Warning: unsupported
```

MetaGen will generate best-effort approximations and note limitations.

## Best Practices

### 1. Match Objective to Modality

| Modality | Recommended Objectives |
|----------|------------------------|
| Text | autoregressive, masked |
| Image (gen) | diffusion |
| Image (cls) | classification, contrastive |
| Audio | autoregressive |
| Multi-modal | contrastive, autoregressive |

### 2. Match Architecture to Task

| Task | Architecture Family |
|------|---------------------|
| Text generation | transformer |
| Image generation | diffusion |
| Image classification | transformer (ViT), cnn |
| Multi-modal | transformer, hybrid |

### 3. Set Appropriate Constraints

Image models typically need:
- Higher memory budget (48GB+ for diffusion)
- Larger parameter budgets
- Offline latency tolerance

## Further Reading

- [Quick Start Guide](quickstart.md) - Getting started
- [Spec Language Reference](spec_language.md) - Complete schema
- [AutoML Guide](automl_guide.md) - Architecture search
