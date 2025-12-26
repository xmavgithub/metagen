# Trainable Models - Technical Specification

**Navigation**: [📚 Docs Home](../README.md) | [🗺️ Roadmap](../project/roadmap.md) | [🏗️ Architecture v2](architecture_v2.md) | [🧪 Testing Strategy](testing_strategy.md)

---

**Versione**: 1.0
**Data**: 2025-12-23
**Status**: Draft

---

## Overview

Questo documento specifica i requisiti tecnici per i modelli trainabili generati da MetaGen. L'obiettivo è produrre codice PyTorch funzionante che possa essere effettivamente allenato su dati (mock o reali), mantenendo il tono satirico del progetto nei documenti di rilascio.

---

## Architetture Supportate

### 1. Text LLM (Causal Transformer)

**Priority**: ⭐⭐⭐ Alta (Implementazione primaria)

**Description**: Modello transformer decoder-only per generazione testo, tipo GPT.

**Components Required**:
- Token embeddings (`nn.Embedding`)
- Position embeddings (learned o sinusoidal)
- Causal transformer blocks (`nn.TransformerDecoderLayer` o custom)
- Layer normalization finale
- LM head (projection to vocabulary)

**Input/Output**:
```python
Input:  input_ids: torch.LongTensor of shape (batch, seq_len)
Output: logits: torch.FloatTensor of shape (batch, seq_len, vocab_size)
```

**Hyperparameters** (from BlueprintState):
- `vocab_size`: int (default 50257 for GPT-2 tokenizer)
- `max_seq_len`: int (from spec.constraints.context_window, es. 2048)
- `hidden_size`: int (from blueprint, es. 4096)
- `layers`: int (from blueprint, es. 32)
- `heads`: int (from blueprint, es. 16)
- `dropout`: float (0.1-0.3)

**Training Objective**: Causal language modeling
```python
loss = nn.functional.cross_entropy(
    logits[:, :-1, :].reshape(-1, vocab_size),  # Predict next token
    input_ids[:, 1:].reshape(-1)                 # Shift labels
)
```

**Evaluation Metrics**:
- **Perplexity**: exp(average cross-entropy loss)
- Token accuracy (optional)
- Satirical metrics: Spec-Fidelity@1, SOTA-Proximity

**Constraints**:
- Must respect `parameter_budget` from spec
- Must support causal masking (no future tokens)
- Must handle variable sequence lengths (up to max_seq_len)

---

### 2. Image Diffusion (UNet)

**Priority**: ⭐⭐ Media (Post text LLM)

**Description**: Modello UNet per diffusion models (tipo Stable Diffusion).

**Components Required**:
- Time embedding MLP
- Text conditioning encoder (CLIP-style)
- Downsampling blocks (conv + attention)
- Middle block (residual + attention)
- Upsampling blocks (transposed conv + attention)
- Skip connections between down/up blocks

**Input/Output**:
```python
Input:  x: (batch, channels, H, W)         # Noisy image
        timesteps: (batch,)                # Noise level
        text_embed: (batch, seq_len, dim)  # Conditioning
Output: noise_pred: (batch, channels, H, W) # Predicted noise
```

**Hyperparameters**:
- `in_channels`: 3 (RGB) or 4 (latent)
- `hidden_sizes`: [128, 256, 512, 1024] (per block)
- `time_embed_dim`: 512
- `text_embed_dim`: 768 (CLIP)
- `num_blocks`: 4 (down/up pairs)

**Training Objective**: Noise prediction
```python
# Sample random timesteps
t = torch.randint(0, num_timesteps, (batch_size,))
# Add noise to images
noisy_x = scheduler.add_noise(x, noise, t)
# Predict noise
noise_pred = model(noisy_x, t, text_embed)
# MSE loss
loss = nn.functional.mse_loss(noise_pred, noise)
```

**Evaluation Metrics**:
- **FID score** (Frechet Inception Distance)
- **IS score** (Inception Score)
- Visual quality (qualitative)

**Constraints**:
- Must support conditioning (text, class labels, etc.)
- Must handle different image sizes (via adaptive layers)
- Must integrate with diffusion scheduler

---

### 3. Multimodal (Image-Text)

**Priority**: ⭐ Bassa (Future work)

**Description**: Dual-encoder per image-text alignment (tipo CLIP).

**Components Required**:
- Image encoder (Vision Transformer or ResNet)
- Text encoder (Transformer encoder)
- Projection heads (to common embedding space)
- Temperature-scaled similarity

**Input/Output**:
```python
Input:  images: (batch, 3, 224, 224)
        text_ids: (batch, max_text_len)
Output: image_embeds: (batch, embed_dim)
        text_embeds: (batch, embed_dim)
        logits: (batch, batch)  # Similarity matrix
```

**Training Objective**: Contrastive loss (InfoNCE)
```python
# Normalize embeddings
image_embeds = F.normalize(image_embeds, dim=-1)
text_embeds = F.normalize(text_embeds, dim=-1)

# Compute similarity matrix
logits = (image_embeds @ text_embeds.T) * temperature

# Contrastive loss (symmetric)
labels = torch.arange(batch_size)
loss = (
    F.cross_entropy(logits, labels) +
    F.cross_entropy(logits.T, labels)
) / 2
```

**Evaluation Metrics**:
- **CLIP score**: Average cosine similarity
- **Retrieval accuracy**: Image→Text and Text→Image
- Zero-shot classification accuracy

---

### 4. Audio/Video (Temporal Models)

**Priority**: ⭐ Bassa (Future work)

**Description**: Modelli con componenti temporali (LSTM, temporal conv, etc.).

**Status**: Specification TBD

---

## Blueprint State Structure

Tutte le architetture ricevono un `BlueprintState` object con:

```python
@dataclass
class BlueprintState:
    """Blueprint state passed from architecture synthesis to code generation"""

    # Core dimensions (computed from spec constraints)
    dims: dict[str, int]  # {hidden_size, layers, heads}

    # Modality-specific parameters
    vocab_size: int | None = None       # For text models
    max_seq_len: int | None = None      # For text models
    num_channels: int | None = None     # For image/video (3 for RGB, 1 for grayscale)
    image_size: int | None = None       # For image models (e.g., 224, 512)
    sample_rate: int | None = None      # For audio models (e.g., 16000, 44100)
    latent_dim: int | None = None       # For diffusion models
    patch_size: int | None = None       # For ViT models

    # Architecture metadata
    family: str = "transformer"         # Architecture family
    components: list[dict] = field(default_factory=list)  # Component list

    # Parameter estimates
    total_params: int = 0               # Total parameter count
    trainable_params: int = 0           # Trainable parameters
    activation_memory_gb: float = 0.0   # Estimated activation memory
    kv_cache_gb: float = 0.0           # Estimated KV cache (for transformers)
```

---

## Code Generation Requirements

### Model Code (model.py)

**Must include**:
1. All necessary imports with graceful fallback:
   ```python
   try:
       import torch
       import torch.nn as nn
   except ImportError:
       torch = None
       nn = object
   ```

2. Model class inheriting from `nn.Module`:
   ```python
   class MetaGenModel(nn.Module if torch else object):
       def __init__(self, ...):
           super().__init__()
           # Initialize all layers

       def forward(self, ...):
           # Forward pass implementation
   ```

3. All layers from blueprint:
   - Use blueprint dimensions (not random)
   - Initialize with proper methods (Xavier, Kaiming, etc.)
   - Add docstrings with expected shapes

4. Architecture-specific components:
   - Text LLM: embeddings, causal blocks, LM head
   - Diffusion: time/text embeddings, UNet blocks
   - Multimodal: dual encoders, projection heads

**Shape documentation**:
```python
def forward(self, input_ids):
    """
    Args:
        input_ids: (batch, seq_len) Long tensor of token IDs

    Returns:
        logits: (batch, seq_len, vocab_size) Next-token prediction logits
    """
```

---

### Data Loader Code (data.py)

**Must include**:
1. `load_data()` function with mode parameter:
   ```python
   def load_data(batch_size=8, mode="mock", split="train"):
       """
       Load data for training/evaluation.

       Args:
           batch_size: Batch size
           mode: "mock" (synthetic) or "real" (download datasets)
           split: "train" or "val"

       Yields:
           batch: dict with tensors (e.g., {"input_ids": torch.LongTensor})
       """
   ```

2. Mock mode (always available):
   - Generate random tensors with correct shapes
   - Use seeded random for reproducibility
   - Instant, no downloads

3. Real mode (future):
   - Download tiny dataset subset (e.g., WikiText-2)
   - Use `datasets` library
   - Cache downloaded files
   - Progress bar for downloads

**Output format**:
- Text: `{"input_ids": LongTensor(batch, seq_len)}`
- Image: `{"images": FloatTensor(batch, C, H, W), "labels": LongTensor(batch)}`
- Multimodal: `{"images": FloatTensor, "input_ids": LongTensor}`

---

### Training Loop Code (train.py)

**Must include**:
1. `train()` function:
   ```python
   def train(
       model,
       data_loader,
       epochs=1,
       lr=1e-4,
       device="cuda",
       max_steps=None,
       checkpoint_dir="checkpoints"
   ):
       """Train the model."""
   ```

2. Complete training logic:
   - Optimizer setup (AdamW recommended)
   - Loss computation (architecture-specific)
   - Backward pass with gradient clipping
   - Logging (loss, step, epoch)
   - Optional checkpointing

3. Device handling:
   - Auto-detect CUDA if available
   - Fallback to CPU with warning
   - Move model and data to device

4. Early stopping:
   - Support `max_steps` parameter for testing
   - Break after N steps if specified

**Example structure**:
```python
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    for step, batch in enumerate(data_loader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)

        # Forward pass
        logits = model(input_ids)

        # Compute loss (architecture-specific)
        loss = compute_loss(logits, input_ids)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Log
        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

        # Early stop for testing
        if max_steps and step >= max_steps:
            break

return model
```

---

### Evaluation Code (eval.py)

**Must include**:
1. `evaluate()` function:
   ```python
   def evaluate(model, data_loader, device="cuda"):
       """Evaluate the model and compute metrics."""
   ```

2. Real metrics (architecture-specific):
   - Text LLM: Perplexity, token accuracy
   - Image diffusion: FID score, IS score
   - Multimodal: CLIP score, retrieval accuracy

3. Satirical metrics (for paper):
   - Spec-Fidelity@1: Always 0.99
   - SOTA-Proximity: Always 0.97
   - Novelty-Per-Parameter: Computed from params

4. Return format:
   ```python
   return {
       # Real metrics
       "perplexity": float,
       "avg_loss": float,
       "total_tokens": int,

       # Satirical metrics (for paper)
       "Spec-Fidelity@1": 0.99,
       "SOTA-Proximity": 0.97,
       "Novelty-Per-Parameter": 1.05,
   }
   ```

---

## Template System

### Template Structure

Templates are Jinja2 files in `src/metagen/synth/templates/`.

**Naming convention**:
- `{architecture}_{component}.py.jinja`
- Examples:
  - `transformer_causal.py.jinja` (model)
  - `dataloader_text.py.jinja` (data)
  - `train_loop.py.jinja` (training)
  - `eval_metrics.py.jinja` (evaluation)

**Template variables** (passed during rendering):
- `spec`: ModelSpec object
- `blueprint`: BlueprintState object
- `seed`: int (for deterministic random choices)
- Additional context as needed

**Example**:
```jinja
class MetaGenModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = {{ blueprint.vocab_size }},
        hidden_size: int = {{ blueprint.dims.hidden_size }},
        layers: int = {{ blueprint.dims.layers }},
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        # ...
```

---

## Testing Requirements

### Unit Tests

Every component must have unit tests:

1. **Blueprint generation**:
   ```python
   def test_blueprint_contains_all_fields():
       blueprint = generate_blueprint(spec, tmp_path, seed=42)
       assert "dims" in blueprint
       assert "vocab_size" in blueprint
       assert blueprint.total_params > 0
   ```

2. **Template rendering**:
   ```python
   def test_template_renders_valid_python():
       code = render_template("transformer_causal.py.jinja", ...)
       compile(code, "<string>", "exec")  # Must not raise
   ```

3. **Model instantiation**:
   ```python
   def test_model_instantiates():
       from model import MetaGenModel
       model = MetaGenModel()
       assert isinstance(model, nn.Module)
   ```

### Integration Tests

End-to-end pipelines:

1. **Code generation → Import**:
   ```python
   def test_generated_code_imports(tmp_path):
       generate_code(spec, tmp_path, blueprint, seed=42)
       sys.path.insert(0, str(tmp_path))
       from model import MetaGenModel  # Must succeed
   ```

2. **Forward pass shapes**:
   ```python
   def test_forward_pass_shapes():
       model = MetaGenModel()
       input_ids = torch.randint(0, vocab_size, (2, 128))
       logits = model(input_ids)
       assert logits.shape == (2, 128, vocab_size)
   ```

3. **1-step training**:
   ```python
   def test_one_step_training():
       model = MetaGenModel()
       data_loader = load_data(batch_size=2, mode="mock")
       initial_params = {n: p.clone() for n, p in model.named_parameters()}

       train(model, data_loader, epochs=1, max_steps=1)

       # Verify parameters changed
       for name, param in model.named_parameters():
           assert not torch.equal(param, initial_params[name])
   ```

### Test Markers

Use pytest markers for slow tests:

```python
@pytest.mark.slow
def test_full_training():
    # This test takes >5s
    train(model, data_loader, epochs=10)
```

Run fast tests: `pytest -m "not slow"`
Run all tests: `pytest`

---

## Mode System

### Mock Mode (default)

**Characteristics**:
- Fast, no downloads
- Synthetic tensors
- Instant data generation
- Backward compatible with current MetaGen

**Use cases**:
- Quick iterations
- CI/CD pipelines
- Development
- Demos

### Trainable Mode

**Characteristics**:
- Blueprint-synchronized code
- Mock data loaders (tensors, not strings)
- Complete training loops
- Real evaluation metrics

**Use cases**:
- Local training experiments
- Testing trainability
- Debugging model architectures

### Full Mode (future)

**Characteristics**:
- Real dataset downloads
- Actual training for N steps
- Real benchmarks
- Production-ready

**Use cases**:
- Research experiments
- Paper results
- Production deployments

---

## Determinism Guarantees

**What IS deterministic**:
- Code generation (same spec + seed → identical code)
- Blueprint dimensions
- Template rendering
- Mock data sequences (if seeded)

**What is NOT deterministic**:
- Training results (CUDA non-determinism, hardware differences)
- Real dataset order (unless seeded)
- Evaluation metrics (vary with training)

**Documentation**:
- Clearly state "code generation is deterministic"
- Note "training results may vary across hardware"
- Add seed to all random operations

---

## Backward Compatibility

**Requirements**:
1. Mode "mock" must behave like current MetaGen
2. All existing tests must pass
3. CLI without `--mode` defaults to "mock"
4. No breaking changes to existing APIs

**Migration path**:
- Users can opt-in to trainable mode
- Existing workflows continue unchanged
- New features additive, not replacing

---

## Dependencies

**Required**:
- `torch>=2.2.0` (already in pyproject.toml)
- `jinja2>=3.1.3` (already present)

**Optional** (for future real mode):
- `transformers>=4.36.0` (tokenizers, models)
- `datasets>=2.16.0` (dataset loading)
- `torchvision>=0.17.0` (image models)
- `torchaudio>=2.2.0` (audio models)

**Installation groups**:
```toml
[project.optional-dependencies]
trainable = ["transformers", "datasets"]
vision = ["torchvision", "pillow"]
audio = ["torchaudio", "librosa"]
full = ["transformers", "datasets", "torchvision", "torchaudio"]
```

---

## Performance Considerations

**Code generation**:
- Template rendering: <100ms per file
- Blueprint computation: <50ms
- Total synthesis time: <1s (mock mode)

**Training** (mock data):
- 1-step training: <1s
- 10-step training: <10s (for testing)
- Scales with model size

**Memory**:
- Small models (<1B params): 2-4GB GPU
- Medium models (1-10B params): 8-16GB GPU
- Large models (>10B params): CPU only or multi-GPU

---

## Future Enhancements

Post-WU-8 considerations:

1. **Checkpoint Management**: Save/load trained models
2. **Distributed Training**: Multi-GPU via DDP
3. **Quantization**: Int8/Int4 for edge deployment
4. **ONNX Export**: For production inference
5. **Hyperparameter Tuning**: Optuna integration
6. **Model Compression**: Pruning, distillation
7. **Real Datasets**: Full integration with HuggingFace datasets
8. **Evaluation Suites**: Comprehensive benchmarking

---

## Examples

### Text LLM Example

**Spec** (`text_llm_8b.yaml`):
```yaml
name: "text_llm_8b"
modality:
  inputs: ["text"]
  outputs: ["text"]
constraints:
  parameter_budget:
    max: "8B"
  context_window: "2048"
```

**Generated Model**:
- vocab_size: 50257
- max_seq_len: 2048
- hidden_size: 4096
- layers: 32
- heads: 16
- Total params: ~6.4B

**Training Command**:
```bash
metagen synth examples/specs/text_llm_8b.yaml --out outputs --mode trainable
cd outputs/run-*/code
python -c "
from model import MetaGenModel
from data import load_data
from train import train

model = MetaGenModel()
data_loader = load_data(batch_size=4, mode='mock')
train(model, data_loader, epochs=1, max_steps=10)
"
```

---

**Document Status**: Draft
**Next Review**: After WU-4 completion
**Maintainer**: Mauro
