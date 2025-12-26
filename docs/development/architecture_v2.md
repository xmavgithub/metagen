# MetaGen Architecture v2 - BlueprintState System

**Navigation**: [📚 Docs Home](../README.md) | [🗺️ Roadmap](../project/roadmap.md) | [📋 Trainable Models Spec](trainable_models_spec.md) | [🧪 Testing Strategy](testing_strategy.md)

---

**Versione**: 2.0
**Data**: 2025-12-23
**Status**: Design Document

---

## Problem Statement

### Current Architecture Issues

**Problema critico**: Inconsistenza dimensionale tra blueprint e codice generato.

**Esempio**:
- `architecture.py::_choose_dims()` calcola: `hidden_size=4096, layers=32, heads=16`
- `architecture.yaml` salvato con queste dimensioni
- `codegen.py::_model_code()` sceglie random: `hidden_size=1024, layers=12`
- **Result**: Model code e blueprint sono completamente disconnessi

**Conseguenze**:
1. Parameter estimate in `params_estimate.json` non corrisponde al codice
2. Paper cita dimensioni diverse dal modello generato
3. Model card ha valori incoerenti
4. Impossible to validate spec constraints sul codice reale

---

## Solution: BlueprintState System

### Core Concept

Creare un **single source of truth** per le dimensioni del modello che fluisce attraverso tutta la pipeline:

```
Spec → Architecture Synthesis → BlueprintState → Code Generation
                                       ↓
                                  All outputs use same dimensions
```

### BlueprintState Dataclass

**Location**: `src/metagen/synth/architecture.py`

```python
from dataclasses import dataclass, field

@dataclass
class BlueprintState:
    """
    Complete blueprint state for model architecture.
    This is the single source of truth for all dimensions and parameters.
    Passed from architecture synthesis to all downstream generators.
    """

    # Core dimensions (always present)
    dims: dict[str, int]  # {hidden_size, layers, heads}

    # Modality-specific parameters
    vocab_size: int | None = None           # Text: vocabulary size
    max_seq_len: int | None = None          # Text: max sequence length
    num_channels: int | None = None         # Image/Video: channels (3=RGB, 1=grayscale)
    image_size: int | None = None           # Image: spatial size (224, 512, etc.)
    sample_rate: int | None = None          # Audio: Hz (16000, 44100, etc.)
    latent_dim: int | None = None           # Diffusion: latent space dimension
    patch_size: int | None = None           # ViT: patch size (16, 32, etc.)

    # Architecture metadata
    family: str = "transformer"             # Architecture family from spec
    components: list[dict] = field(default_factory=list)  # Component graph

    # Parameter estimates
    total_params: int = 0                   # Total parameter count
    trainable_params: int = 0               # Trainable params (usually same)
    activation_memory_gb: float = 0.0       # Forward pass memory
    kv_cache_gb: float = 0.0               # Attention cache (transformers only)

    # Seed and determinism
    seed: int = 42                          # Random seed used for generation
```

**Rationale**:
- Dataclass for clean API and type checking
- Optional fields for modality-specific params (only set if relevant)
- Includes everything needed for code generation
- Carries metadata for documentation and validation

---

## Architecture Changes

### 1. architecture.py Modifications

**Current**:
```python
def generate_blueprint(spec: ModelSpec, out_dir: Path, seed: int) -> dict[str, float]:
    dims, summary = estimate_summary(spec, seed)
    # Write files...
    return summary  # Just params_billion, memory, kv_cache
```

**New**:
```python
def generate_blueprint(
    spec: ModelSpec,
    out_dir: Path,
    seed: int
) -> BlueprintState:
    """
    Generate architecture blueprint and return complete state.

    Returns:
        BlueprintState: Complete blueprint with all dimensions and metadata
    """
    # Compute all dimensions
    dims = _choose_dims(spec)

    # Compute modality-specific params
    blueprint = _build_blueprint_state(spec, dims, seed)

    # Write blueprint files (architecture.yaml, graph.json, etc.)
    _write_blueprint_files(spec, blueprint, out_dir)

    return blueprint
```

**New helper function**:
```python
def _build_blueprint_state(
    spec: ModelSpec,
    dims: dict[str, int],
    seed: int
) -> BlueprintState:
    """Build complete BlueprintState from spec and computed dimensions."""

    # Determine modality-specific params
    vocab_size = None
    max_seq_len = None
    num_channels = None
    image_size = None

    if "text" in spec.modality.inputs or "text" in spec.modality.outputs:
        vocab_size = 50257  # GPT-2 tokenizer default
        # Parse context_window from spec
        ctx = spec.constraints.context_window
        if "k" in ctx.lower():
            max_seq_len = int(ctx.lower().replace("k", "")) * 1024
        else:
            max_seq_len = int(ctx) if ctx.isdigit() else 2048

    if "image" in spec.modality.inputs or "image" in spec.modality.outputs:
        num_channels = 3  # RGB default
        image_size = 224  # Standard image size

    # TODO: Add audio, video, 3d params

    # Estimate parameters
    params_b = _estimate_params(
        dims["hidden_size"],
        dims["layers"],
        spec.architecture.family.lower()
    )

    # Build BlueprintState
    blueprint = BlueprintState(
        dims=dims,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        num_channels=num_channels,
        image_size=image_size,
        family=spec.architecture.family.lower(),
        components=[c.model_dump() for c in spec.architecture.components],
        total_params=int(params_b * 1e9),
        trainable_params=int(params_b * 1e9),
        activation_memory_gb=round(params_b * 0.5, 2),
        kv_cache_gb=round(dims["layers"] * dims["heads"] * 0.01, 2),
        seed=seed,
    )

    return blueprint
```

**Enhanced `_choose_dims()`**:
```python
def _choose_dims(spec: ModelSpec) -> dict[str, int]:
    """
    Derive model dimensions from spec constraints.

    Extended to return more complete dimension dictionary.
    """
    family = spec.architecture.family.lower()
    latency = spec.constraints.latency
    device = spec.constraints.device

    # Base hidden size computation (existing logic)
    base_hidden = 4096 if family == "transformer" else 2048
    if latency == "real-time" or device in {"edge", "cpu"}:
        base_hidden = int(base_hidden * 0.5)
    elif device == "datacenter_gpu":
        base_hidden = int(base_hidden * 1.5)

    # Layer count computation (existing logic)
    layers = 32 if family == "transformer" else 24
    if spec.task.domain == "video":
        layers += 8
    if spec.task.domain == "audio":
        layers -= 4
    if "tiny" in spec.name:
        layers = max(12, int(layers * 0.6))

    # Heads computation (existing logic)
    heads = max(8, base_hidden // 256)

    return {
        "hidden_size": base_hidden,
        "layers": layers,
        "heads": heads,
    }
```

---

### 2. engine.py Modifications

**Current**:
```python
def synthesize(...):
    arch_summary = architecture.generate_blueprint(spec, bp_dir, resolved_seed)
    codegen.generate_code(spec, code_dir, resolved_seed)
    # arch_summary and codegen are disconnected
```

**New**:
```python
def synthesize(...):
    # Generate blueprint and get complete state
    blueprint = architecture.generate_blueprint(spec, bp_dir, resolved_seed)

    # Pass blueprint to all downstream generators
    codegen.generate_code(spec, code_dir, blueprint, resolved_seed)
    bench_summary = benchmarks.generate_reports(spec, run_folder, blueprint, resolved_seed)
    paper_gen.generate_paper(spec, paper_dir, bench_summary, blueprint, resolved_seed)

    # Use blueprint for docs
    _write_docs(spec, docs_dir, blueprint, bench_summary)
```

**Signature change**:
```python
# Before
def synthesize(spec_path: Path, out_dir: Path, run_id: str | None = None, base_seed: int = 42) -> Path:

# After (no change to signature, but internal flow updated)
def synthesize(spec_path: Path, out_dir: Path, run_id: str | None = None, base_seed: int = 42) -> Path:
```

**Internal updates**:
- Store `blueprint` instead of `arch_summary`
- Pass `blueprint` to all generators
- Extract summary dict from blueprint when needed for backwards compat

---

### 3. codegen.py Modifications

**Current**:
```python
def generate_code(spec: ModelSpec, out_dir: Path, seed: int) -> None:
    rnd = random.Random(seed)
    model_code = HEADER + _model_code(spec, rnd)  # Uses random dims!
```

**New**:
```python
def generate_code(
    spec: ModelSpec,
    out_dir: Path,
    blueprint: BlueprintState,
    seed: int
) -> None:
    """
    Generate model code using blueprint dimensions.

    Args:
        spec: Model specification
        out_dir: Output directory
        blueprint: Blueprint state with all dimensions
        seed: Random seed (for minor variations like dropout)
    """
    ensure_dir(out_dir)
    rnd = random.Random(seed)

    # Generate code using blueprint dimensions
    model_code = HEADER + _model_code(spec, blueprint, rnd)
    train_code = HEADER + _train_code(spec, blueprint)
    data_code = HEADER + _data_code(spec, blueprint)
    eval_code = HEADER + _eval_code(spec, blueprint)

    write_text(out_dir / "model.py", model_code)
    write_text(out_dir / "train.py", train_code)
    write_text(out_dir / "data.py", data_code)
    write_text(out_dir / "eval.py", eval_code)
    write_text(out_dir / "__init__.py", "from .model import MetaGenModel\n")
```

**Updated `_model_code()`**:
```python
def _model_code(spec: ModelSpec, blueprint: BlueprintState, rnd: random.Random) -> str:
    """Generate model code using blueprint dimensions."""

    # Get dimensions from blueprint (NOT random!)
    hidden_size = blueprint.dims["hidden_size"]
    layers = blueprint.dims["layers"]
    heads = blueprint.dims["heads"]

    # Only random choices for minor variations
    dropout = rnd.choice([0.1, 0.2, 0.3])

    return f'''try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = object


class MetaGenModel(nn.Module if torch else object):
    """
    Blueprint model for {spec.name}.
    Inputs: {spec.modality.inputs}
    Outputs: {spec.modality.outputs}

    Architecture:
      hidden_size: {hidden_size}
      layers: {layers}
      heads: {heads}
      dropout: {dropout}
    """

    def __init__(
        self,
        hidden_size: int = {hidden_size},
        layers: int = {layers},
    ):
        if torch:
            super().__init__()
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=max(4, hidden_size // 256),
                    dropout={dropout},
                )
                for _ in range(layers)
            ])
            self.norm = nn.LayerNorm(hidden_size)
        else:
            self.layers = []
            self.norm = None

    def forward(self, x):
        if not torch:
            raise RuntimeError("PyTorch not installed; MetaGen generates code as text.")
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
'''
```

**Key change**: `hidden_size = blueprint.dims["hidden_size"]` instead of `hidden_size = rnd.choice([1024, 2048, 4096])`

---

### 4. benchmarks.py Modifications

**Current**:
```python
def generate_reports(spec: ModelSpec, out_dir: Path, seed: int) -> dict:
    # Doesn't know actual model params
```

**New**:
```python
def generate_reports(
    spec: ModelSpec,
    out_dir: Path,
    blueprint: BlueprintState,
    seed: int
) -> dict:
    """
    Generate benchmark reports using blueprint parameters.

    Args:
        spec: Model specification
        out_dir: Output directory
        blueprint: Blueprint state (for param counts, etc.)
        seed: Random seed for score generation
    """
    # Use blueprint.total_params for realistic metrics
    # Can compute Novelty-Per-Parameter accurately now
```

---

### 5. paper_gen.py Modifications

**Current**:
```python
def generate_paper(spec, paper_dir, bench_summary, arch_summary, seed):
    # Uses separate arch_summary and bench_summary
```

**New**:
```python
def generate_paper(
    spec: ModelSpec,
    paper_dir: Path,
    bench_summary: dict,
    blueprint: BlueprintState,
    seed: int
) -> None:
    """
    Generate LaTeX paper using blueprint dimensions.

    Args:
        spec: Model specification
        paper_dir: Output directory
        bench_summary: Benchmark results
        blueprint: Blueprint state with dimensions
        seed: Random seed
    """
    # Access blueprint.dims, blueprint.total_params, etc.
    # Ensure paper cites correct dimensions from blueprint
```

---

## Data Flow Diagram

### Before (v1)

```
Spec
  ↓
architecture.py::generate_blueprint()
  ├→ _choose_dims() → dims = {hidden_size: 4096, layers: 32}
  ├→ Write architecture.yaml (hidden_size: 4096)
  └→ Return arch_summary

codegen.py::generate_code()
  └→ _model_code()
      └→ random.choice([1024, 2048, 4096]) → hidden_size = 1024  ❌ MISMATCH!
      └→ Write model.py (hidden_size: 1024)

Result: architecture.yaml says 4096, model.py has 1024
```

### After (v2)

```
Spec
  ↓
architecture.py::generate_blueprint()
  ├→ _choose_dims() → dims = {hidden_size: 4096, layers: 32}
  ├→ _build_blueprint_state() → BlueprintState(dims={...}, vocab_size=50257, ...)
  ├→ Write architecture.yaml (hidden_size: 4096)
  └→ Return BlueprintState

engine.py::synthesize()
  ├→ blueprint = generate_blueprint(...)
  └→ generate_code(..., blueprint)  ← Pass blueprint!

codegen.py::generate_code(blueprint)
  └→ _model_code(blueprint)
      └→ hidden_size = blueprint.dims["hidden_size"]  → 4096  ✓ CONSISTENT!
      └→ Write model.py (hidden_size: 4096)

Result: architecture.yaml and model.py both have 4096
```

---

## Migration Strategy

### Phase 1: Add BlueprintState (Non-breaking)

1. Add `BlueprintState` dataclass to `architecture.py`
2. Create `_build_blueprint_state()` helper
3. Update `generate_blueprint()` to return `BlueprintState`
4. Keep backward compat by allowing callers to extract dict if needed

**Backward compat**:
```python
# Old code still works
arch_summary = generate_blueprint(...)  # Returns BlueprintState
params_b = arch_summary.total_params / 1e9  # Can access as object

# Can also convert to dict for old code
summary_dict = {
    "params_billion": arch_summary.total_params / 1e9,
    "activation_memory_gb": arch_summary.activation_memory_gb,
    "kv_cache_gb": arch_summary.kv_cache_gb,
}
```

### Phase 2: Update Consumers

2. Update `engine.py` to pass blueprint to downstream
3. Update `codegen.py` to accept and use blueprint
4. Update `benchmarks.py` signature (optional use of blueprint)
5. Update `paper_gen.py` signature (optional use of blueprint)

**Each change is isolated and testable**

### Phase 3: Update Tests

6. Update `test_architecture.py` to verify BlueprintState
7. Add test for dimension consistency
8. Ensure all existing tests pass (no regression)

---

## Validation and Testing

### New Tests

**`tests/test_architecture.py`**:
```python
def test_blueprint_state_creation():
    """Test BlueprintState is created with all required fields"""
    spec = load_spec("examples/specs/text_llm_8b.yaml")
    blueprint = generate_blueprint(spec, tmp_path, seed=42)

    # Verify type
    assert isinstance(blueprint, BlueprintState)

    # Verify core dims
    assert "hidden_size" in blueprint.dims
    assert "layers" in blueprint.dims
    assert "heads" in blueprint.dims

    # Verify text-specific params
    assert blueprint.vocab_size == 50257
    assert blueprint.max_seq_len > 0

    # Verify param estimates
    assert blueprint.total_params > 0
    assert blueprint.activation_memory_gb > 0


def test_blueprint_dimensions_deterministic():
    """Test same spec + seed produces same dimensions"""
    spec = load_spec("examples/specs/text_llm_8b.yaml")

    bp1 = generate_blueprint(spec, tmp_path / "run1", seed=42)
    bp2 = generate_blueprint(spec, tmp_path / "run2", seed=42)

    assert bp1.dims == bp2.dims
    assert bp1.vocab_size == bp2.vocab_size
    assert bp1.total_params == bp2.total_params
```

**`tests/test_codegen.py`**:
```python
def test_blueprint_dimensions_in_generated_code(tmp_path):
    """Verify generated code uses blueprint dimensions"""
    spec = load_spec("examples/specs/text_llm_8b.yaml")
    blueprint = generate_blueprint(spec, tmp_path / "bp", seed=42)
    generate_code(spec, tmp_path / "code", blueprint, seed=42)

    # Read generated model.py
    model_code = (tmp_path / "code" / "model.py").read_text()

    # Verify dimensions from blueprint appear in code
    assert f"hidden_size: int = {blueprint.dims['hidden_size']}" in model_code
    assert f"layers: int = {blueprint.dims['layers']}" in model_code

    # Verify no hardcoded random dimensions
    assert "hidden_size: int = 1024" not in model_code or blueprint.dims["hidden_size"] == 1024
    assert "hidden_size: int = 2048" not in model_code or blueprint.dims["hidden_size"] == 2048


def test_code_generation_consistent_with_blueprint(tmp_path):
    """Integration test: blueprint → code → architecture.yaml consistency"""
    spec = load_spec("examples/specs/text_llm_8b.yaml")

    bp_dir = tmp_path / "blueprint"
    code_dir = tmp_path / "code"

    blueprint = generate_blueprint(spec, bp_dir, seed=42)
    generate_code(spec, code_dir, blueprint, seed=42)

    # Load architecture.yaml
    import yaml
    arch_yaml = yaml.safe_load((bp_dir / "architecture.yaml").read_text())

    # Parse model.py for dimensions
    model_code = (code_dir / "model.py").read_text()

    # Extract hidden_size from model.py default value
    import re
    match = re.search(r"hidden_size: int = (\d+)", model_code)
    assert match, "Could not find hidden_size in model.py"
    code_hidden_size = int(match.group(1))

    # Verify consistency
    assert arch_yaml["hidden_size"] == code_hidden_size
    assert arch_yaml["hidden_size"] == blueprint.dims["hidden_size"]
```

---

## Benefits of BlueprintState System

### 1. Consistency
- All artifacts (YAML, code, paper, docs) use identical dimensions
- No more mismatches between blueprint and generated code
- Single source of truth enforced by type system

### 2. Maintainability
- Clear API: pass `BlueprintState` instead of multiple dicts
- Type hints prevent errors (IDE autocomplete, mypy)
- Easier to add new fields (just extend dataclass)

### 3. Testability
- Easy to verify dimension consistency
- Can test blueprint generation independently
- Mock blueprints for testing downstream components

### 4. Extensibility
- New modalities just add optional fields
- Template rendering gets richer context
- Blueprint can carry more metadata without breaking API

### 5. Debugging
- Inspect blueprint object to see all computed dimensions
- Logs can print blueprint for reproducibility
- Clear data flow through pipeline

---

## Open Questions

1. **Backward compatibility**: Should we support old `arch_summary` dict return type?
   - **Decision**: Return `BlueprintState` but allow dict extraction

2. **Serialization**: Should `BlueprintState` be JSON-serializable?
   - **Decision**: Yes, add `to_dict()` and `from_dict()` methods

3. **Validation**: Should `BlueprintState` validate param budgets match spec?
   - **Decision**: Yes, add `validate()` method that checks constraints

4. **Immutability**: Should `BlueprintState` be frozen (immutable)?
   - **Decision**: Yes, use `@dataclass(frozen=True)` to prevent accidental mutations

---

## Implementation Checklist

### WU-2: BlueprintState Implementation

- [ ] Add `BlueprintState` dataclass to `architecture.py`
- [ ] Implement `_build_blueprint_state()` helper
- [ ] Update `generate_blueprint()` return type
- [ ] Update `_choose_dims()` if needed (already good)
- [ ] Update `engine.py` to pass blueprint
- [ ] Update `codegen.py` to accept blueprint
- [ ] Remove random dimension choices in `_model_code()`
- [ ] Update tests for dimension consistency
- [ ] Verify all existing tests pass
- [ ] Update documentation

---

**Document Status**: Design Complete
**Implementation**: WU-2
**Maintainer**: Mauro
