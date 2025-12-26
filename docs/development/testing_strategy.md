# MetaGen Testing Strategy

**Navigation**: [📚 Docs Home](../README.md) | [🗺️ Roadmap](../project/roadmap.md) | [🏗️ Architecture v2](architecture_v2.md) | [📋 Trainable Models Spec](trainable_models_spec.md)

---

**Versione**: 1.0
**Data**: 2025-12-23
**Status**: Active

---

## Overview

Questa strategia di testing definisce come testiamo MetaGen durante la trasformazione da "mock generator" a "trainable model generator". L'obiettivo è mantenere alta qualità del codice con test veloci e completi.

---

## Testing Principles

### 1. **Test-Driven Development**
- Scrivere test PRIMA di implementare feature
- Ogni WU ha criteri di successo misurabili via test
- No merge senza test passing

### 2. **Fast Feedback**
- Unit tests < 1s ciascuno
- Integration tests < 5s
- Slow tests marcati e opzionali

### 3. **Backward Compatibility**
- Tutti i test esistenti devono passare
- Nuove feature non devono rompere vecchie
- Mode "mock" mantiene comportamento originale

### 4. **Realism**
- Test devono usare spec realistiche
- Test di training devono verificare che parametri cambino
- Test di evaluation devono validare metriche

---

## Test Levels

### Level 1: Unit Tests (< 1s each)

**Scope**: Funzioni e classi isolate

**Characteristics**:
- No I/O (file system, network)
- Mock di dipendenze pesanti
- Deterministici (no randomness senza seed)
- Run su ogni commit

**Examples**:

```python
# Test spec validation
def test_validate_minimal_spec():
    spec = {"metagen_version": "1.0", "name": "test"}
    validated = validate_spec(spec)
    assert validated.name == "test"

# Test dimension computation
def test_choose_dims_transformer():
    spec = ModelSpec(architecture=Architecture(family="transformer"))
    dims = _choose_dims(spec)
    assert dims["hidden_size"] > 0
    assert dims["layers"] > 0

# Test blueprint state creation
def test_blueprint_state_has_required_fields():
    blueprint = BlueprintState(dims={"hidden_size": 4096})
    assert blueprint.dims["hidden_size"] == 4096
    assert blueprint.family == "transformer"
```

**Coverage target**: 80%+ for core modules (specs, architecture, utils)

---

### Level 2: Integration Tests (1-5s each)

**Scope**: Multiple components working together

**Characteristics**:
- Minimal I/O (small temp files OK)
- Real components (no heavy mocking)
- Test pipelines end-to-end
- Run in CI on every PR

**Examples**:

```python
# Test blueprint → code consistency
def test_blueprint_dimensions_in_code(tmp_path):
    spec = load_spec("examples/specs/text_llm_8b.yaml")
    blueprint = generate_blueprint(spec, tmp_path / "bp", seed=42)
    generate_code(spec, tmp_path / "code", blueprint, seed=42)

    model_code = (tmp_path / "code" / "model.py").read_text()
    assert f"hidden_size: int = {blueprint.dims['hidden_size']}" in model_code

# Test generated code imports
def test_generated_model_imports(tmp_path):
    spec = load_spec("examples/specs/text_llm_8b.yaml")
    blueprint = generate_blueprint(spec, tmp_path / "bp", seed=42)
    generate_code(spec, tmp_path / "code", blueprint, seed=42)

    sys.path.insert(0, str(tmp_path / "code"))
    from model import MetaGenModel
    assert MetaGenModel is not None

# Test 1-step training
def test_one_step_training(tmp_path):
    # Generate code
    spec = load_spec("examples/specs/text_llm_8b.yaml")
    blueprint = generate_blueprint(spec, tmp_path / "bp", seed=42)
    generate_code(spec, tmp_path / "code", blueprint, seed=42)

    # Import and train
    sys.path.insert(0, str(tmp_path / "code"))
    from model import MetaGenModel
    from data import load_data
    from train import train

    model = MetaGenModel()
    data_loader = load_data(batch_size=2, mode="mock")

    initial_params = {n: p.clone() for n, p in model.named_parameters()}
    train(model, data_loader, epochs=1, max_steps=1)

    # Verify params changed
    for name, param in model.named_parameters():
        assert not torch.equal(param, initial_params[name])
```

**Coverage target**: 90%+ for synthesis pipeline (engine, codegen, architecture)

---

### Level 3: Slow Tests (> 5s, marked)

**Scope**: Full training, dataset downloads, expensive operations

**Characteristics**:
- Run manually or in nightly CI
- Marked with `@pytest.mark.slow`
- Can involve network I/O, GPU training
- Optional but valuable for real-world validation

**Examples**:

```python
@pytest.mark.slow
def test_full_training_10_steps(tmp_path):
    """Train for 10 steps and verify loss decreases"""
    spec = load_spec("examples/specs/text_llm_8b.yaml")
    blueprint = generate_blueprint(spec, tmp_path / "bp", seed=42)
    generate_code(spec, tmp_path / "code", blueprint, seed=42)

    sys.path.insert(0, str(tmp_path / "code"))
    from model import MetaGenModel
    from data import load_data
    from train import train

    model = MetaGenModel()
    data_loader = load_data(batch_size=4, mode="mock")

    # Train for 10 steps
    trained_model = train(model, data_loader, epochs=1, max_steps=10)
    # Note: With mock random data, loss may not decrease
    # But training should complete without errors


@pytest.mark.slow
@pytest.mark.requires_network
def test_real_dataset_download(tmp_path):
    """Test downloading real WikiText-2 dataset"""
    # This test downloads ~10MB from HuggingFace
    # Only run when testing real data mode
    spec = load_spec("examples/specs/text_llm_8b.yaml")
    blueprint = generate_blueprint(spec, tmp_path / "bp", seed=42)
    generate_code(spec, tmp_path / "code", blueprint, seed=42)

    sys.path.insert(0, str(tmp_path / "code"))
    from data import load_data

    loader = load_data(batch_size=2, mode="real", max_samples=100)
    batch = next(iter(loader))

    assert "input_ids" in batch
    assert batch["input_ids"].dtype == torch.long
```

**Run command**:
```bash
# Skip slow tests (default)
pytest -m "not slow"

# Run only slow tests
pytest -m slow

# Run all tests
pytest
```

---

## Test Organization

### File Structure

```
tests/
├── test_schema.py              # Spec validation tests
├── test_loader.py              # Spec loading tests
├── test_seed.py                # Determinism tests
├── test_architecture.py        # Blueprint generation tests
├── test_codegen.py             # Code generation tests
├── test_engine.py              # Synthesis pipeline tests
├── test_cli.py                 # CLI tests
├── test_templates.py           # NEW: Template rendering tests
├── test_dataloaders.py         # NEW: Data loader tests
├── test_trainable_text.py      # NEW: Text LLM trainability tests
├── test_trainable_image.py     # NEW: Image diffusion tests (future)
├── test_trainable_multimodal.py # NEW: Multimodal tests (future)
└── conftest.py                 # Shared fixtures
```

### Fixtures (conftest.py)

```python
import pytest
from pathlib import Path
import tempfile

@pytest.fixture
def tmp_synthesis_dir():
    """Temporary directory for synthesis outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text_spec():
    """Load text_llm_8b spec for testing"""
    return load_spec("examples/specs/text_llm_8b.yaml")


@pytest.fixture
def sample_blueprint(tmp_path, sample_text_spec):
    """Generate blueprint for testing"""
    return generate_blueprint(sample_text_spec, tmp_path, seed=42)


@pytest.fixture
def mock_model_code(tmp_path, sample_text_spec, sample_blueprint):
    """Generate model code for import testing"""
    code_dir = tmp_path / "code"
    generate_code(sample_text_spec, code_dir, sample_blueprint, seed=42)
    return code_dir
```

---

## Test Markers

### Configuration (pyproject.toml)

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "requires_network: marks tests requiring network access",
    "requires_gpu: marks tests requiring CUDA GPU",
    "integration: marks integration tests",
    "unit: marks unit tests",
]
```

### Usage

```python
import pytest

# Unit test (fast)
def test_fast_function():
    assert True


# Integration test
@pytest.mark.integration
def test_pipeline():
    ...


# Slow test
@pytest.mark.slow
def test_full_training():
    ...


# Network required
@pytest.mark.requires_network
def test_dataset_download():
    ...


# GPU required
@pytest.mark.requires_gpu
def test_gpu_training():
    ...
```

### Run Commands

```bash
# Fast tests only (default for CI)
pytest -m "not slow"

# All tests including slow
pytest

# Only integration tests
pytest -m integration

# Skip network tests (for offline dev)
pytest -m "not requires_network"

# Only GPU tests
pytest -m requires_gpu
```

---

## Test Scenarios per WU

### WU-1: Documentation
- No code tests (only doc validation)
- Verify markdown files exist and are parseable

### WU-2: Blueprint Sync
**Tests to add**:
- [ ] `test_blueprint_state_creation()` - Verify BlueprintState has all fields
- [ ] `test_blueprint_dimensions_deterministic()` - Same seed → same dims
- [ ] `test_blueprint_dimensions_in_code()` - Blueprint dims appear in generated code
- [ ] `test_code_architecture_yaml_consistency()` - architecture.yaml matches model.py

**Existing tests to verify**:
- [ ] All tests in `test_architecture.py` still pass
- [ ] All tests in `test_codegen.py` still pass
- [ ] `make test` at 100%

### WU-3: Templates
**Tests to add**:
- [ ] `test_template_loader_exists()` - Template loader module exists
- [ ] `test_template_rendering()` - Render template with blueprint
- [ ] `test_rendered_code_valid_python()` - Rendered code compiles
- [ ] `test_template_variables_substituted()` - Variables replaced correctly

### WU-4: Model Complete
**Tests to add**:
- [ ] `test_model_imports()` - Generated model imports successfully
- [ ] `test_model_instantiation()` - Can create model instance
- [ ] `test_model_has_embeddings()` - Model has token_embed layer
- [ ] `test_model_has_lm_head()` - Model has output projection
- [ ] `test_forward_pass_shapes()` - Forward pass returns correct shape
- [ ] `test_model_parameters_count()` - Param count roughly matches blueprint

### WU-5: Data Loaders
**Tests to add**:
- [ ] `test_mock_data_loader_yields_tensors()` - Returns tensors not strings
- [ ] `test_data_loader_batch_shape()` - Correct batch dimensions
- [ ] `test_data_loader_dtype()` - input_ids are LongTensor
- [ ] `test_data_loader_iteration()` - Can iterate multiple batches

### WU-6: Training Loop
**Tests to add**:
- [ ] `test_training_loop_runs()` - Training completes without error
- [ ] `test_parameters_change_after_training()` - Model params updated
- [ ] `test_loss_computed()` - Loss is calculated
- [ ] `test_gradient_flow()` - Gradients propagate to all layers
- [ ] `test_max_steps_respected()` - Training stops at max_steps

### WU-7: Evaluation
**Tests to add**:
- [ ] `test_evaluation_runs()` - Eval completes without error
- [ ] `test_perplexity_computed()` - Perplexity is a positive float
- [ ] `test_satirical_metrics_present()` - Spec-Fidelity etc. included
- [ ] `test_eval_on_validation_split()` - Can eval on different splits

### WU-8: Mode System
**Tests to add**:
- [ ] `test_mode_mock_backward_compat()` - Mock mode behaves like before
- [ ] `test_mode_trainable_generates_complete_code()` - Trainable has all components
- [ ] `test_cli_mode_flag()` - CLI accepts --mode parameter
- [ ] `test_cli_trainable_output()` - Trainable mode produces expected files

---

## Regression Testing

### Baseline Tests (Must Always Pass)

These tests protect against regressions:

1. **All existing 13 tests** must continue to pass
2. **Demo command** must work: `metagen demo`
3. **Spec validation** for all example specs must succeed
4. **Determinism** tests must pass (same spec+seed → same output)
5. **Linting** must pass: `make lint`

### Test Before Merge

Every PR must:
- [ ] Pass `make test` (all fast tests)
- [ ] Pass `make lint`
- [ ] Have >80% code coverage for new code
- [ ] Include tests for new functionality
- [ ] Not break any existing tests

---

## Continuous Integration

### GitHub Actions Pipeline

```yaml
name: CI

on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]

      - name: Lint
        run: ruff check src tests

      - name: Fast Tests
        run: pytest -m "not slow" --cov=src/metagen --cov-report=xml

      - name: Demo
        run: metagen demo

      - name: Coverage
        run: |
          pip install coverage
          coverage report --fail-under=80
```

### Nightly/Weekly Tests

Separate job for slow tests:

```yaml
name: Nightly Tests

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily

jobs:
  slow-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5

      - name: Install
        run: pip install -e .[dev,full]

      - name: Slow Tests
        run: pytest -m slow --timeout=600
```

---

## Coverage Goals

### Per-Module Targets

| Module | Target Coverage | Priority |
|--------|----------------|----------|
| `specs/schema.py` | 90% | High |
| `specs/loader.py` | 85% | High |
| `synth/architecture.py` | 90% | High |
| `synth/codegen.py` | 85% | High |
| `synth/engine.py` | 90% | High |
| `synth/benchmarks.py` | 75% | Medium |
| `synth/paper_gen.py` | 70% | Medium |
| `utils/seed.py` | 95% | High |
| `utils/io.py` | 85% | Medium |
| `cli.py` | 80% | Medium |

### Coverage Commands

```bash
# Run with coverage
pytest --cov=src/metagen --cov-report=html

# View report
open htmlcov/index.html

# Coverage summary
coverage report

# Fail if below threshold
coverage report --fail-under=80
```

---

## Performance Testing

### Benchmarks (Future)

Track performance over time:

```python
@pytest.mark.benchmark
def test_synthesis_speed(benchmark, tmp_path):
    """Benchmark full synthesis time"""
    spec = load_spec("examples/specs/text_llm_8b.yaml")

    def run_synthesis():
        synthesize(spec, tmp_path, seed=42)

    result = benchmark(run_synthesis)
    # Should complete in < 2s for mock mode
    assert result.stats.mean < 2.0
```

### Memory Profiling

For large models:

```bash
# Profile memory usage
python -m memory_profiler scripts/profile_synthesis.py

# Trace allocations
python -m tracemalloc scripts/synthesis_trace.py
```

---

## Test Data

### Example Specs

Use existing specs for testing:
- `examples/specs/text_llm_8b.yaml` - Text LLM
- `examples/specs/image_diffusion_sdxl_like.yaml` - Image diffusion
- `examples/specs/edge_tiny_agent.yaml` - Tiny model

### Synthetic Test Specs

Create minimal specs for specific scenarios:

```python
# Minimal spec for testing
MINIMAL_SPEC = {
    "metagen_version": "1.0",
    "name": "test_model",
    "modality": {"inputs": ["text"], "outputs": ["text"]}
}

# Spec with all fields
COMPLETE_SPEC = {
    "metagen_version": "1.0",
    "name": "complete_test",
    "modality": {"inputs": ["text"], "outputs": ["text"]},
    "task": {"type": "generation", "domain": "text"},
    "constraints": {
        "latency": "near-real-time",
        "device": "consumer_gpu",
        "parameter_budget": {"max": "8B"},
    },
    # ... all other fields
}
```

---

## Debugging Tests

### Useful pytest flags

```bash
# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Run specific test
pytest tests/test_architecture.py::test_blueprint_state_creation

# Debug with pdb
pytest --pdb

# Show slow tests
pytest --durations=10
```

### Test Isolation

Ensure tests don't interfere:

```python
# Use tmp_path fixture (auto-cleanup)
def test_with_temp_dir(tmp_path):
    output = tmp_path / "output"
    # tmp_path is cleaned up automatically


# Reset global state if needed
def test_with_global_state():
    import metagen.utils.seed as seed_utils
    original_seed = seed_utils._global_seed
    try:
        # Test code
        seed_utils.set_global_seed(42)
        ...
    finally:
        seed_utils._global_seed = original_seed
```

---

## Best Practices

### 1. Test Names

Use descriptive names:
- ✅ `test_blueprint_dimensions_match_code()`
- ❌ `test_bp1()`

### 2. Assertions

Use informative assertion messages:
```python
# Good
assert param_count > 0, f"Expected positive params, got {param_count}"

# Bad
assert param_count > 0
```

### 3. Test Data

Use realistic data:
```python
# Good - real spec from examples
spec = load_spec("examples/specs/text_llm_8b.yaml")

# OK - minimal valid spec
spec = ModelSpec(name="test", modality=Modality())

# Bad - invalid spec
spec = {"name": "test"}  # Missing required fields
```

### 4. DRY (Don't Repeat Yourself)

Use fixtures for common setup:
```python
@pytest.fixture
def trained_model(tmp_path):
    """Fixture providing a trained model"""
    # Setup code...
    return model
```

### 5. Independent Tests

Each test should be runnable in isolation:
```python
# Good - self-contained
def test_feature(tmp_path):
    # Create all needed data
    spec = ModelSpec(...)
    result = function(spec)
    assert result

# Bad - depends on test order
def test_step1():
    global shared_data
    shared_data = ...

def test_step2():
    # Assumes test_step1 ran first
    assert shared_data
```

---

## Summary

### Key Points

1. **Three test levels**: Unit (fast), Integration (medium), Slow (marked)
2. **Test markers**: Use `@pytest.mark.slow` for expensive tests
3. **Coverage goal**: 80%+ overall, 90%+ for core modules
4. **CI requirement**: All fast tests must pass before merge
5. **Backward compat**: Existing tests must continue passing

### Commands Reference

```bash
# Fast tests (CI default)
pytest -m "not slow"

# All tests
pytest

# With coverage
pytest --cov=src/metagen

# Lint
make lint

# Full quality check
make test && make lint
```

---

**Document Status**: Active
**Next Review**: After WU-4 completion
**Maintainer**: Mauro
