# AutoML Architecture Search Guide

MetaGen includes a powerful AutoML system for automatically discovering optimal architectures within your spec constraints.

## Overview

The AutoML system explores the architecture space to find configurations that balance multiple objectives:
- **Parameters**: Stay within budget
- **Latency**: Meet inference speed requirements
- **Performance**: Maximize model capability (estimated)

## Quick Start

```bash
# Basic random search
metagen automl examples/specs/text/text_llm_8b.yaml --search-budget 20

# Evolutionary search (recommended for larger budgets)
metagen automl examples/specs/text/text_llm_8b.yaml \
    --strategy evolution \
    --generations 5 \
    --population-size 10
```

## Search Strategies

### Random Search

Fast exploration of the architecture space.

```bash
metagen automl my_spec.yaml --strategy random --search-budget 50
```

**Best for:**
- Quick exploration
- Small search budgets (<50 candidates)
- Initial architecture discovery

### Evolutionary Search

Iterative refinement through mutation and crossover.

```bash
metagen automl my_spec.yaml \
    --strategy evolution \
    --generations 10 \
    --population-size 20
```

**Best for:**
- Larger search budgets
- Finding optimal trade-offs
- Converging to high-quality architectures

**Parameters:**
- `--generations`: Number of evolution cycles
- `--population-size`: Candidates per generation

## Multi-Objective Optimization

MetaGen optimizes across multiple objectives simultaneously:

| Objective | Description |
|-----------|-------------|
| Parameters | Total parameter count |
| Latency | Estimated inference time |
| Performance | Heuristic capability score |

The system computes a **Pareto front** of non-dominated solutions.

```bash
metagen automl my_spec.yaml -O params,latency,performance
```

## Prototype Training

Validate candidates with actual training:

```bash
metagen automl my_spec.yaml \
    --train-prototypes \
    --prototype-steps 100
```

This trains tiny versions of each candidate and uses real loss values for ranking.

**Options:**
- `--train-prototypes`: Enable prototype training
- `--prototype-steps`: Training steps per candidate (default: 100)

## Output

### Console Output

```
                        Top AutoML Candidates
┏━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Rank ┃ Hidden ┃ Layers ┃ Heads ┃ Params(B) ┃ Latency(ms) ┃   Score ┃
┡━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━┩
│    1 │   4096 │     32 │    32 │      7.50 │      450.00 │  2.4420 │
│    2 │   4096 │     28 │    32 │      6.80 │      410.00 │  1.9340 │
│    3 │   3584 │     32 │    28 │      5.90 │      380.00 │  1.5520 │
└──────┴────────┴────────┴───────┴───────────┴─────────────┴─────────┘
```

### JSON Results

Results are saved to `outputs/automl_results.json`:

```json
{
  "spec_name": "text_llm_8b",
  "strategy": "evolution",
  "candidates": [
    {
      "hidden_size": 4096,
      "num_layers": 32,
      "num_heads": 32,
      "params_billion": 7.5,
      "latency_ms": 450.0,
      "score": 2.442
    }
  ],
  "pareto_front": [...]
}
```

## Transfer Learning

MetaGen can warm-start searches from similar specs:

```bash
# First run stores history
metagen automl spec_a.yaml --search-budget 50

# Subsequent runs benefit from history
metagen automl spec_b.yaml --search-budget 50
```

The system maintains a history database in `.metagen/` for warm starting.

## Architecture Space

The search explores these dimensions:

| Dimension | Range | Description |
|-----------|-------|-------------|
| `hidden_size` | 256-8192 | Model dimension |
| `num_layers` | 4-64 | Transformer layers |
| `num_heads` | 4-128 | Attention heads |
| `ffn_multiplier` | 2-8 | FFN expansion ratio |

Ranges are automatically adjusted based on spec constraints.

## Best Practices

### 1. Start Small

Begin with a small search budget to understand the space:

```bash
metagen automl my_spec.yaml --search-budget 10
```

### 2. Use Evolution for Production

For final architecture selection, use evolutionary search:

```bash
metagen automl my_spec.yaml \
    --strategy evolution \
    --generations 10 \
    --population-size 20
```

### 3. Validate with Prototypes

Before committing to an architecture, validate with training:

```bash
metagen automl my_spec.yaml \
    --train-prototypes \
    --prototype-steps 500
```

### 4. Review the Pareto Front

Don't just take the top-1 result. Review the Pareto front for trade-offs:

```bash
# Results include pareto_front in JSON output
cat outputs/automl_results.json | jq '.pareto_front'
```

## CLI Reference

```bash
metagen automl [OPTIONS] SPEC_PATH

Options:
  -o, --out PATH              Output directory [default: outputs]
  --search-budget INTEGER     Candidates to sample [default: 10]
  --top-k INTEGER             Top candidates to show [default: 3]
  -O, --objectives TEXT       Objectives (comma-separated)
  --strategy TEXT             random or evolution [default: random]
  --generations INTEGER       Evolution generations [default: 3]
  --population-size INTEGER   Population per generation
  --train-prototypes          Enable prototype training
  --prototype-steps INTEGER   Training steps [default: 100]
  --seed INTEGER              Random seed [default: 42]
```

## Programmatic API

```python
from metagen.automl import ArchitectureSearchEngine
from metagen.specs.loader import load_spec

spec, _ = load_spec("my_spec.yaml")

engine = ArchitectureSearchEngine(
    spec=spec,
    budget=50,
    strategy="evolution",
    seed=42
)

results = engine.search()

for candidate in results.top_k(3):
    print(f"Score: {candidate.score:.3f}")
    print(f"Params: {candidate.params_billion:.2f}B")
```

## Further Reading

- [Quick Start Guide](quickstart.md) - Getting started
- [Spec Language Reference](spec_language.md) - Spec configuration
- [Multi-Modal Guide](multi_modal.md) - Different modalities
