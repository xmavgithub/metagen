# Tutorial 3: Architecture Search with AutoML

Learn to use MetaGen's AutoML system to discover optimal architectures.

## Prerequisites

- Completed [Tutorial 1](01_first_synthesis.md) and [Tutorial 2](02_custom_spec.md)
- A spec file to optimize

## Step 1: Quick Random Search

Start with a fast exploration:

```bash
metagen automl examples/specs/text/text_llm_8b.yaml --search-budget 10
```

You'll see a table of top candidates:

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

## Step 2: Understand the Output

Results are saved to `outputs/automl_results.json`:

```bash
cat outputs/automl_results.json | python -m json.tool | head -30
```

Key fields:
- `hidden_size`: Model dimension
- `num_layers`: Transformer layers
- `num_heads`: Attention heads
- `params_billion`: Total parameters
- `latency_ms`: Estimated inference latency
- `score`: Combined objective score

## Step 3: Increase Search Budget

More candidates = better exploration:

```bash
metagen automl examples/specs/text/text_llm_8b.yaml --search-budget 50
```

## Step 4: Use Evolutionary Search

For better results, use evolution:

```bash
metagen automl examples/specs/text/text_llm_8b.yaml \
    --strategy evolution \
    --generations 5 \
    --population-size 10
```

This runs 5 generations of:
1. Evaluate population
2. Select best parents
3. Mutate and crossover
4. Repeat

## Step 5: Show More Candidates

Display top 10 instead of top 3:

```bash
metagen automl examples/specs/text/text_llm_8b.yaml \
    --search-budget 30 \
    --top-k 10
```

## Step 6: Validate with Prototype Training

Train tiny models to get real loss values:

```bash
metagen automl examples/specs/text/text_llm_8b.yaml \
    --train-prototypes \
    --prototype-steps 100
```

This takes longer but gives more reliable rankings.

## Step 7: Analyze the Pareto Front

The search finds Pareto-optimal trade-offs:

```bash
cat outputs/automl_results.json | python -c "
import json, sys
data = json.load(sys.stdin)
print('Pareto Front:')
for c in data.get('pareto_front', [])[:5]:
    print(f\"  {c['params_billion']:.1f}B params, {c['latency_ms']:.0f}ms latency\")
"
```

## Step 8: Apply the Best Architecture

Take the best candidate and update your spec:

```yaml
# my_optimized_spec.yaml
metagen_version: "1.0"
name: "optimized_model"
modality:
  inputs: ["text"]
  outputs: ["text"]
constraints:
  parameter_budget:
    max: "7.5B"  # From search results
architecture:
  family: "transformer"
# Note: MetaGen will derive dimensions from constraints
```

Then synthesize:

```bash
metagen synth my_optimized_spec.yaml --out optimized_output/
```

## Step 9: Compare Strategies

Run both strategies and compare:

```bash
# Random search
metagen automl my_spec.yaml \
    --strategy random \
    --search-budget 50 \
    --out outputs/random/

# Evolutionary search
metagen automl my_spec.yaml \
    --strategy evolution \
    --generations 10 \
    --population-size 10 \
    --out outputs/evolution/
```

## Step 10: Iterate

AutoML benefits from iteration:

1. Run initial search
2. Analyze results
3. Adjust constraints
4. Search again with refined spec

```bash
# First pass: wide exploration
metagen automl spec.yaml --search-budget 100

# Refine spec based on results, then:
metagen automl spec_v2.yaml --strategy evolution --generations 20
```

## Tips for Effective Search

### 1. Start Broad, Then Narrow

Begin with a large parameter budget and generous constraints, then tighten:

```yaml
# Round 1: Explore
constraints:
  parameter_budget:
    max: "20B"

# Round 2: Constrain (after seeing results)
constraints:
  parameter_budget:
    max: "8B"
```

### 2. Use Appropriate Search Budget

| Goal | Budget | Strategy |
|------|--------|----------|
| Quick check | 10-20 | random |
| Exploration | 50-100 | random |
| Optimization | 100+ | evolution |

### 3. Consider All Objectives

Review the Pareto front, not just top-1:

```bash
# Show all Pareto-optimal candidates
cat outputs/automl_results.json | jq '.pareto_front'
```

### 4. Validate Important Decisions

For production architectures, use prototype training:

```bash
metagen automl spec.yaml \
    --train-prototypes \
    --prototype-steps 500
```

## Common Workflows

### Finding Minimal Architecture

```bash
# Search for smallest model meeting constraints
metagen automl spec.yaml \
    --search-budget 100 \
    -O params  # Optimize for parameters
```

### Finding Fastest Architecture

```bash
# Optimize for latency
metagen automl spec.yaml \
    --search-budget 100 \
    -O latency
```

### Balanced Search

```bash
# Default: balance all objectives
metagen automl spec.yaml \
    --strategy evolution \
    --generations 10
```

## What You Learned

- Running random and evolutionary search
- Interpreting AutoML results
- Using prototype training for validation
- Analyzing Pareto fronts
- Iterating on architecture search

## Next Steps

- [AutoML Guide](../user-guide/automl_guide.md) - Complete reference
- [Multi-Modal Guide](../user-guide/multi_modal.md) - Different modalities
