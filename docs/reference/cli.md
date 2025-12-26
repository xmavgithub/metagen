# CLI Reference

Complete reference for all MetaGen commands and options.

## Global Usage

```bash
metagen [OPTIONS] COMMAND [ARGS]
```

## Commands

### synth

Synthesize a complete release package from a specification.

```bash
metagen synth [OPTIONS] SPEC_PATH
```

**Arguments:**
- `SPEC_PATH` - Path to model spec YAML/JSON (required)

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --out` | PATH | `outputs` | Output directory |
| `--run-id` | TEXT | auto | Custom run identifier |
| `--seed` | INTEGER | 42 | Base seed for determinism |

**Examples:**

```bash
# Basic synthesis
metagen synth examples/specs/text_llm_8b.yaml

# Custom output directory
metagen synth my_spec.yaml --out my_outputs/

# Specific seed for reproducibility
metagen synth my_spec.yaml --seed 123
```

**Output:**
Creates a directory with blueprint, code, docs, paper, and logs.

---

### demo

Run demo synthesis on bundled example specs.

```bash
metagen demo
```

**Options:** None

**Output:**
Synthesizes packages for multiple example specs and displays summaries.

---

### paper

Generate only the LaTeX paper project (without full synthesis).

```bash
metagen paper [OPTIONS] SPEC_PATH
```

**Arguments:**
- `SPEC_PATH` - Path to model spec (required)

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --out` | PATH | `paper` | Paper output directory |
| `--seed` | INTEGER | 42 | Base seed |

**Examples:**

```bash
# Generate paper
metagen paper examples/specs/text_llm_8b.yaml --out paper/

# Build PDF (requires LaTeX)
cd paper && make pdf
```

**Output:**
- `main.tex` - Main LaTeX file
- `sections/` - Individual section files
- `figures/` - Generated plots
- `bibliography.bib` - References
- `Makefile` - Build automation

---

### automl

Run AutoML architecture search.

```bash
metagen automl [OPTIONS] SPEC_PATH
```

**Arguments:**
- `SPEC_PATH` - Path to model spec (required)

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --out` | PATH | `outputs` | Output directory |
| `--search-budget` | INTEGER | 10 | Candidates to sample |
| `--top-k` | INTEGER | 3 | Top candidates to display |
| `-O, --objectives` | TEXT | all | Objectives (comma-separated) |
| `--strategy` | TEXT | `random` | Search strategy: random, evolution |
| `--generations` | INTEGER | 3 | Evolution generations |
| `--population-size` | INTEGER | auto | Population per generation |
| `--train-prototypes` | FLAG | off | Train tiny prototypes |
| `--prototype-steps` | INTEGER | 100 | Steps per prototype |
| `--seed` | INTEGER | 42 | Random seed |

**Examples:**

```bash
# Quick random search
metagen automl my_spec.yaml --search-budget 20

# Evolutionary search
metagen automl my_spec.yaml \
    --strategy evolution \
    --generations 10 \
    --population-size 20

# With prototype training
metagen automl my_spec.yaml \
    --train-prototypes \
    --prototype-steps 200
```

**Output:**
- Console table with top candidates
- `automl_results.json` with full results

---

### validate

Validate a specification without synthesis.

```bash
metagen validate SPEC_PATH
```

**Arguments:**
- `SPEC_PATH` - Path to model spec (required)

**Examples:**

```bash
metagen validate my_spec.yaml
# Output: "Spec valid: my_model"

metagen validate invalid_spec.yaml
# Output: Validation errors
```

---

### schema

Print the JSON schema for MetaGen specifications.

```bash
metagen schema
```

**Options:** None

**Output:**
JSON schema to stdout. Pipe to file or jq for formatting:

```bash
metagen schema > schema.json
metagen schema | jq .
```

---

### train

Generate model code and run training.

```bash
metagen train [OPTIONS] SPEC_PATH
```

**Arguments:**
- `SPEC_PATH` - Path to model spec (required)

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --out` | PATH | `outputs` | Output directory |
| `--data` | PATH | required | Training data path |
| `--epochs` | INTEGER | 1 | Training epochs |
| `--batch-size` | INTEGER | 4 | Batch size |

---

## Common Patterns

### Reproducibility

Use the same seed for identical outputs:

```bash
metagen synth spec.yaml --seed 42
metagen synth spec.yaml --seed 42  # Identical output
```

### Pipeline

Full workflow from spec to PDF:

```bash
# Synthesize
metagen synth my_spec.yaml --out outputs/

# Find the run
RUN=$(ls -t outputs/ | head -1)

# Build paper PDF
cd outputs/$RUN/paper && make pdf
```

### Architecture Exploration

Iterative search workflow:

```bash
# Quick exploration
metagen automl spec.yaml --search-budget 10

# Detailed search
metagen automl spec.yaml \
    --strategy evolution \
    --generations 20 \
    --train-prototypes

# Apply best architecture
# (manually update spec based on results)
metagen synth updated_spec.yaml
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `METAGEN_SEED` | Default seed (overridden by --seed) |
| `METAGEN_OUT` | Default output directory |

## Further Reading

- [Quick Start Guide](../user-guide/quickstart.md)
- [Spec Language Reference](../user-guide/spec_language.md)
- [AutoML Guide](../user-guide/automl_guide.md)
