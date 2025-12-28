# Tutorial 1: Your First Synthesis

In this tutorial, you'll synthesize a complete model release package from a specification.

## Prerequisites

- MetaGen installed (`pip install -e .`)
- Terminal access

## Step 1: Explore Example Specs

MetaGen includes several example specifications:

```bash
ls examples/specs/
```

You'll see specs for different modalities and architectures.

## Step 2: Examine a Spec

Let's look at a text LLM specification:

```bash
cat examples/specs/text/text_llm_8b.yaml
```

Key sections:
- `modality`: What the model processes (text→text)
- `constraints`: Parameter budget, latency, device
- `training`: Objectives and compute
- `architecture`: Model family

## Step 3: Run Synthesis

Generate a complete release package:

```bash
metagen synth examples/specs/text/text_llm_8b.yaml --out my_first_run/
```

You'll see a summary table with:
- Spec name
- Seed used
- Estimated parameters
- Benchmark scores

## Step 4: Explore the Output

```bash
ls my_first_run/run-*/
```

The output includes:

### Blueprint (`blueprint/`)

```bash
cat my_first_run/run-*/blueprint/architecture.yaml
```

Architecture configuration with dimensions, layers, and components.

### Code (`code/`)

```bash
ls my_first_run/run-*/code/
```

PyTorch skeleton files:
- `model.py` - Model definition
- `train.py` - Training loop
- `data.py` - Data loading
- `eval.py` - Evaluation
- `infer.py` - Inference

### Documentation (`docs/`)

```bash
cat my_first_run/run-*/docs/model_card.md
```

Includes model card, data card, and limitations.

### Paper (`paper/`)

```bash
ls my_first_run/run-*/paper/
```

Complete LaTeX project ready for compilation.

## Step 5: Build the Paper (Optional)

If you have LaTeX installed:

```bash
cd my_first_run/run-*/paper
make pdf
```

This generates a publication-ready PDF.

## Step 6: Verify Determinism

Run synthesis again with the same seed:

```bash
metagen synth examples/specs/text/text_llm_8b.yaml --out second_run/ --seed 42
```

Compare outputs:

```bash
diff my_first_run/run-*/blueprint/architecture.yaml \
     second_run/run-*/blueprint/architecture.yaml
```

The outputs should be identical.

## What You Learned

- How to synthesize from a spec
- Structure of the output package
- MetaGen's deterministic behavior

## Next Steps

- [Tutorial 2: Creating Custom Specs](02_custom_spec.md)
- [Tutorial 3: Architecture Search](03_architecture_search.md)
