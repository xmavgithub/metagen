# Tutorial 2: Creating Custom Specs

Learn to create your own MetaGen specifications from scratch.

## Prerequisites

- Completed [Tutorial 1](01_first_synthesis.md)
- Text editor

## Step 1: Minimal Spec

Create a file `my_spec.yaml`:

```yaml
metagen_version: "1.0"
name: "my_first_model"
modality:
  inputs: ["text"]
  outputs: ["text"]
```

This is the minimal valid spec. MetaGen fills defaults for everything else.

## Step 2: Synthesize the Minimal Spec

```bash
metagen synth my_spec.yaml --out minimal_output/
```

Examine the blueprint to see what defaults were applied:

```bash
cat minimal_output/run-*/blueprint/architecture.yaml
```

## Step 3: Add Constraints

Update your spec with constraints:

```yaml
metagen_version: "1.0"
name: "my_constrained_model"
description: "A small, fast text model."
modality:
  inputs: ["text"]
  outputs: ["text"]
constraints:
  parameter_budget:
    max: "500M"
  latency: "real-time"
  device: "consumer_gpu"
  memory_budget: "8GB"
  context_window: "32k"
```

Synthesize and compare:

```bash
metagen synth my_spec.yaml --out constrained_output/
```

The architecture will be smaller to fit the 500M parameter budget.

## Step 4: Specify Training

Add training configuration:

```yaml
metagen_version: "1.0"
name: "my_trained_model"
modality:
  inputs: ["text"]
  outputs: ["text"]
constraints:
  parameter_budget:
    max: "1B"
training:
  objective: ["autoregressive"]
  data:
    sources: ["synthetic", "licensed"]
    size: "100B tokens"
    governance:
      pii: "filtered"
      copyright: "licensed"
  compute:
    hardware: "8xA100"
    duration: "5 days"
```

## Step 5: Create an Image Model

Try a different modality:

```yaml
metagen_version: "1.0"
name: "my_image_model"
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

This creates a Vision Transformer (ViT) architecture.

## Step 6: Create a Diffusion Model

For image generation:

```yaml
metagen_version: "1.0"
name: "my_diffusion_model"
modality:
  inputs: ["text", "image"]
  outputs: ["image"]
constraints:
  parameter_budget:
    max: "2B"
  memory_budget: "24GB"
training:
  objective: ["diffusion"]
architecture:
  family: "diffusion"
```

## Step 7: Create a Multi-Modal Model

CLIP-style contrastive learning:

```yaml
metagen_version: "1.0"
name: "my_clip_model"
modality:
  inputs: ["text", "image"]
  outputs: ["text", "image"]
constraints:
  parameter_budget:
    max: "400M"
training:
  objective: ["contrastive"]
architecture:
  family: "transformer"
```

## Step 8: Validate Your Spec

Check for errors before synthesis:

```bash
metagen validate my_spec.yaml
```

## Common Patterns

### Edge Deployment

```yaml
constraints:
  parameter_budget:
    max: "50M"
  latency: "real-time"
  device: "edge"
  memory_budget: "512MB"
```

### Large Language Model

```yaml
constraints:
  parameter_budget:
    max: "70B"
  latency: "batch"
  device: "datacenter_gpu"
  context_window: "128k"
training:
  objective: ["autoregressive"]
  compute:
    hardware: "512xH100"
    duration: "30 days"
```

### Audio Generation

```yaml
modality:
  inputs: ["audio"]
  outputs: ["audio"]
training:
  objective: ["autoregressive"]
architecture:
  family: "transformer"
```

## Best Practices

1. **Start minimal** - Only add what differs from defaults
2. **Use descriptive names** - They appear in all artifacts
3. **Match objective to architecture** - diffusion+diffusion, autoregressive+transformer
4. **Be realistic** - Constraints affect architecture sizing

## Troubleshooting

### "Validation Error"

Check your YAML syntax:

```bash
python -c "import yaml; yaml.safe_load(open('my_spec.yaml'))"
```

### "Unsupported modality warning"

Some modalities like `taste` are intentionally unsupported. MetaGen will generate best-effort output with a warning.

## What You Learned

- Creating specs from scratch
- Setting constraints
- Configuring training
- Different modality patterns

## Next Steps

- [Tutorial 3: Architecture Search](03_architecture_search.md)
- [Spec Language Reference](../user-guide/spec_language.md)
