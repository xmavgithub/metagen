# Architecture Synthesis

MetaGen maps specifications to a structured architecture blueprint. The system
uses a deterministic rule set to select families, blocks, and resource estimates.

## Families
- Transformer
- Diffusion
- Hybrid (cross-modal or multi-objective)
- Other (fallback when constraints conflict)

## Components
Each blueprint includes:
- SpecEncoder
- ModelLatent
- ArchitectureSynth
- LossComposer
- PaperHead

## Parameter estimation
A lightweight formula estimates parameter count and memory footprint. These
estimates are used consistently across the blueprint, paper, and reports.

## Outputs
- `blueprint/architecture.yaml` with topology details
- `blueprint/graph.json` for component flow
- `blueprint/params_estimate.json` with budgets and notes

## Implementation Details
The architecture logic is encapsulated in a deterministic `BlueprintState` object. This frozen dataclass serves as the single source of truth for all model dimensions (hidden size, layers, heads) throughout the pipeline, ensuring that the generated code, benchmarks, and papers are mathematically consistent.
