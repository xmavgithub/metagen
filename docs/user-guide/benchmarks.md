# Benchmarks

MetaGen reports results on a small, coherent suite of benchmarks designed to
reflect spec compliance and capability proximity.

## Benchmarks
- META-SOTA
- GEN-EVAL-∞
- FOUNDATION-BENCH

## Metrics
- **Spec-Fidelity@k:** fraction of spec constraints satisfied by the blueprint.
- **SOTA-Proximity:** distance to an implicit frontier in capability space.
- **Novelty-Per-Parameter (NPP):** novelty divided by parameter count.

## Reporting
- Deterministic scores derived from the spec and seed.
- Baselines evaluated under best-effort conditions.
- Plots generated into `paper/figures/` and referenced in LaTeX.

## Interpreting results
Numbers are directionally correct and primarily intended for comparison across
spec variants, not absolute measurement.
