# Paper Generation

MetaGen produces a full LaTeX paper project with figures, tables, and references.

## Structure
- `main.tex`
- `sections/` (abstract, intro, method, experiments, limitations, ethics)
- `figures/` (pipeline, ablation)
- `bibliography.bib`

## Build
From the repo root:

```sh
make paper
```

Or directly:

```sh
make -C paper pdf
```

## Notes
The paper is deterministic given the spec and seed, and references the same
parameter estimates and benchmark results as the generated reports.
