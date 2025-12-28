# AGENTS.md

Project: MetaGen

## Overview
MetaGen is a spec-to-model synthesizer that generates release artifacts from YAML specs.
It does not train weights by default; it emits blueprints, PyTorch skeletons, training
recipes, benchmark reports, and LaTeX papers. The generated code can be trained separately.

## Key Commands
Setup:
- make setup
- pip install -e .

Tests and lint:
- make test
- pytest
- make lint
- ruff check src tests

Synthesis and docs:
- metagen demo
- metagen synth examples/specs/text/text_llm_8b.yaml --out outputs
- metagen paper examples/specs/text/text_llm_8b.yaml --out paper
- metagen validate examples/specs/text/text_llm_8b.yaml
- metagen schema

## Trainable Code Path
1) Generate code:
   metagen synth <spec.yaml> --out <out_dir>
2) Run training from the generated run folder:
   python <out_dir>/<run_id>/code/train.py --dataset auto --dataset-size 512 --epochs 1 --batch-size 4
3) Optional:
   python <out_dir>/<run_id>/code/train.py --list-datasets
   python <out_dir>/<run_id>/code/train.py --sample-data auto --sample-size 256 --epochs 1 --batch-size 4
   python <out_dir>/<run_id>/code/train.py --data <path> --epochs 1 --batch-size 4

## Determinism
Outputs are deterministic for the same spec content and base seed.

## Output Layout
<out_dir>/<run_id>/
  blueprint/   architecture.yaml, params_estimate.json, graph.json, ablations.yaml
  code/        model.py, train.py, data.py, eval.py
  docs/        model_card.md, data_card.md, eval_report.md, limitations.md, summary.json
  paper/       LaTeX project
  logs/        metagen.log

## Code Style
- Python 3.11+
- Ruff line length 100
- Import order: stdlib, third-party, first-party
