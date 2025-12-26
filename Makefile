.PHONY: setup lint test demo paper

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -e .[dev]

lint:
	.venv/bin/ruff check src tests

test:
	.venv/bin/pytest

demo:
	metagen demo

paper:
	metagen paper examples/specs/text_llm_8b.yaml --out paper
	make -C paper pdf
