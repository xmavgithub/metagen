# Contributing

Thanks for your interest in MetaGen. Contributions are welcome.

## Development setup

```sh
make setup
```

Or manually:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Running checks

```sh
make lint
make test
make demo
```

## Style
- Python 3.11+
- Lint: `ruff`
- Tests: `pytest`

## Pull requests
- Keep changes focused and well documented.
- Add tests for new behavior.
- Ensure `make lint` and `make test` pass.
