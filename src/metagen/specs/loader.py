from __future__ import annotations

import json
from pathlib import Path

import yaml

from metagen.specs.schema import ModelSpec, validate_spec
from metagen.utils.io import ensure_dir, write_yaml


def load_raw_spec(path: Path) -> tuple[dict, str]:
    """Load spec file (YAML or JSON) and return dict and raw text."""
    text = Path(path).read_text(encoding="utf-8")
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported spec format: {suffix}")
    return data, text


def load_spec(path: Path) -> tuple[ModelSpec, str]:
    """Load and validate a spec, returning the model and raw text."""
    data, raw = load_raw_spec(path)
    spec = validate_spec(data)
    return spec, raw


def dump_resolved_spec(spec: ModelSpec, out_path: Path) -> None:
    """Write resolved spec to YAML."""
    ensure_dir(out_path.parent)
    write_yaml(out_path, spec.model_dump())


def serialize_spec(spec: ModelSpec) -> str:
    """Serialize spec deterministically for hashing."""
    return yaml.safe_dump(spec.model_dump(), sort_keys=True)
