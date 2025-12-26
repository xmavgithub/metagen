import json
from pathlib import Path

from metagen.specs.loader import load_spec


def test_load_yaml(tmp_path: Path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("name: yaml_spec\n", encoding="utf-8")
    spec, raw = load_spec(spec_path)
    assert spec.name == "yaml_spec"
    assert "yaml_spec" in raw


def test_load_json(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"name": "json_spec"}), encoding="utf-8")
    spec, raw = load_spec(spec_path)
    assert spec.name == "json_spec"
    assert "json_spec" in raw
