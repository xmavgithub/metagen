from metagen.specs.schema import validate_spec


def test_validate_minimal_spec():
    spec = validate_spec({"name": "minimal"})
    assert spec.name == "minimal"
    assert spec.metagen_version == "1.0"


def test_warning_unsupported_modality():
    spec = validate_spec({"name": "tastey", "modality": {"outputs": ["taste"]}})
    assert any("Unsupported modalities" in w for w in spec.warnings)


def test_warning_infinite_context():
    spec = validate_spec({"name": "infinite", "constraints": {"context_window": "∞"}})
    assert any("Infinite context" in w for w in spec.warnings)
