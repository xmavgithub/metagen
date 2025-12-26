from metagen.utils.seed import derive_seed


def test_derive_seed_deterministic():
    assert derive_seed("spec", 42) == derive_seed("spec", 42)


def test_derive_seed_changes_with_content():
    assert derive_seed("spec-a", 42) != derive_seed("spec-b", 42)
