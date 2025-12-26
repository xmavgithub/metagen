from pathlib import Path

from metagen.synth.engine import synthesize


def test_synthesize_outputs(tmp_path: Path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("name: engine_test\n", encoding="utf-8")
    out_dir = tmp_path / "outputs"
    run_folder = synthesize(spec_path, out_dir, run_id="run1", base_seed=42)
    assert run_folder.exists()
    assert (run_folder / "spec_resolved.yaml").exists()
    assert (run_folder / "blueprint" / "architecture.yaml").exists()
    assert (run_folder / "code" / "model.py").exists()
    assert (run_folder / "docs" / "model_card.md").exists()
    assert (run_folder / "paper" / "main.tex").exists()
