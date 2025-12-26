from pathlib import Path

from typer.testing import CliRunner

from metagen.cli import app

runner = CliRunner()


def test_cli_schema():
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0
    assert "ModelSpec" in result.stdout or "model" in result.stdout.lower()


def test_cli_validate(tmp_path: Path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("name: cli_spec\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(spec_path)])
    assert result.exit_code == 0
    assert "Spec valid" in result.stdout


def test_cli_paper(tmp_path: Path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("name: paper_spec\n", encoding="utf-8")
    out_dir = tmp_path / "paper"
    result = runner.invoke(app, ["paper", str(spec_path), "--out", str(out_dir)])
    assert result.exit_code == 0
    assert (out_dir / "main.tex").exists()
    assert (out_dir / "figures" / "pipeline.pdf").exists()


def test_cli_automl(tmp_path: Path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("name: automl_spec\n", encoding="utf-8")
    out_dir = tmp_path / "automl"

    result = runner.invoke(
        app,
        [
            "automl",
            str(spec_path),
            "--out",
            str(out_dir),
            "--search-budget",
            "3",
            "--top-k",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert (out_dir / "automl_results.json").exists()
    assert "AutoML search complete" in result.stdout
