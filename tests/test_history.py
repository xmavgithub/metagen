import pytest

from metagen.automl.history import HistoryDatabase


def test_history_database_save_load(tmp_path) -> None:
    db = HistoryDatabase(root_dir=tmp_path)

    db.save_run(
        "spec-a",
        {"id": 1},
        {"loss": 0.8},
        timestamp="2024-01-01T00:00:00+00:00",
    )
    db.save_run(
        "spec-a",
        {"id": 2},
        {"loss": 0.4},
        timestamp="2024-01-02T00:00:00+00:00",
    )
    db.save_run(
        "spec-b",
        {"id": 3},
        {"loss": 0.1},
        timestamp="2024-01-03T00:00:00+00:00",
    )

    runs = db.load_runs("spec-a")
    assert [run.blueprint["id"] for run in runs] == [2, 1]
    assert runs[0].metrics["loss"] == 0.4

    recent = db.load_runs(limit=2)
    assert len(recent) == 2
    assert recent[0].spec_hash == "spec-b"


def test_history_database_limit_validation(tmp_path) -> None:
    db = HistoryDatabase(root_dir=tmp_path)

    with pytest.raises(ValueError):
        db.load_runs(limit=0)
