"""The results database (ttf/results.py): records, supersedes, ingest, the run log."""
import json

import pytest

from ttf import results as R


def test_make_validate_append_load(tmp_path):
    db = str(tmp_path / "runs.jsonl")
    rid = R.add("run", db, name="x", config={"a": 1})
    R.add("result", db, run=rid, metric="val_train", value=2.5)
    recs = R.load(db)
    assert [r["type"] for r in recs] == ["run", "result"] and recs[1]["run"] == rid
    with pytest.raises(ValueError):
        R.make("nonsense")


def test_supersedes_and_ingest(tmp_path):
    db, other = str(tmp_path / "db.jsonl"), str(tmp_path / "rec.jsonl")
    n1 = R.add("note", db, text="a")
    R.add("note", db, text="b", supersedes=n1)
    assert R.current(R.load(db))[n1]["text"] == "b"
    rid = R.add("run", other, name="y")
    assert R.ingest([other], db) == (1, 0)
    assert R.ingest([other], db) == (0, 1)          # idempotent
    assert rid in {r["id"] for r in R.load(db)}


def test_damaged_line_skipped(tmp_path):
    db = tmp_path / "db.jsonl"
    db.write_text(json.dumps(R.make("note", text="ok")) + "\n{broken\n")
    assert len(R.load(str(db))) == 1


def test_runlog(tmp_path):
    lg = R.RunLog(str(tmp_path), "run1", {"lr": 1e-3}, steps=10)
    lg.event("running", 0); lg.result("val_train", 3.0, step=10); lg.event("finished", 10)
    recs = R.load(str(tmp_path / "records.jsonl"))
    assert [r["type"] for r in recs] == ["run", "event", "result", "event"]
    assert R.run_of(str(tmp_path))[0] == lg.id
