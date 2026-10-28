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
    n3 = R.add("note", db, text="c", supersedes=R.load(db)[-1]["id"])   # a correction of the correction
    cur = R.current(R.load(db))
    assert cur[n1]["text"] == "c" and cur[n1]["id"] == n3 and len(cur) == 1
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


def test_notes_apart(tmp_path, monkeypatch):
    """without a path, notes go to NOTES and the measurements to DB; load() reads both"""
    monkeypatch.setattr(R, "DB", str(tmp_path / "runs.jsonl")); monkeypatch.setattr(R, "NOTES", str(tmp_path / "notes.jsonl"))
    rid = R.add("run", name="z")
    R.append([R.make("note", text="why", runs=[rid]), R.make("result", run=rid, metric="m", value=1.0)])
    assert [r["type"] for r in R.load(R.DB)] == ["run", "result"]
    assert [r["type"] for r in R.load(R.NOTES)] == ["note"]
    assert len(R.load()) == 3


@pytest.mark.parametrize("code,status", [(0, "stopped"), ("bad option", "failed")])
def test_cli_records_an_exit(tmp_path, monkeypatch, code, status):
    """a SystemExit after the run record (--profile, a plug-in) leaves an end event; a normal end is not doubled"""
    from ttf import train as T
    def fake_main():
        T.RUNLOG[0] = R.RunLog(str(tmp_path), "r", {})
        T.RUNLOG[0].event("running", 0)
        raise SystemExit(code)
    monkeypatch.setattr(T, "main", fake_main)
    with pytest.raises(SystemExit):
        T.cli()
    ev = [r for r in R.load(str(tmp_path / "records.jsonl")) if r["type"] == "event"]
    assert [e["status"] for e in ev] == ["running", status]
    T.RUNLOG[0] = None


def test_cli_does_not_double_an_end(tmp_path, monkeypatch):
    from ttf import train as T
    def fake_main():
        T.RUNLOG[0] = R.RunLog(str(tmp_path), "r", {})
        T.RUNLOG[0].event("stopped", 3, "signal")
        raise SystemExit(0)
    monkeypatch.setattr(T, "main", fake_main)
    with pytest.raises(SystemExit):
        T.cli()
    assert [r["status"] for r in R.load(str(tmp_path / "records.jsonl")) if r["type"] == "event"] == ["stopped"]
    T.RUNLOG[0] = None


def test_current_skips_records_without_id():
    """a record without an id (hand-written notes, 23 Oct) is left out of current() for validate to report; it raised"""
    from ttf import results as R
    cur = R.current([{"type": "note", "text": "x"}, {"id": "a", "type": "note", "text": "y"}])
    assert list(cur) == ["a"]
