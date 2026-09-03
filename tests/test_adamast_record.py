"""AdaMAST record shape and the local pre-flight validator. FREE: no model calls."""

from __future__ import annotations

from gepa_taxonomy.adamast_trace import REQUIRED_FIELDS, AdamastRecord, validate, write_jsonl


def _rec(pid: str, trajectory: str = "[STEP 1]\nhello") -> AdamastRecord:
    return AdamastRecord(problem_id=pid, task="t", raw_trajectory=trajectory, metadata={"score": 0.0})


def test_record_carries_the_required_fields():
    d = _rec("a").to_dict()
    assert all(d.get(f) for f in REQUIRED_FIELDS)
    assert d["metadata"] == {"score": 0.0}


def test_written_file_passes_validation(tmp_path):
    p = write_jsonl([_rec("a"), _rec("b")], tmp_path / "traces.jsonl")
    assert validate(p)["ok"]


def test_validation_catches_empty_trajectories_and_duplicates(tmp_path):
    p = write_jsonl([_rec("a", "   "), _rec("b"), _rec("b"), _rec("", "x")], tmp_path / "bad.jsonl")
    report = validate(p)
    assert not report["ok"]
    assert report["empty_trajectories"] == 1
    assert report["duplicate_problem_ids"] == 1
    assert report["missing_required_fields"] == 1
