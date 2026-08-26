"""Tests for the trace-harvest half of the stage boundary."""

from __future__ import annotations

import json

import pytest

from failure_taxonomy import harvest_traces, trace_report, write_generation_traces


class _Batch:
    def __init__(self, trajectories, scores=None):
        self.trajectories = trajectories
        self.scores = scores or [0.0] * len(trajectories)


def _traj(instance_id, components=("solver", "refiner")):
    return {
        "instance_id": instance_id,
        "task": "do the thing",
        "module_calls": [{"component": c, "prompt": f"{c} prompt", "output": f"{c} output"} for c in components],
    }


def test_harvest_uses_instance_ids_from_trajectories():
    traces = harvest_traces(_Batch([_traj("a"), _traj("b")]))
    assert [t.trace_id for t in traces] == ["a", "b"]


def test_scores_land_in_metadata_not_in_the_trajectory():
    """A generator shown the outcome writes codes about being wrong rather than
    about observable behaviour, and those cannot be judged at optimization time."""
    traces = harvest_traces(_Batch([_traj("a")], scores=[1.0]))
    assert traces[0].metadata["score"] == 1.0
    assert "1.0" not in traces[0].render()
    assert "score" not in traces[0].render().lower()


def test_generation_record_names_the_real_components():
    traces = harvest_traces(_Batch([_traj("a")]))
    record = traces[0].to_generation_record()
    assert record["metadata"]["components"] == ["solver", "refiner"]
    assert "[COMPONENT: solver]" in record["raw_trajectory"]


def test_write_round_trips_as_jsonl(tmp_path):
    path = write_generation_traces(harvest_traces(_Batch([_traj("a"), _traj("b")])), tmp_path / "t.jsonl")
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["problem_id"] == "a"


def test_empty_trajectory_is_refused_rather_than_written(tmp_path):
    """Discovering this generator-side, after paying for the rollouts, is the
    expensive way to find out."""
    traces = harvest_traces(_Batch([{}]))
    with pytest.raises(ValueError, match="empty trajectory"):
        write_generation_traces(traces, tmp_path / "t.jsonl")


def test_unsegmented_dict_trajectory_renders_as_readable_json():
    """Adapters commonly store a trajectory as a plain dict. Python's repr of a
    dict is materially worse for a judge to read than indented JSON."""
    traces = harvest_traces(_Batch([{"instance_id": "a", "answer": "42", "steps": ["one", "two"]}]))
    rendered = traces[0].render()
    assert '"answer": "42"' in rendered
    assert "'answer'" not in rendered


def test_report_flags_unsegmented_traces():
    batch = _Batch([_traj("a"), {"instance_id": "b", "text": "no structure"}])
    report = trace_report(harvest_traces(batch))
    assert report["traces"] == 2
    assert report["segmented"] == 1
    assert report["unsegmented"] == 1
    assert report["components"] == {"refiner": 1, "solver": 1}
