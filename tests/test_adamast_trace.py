"""Traces must be generation-grade: the format AdaMAST actually ingests.

Verified against multi-agent-systems-failure-taxonomy/AdaMAST
docs/TRACE_FORMATS.md. Getting this wrong is expensive: the base-val rollouts
are paid for once, and a trace without a trajectory cannot be re-derived.
"""

from __future__ import annotations

import json

import pytest

from gepa_taxonomy.adamast_trace import (
    REQUIRED_FIELDS,
    build_record,
    render_trajectory,
    validate,
    write_jsonl,
)
from gepa_taxonomy.program import PatchFeedback, Rollout
from gepa_taxonomy.tasks import split_row
from tests.test_gold_blindness import RAW_ROW


@pytest.fixture
def instance():
    return split_row(RAW_ROW)


@pytest.fixture
def rollout():
    return Rollout(
        instance_id="django__django-11099",
        retrieved_paths=["django/contrib/auth/validators.py"],
        solver_patch="--- a/v.py\n+++ b/v.py\n@@\n-a\n+b\n",
        feedback=PatchFeedback(True, None, None, ()),
        refiner_patch="--- a/v.py\n+++ b/v.py\n@@\n-a\n+c\n",
        solver_tokens=(22219, 444),
        refiner_tokens=(22442, 200),
        cost_usd=0.0948,
        solver_prompt="SOLVER PROMPT TEXT",
        refiner_prompt="REFINER PROMPT TEXT",
    )


def test_record_has_the_required_fields(rollout, instance):
    rec = build_record(rollout=rollout, task=instance.task, score=0.0, grading={"resolved": False}).to_dict()
    for f in REQUIRED_FIELDS:
        assert rec.get(f), f"AdaMAST requires a non-empty {f!r}"


def test_trajectory_contains_the_actual_content_not_a_digest(rollout, instance):
    """The regression that motivated this module: we stored sha256 digests."""
    rec = build_record(rollout=rollout, task=instance.task, score=0.0, grading={}).to_dict()
    traj = rec["raw_trajectory"]
    assert "SOLVER PROMPT TEXT" in traj
    assert "REFINER PROMPT TEXT" in traj
    assert rollout.solver_patch in traj
    assert rollout.refiner_patch in traj
    assert "sha256" not in traj


def test_task_carries_the_problem_statement(rollout, instance):
    rec = build_record(rollout=rollout, task=instance.task, score=0.0, grading={}).to_dict()
    assert rec["task"] == instance.task.problem_statement


def test_outcome_lives_in_metadata_not_the_trajectory(rollout, instance):
    """AdaMAST's checklist: keep oracle outcomes out of what the judge reads."""
    rec = build_record(rollout=rollout, task=instance.task, score=1.0, grading={"resolved": True}).to_dict()
    assert rec["metadata"]["resolved"] is True
    traj = rec["raw_trajectory"].lower()
    assert "resolved" not in traj
    assert "fail_to_pass" not in traj


def test_trajectory_is_gold_free(rollout, instance):
    from gepa_taxonomy.tasks import assert_gold_free

    rec = build_record(rollout=rollout, task=instance.task, score=0.0, grading={}).to_dict()
    assert_gold_free(rec["raw_trajectory"], where="adamast trajectory", gold=instance.gold)


def test_rollout_without_prompts_is_rejected(instance):
    """A digest-only rollout must fail loudly, not emit an empty trajectory."""
    r = Rollout(
        instance_id="x",
        retrieved_paths=[],
        solver_patch="p",
        feedback=PatchFeedback(True, None, None, ()),
        refiner_patch="q",
        solver_tokens=(1, 1),
        refiner_tokens=(1, 1),
        cost_usd=0.0,
    )  # prompts default to ""
    with pytest.raises(ValueError, match="no solver prompt"):
        build_record(rollout=r, task=instance.task, score=0.0, grading={})


def test_metadata_carries_difficulty_for_the_taxonomy(rollout, instance):
    rec = build_record(rollout=rollout, task=instance.task, score=0.0, grading={}, difficulty="1-4 hours").to_dict()
    assert rec["metadata"]["difficulty"] == "1-4 hours"
    assert rec["metadata"]["benchmark"] == "SWE-bench_Verified"


def test_written_file_passes_local_validation(rollout, instance, tmp_path):
    recs = [build_record(rollout=rollout, task=instance.task, score=0.0, grading={})]
    p = write_jsonl(recs, tmp_path / "traces.jsonl")
    report = validate(p)
    assert report["ok"], report
    assert report["trace_count"] == 1
    assert report["empty_trajectories"] == 0


def test_validation_catches_an_empty_trajectory(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"problem_id": "a", "raw_trajectory": ""}) + "\n")
    assert validate(p)["ok"] is False


def test_validation_catches_duplicate_problem_ids(rollout, instance, tmp_path):
    recs = [
        build_record(rollout=rollout, task=instance.task, score=0.0, grading={}),
        build_record(rollout=rollout, task=instance.task, score=1.0, grading={}),
    ]
    p = write_jsonl(recs, tmp_path / "dupe.jsonl")
    r = validate(p)
    assert r["duplicate_problem_ids"] == 1 and r["ok"] is False


def test_render_is_readable_and_ordered():
    t = render_trajectory(
        solver_prompt="SP",
        solver_output="SO",
        feedback="FB",
        refiner_prompt="RP",
        refiner_output="RO",
    )
    assert t.index("SP") < t.index("SO") < t.index("FB") < t.index("RP") < t.index("RO")
