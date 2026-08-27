"""Harness-artifact enrichment: grading detail carries failing-test
names and the test-output tail, so reflection sees outcome substance.

Artifact shapes are mirrored from a real harness tree
(``results/base_val_work/logs/run_evaluation/<run_id>/gepa/<instance_id>/``:
``report.json`` with ``tests_status``, plus ``test_output.txt``), not assumed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from gepa_taxonomy.grading import (
    MAX_FAILING_TESTS,
    MODEL_TAG,
    TEST_OUTPUT_TAIL_CHARS,
    LocalDockerGrader,
)
from gepa_taxonomy.tasks import split_row
from tests.test_gold_blindness import RAW_ROW

RUN_ID = "gepa-0123456789"
IID = "django__django-11099"


def _write_artifacts(work_dir, instance_id, *, f2p_failures=(), p2p_failures=(), output=None, report_text=None):
    d = work_dir / "logs" / "run_evaluation" / RUN_ID / MODEL_TAG / instance_id
    d.mkdir(parents=True)
    if report_text is None:
        report_text = json.dumps(
            {
                instance_id: {
                    "resolved": False,
                    "tests_status": {
                        "FAIL_TO_PASS": {"success": ["test_that_already_passes"], "failure": list(f2p_failures)},
                        "PASS_TO_PASS": {"success": [], "failure": list(p2p_failures)},
                        "FAIL_TO_FAIL": {"success": [], "failure": []},
                        "PASS_TO_FAIL": {"success": [], "failure": []},
                    },
                }
            }
        )
    (d / "report.json").write_text(report_text)
    if output is not None:
        (d / "test_output.txt").write_text(output)
    return d


def test_failing_tests_and_tail_are_extracted(tmp_path):
    _write_artifacts(
        tmp_path,
        IID,
        f2p_failures=["test_a (mod.Case)", "test_b (mod.Case)"],
        p2p_failures=["test_regressed (mod.Other)"],
        output="ran everything\nAssertionError: boom\n",
    )
    detail = LocalDockerGrader(work_dir=tmp_path)._instance_substance(RUN_ID, IID)

    # Target tests first, regressions after -- both "should have passed".
    assert detail["failing_tests"] == [
        "test_a (mod.Case)",
        "test_b (mod.Case)",
        "test_regressed (mod.Other)",
    ]
    assert "AssertionError: boom" in detail["test_output_tail"]


def test_failing_tests_are_capped(tmp_path):
    _write_artifacts(tmp_path, IID, f2p_failures=[f"test_{n}" for n in range(9)])
    detail = LocalDockerGrader(work_dir=tmp_path)._instance_substance(RUN_ID, IID)
    assert detail["failing_tests"] == [f"test_{n}" for n in range(MAX_FAILING_TESTS)]


def test_tail_is_bounded_and_ansi_free(tmp_path):
    noise = "x" * 5000
    _write_artifacts(
        tmp_path,
        IID,
        f2p_failures=["test_a"],
        output=noise + "\x1b[31mFAILED\x1b[0m tests/test_mod.py::test_a\n",
    )
    detail = LocalDockerGrader(work_dir=tmp_path)._instance_substance(RUN_ID, IID)
    tail = detail["test_output_tail"]
    assert len(tail) <= TEST_OUTPUT_TAIL_CHARS
    assert "\x1b" not in tail, "ANSI escapes must be stripped, not just truncated"
    assert "FAILED" in tail


def test_missing_artifact_tree_yields_no_fields(tmp_path):
    detail = LocalDockerGrader(work_dir=tmp_path)._instance_substance(RUN_ID, IID)
    assert detail == {}


def test_each_artifact_degrades_independently(tmp_path):
    """A malformed report must not cost us the tail, and vice versa."""
    _write_artifacts(tmp_path, IID, report_text="{not json", output="AssertionError: still here\n")
    detail = LocalDockerGrader(work_dir=tmp_path)._instance_substance(RUN_ID, IID)
    assert "failing_tests" not in detail
    assert "AssertionError: still here" in detail["test_output_tail"]

    # And a report without tests_status is tolerated too.
    other = "django__django-99999"
    _write_artifacts(tmp_path, other, report_text=json.dumps({other: {"resolved": False}}))
    detail = LocalDockerGrader(work_dir=tmp_path)._instance_substance(RUN_ID, other)
    assert detail == {}


def test_grade_batch_carries_substance_into_detail(tmp_path, monkeypatch):
    """The wiring, not just the parser: grade_batch's detail must carry the
    substance. The fake harness process writes exactly what the real one
    writes -- the aggregate report plus the per-instance artifact tree."""
    inst = split_row(RAW_ROW)
    iid = inst.task.instance_id
    grader = LocalDockerGrader(work_dir=tmp_path)

    def fake_harness(cmd, **kwargs):
        run_id = cmd[cmd.index("--run_id") + 1]
        (tmp_path / f"{MODEL_TAG}.{run_id}.json").write_text(
            json.dumps({"resolved_ids": [], "error_ids": [], "empty_patch_ids": []})
        )
        d = tmp_path / "logs" / "run_evaluation" / run_id / MODEL_TAG / iid
        d.mkdir(parents=True)
        (d / "report.json").write_text(
            json.dumps(
                {
                    iid: {
                        "tests_status": {
                            "FAIL_TO_PASS": {"success": [], "failure": ["test_newline_rejected (auth.Case)"]},
                            "PASS_TO_PASS": {"success": ["test_ok"], "failure": []},
                        }
                    }
                }
            )
        )
        (d / "test_output.txt").write_text("AssertionError: trailing newline accepted\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("gepa_taxonomy.grading.subprocess.run", fake_harness)
    out = grader.grade_batch([(inst.task, inst.gold, "--- a/x.py\n+++ b/x.py\n@@\n-a\n+b\n")])

    score, detail = out[iid]
    assert score == 0.0
    assert detail["resolved"] is False
    assert detail["failing_tests"] == ["test_newline_rejected (auth.Case)"]
    assert "AssertionError: trailing newline accepted" in detail["test_output_tail"]
    assert json.dumps(detail), "detail must stay JSON-serializable"
