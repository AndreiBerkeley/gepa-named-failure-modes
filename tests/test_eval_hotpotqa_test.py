"""Tests for candidate resolution in the HotpotQA test evaluation.

This script produces the headline number, and picking the wrong candidate would
not raise -- it would quietly evaluate a different program and report it as the
result. gepa orders candidates by DISCOVERY, not by score, so "the last one" is
not "the best one", and that was the tempting fallback.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("_ev", REPO / "scripts" / "eval_hotpotqa_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_dir(tmp_path: Path, candidates, summary=None, log=None) -> Path:
    d = tmp_path / "run"
    d.mkdir(exist_ok=True)
    (d / "candidates.json").write_text(json.dumps(candidates), encoding="utf-8")
    if summary is not None:
        (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if log is not None:
        (d / "run_log.txt").write_text(log, encoding="utf-8")
    return d


CANDS = [{"final_answer": "base"}, {"final_answer": "one"}, {"final_answer": "two"}, {"final_answer": "three"}]


# -- spec parsing ----------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("base=results/runs/x", ("base", "results/runs/x", None)),
        ("base=results/runs/x#4", ("base", "results/runs/x", 4)),
        ("tax=a/b#0", ("tax", "a/b", 0)),
    ],
)
def test_spec_parsing(ev, spec, expected):
    label, path, index = ev.parse_candidate_spec(spec)
    assert (label, path.as_posix(), index) == expected


def test_a_spec_without_a_label_is_rejected(ev):
    """Labels name the arms in the paired comparison; an unlabelled one would
    make the output ambiguous about which arm is which."""
    with pytest.raises(ev.CandidateSelectionError, match="label"):
        ev.parse_candidate_spec("results/runs/x")


# -- resolution ------------------------------------------------------------


def test_explicit_index_wins(ev, tmp_path):
    d = _run_dir(tmp_path, CANDS, summary={"best_candidate_index": 1, "best_val_score": 0.9})
    candidate, index, _val = ev.resolve_candidate(d, 2)
    assert index == 2 and candidate == {"final_answer": "two"}


def test_summary_index_is_used_when_no_index_given(ev, tmp_path):
    d = _run_dir(tmp_path, CANDS, summary={"best_candidate_index": 3, "best_val_score": 0.73})
    candidate, index, val = ev.resolve_candidate(d, None)
    assert index == 3 and candidate == {"final_answer": "three"} and val == 0.73


def test_the_best_is_not_the_last(ev, tmp_path):
    """The specific bug this guards: candidate 1 is best, candidate 3 is newest."""
    d = _run_dir(tmp_path, CANDS, summary={"best_candidate_index": 1, "best_val_score": 0.8})
    _candidate, index, _val = ev.resolve_candidate(d, None)
    assert index == 1, "resolution must follow the recorded best, not the newest"


def test_falls_back_to_the_run_log_for_runs_predating_the_index(ev, tmp_path):
    """Seed 1 was launched before best_candidate_index was recorded."""
    log = (
        "Iteration 1: Selected program 0 score: 0.5427\n"
        "Iteration 2: Selected program 1 score: 0.5592\n"
        "Iteration 3: Selected program 2 score: 0.7248\n"
        "Iteration 4: Selected program 0 score: 0.5427\n"
    )
    d = _run_dir(tmp_path, CANDS, summary={"best_val_score": 0.7248}, log=log)
    _candidate, index, val = ev.resolve_candidate(d, None)
    assert index == 2 and val == pytest.approx(0.7248)


def test_refuses_rather_than_guessing_when_nothing_identifies_the_best(ev, tmp_path):
    d = _run_dir(tmp_path, CANDS, summary={"best_val_score": 0.7}, log="no scores here")
    with pytest.raises(ev.CandidateSelectionError, match="index"):
        ev.resolve_candidate(d, None)


def test_missing_summary_is_an_error_not_a_guess(ev, tmp_path):
    d = _run_dir(tmp_path, CANDS)
    with pytest.raises(ev.CandidateSelectionError, match="summary.json"):
        ev.resolve_candidate(d, None)


def test_out_of_range_index_is_rejected(ev, tmp_path):
    d = _run_dir(tmp_path, CANDS)
    with pytest.raises(ev.CandidateSelectionError, match="out of range"):
        ev.resolve_candidate(d, 99)


def test_empty_candidates_file_is_rejected(ev, tmp_path):
    d = _run_dir(tmp_path, [])
    with pytest.raises(ev.CandidateSelectionError, match="empty"):
        ev.resolve_candidate(d, 0)


def test_log_fallback_reports_the_max_not_the_last_seen(ev, tmp_path):
    """Parent selection revisits earlier candidates, so the last line in the log
    is routinely NOT the best one."""
    log = "Iteration 1: Selected program 3 score: 0.7300\nIteration 2: Selected program 0 score: 0.5427\n"
    d = _run_dir(tmp_path, CANDS, log=log)
    index, val = ev._best_from_log(d)
    assert index == 3 and val == pytest.approx(0.73)
