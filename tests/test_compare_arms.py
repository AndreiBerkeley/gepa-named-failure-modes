"""Paired arm comparison. FREE: synthetic score vectors, no network, no spend.

This is the script that turns six expensive runs into an answer, so its failure
modes matter more than most: a silently-wrong pairing or a p-value computed over
the wrong differences would not crash, it would just produce a confident number.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_arms.py"
spec = importlib.util.spec_from_file_location("compare_arms", SCRIPT)
compare_arms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_arms)


class TestLoadPerInstance:
    def test_reads_the_ifbench_format(self, tmp_path):
        p = tmp_path / "e.json"
        p.write_text(json.dumps({"per_instance": {"a": 1.0, "b": 0.5}}), encoding="utf-8")
        assert compare_arms.load_per_instance(p) == {"a": 1.0, "b": 0.5}

    def test_reads_the_hotpotqa_format(self, tmp_path):
        p = tmp_path / "e.json"
        p.write_text(
            json.dumps({"instances": [{"example_id": "a", "baseline-s1": 0.25}]}),
            encoding="utf-8",
        )
        assert compare_arms.load_per_instance(p) == {"a": 0.25}

    def test_a_multi_candidate_file_demands_an_explicit_label(self, tmp_path):
        """Guessing which column is 'the' score would silently compare the wrong
        pair of arms."""
        p = tmp_path / "e.json"
        p.write_text(
            json.dumps({"instances": [{"example_id": "a", "baseline": 0.2, "taxonomy": 0.9}]}),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit, match="--label"):
            compare_arms.load_per_instance(p)
        assert compare_arms.load_per_instance(p, "taxonomy") == {"a": 0.9}

    def test_a_missing_file_is_none_not_an_exception(self, tmp_path):
        assert compare_arms.load_per_instance(tmp_path / "nope.json") is None


class TestPaired:
    def test_a_clear_improvement_is_measured_and_signed_correctly(self):
        base = {f"i{k}": 0.0 for k in range(20)}
        treat = {f"i{k}": 1.0 for k in range(20)}
        r = compare_arms.paired(base, treat)
        assert r["mean_difference"] == pytest.approx(1.0)
        assert r["treatment_better"] == 20
        assert r["baseline_better"] == 0
        assert r["wilcoxon_p"] is not None and r["wilcoxon_p"] < 0.01

    def test_the_sign_convention_is_treatment_minus_baseline(self):
        """Backwards, this reports a win as a loss -- and nothing would crash."""
        r = compare_arms.paired({"a": 1.0}, {"a": 0.0})
        assert r["mean_difference"] == pytest.approx(-1.0)
        assert r["baseline_better"] == 1

    def test_identical_arms_produce_no_difference_and_no_p_value(self):
        same = {f"i{k}": 0.5 for k in range(10)}
        r = compare_arms.paired(same, dict(same))
        assert r["mean_difference"] == 0.0
        assert r["instances_differing"] == 0
        assert r["wilcoxon_p"] is None, "a p-value over all-zero differences is meaningless"

    def test_only_shared_instances_are_compared(self):
        """Pairing must be BY INSTANCE. If the two arms' vectors were zipped
        positionally instead, a single missing instance would silently misalign
        every subsequent pair."""
        r = compare_arms.paired({"a": 0.0, "b": 0.0, "c": 9.0}, {"a": 1.0, "b": 1.0})
        assert r["n"] == 2
        assert r["mean_difference"] == pytest.approx(1.0)

    def test_no_overlap_reports_nothing_rather_than_dividing_by_zero(self):
        assert compare_arms.paired({"a": 1.0}, {"b": 1.0}) == {"n": 0}

    def test_means_are_computed_over_the_shared_set_only(self):
        r = compare_arms.paired({"a": 0.0, "z": 1.0}, {"a": 0.0})
        assert r["baseline_mean"] == pytest.approx(0.0), "the unshared instance must not shift the mean"
