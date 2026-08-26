"""Arming the optimize_anything evaluator. FREE: no network, no LLM.

The three arms must differ in exactly one respect -- what feedback reaches
reflection -- and be identical in every other respect that touches the search.
Most of these tests exist to pin that "every other respect" down, because an
arm that also perturbs candidate selection would produce a difference that
looks like a feedback effect and is not.
"""

from __future__ import annotations

import inspect
import json

import pytest

from failure_taxonomy.judge import Occurrence
from gepa_taxonomy.oa.asi import (
    FAILURE_MODES_KEY,
    SCORE_ONLY,
    STOCK,
    TAXONOMY,
    ArmedEvaluator,
    TraceSink,
    trace_from_side_info,
)

STOCK_SIDE_INFO = {
    "scores": {"sum_radii": 2.61},
    "metrics": {"mean": 2.4, "max": 2.61},
    "code": "def main(): ...",
    "circles": [[0.5, 0.5, 0.1]],
    "stdout": "iter 1 score 2.4",
    "error": None,
    "traceback": None,
    "validation_details": {"sum_radii": 2.61, "valid": True},
}


def inner(candidate, opt_state=None):
    return 2.61, dict(STOCK_SIDE_INFO)


class FakeJudge:
    """Returns one occurrence per trace, and records what it was shown."""

    def __init__(self, occurrences=None, explode=False):
        self.explode = explode
        self.seen = []
        self._occurrences = occurrences

    def judge(self, traces):
        if self.explode:
            raise RuntimeError("judge unavailable")
        self.seen.extend(traces)
        found = self._occurrences
        if found is None:
            found = [Occurrence(code="A.1", name="Overlapping_Circles", evidence="valid: True")]
        return {t.trace_id: list(found) for t in traces}


class TestStockArm:
    def test_side_info_passes_through_untouched(self):
        armed = ArmedEvaluator(inner=inner, arm=STOCK)
        _, side_info = armed("code")
        assert side_info == STOCK_SIDE_INFO, "the baseline arm must be their evaluator, verbatim"

    def test_writes_one_trace_per_evaluation(self, tmp_path):
        sink = TraceSink(path=tmp_path / "traces.jsonl")
        armed = ArmedEvaluator(inner=inner, arm=STOCK, sink=sink)
        for _ in range(3):
            armed("code")
        lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        assert {json.loads(line)["problem_id"] for line in lines} == {"eval-00000", "eval-00001", "eval-00002"}

    def test_needs_no_judge(self):
        """The arm that harvests traces cannot depend on a taxonomy that does
        not exist yet -- it is what the taxonomy is generated from."""
        ArmedEvaluator(inner=inner, arm=STOCK)

    def test_written_records_match_the_format_adamast_already_consumed(self, tmp_path):
        """Pinned against results/base_val/base_val.traces.jsonl, the bundle a
        taxonomy was successfully generated from. A format mismatch here is only
        discoverable after paying for the whole stock run."""
        sink = TraceSink(path=tmp_path / "t.jsonl")
        ArmedEvaluator(inner=inner, arm=STOCK, sink=sink, task_text="pack circles")("code")
        rec = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8").strip())
        assert sorted(rec) == ["metadata", "problem_id", "raw_trajectory", "task"]
        assert "components" in rec["metadata"]
        assert rec["metadata"]["score"] == 2.61, "generation benefits from knowing which runs went badly"
        assert rec["task"] == "pack circles"
        assert rec["raw_trajectory"]


class TestTaxonomyArm:
    def test_replaces_rather_than_enriches(self):
        armed = ArmedEvaluator(inner=inner, arm=TAXONOMY, judge=FakeJudge())
        _, side_info = armed("code")
        assert FAILURE_MODES_KEY in side_info
        for dropped in ("metrics", "code", "circles", "stdout", "traceback", "validation_details"):
            assert dropped not in side_info, f"{dropped} survived; this is the enrichment design, not replacement"

    def test_occurrences_carry_name_and_evidence_but_not_the_code_id(self):
        armed = ArmedEvaluator(inner=inner, arm=TAXONOMY, judge=FakeJudge())
        _, side_info = armed("code")
        assert side_info[FAILURE_MODES_KEY] == [{"name": "Overlapping_Circles", "evidence": "valid: True"}]

    def test_a_clean_trace_yields_no_failure_modes_key(self):
        armed = ArmedEvaluator(inner=inner, arm=TAXONOMY, judge=FakeJudge(occurrences=[]))
        _, side_info = armed("code")
        assert FAILURE_MODES_KEY not in side_info, "an empty list would tell reflection 'diagnosed, nothing found'"
        assert side_info == {"scores": {"sum_radii": 2.61}}

    def test_judge_failure_is_soft_and_counted(self):
        armed = ArmedEvaluator(inner=inner, arm=TAXONOMY, judge=FakeJudge(explode=True))
        score, side_info = armed("code")
        assert score == 2.61, "a lost diagnosis must never cost the evaluation"
        assert side_info == {"scores": {"sum_radii": 2.61}}
        assert armed.summary()["judge_failures"] == 1

    def test_refuses_to_run_without_a_judge(self):
        with pytest.raises(ValueError, match="needs a judge"):
            ArmedEvaluator(inner=inner, arm=TAXONOMY)

    def test_judge_is_not_handed_a_labelled_score_block(self):
        """The judge should diagnose from evidence, not read off the verdict.
        What is enforceable is that the dedicated `scores` field is not
        rendered as its own section."""
        judge = FakeJudge()
        armed = ArmedEvaluator(inner=inner, arm=TAXONOMY, judge=judge)
        armed("code")
        assert "[SCORES]" not in judge.seen[0].render()

    def test_score_values_can_still_leak_via_diagnostics(self):
        """Documented, not fixed. `validation_details` embeds `sum_radii` and
        `metrics` carries its running mean -- both are hand-authored fields
        whose diagnostic content is exactly what the judge needs. Stripping
        score-shaped values out of arbitrary nested payloads is not possible in
        general, and blinding only our arm would handicap it against an ASI
        baseline that shows the score prominently. This test exists so the leak
        is a recorded property rather than a surprise in the writeup.
        """
        judge = FakeJudge()
        armed = ArmedEvaluator(inner=inner, arm=TAXONOMY, judge=judge)
        armed("code")
        assert "sum_radii" in judge.seen[0].render()


class TestScoreOnlyArm:
    def test_keeps_only_the_preserved_keys(self):
        armed = ArmedEvaluator(inner=inner, arm=SCORE_ONLY)
        _, side_info = armed("code")
        assert side_info == {"scores": {"sum_radii": 2.61}}


class TestSearchIsUnperturbed:
    @pytest.mark.parametrize("arm", [STOCK, TAXONOMY, SCORE_ONLY])
    def test_scores_survive_every_arm(self, arm):
        """Under frontier_type='objective' the Pareto frontier is built from
        side_info['scores']. Drop it in one arm and the arms differ in their
        SEARCH, not their feedback, and the comparison means nothing."""
        judge = FakeJudge() if arm == TAXONOMY else None
        armed = ArmedEvaluator(inner=inner, arm=arm, judge=judge)
        _, side_info = armed("code")
        assert side_info["scores"] == {"sum_radii": 2.61}

    @pytest.mark.parametrize("arm", [STOCK, TAXONOMY, SCORE_ONLY])
    def test_score_is_identical_across_arms(self, arm):
        judge = FakeJudge() if arm == TAXONOMY else None
        armed = ArmedEvaluator(inner=inner, arm=arm, judge=judge)
        score, _ = armed("code")
        assert score == 2.61


class TestSignatureContract:
    def test_wrapper_advertises_the_inner_signature(self):
        """gepa inspects the evaluator to decide which kwargs to pass. A
        wrapper that advertised **kwargs would be handed `example` even in
        single-task mode, where the inner evaluator takes no such argument."""
        armed = ArmedEvaluator(inner=inner, arm=STOCK)
        assert list(inspect.signature(armed).parameters) == ["candidate", "opt_state"]

    def test_unaccepted_kwargs_are_not_forwarded(self):
        def strict(candidate):
            return 1.0, {"scores": {"x": 1.0}}

        armed = ArmedEvaluator(inner=strict, arm=STOCK)
        score, _ = armed("code", example={"id": 1}, opt_state=None)
        assert score == 1.0

    def test_a_score_only_return_is_tolerated(self):
        def bare(candidate):
            return 0.5

        armed = ArmedEvaluator(inner=bare, arm=SCORE_ONLY)
        assert armed("code") == (0.5, {})


class TestTraceRendering:
    def test_unserialisable_values_do_not_lose_the_trace(self):
        """numpy arrays and live objects appear in real side_info. A raising
        renderer would drop the trace silently, and a dropped trace is a
        diagnosis that never happens."""

        class Hostile:
            def __repr__(self):
                return "<hostile>"

        trace = trace_from_side_info({"thing": Hostile()}, trace_id="t1")
        assert "hostile" in trace.render()

    def test_trace_id_and_task_are_not_rendered_as_evidence(self):
        trace = trace_from_side_info({"stdout": "ok"}, trace_id="eval-00007", task="pack circles")
        rendered = trace.render()
        assert "eval-00007" not in rendered
        assert rendered.count("pack circles") == 1, "task belongs in [TASK], not duplicated into the body"

    def test_long_values_are_truncated(self):
        trace = trace_from_side_info({"stdout": "x" * 50_000}, trace_id="t1")
        assert "[truncated]" in trace.render()
        assert len(trace.render()) < 10_000
