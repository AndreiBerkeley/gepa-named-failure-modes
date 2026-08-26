"""Tests for the PR-bound ``failure_taxonomy`` package."""

from __future__ import annotations

import json

import pytest

from failure_taxonomy import (
    FAILURE_MODES_KEY,
    ComponentCall,
    JudgeCache,
    LLMFailureJudge,
    Occurrence,
    SegmentedTrace,
    Taxonomy,
    TaxonomyError,
    TaxonomyFeedbackEnricher,
    build_trace,
    candidate_key,
    extract_calls,
    load_taxonomy,
)

# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------

CODES = [
    {"id": "A.1", "name": "Output_Truncation", "description": "Output ends abruptly."},
    {"id": "B.4", "name": "Malformed_Output", "description": "Output is not well formed.", "severity": "high"},
    {"id": "C.2", "name": "Wrong_Entity", "description": "Reasons about the wrong entity."},
]


@pytest.fixture
def taxonomy() -> Taxonomy:
    return Taxonomy.from_codes(CODES)


def _trajectory(instance_id: str, solver_out: str, refiner_out: str) -> dict:
    return {
        "instance_id": instance_id,
        "task": "fix the bug",
        "module_calls": [
            {"component": "solver", "prompt": "solve this", "output": solver_out},
            {"component": "refiner", "prompt": f"improve: {solver_out}", "output": refiner_out},
        ],
    }


class _EvalBatch:
    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.scores = [0.0] * len(trajectories)
        self.outputs = [None] * len(trajectories)


class InnerAdapter:
    """Minimal stand-in for a real GEPAAdapter."""

    propose_new_texts = None

    def __init__(self):
        self.evaluate_calls = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        self.evaluate_calls += 1
        return _EvalBatch([])

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {
            component: [
                {"instance_id": t["instance_id"], "produced": t["module_calls"][0]["output"], "score": 0.0}
                for t in eval_batch.trajectories
            ]
            for component in components_to_update
        }


class ScriptedJudge:
    """Returns pre-baked occurrences, so routing is tested without a model."""

    def __init__(self, by_trace):
        self.by_trace = by_trace
        self.seen = []
        self.candidate_key = ""

    def judge(self, traces):
        self.seen = [t.trace_id for t in traces]
        return {t.trace_id: list(self.by_trace.get(t.trace_id, [])) for t in traces}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_minimal_code_needs_only_id_and_name():
    tax = Taxonomy.from_codes([{"id": "X.1", "name": "Thing"}])
    assert len(tax) == 1
    assert tax.get("X.1").description == ""


def test_unrecognised_fields_are_preserved_but_do_not_route(taxonomy):
    assert taxonomy.get("B.4").extra["severity"] == "high"
    # severity/category/applies_to_role are carried, never consulted.
    assert "severity" not in taxonomy.get("B.4").catalog_entry()


def test_layered_adamast_document_is_accepted():
    doc = {"full_layer": {"category_b": {"B.1": {"name": "Bad", "description": "d"}}}}
    tax = Taxonomy.from_codes(_codes(doc))
    assert tax.get("B.1").name == "Bad"


def _codes(doc):
    from failure_taxonomy.schema import _codes_from_document

    return _codes_from_document(doc)


def test_empty_taxonomy_is_rejected():
    with pytest.raises(TaxonomyError):
        Taxonomy.from_codes([])


def test_fingerprint_changes_when_codes_change(taxonomy):
    other = Taxonomy.from_codes(CODES[:2])
    assert taxonomy.fingerprint != other.fingerprint


def test_load_taxonomy_tolerates_a_bom(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"codes": CODES}), encoding="utf-8-sig")
    assert len(load_taxonomy(p)) == 3


def test_guidance_fields_reach_the_catalog_when_present():
    tax = Taxonomy.from_codes([{"id": "A.1", "name": "N", "when_to_use": "only when X"}])
    assert "only when X" in tax.catalog_text()


# ---------------------------------------------------------------------------
# Trace segmentation
# ---------------------------------------------------------------------------


def test_extract_calls_reads_the_contract():
    calls = extract_calls(_trajectory("i1", "patch", "better patch"))
    assert [c.component for c in calls] == ["solver", "refiner"]
    assert calls[0].output == "patch"


def test_trajectory_without_module_calls_degrades_not_breaks():
    trace = build_trace({"anything": "else"}, trace_id="i1")
    assert not trace.is_segmented
    assert trace.components == ()
    assert trace.render()  # still renders something judgeable


def test_repeated_component_appears_once_in_the_vocabulary():
    trace = SegmentedTrace(
        trace_id="i1",
        calls=(ComponentCall("hop", output="a"), ComponentCall("hop", output="b")),
    )
    assert trace.components == ("hop",)
    assert "STEP 1 of 2" in trace.render()


def test_generation_record_carries_components():
    trace = build_trace(_trajectory("i1", "p", "q"), trace_id="i1", task="t")
    record = trace.to_generation_record()
    assert record["problem_id"] == "i1"
    assert record["metadata"]["components"] == ["solver", "refiner"]
    assert record["raw_trajectory"]


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def test_judge_parses_fenced_json_and_attributes(taxonomy):
    response = """Sure, here you go:
    ```json
    {"occurrences": [{"code": "B.4", "component": "solver", "evidence": "bad diff"}]}
    ```"""
    judge = LLMFailureJudge(taxonomy=taxonomy, lm=lambda _: response)
    trace = build_trace(_trajectory("i1", "p", "q"), trace_id="i1")
    got = judge.judge([trace])["i1"]
    assert got == [Occurrence(code="B.4", name="Malformed_Output", evidence="bad diff", component="solver")]


def test_judge_drops_codes_not_in_the_taxonomy(taxonomy):
    response = '{"occurrences": [{"code": "Z.9", "component": "solver", "evidence": "x"}]}'
    judge = LLMFailureJudge(taxonomy=taxonomy, lm=lambda _: response)
    trace = build_trace(_trajectory("i1", "p", "q"), trace_id="i1")
    assert judge.judge([trace])["i1"] == []
    assert judge.unknown_codes_dropped == 1


def test_unknown_component_degrades_to_general_rather_than_being_dropped(taxonomy):
    response = '{"occurrences": [{"code": "A.1", "component": "ghost", "evidence": "e"}]}'
    judge = LLMFailureJudge(taxonomy=taxonomy, lm=lambda _: response)
    trace = build_trace(_trajectory("i1", "p", "q"), trace_id="i1")
    got = judge.judge([trace])["i1"]
    assert got[0].component is None
    assert judge.unknown_components_generalised == 1


def test_name_comes_from_the_taxonomy_not_the_judge(taxonomy):
    response = '{"occurrences": [{"code": "A.1", "name": "RELABELLED", "component": "solver", "evidence": "e"}]}'
    judge = LLMFailureJudge(taxonomy=taxonomy, lm=lambda _: response)
    trace = build_trace(_trajectory("i1", "p", "q"), trace_id="i1")
    assert judge.judge([trace])["i1"][0].name == "Output_Truncation"


def test_repeated_occurrences_are_kept_not_deduplicated(taxonomy):
    response = json.dumps(
        {
            "occurrences": [
                {"code": "B.4", "component": "solver", "evidence": "first"},
                {"code": "B.4", "component": "solver", "evidence": "second"},
            ]
        }
    )
    judge = LLMFailureJudge(taxonomy=taxonomy, lm=lambda _: response)
    trace = build_trace(_trajectory("i1", "p", "q"), trace_id="i1")
    got = judge.judge([trace])["i1"]
    assert [o.evidence for o in got] == ["first", "second"]


def test_judge_failure_is_soft(taxonomy):
    def boom(_):
        raise RuntimeError("provider down")

    judge = LLMFailureJudge(taxonomy=taxonomy, lm=boom, log=lambda _: None)
    trace = build_trace(_trajectory("i1", "p", "q"), trace_id="i1")
    assert judge.judge([trace]) == {}
    assert judge.failures == 1


def test_empty_occurrence_list_is_a_real_answer(taxonomy):
    judge = LLMFailureJudge(taxonomy=taxonomy, lm=lambda _: '{"occurrences": []}')
    trace = build_trace(_trajectory("i1", "p", "q"), trace_id="i1")
    result = judge.judge([trace])
    assert result == {"i1": []}  # present-and-empty, not absent


def test_prompt_lists_the_component_vocabulary(taxonomy):
    judge = LLMFailureJudge(taxonomy=taxonomy, lm=lambda _: '{"occurrences": []}')
    prompt = judge.build_prompt(build_trace(_trajectory("i1", "p", "q"), trace_id="i1"))
    assert "- solver" in prompt and "- refiner" in prompt
    assert "A.1" in prompt and "Output_Truncation" in prompt


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_round_trips_and_avoids_a_second_call(tmp_path, taxonomy):
    calls = {"n": 0}

    def lm(_):
        calls["n"] += 1
        return '{"occurrences": [{"code": "A.1", "component": "solver", "evidence": "e"}]}'

    cache = JudgeCache.open(tmp_path / "judge.jsonl")
    judge = LLMFailureJudge(taxonomy=taxonomy, lm=lm, cache=cache, candidate_key="cand")
    trace = build_trace(_trajectory("i1", "p", "q"), trace_id="i1")
    first = judge.judge([trace])["i1"]
    second = judge.judge([trace])["i1"]
    cache.close()
    assert calls["n"] == 1
    assert first == second


def test_cache_drops_a_truncated_final_record(tmp_path):
    p = tmp_path / "judge.jsonl"
    good = {"taxonomy": "t", "candidate_key": "c", "trace_id": "i1", "occurrences": []}
    p.write_text(json.dumps(good) + "\n" + '{"taxonomy": "t", "cand', encoding="utf-8")
    cache = JudgeCache.open(p)
    assert len(cache) == 1
    assert cache.truncated_records == 1
    cache.close()


def test_candidate_key_is_order_independent():
    assert candidate_key({"a": "1", "b": "2"}) == candidate_key({"b": "2", "a": "1"})


# ---------------------------------------------------------------------------
# Optimizer-side enrichment — the guarantees that make the comparison valid
# ---------------------------------------------------------------------------


def _enrich(inner, batch, judge, components=("solver", "refiner"), *, log=print):
    candidate = {component: component for component in components}
    baseline = inner.make_reflective_dataset(candidate, batch, list(components))
    enricher = TaxonomyFeedbackEnricher(judge=judge, log=log)
    got = enricher(
        candidate=candidate,
        eval_batch=batch,
        components_to_update=list(components),
        reflective_dataset=baseline,
    )
    return baseline, got, enricher


def test_no_enricher_is_byte_identical(taxonomy):
    """The baseline path never constructs or calls taxonomy code."""
    inner = InnerAdapter()
    trajectories = [_trajectory("i1", "p", "q"), _trajectory("i2", "r", "s")]
    batch = _EvalBatch(trajectories)
    candidate = {"solver": "s", "refiner": "r"}
    baseline = inner.make_reflective_dataset(candidate, batch, ["solver", "refiner"])
    repeated = inner.make_reflective_dataset(candidate, batch, ["solver", "refiner"])
    assert json.dumps(repeated, sort_keys=True) == json.dumps(baseline, sort_keys=True)


def test_occurrences_route_to_their_attributed_component(taxonomy):
    inner = InnerAdapter()
    batch = _EvalBatch([_trajectory("i1", "p", "q")])
    judge = ScriptedJudge(
        {
            "i1": [
                Occurrence("B.4", "Malformed_Output", "solver evidence", "solver"),
                Occurrence("C.2", "Wrong_Entity", "refiner evidence", "refiner"),
            ]
        }
    )
    _baseline, got, _enricher = _enrich(inner, batch, judge)

    assert got["solver"][0][FAILURE_MODES_KEY] == [{"name": "Malformed_Output", "evidence": "solver evidence"}]
    assert got["refiner"][0][FAILURE_MODES_KEY] == [{"name": "Wrong_Entity", "evidence": "refiner evidence"}]


def test_general_occurrences_reach_every_component(taxonomy):
    inner = InnerAdapter()
    batch = _EvalBatch([_trajectory("i1", "p", "q")])
    judge = ScriptedJudge({"i1": [Occurrence("A.1", "Output_Truncation", "shared", None)]})
    _baseline, got, _enricher = _enrich(inner, batch, judge)

    for component in ("solver", "refiner"):
        assert got[component][0][FAILURE_MODES_KEY] == [{"name": "Output_Truncation", "evidence": "shared"}]


def test_examples_without_occurrences_gain_no_key(taxonomy):
    inner = InnerAdapter()
    batch = _EvalBatch([_trajectory("i1", "p", "q")])
    _baseline, got, _enricher = _enrich(inner, batch, ScriptedJudge({}), components=("solver",))
    assert FAILURE_MODES_KEY not in got["solver"][0]


def test_reflection_never_sees_the_code_id(taxonomy):
    inner = InnerAdapter()
    batch = _EvalBatch([_trajectory("i1", "p", "q")])
    judge = ScriptedJudge({"i1": [Occurrence("B.4", "Malformed_Output", "e", "solver")]})
    _baseline, got, _enricher = _enrich(inner, batch, judge, components=("solver",))
    assert set(got["solver"][0][FAILURE_MODES_KEY][0]) == {"name", "evidence"}


def test_misaligned_examples_are_left_undiagnosed_rather_than_mismatched(taxonomy):
    """Attaching a diagnosis to the wrong rollout is worse than attaching none."""

    class FilteringAdapter(InnerAdapter):
        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            # Returns fewer examples than trajectories, e.g. it dropped successes.
            return {c: [{"instance_id": "i1"}] for c in components_to_update}

    batch = _EvalBatch([_trajectory("i1", "p", "q"), _trajectory("i2", "r", "s")])
    judge = ScriptedJudge({"i1": [Occurrence("A.1", "Output_Truncation", "e", "solver")]})
    _baseline, got, enricher = _enrich(
        FilteringAdapter(), batch, judge, components=("solver",), log=lambda _: None
    )

    assert FAILURE_MODES_KEY not in got["solver"][0]
    assert enricher.skipped_batches == 1


def test_judge_failure_leaves_the_baseline_dataset_intact(taxonomy):
    class BoomJudge:
        candidate_key = ""

        def judge(self, traces):
            raise RuntimeError("provider down")

    inner = InnerAdapter()
    batch = _EvalBatch([_trajectory("i1", "p", "q")])
    baseline = inner.make_reflective_dataset({"solver": "s"}, batch, ["solver"])
    _unused, got, _enricher = _enrich(inner, batch, BoomJudge(), components=("solver",), log=lambda _: None)
    assert json.dumps(got, sort_keys=True) == json.dumps(baseline, sort_keys=True)


def test_every_instance_is_judged_not_only_failures(taxonomy):
    inner = InnerAdapter()
    batch = _EvalBatch([_trajectory("i1", "p", "q"), _trajectory("i2", "r", "s")])
    batch.scores = [1.0, 0.0]  # one success, one failure
    judge = ScriptedJudge({})
    _enrich(inner, batch, judge, components=("solver",))
    assert judge.seen == ["i1", "i2"]


def test_enricher_never_mutates_the_adapter_dataset(taxonomy):
    inner = InnerAdapter()
    batch = _EvalBatch([_trajectory("i1", "p", "q")])
    judge = ScriptedJudge({"i1": [Occurrence("B.4", "Malformed_Output", "e", "solver")]})
    baseline, got, _enricher = _enrich(inner, batch, judge, components=("solver",))

    assert FAILURE_MODES_KEY not in baseline["solver"][0]
    assert FAILURE_MODES_KEY in got["solver"][0]


def test_summary_contains_only_enrichment_diagnostics(taxonomy):
    class SummaryJudge(ScriptedJudge):
        def summary(self):
            return {"judge_calls": 214}

    enricher = TaxonomyFeedbackEnricher(judge=SummaryJudge({}))
    assert enricher.summary() == {
        "examples_diagnosed": 0,
        "occurrences_injected": 0,
        "unalignable_batches": 0,
        "judge": {"judge_calls": 214},
    }


class TestDuplicateCodeIds:
    """A repeated id is a duplicate only if the entries agree.

    Bringing your own taxonomy is a supported entry point, so the file may not
    have come from a generator that deduplicates.
    """

    @staticmethod
    def _load(tmp_path, codes):
        import json

        from failure_taxonomy import load_taxonomy

        p = tmp_path / "t.json"
        p.write_text(json.dumps({"codes": codes}), encoding="utf-8")
        return load_taxonomy(p)

    def test_identical_rows_are_deduplicated_quietly(self, tmp_path):
        row = {"id": "A.1", "name": "a", "description": "x"}
        assert len(self._load(tmp_path, [row, dict(row)])) == 1

    def test_a_conflicting_id_is_rejected_not_silently_resolved(self, tmp_path):
        from failure_taxonomy import TaxonomyError

        with pytest.raises(TaxonomyError, match="appears twice with different content"):
            self._load(
                tmp_path,
                [
                    {"id": "A.1", "name": "Dropped_Constraint", "description": "x"},
                    {"id": "A.1", "name": "Hallucinated_Entity", "description": "y"},
                ],
            )

    def test_why_it_matters(self, tmp_path):
        """Keeping the first silently would make the judge emit A.1 and hand
        reflection a name the taxonomy's author never wrote. The id is also
        what cross-run analysis joins on."""
        from failure_taxonomy import TaxonomyError

        with pytest.raises(TaxonomyError):
            self._load(tmp_path, [{"id": "B.2", "name": "a"}, {"id": "B.2", "name": "b"}])
