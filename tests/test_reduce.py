"""Evidence-based reduction: pure, offline, deterministic, fully accounted."""

from __future__ import annotations

import json

import pytest

from failure_taxonomy import (
    FrozenTraceLeakError,
    assert_generation_disjoint,
    load_taxonomy,
    measure_support,
    reduce_taxonomy,
)

DOC = {
    "schema_version": 1,
    "status": "accepted",
    "codes": [
        {"id": "A.1", "name": "Truncation", "description": "d"},
        {"id": "B.1", "name": "Solver_Miss", "description": "d"},
        {"id": "B.2", "name": "Checker_Miss", "description": "d"},
        {"id": "C.1", "name": "Never_Fires", "description": "d"},
        {"id": "C.2", "name": "Single_Citation", "description": "d"},
    ],
}


def judgements(**per_trace):
    return {tid: [{"code": c, "name": c, "evidence": "e"} for c in codes] for tid, codes in per_trace.items()}


def test_support_counts_distinct_traces_not_mentions():
    j = {"t1": [{"code": "B.1"}, {"code": "B.1"}, {"code": "B.1"}], "t2": [{"code": "B.1"}]}
    assert measure_support(j) == {"B.1": {"t1", "t2"}}


def test_grounding_filter_alone_when_cap_does_not_bind():
    # Regression guard: the cap must not be the active filter here.
    j = judgements(t1=["A.1", "B.1"], t2=["A.1", "B.1", "B.2"], t3=["B.2", "C.2"])
    result = reduce_taxonomy(DOC, j, min_support=2, max_codes=25)
    kept = [c["id"] for c in result.document["codes"]]
    assert kept == ["A.1", "B.1", "B.2"]
    assert result.document["reduction"]["cap_bound"] is False
    tallies = result.report["tallies"]
    assert tallies == {"retained": 3, "ungrounded": 2, "over_cap": 0}


def test_every_code_accounted_and_sums_match():
    j = judgements(t1=["A.1"], t2=["A.1"], t3=["B.1"], t4=["B.1"], t5=["B.2"], t6=["B.2"], t7=["C.2"])
    result = reduce_taxonomy(DOC, j, min_support=2, max_codes=2)
    tallies = result.report["tallies"]
    assert sum(tallies.values()) == len(DOC["codes"])
    assert {e["outcome"] for e in result.report["codes"]} == {"retained", "ungrounded", "over_cap"}
    assert result.document["reduction"]["cap_bound"] is True


def test_deterministic_ordering_and_tie_break():
    j = judgements(t1=["A.1", "B.1"], t2=["A.1", "B.1"])
    a = reduce_taxonomy(DOC, j, min_support=2, max_codes=1)
    b = reduce_taxonomy(DOC, dict(reversed(list(j.items()))), min_support=2, max_codes=1)
    # equal support: code id ascending is the tie-break, input order irrelevant
    assert [c["id"] for c in a.document["codes"]] == ["A.1"]
    assert a.report == b.report


def test_reduced_document_keeps_original_code_order():
    j = judgements(t1=["C.2", "A.1"], t2=["C.2", "A.1"], t3=["C.2"])
    result = reduce_taxonomy(DOC, j, min_support=2, max_codes=25)
    assert [c["id"] for c in result.document["codes"]] == ["A.1", "C.2"]


def test_load_taxonomy_carries_generation_trace_ids(tmp_path):
    j = judgements(t1=["A.1"], t2=["A.1"])
    result = reduce_taxonomy(DOC, j, min_support=2, max_codes=25)
    p = tmp_path / "taxonomy.json"
    p.write_text(json.dumps(result.document))
    taxonomy = load_taxonomy(p)
    assert taxonomy.generation_trace_ids == frozenset({"t1", "t2"})


def test_disjointness_assertion_names_the_leak():
    with pytest.raises(FrozenTraceLeakError, match="t2"):
        assert_generation_disjoint(frozenset({"t1", "t2"}), ["t2", "t9"], context="minibatch")
    assert_generation_disjoint(frozenset({"t1"}), ["t9"])  # disjoint: no raise


def test_rejects_empty_and_bad_thresholds():
    with pytest.raises(ValueError):
        reduce_taxonomy({"codes": []}, {})
    with pytest.raises(ValueError):
        reduce_taxonomy(DOC, {}, min_support=0)
    with pytest.raises(ValueError):
        reduce_taxonomy(DOC, {}, max_codes=0)
