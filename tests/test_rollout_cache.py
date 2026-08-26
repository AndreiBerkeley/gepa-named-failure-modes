"""Rollout-granularity durability (O011).

gepa saves state once per iteration, so without this an interruption discards
the whole in-flight iteration's paid work (~$2.02, ~28 rollouts, ~50 min of
local container time at measured rates).
"""

from __future__ import annotations

import json

import pytest

from gepa_taxonomy.rollout_cache import RolloutCache
from gepa_taxonomy.seed_cache import candidate_hash

CAND = {"solver_instruction": "a", "refiner_instruction": "b"}
OTHER = {"solver_instruction": "a", "refiner_instruction": "CHANGED"}


@pytest.fixture
def cache(tmp_path):
    c = RolloutCache.open(tmp_path / "rollouts.jsonl")
    yield c
    c.close()


def test_put_then_get_roundtrips(cache):
    cache.put(CAND, "i1", score=1.0, output={"patch": "p"}, cost_usd=0.07)
    rec = cache.get(CAND, "i1")
    assert rec["score"] == 1.0
    assert rec["output"] == {"patch": "p"}
    assert rec["cost_usd"] == 0.07


def test_miss_returns_none(cache):
    assert cache.get(CAND, "nope") is None


def test_keyed_on_candidate_not_just_instance(cache):
    """A mutated candidate must NOT reuse another candidate's result."""
    cache.put(CAND, "i1", score=1.0, output={})
    assert cache.get(OTHER, "i1") is None


def test_key_matches_gepas_own_scheme(cache):
    cache.put(CAND, "i1", score=1.0, output={})
    rec = json.loads((cache.path).read_text().splitlines()[0])
    assert rec["candidate_hash"] == candidate_hash(CAND)


def test_survives_reopen(tmp_path):
    """The core durability property: a fresh process sees prior work."""
    c1 = RolloutCache.open(tmp_path / "r.jsonl")
    c1.put(CAND, "i1", score=1.0, output={"patch": "p"}, cost_usd=0.07)
    c1.put(CAND, "i2", score=0.0, output={"patch": ""}, cost_usd=0.07)
    c1.close()

    c2 = RolloutCache.open(tmp_path / "r.jsonl")
    assert len(c2) == 2
    assert c2.get(CAND, "i1")["score"] == 1.0
    assert c2.get(CAND, "i2")["score"] == 0.0
    assert c2.recovered_usd == pytest.approx(0.14)
    c2.close()


def test_truncated_final_line_is_survivable(tmp_path):
    """Simulates a kill mid-append.

    A partial last record must cost at most that one rollout -- it must not
    prevent the cache from loading, which would discard everything.
    """
    p = tmp_path / "r.jsonl"
    c1 = RolloutCache.open(p)
    c1.put(CAND, "i1", score=1.0, output={"patch": "p"}, cost_usd=0.07)
    c1.put(CAND, "i2", score=1.0, output={"patch": "q"}, cost_usd=0.07)
    c1.close()

    with p.open("a") as fh:  # a half-written third record
        fh.write('{"candidate_hash": "abc", "instance_i')

    c2 = RolloutCache.open(p)
    assert len(c2) == 2, "complete records must survive a truncated tail"
    assert c2.truncated_records == 1
    assert c2.get(CAND, "i1")["score"] == 1.0
    c2.close()


def test_entries_are_flushed_immediately(tmp_path):
    """Durability requires the record to be on disk BEFORE the next rollout,
    not buffered until close()."""
    p = tmp_path / "r.jsonl"
    c = RolloutCache.open(p)
    c.put(CAND, "i1", score=1.0, output={})
    # Read from a separate handle without closing the writer.
    assert p.read_text().strip(), "record was still buffered -- a kill would lose it"
    assert json.loads(p.read_text().splitlines()[0])["instance_id"] == "i1"
    c.close()


def test_hits_are_counted(cache):
    cache.put(CAND, "i1", score=1.0, output={})
    assert cache.hits == 0
    cache.get(CAND, "i1")
    cache.get(CAND, "i1")
    assert cache.hits == 2


def test_recovered_usd_reports_work_not_repaid(tmp_path):
    c = RolloutCache.open(tmp_path / "r.jsonl")
    for i in range(28):  # one full iteration at our measured rates
        c.put(CAND, f"i{i}", score=0.0, output={}, cost_usd=0.0714)
    assert c.recovered_usd == pytest.approx(28 * 0.0714)
    c.close()


# --------------------------------------------------------------------------
# Integration with the adapter
# --------------------------------------------------------------------------


def test_adapter_serves_from_cache_without_rerunning(tmp_path):
    from gepa_taxonomy.adapter import SweBenchAdapter
    from gepa_taxonomy.cost import CostMeter
    from gepa_taxonomy.program import SEED_CANDIDATE, SolverRefinerProgram
    from gepa_taxonomy.tasks import split_row
    from tests.test_adapter import HAIKU, SONNET, CountingLM, FakeGrader, FakeRetriever
    from tests.test_gold_blindness import RAW_ROW

    inst = split_row(RAW_ROW)
    iid = inst.task.instance_id
    solver, refiner = CountingLM(), CountingLM()
    grader = FakeGrader()
    program = SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=solver,
        refiner_lm=refiner,
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    rc = RolloutCache.open(tmp_path / "r.jsonl")
    ad = SweBenchAdapter(
        program=program,
        grader=grader,
        instances={iid: inst},
        rollout_cache=rc,
        skip_ungradeable_patches=False,
    )

    ad.evaluate([iid], SEED_CANDIDATE)
    assert solver.calls == 1 and grader.calls == 1

    # Second evaluation of the same (candidate, instance) must be free.
    ad.evaluate([iid], SEED_CANDIDATE)
    assert solver.calls == 1, "cached rollout re-ran the solver -- paid twice"
    assert grader.calls == 1, "cached rollout re-graded -- container run wasted"
    rc.close()
