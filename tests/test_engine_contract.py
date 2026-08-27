"""End-to-end contract test: the REAL adapter through gepa's own engine. FREE.

Why this file exists
--------------------
A seed-1 launch died instantly at ``engine.py:129``::

    return eval_result.outputs, eval_result.scores, eval_result.objective_scores
    AttributeError: 'EvaluationBatchResult' object has no attribute 'objective_scores'

Our adapter returned a look-alike result type with three fields; gepa's contract
has five. Every existing adapter test called ``adapter.evaluate()`` *directly*,
so none of them touched the engine's own consumption of the result -- the
mismatch was invisible until a paid launch.

A second omission, ``num_metric_calls`` (read at
``reflective_mutation.py:329``), would have fired later still: during reflective
mutation, i.e. **after** LM calls had been paid for.

These tests drive the real ``SweBenchAdapter`` through ``gepa.optimize()`` so the
engine itself consumes the result. The seed cache replays every val instance, so
no LM call and no container run occurs -- a contract break fails here, free,
instead of at launch.
"""

from __future__ import annotations

import dataclasses

import gepa
import pytest
from gepa.core.adapter import EvaluationBatch

from gepa_taxonomy.adapter import SweBenchAdapter
from gepa_taxonomy.cost import CostMeter
from gepa_taxonomy.program import SEED_CANDIDATE, RetrievedFile, SolverRefinerProgram
from gepa_taxonomy.seed_cache import SeedEvaluationCache
from gepa_taxonomy.tasks import split_row
from tests.test_gold_blindness import RAW_ROW

HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET = "us.anthropic.claude-sonnet-5"

VAL_N = 6


class ExplodingLM:
    """Any LM call here means the replay path failed and we would have paid."""

    def complete(self, prompt: str, *, max_tokens: int = 4096):
        raise AssertionError("an LM call was made; the seed cache should have replayed this")


class ExplodingGrader:
    """Any grading call means a container would have been started."""

    def grade_batch(self, items):
        raise AssertionError("the grader was called; the seed cache should have replayed this")

    def grade(self, task, gold, patch):
        raise AssertionError("the grader was called; the seed cache should have replayed this")


class FakeRetriever:
    def retrieve(self, task, *, k):
        return [RetrievedFile(path="m.py", content="x = 1\n")]


def _instances(n: int):
    out = {}
    for i in range(n):
        row = dict(RAW_ROW)
        row["instance_id"] = f"proj__proj-{i}"
        inst = split_row(row)
        out[inst.task.instance_id] = inst
    return out


@pytest.fixture
def wired(tmp_path):
    """The real adapter, with a seed cache covering every val instance."""
    instances = _instances(VAL_N)
    ids = sorted(instances)

    seed_cache = SeedEvaluationCache.build(
        SEED_CANDIDATE,
        {
            iid: {
                "score": 1.0 if n % 2 == 0 else 0.0,
                "output": {"patch": "--- a/m.py\n+++ b/m.py\n", "instance_id": iid},
                "trace": {"instance_id": iid},
            }
            for n, iid in enumerate(ids)
        },
    )

    program = SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=ExplodingLM(),
        refiner_lm=ExplodingLM(),
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    adapter = SweBenchAdapter(
        program=program,
        grader=ExplodingGrader(),
        instances=instances,
        seed_cache=seed_cache,
    )
    return adapter, ids, tmp_path


# --------------------------------------------------------------------------
# The contract itself
# --------------------------------------------------------------------------


def test_adapter_returns_gepas_own_result_type(wired):
    adapter, ids, _ = wired
    res = adapter.evaluate(ids, SEED_CANDIDATE)
    assert isinstance(res, EvaluationBatch), "must be gepa's type, not a look-alike"


@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(EvaluationBatch)])
def test_result_exposes_every_field_gepa_declares(wired, field):
    """Pins the whole contract, not just the field that happened to crash."""
    adapter, ids, _ = wired
    res = adapter.evaluate(ids, SEED_CANDIDATE)
    assert hasattr(res, field), f"gepa's engine may read .{field}"


def test_engine_evaluator_closure_works(wired):
    """Reproduces engine.py:126-129 verbatim -- the exact line that crashed."""
    adapter, ids, _ = wired

    def evaluator(batch, program):
        eval_result = adapter.evaluate(batch, program, capture_traces=False)
        return eval_result.outputs, eval_result.scores, eval_result.objective_scores

    outputs, scores, objective_scores = evaluator(ids, SEED_CANDIDATE)
    assert len(outputs) == len(scores) == VAL_N
    assert objective_scores is None


def test_num_metric_calls_is_reported(wired):
    """Read at reflective_mutation.py:329; would have crashed after paid calls."""
    adapter, ids, _ = wired
    res = adapter.evaluate(ids, SEED_CANDIDATE)
    # Everything was replayed from the seed cache, so nothing actually ran.
    assert res.num_metric_calls == 0


# --------------------------------------------------------------------------
# End-to-end through gepa.optimize()
# --------------------------------------------------------------------------


def test_real_adapter_survives_the_engines_seed_valset_evaluation(wired):
    """The launch path, end to end, for free.

    ``max_metric_calls=1`` lets the seed valset evaluation run and then stops
    the loop immediately, so this exercises exactly the code path that failed
    at launch without reaching reflection.
    """
    adapter, ids, tmp_path = wired

    result = gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=list(ids),
        valset=list(ids),
        adapter=adapter,
        reflection_lm=lambda prompt: "unused",
        max_metric_calls=1,
        run_dir=str(tmp_path / "run"),
        seed=1,
        display_progress_bar=False,
    )

    assert result.candidates[0] == SEED_CANDIDATE
    assert len(result.val_subscores[0]) == VAL_N


def test_launch_path_spends_nothing_when_the_seed_cache_covers_val(wired):
    """The replay must issue no LM call and start no container.

    ExplodingLM/ExplodingGrader turn any such call into a test failure, so this
    also proves the free dry-run is genuinely free.
    """
    adapter, ids, tmp_path = wired

    gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=list(ids),
        valset=list(ids),
        adapter=adapter,
        reflection_lm=lambda prompt: "unused",
        max_metric_calls=1,
        run_dir=str(tmp_path / "run2"),
        seed=1,
        display_progress_bar=False,
    )

    assert adapter.program.solver_meter.budgeted_usd == 0.0
    assert adapter.program.refiner_meter.budgeted_usd == 0.0


def test_seed_scores_reach_gepa_unchanged(wired):
    """gepa must see the replayed scores verbatim, or runs would not start from
    identical state.

    Note gepa keys ``val_subscores`` by the valset's **positional index**, not by
    our instance id -- a plain-list valset gets integer DataIds. So the mapping
    back to instance ids is positional, and the valset order must stay stable
    across runs for those keys to mean the same thing. Our manifests are sorted,
    which gives exactly that.
    """
    adapter, ids, tmp_path = wired
    expected = {n: (1.0 if n % 2 == 0 else 0.0) for n, _iid in enumerate(ids)}

    result = gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=list(ids),
        valset=list(ids),
        adapter=adapter,
        reflection_lm=lambda prompt: "unused",
        max_metric_calls=1,
        run_dir=str(tmp_path / "run3"),
        seed=1,
        display_progress_bar=False,
    )
    assert dict(result.val_subscores[0]) == expected
