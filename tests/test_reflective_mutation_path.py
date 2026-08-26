"""Drive the REFLECTIVE-MUTATION path end to end, for free.

Why this exists
---------------
Two launches died past the point earlier tests reached:

1. ``engine.py:129`` -- result-type contract (fixed, covered by
   ``test_engine_contract.py``).
2. ``seed_cache.get`` raising on a TRAIN instance during a parent evaluation --
   a **routing** bug, not an LM one.

I said after the first failure that the reflective-mutation path "still isn't
exercised end-to-end, because reaching it requires a real mutation". That was
wrong, and the second crash proved it: the failure was in routing, so the path
is perfectly testable with a stubbed LM client. Nothing about it needed a real
model.

This module drives ``gepa.optimize()`` far enough to perform reflective
mutation -- parent evaluations on train minibatches, candidate proposal, and val
evaluation of the proposed candidate -- using the real adapter, the real seed
cache, and the real committed manifests. Only the LM client and the grader are
stubbed, so no token is spent and no container starts.
"""

from __future__ import annotations

import json
from pathlib import Path

import gepa
import pytest

from gepa_taxonomy.adapter import SweBenchAdapter
from gepa_taxonomy.cost import CostMeter
from gepa_taxonomy.program import SEED_CANDIDATE, RetrievedFile, SolverRefinerProgram
from gepa_taxonomy.seed_cache import SeedEvaluationCache
from gepa_taxonomy.splits import load_manifest
from gepa_taxonomy.tasks import split_row
from tests.test_gold_blindness import RAW_ROW

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = REPO_ROOT / "manifests" / "swebench_verified"

HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET = "us.anthropic.claude-sonnet-5"

PATCH = "--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n"


class StubLM:
    """Deterministic stand-in for BedrockLM. Records what it was asked to do."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str, *, max_tokens: int = 4096):
        self.calls += 1
        return PATCH, 20_000, 400


class StubGrader:
    """Stands in for LocalDockerGrader. No container is ever started."""

    def __init__(self):
        self.graded: list[str] = []

    def grade_batch(self, items):
        self.graded += [t.instance_id for t, _g, _p in items]
        # Deterministic, and varied enough that some candidate can beat another.
        return {
            t.instance_id: (1.0 if (sum(map(ord, t.instance_id)) % 3 == 0) else 0.0, {"resolved": False})
            for t, _g, _p in items
        }


class StubRetriever:
    def retrieve(self, task, *, k):
        return [RetrievedFile(path="m.py", content="x = 1\n")]


def _instances(ids: list[str]) -> dict:
    """Real Task/Gold objects for the real manifest ids."""
    out = {}
    for iid in ids:
        row = dict(RAW_ROW)
        row["instance_id"] = iid
        inst = split_row(row)
        out[iid] = inst
    return out


@pytest.fixture
def wired(tmp_path):
    """Real adapter + real seed cache + REAL committed manifests."""
    val_ids = load_manifest(MANIFESTS / "val.json")
    train_ids = load_manifest(MANIFESTS / "train.json")
    # Trim the train side so the test stays fast; the routing behaviour under
    # test does not depend on how many train instances exist.
    train_ids = train_ids[:20]

    instances = _instances(val_ids + train_ids)

    # The seed cache covers val ONLY -- exactly as build_base_val produces it.
    seed_cache = SeedEvaluationCache.build(
        SEED_CANDIDATE,
        {
            iid: {
                "score": 1.0 if n % 4 == 0 else 0.0,
                "output": {"patch": PATCH, "instance_id": iid},
                "trace": {"instance_id": iid},
            }
            for n, iid in enumerate(val_ids)
        },
    )

    solver, refiner = StubLM(), StubLM()
    program = SolverRefinerProgram(
        retriever=StubRetriever(),
        solver_lm=solver,
        refiner_lm=refiner,
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    grader = StubGrader()
    adapter = SweBenchAdapter(
        program=program,
        grader=grader,
        instances=instances,
        seed_cache=seed_cache,
        skip_ungradeable_patches=False,
    )
    return adapter, val_ids, train_ids, solver, grader, tmp_path


def _run(adapter, val_ids, train_ids, tmp_path, *, max_metric_calls, name="run"):
    return gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=list(train_ids),
        valset=list(val_ids),
        adapter=adapter,
        reflection_lm=lambda prompt: "a mutated solver instruction",
        max_metric_calls=max_metric_calls,
        run_dir=str(tmp_path / name),
        seed=1,
        reflection_minibatch_size=3,
        display_progress_bar=False,
    )


# --------------------------------------------------------------------------
# The routing failure that killed launch #2
# --------------------------------------------------------------------------


def test_reflective_mutation_reaches_train_minibatches_without_crashing(wired):
    """The exact scenario: base candidate evaluated on a TRAIN minibatch.

    Before the scoping fix this raised
    ``KeyError: 'django__django-11099'`` from seed_cache.get.
    """
    adapter, val_ids, train_ids, solver, grader, tmp_path = wired
    result = _run(adapter, val_ids, train_ids, tmp_path, max_metric_calls=140)

    assert result is not None
    # Reaching mutation at all means parent evals on train ids succeeded.
    assert solver.calls > 0, "no live rollout ran; the mutation path was not reached"


def test_train_instances_take_the_live_path_and_val_takes_the_cache(wired):
    """Both halves of the routing rule, asserted separately."""
    adapter, val_ids, train_ids, solver, grader, tmp_path = wired
    _run(adapter, val_ids, train_ids, tmp_path, max_metric_calls=140, name="r2")

    graded = set(grader.graded)
    assert graded, "nothing was graded; the live path never ran"
    # Everything that reached the grader must be a train instance: val results
    # for the base candidate come from the cache and never re-grade.
    assert graded <= set(train_ids), f"val instances were re-graded: {graded & set(val_ids)}"


def test_seed_candidate_val_evaluation_is_still_fully_replayed(wired):
    """The scoping fix must not have broken D009's replay."""
    adapter, val_ids, train_ids, solver, grader, tmp_path = wired
    calls_before = solver.calls
    result = _run(adapter, val_ids, train_ids, tmp_path, max_metric_calls=1, name="r3")

    # max_metric_calls=1 stops right after the seed valset evaluation.
    assert solver.calls == calls_before, "the seed valset evaluation was not replayed"
    assert len(result.val_subscores[0]) == len(val_ids)


def test_a_mutated_candidate_is_evaluated_live_on_val(wired):
    """A mutated candidate is NOT the base candidate, so val must run live."""
    adapter, val_ids, train_ids, solver, grader, tmp_path = wired
    mutated = {**SEED_CANDIDATE, "solver_instruction": "mutated"}

    res = adapter.evaluate(val_ids[:3], mutated)
    assert solver.calls >= 3, "a mutated candidate must not be served from the seed cache"
    assert len(res.scores) == 3


def test_run_completes_and_writes_resumable_state(wired):
    """A full small run leaves the artifacts a resume depends on."""
    adapter, val_ids, train_ids, solver, grader, tmp_path = wired
    _run(adapter, val_ids, train_ids, tmp_path, max_metric_calls=140, name="r5")

    run_dir = tmp_path / "r5"
    assert (run_dir / "gepa_state.bin").exists()
    candidates = json.loads((run_dir / "candidates.json").read_text())
    assert candidates[0] == SEED_CANDIDATE


def test_nothing_was_spent(wired):
    """Stub LMs report tokens, but the run must issue no real call.

    The meters here prove the accounting path works end to end; the point is
    that this whole file runs with no provider and no Docker.
    """
    adapter, val_ids, train_ids, solver, grader, tmp_path = wired
    _run(adapter, val_ids, train_ids, tmp_path, max_metric_calls=140, name="r6")
    # Metered spend is attributed, and it is all from the stub.
    assert adapter.program.solver_meter.calls == solver.calls
