"""Full-launch smoke test with ONLY the network faked. FREE.

Three launches have now died in three different places, each past the point the
previous tests reached:

1. ``engine.py:129``          -- result-type contract
2. ``seed_cache.get``         -- replay routing on train minibatches
3. ``lm.py:231``              -- reflection LM not callable, and gepa SWALLOWS
                                 the TypeError, so the run burns budget forever
                                 while proposing nothing

Every one of them was reachable without a real model. The pattern is that
stubbing too high up -- replacing our own classes with fakes -- skips the very
seams that break. So this module stubs at the **lowest possible layer**:
``litellm.completion`` itself.

Everything above it is the real thing: the real ``BedrockLM`` (routing, token
extraction), the real ``MeteredReflectionLM``, the real adapter, the real seed
cache, the real manifests, the real cost meters and stopper, and gepa's real
reflection machinery. Only the HTTP call is replaced.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gepa
import pytest

from gepa_taxonomy.adapter import SweBenchAdapter
from gepa_taxonomy.bedrock import BedrockLM, MeteredReflectionLM
from gepa_taxonomy.cost import REFINER_MODEL, SOLVER_MODEL, CostMeter, MaxTotalCostStopper
from gepa_taxonomy.program import SEED_CANDIDATE, RetrievedFile, SolverRefinerProgram
from gepa_taxonomy.seed_cache import SeedEvaluationCache
from gepa_taxonomy.splits import load_manifest
from gepa_taxonomy.tasks import split_row
from tests.test_gold_blindness import RAW_ROW

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = REPO_ROOT / "manifests" / "swebench_verified"

PATCH = "--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n"
BETTER_PATCH = "--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 3\n"
MUTATION_MARKER = "an improved solver instruction"
#: Last line of our solver/refiner prompt templates -- an exact, unambiguous tell.
ROLLOUT_TEMPLATE_MARKER = "unified diff patch and nothing else."


class FakeNetwork:
    """Stands in for ``litellm.completion``. Records every call it receives."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        # Discriminate on OUR OWN template rather than guessing from keywords:
        # gepa's reflection prompt embeds our rollout traces (patches and all),
        # so any "does it mention a diff" heuristic misclassifies it.
        is_rollout = ROLLOUT_TEMPLATE_MARKER in prompt
        if not is_rollout:
            content = MUTATION_MARKER
        else:
            # A rollout driven by a MUTATED instruction produces a different
            # patch, so the grader below can reward it and gepa can accept the
            # candidate. Without this the stub scores every candidate 0.0 and
            # nothing is ever accepted -- which looks identical to reflection
            # being broken.
            content = BETTER_PATCH if MUTATION_MARKER in prompt else PATCH
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=20_000, completion_tokens=400),
        )

    @property
    def models_called(self) -> set[str]:
        return {c["model"] for c in self.calls}


class StubRetriever:
    def retrieve(self, task, *, k):
        return [RetrievedFile(path="m.py", content="x = 1\n")]


class StubGrader:
    """Rewards the patch a mutated instruction produces, so acceptance can happen."""

    def __init__(self):
        self.graded: list[str] = []

    def grade_batch(self, items):
        self.graded += [t.instance_id for t, _g, _p in items]
        # Match on content, not byte-equality: extract_patch() strips trailing
        # whitespace, so `p == BETTER_PATCH` is false even when it is the same
        # patch. That mismatch made this test look like a reflection failure.
        def wins(p: str) -> bool:
            return "+x = 3" in p

        return {t.instance_id: (1.0 if wins(p) else 0.0, {"resolved": wins(p)}) for t, _g, p in items}


@pytest.fixture
def net(monkeypatch):
    """Replace only the HTTP boundary; everything above it stays real."""
    import litellm

    fake = FakeNetwork()
    monkeypatch.setattr(litellm, "completion", fake)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "x" * 32)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    return fake


@pytest.fixture
def launch(net, tmp_path):
    """Assemble exactly what run_seed.py assembles."""
    val_ids = load_manifest(MANIFESTS / "val.json")[:8]
    train_ids = load_manifest(MANIFESTS / "train.json")[:12]

    instances = {}
    for iid in val_ids + train_ids:
        row = dict(RAW_ROW)
        row["instance_id"] = iid
        instances[iid] = split_row(row)

    seed_cache = SeedEvaluationCache.build(
        SEED_CANDIDATE,
        {iid: {"score": 0.0, "output": {"patch": PATCH, "instance_id": iid}, "trace": {}} for iid in val_ids},
    )

    solver_meter, refiner_meter, reflection_meter = CostMeter(), CostMeter(), CostMeter()
    program = SolverRefinerProgram(
        retriever=StubRetriever(),
        solver_lm=BedrockLM(model=SOLVER_MODEL),      # REAL client
        refiner_lm=BedrockLM(model=REFINER_MODEL),    # REAL client
        solver_meter=solver_meter,
        refiner_meter=refiner_meter,
        solver_model=SOLVER_MODEL,
        refiner_model=REFINER_MODEL,
    )
    grader = StubGrader()
    adapter = SweBenchAdapter(
        program=program, grader=grader, instances=instances,
        seed_cache=seed_cache, skip_ungradeable_patches=False,
    )
    reflection_lm = MeteredReflectionLM(       # REAL wrapper
        lm=BedrockLM(model=REFINER_MODEL), meter=reflection_meter, model=REFINER_MODEL
    )
    return SimpleNamespace(
        adapter=adapter, reflection_lm=reflection_lm, grader=grader, net=net,
        val_ids=val_ids, train_ids=train_ids, tmp_path=tmp_path,
        solver_meter=solver_meter, refiner_meter=refiner_meter, reflection_meter=reflection_meter,
    )


def _optimize(L, *, budget=5.0, name="run"):
    stopper = MaxTotalCostStopper(budget, meters=[L.solver_meter, L.refiner_meter, L.reflection_meter])
    return gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=list(L.train_ids),
        valset=list(L.val_ids),
        adapter=L.adapter,
        reflection_lm=L.reflection_lm,
        stop_callbacks=stopper,
        run_dir=str(L.tmp_path / name),
        seed=1,
        reflection_minibatch_size=3,
        display_progress_bar=False,
    )


# --------------------------------------------------------------------------
# The failure that killed launch #3
# --------------------------------------------------------------------------


def test_a_mutated_candidate_is_actually_evaluated(launch):
    """The decisive assertion.

    A non-callable reflection LM does not crash the run -- gepa swallows the
    TypeError and keeps iterating, proposing nothing. So "the run completed"
    proves nothing. What proves reflection worked is that a candidate DIFFERENT
    from the seed reached evaluation.

    This is asserted separately from acceptance: whether a proposal wins on
    score is the search's business, but a proposal must at least be produced
    and tried.
    """
    seen: list[dict] = []
    real_evaluate = launch.adapter.evaluate

    def spy(batch, candidate, capture_traces=False):
        seen.append(dict(candidate))
        return real_evaluate(batch, candidate, capture_traces=capture_traces)

    launch.adapter.evaluate = spy
    _optimize(launch, budget=3.0)

    mutated = [c for c in seen if c != SEED_CANDIDATE]
    assert mutated, (
        "only the seed candidate was ever evaluated -- reflection is silently "
        "failing, which is exactly the $100-for-nothing pattern"
    )


def test_the_candidate_pool_grows_when_a_proposal_wins(launch):
    """End of the chain: propose -> evaluate -> accept -> pool grows."""
    result = _optimize(launch, budget=3.0, name="grow")
    assert len(result.candidates) > 1, "no proposal was ever accepted into the pool"


def test_reflection_lm_is_invoked_and_metered(launch):
    _optimize(launch, budget=3.0, name="r2")
    assert launch.reflection_lm.calls > 0, "reflection LM was never called"
    assert launch.reflection_meter.budgeted_usd > 0, "reflection spend was not metered"


def test_reflection_spend_counts_toward_the_budget(launch):
    """cost.py says the budget covers reflection; before this it did not."""
    _optimize(launch, budget=3.0, name="r3")
    total = (
        launch.solver_meter.budgeted_usd
        + launch.refiner_meter.budgeted_usd
        + launch.reflection_meter.budgeted_usd
    )
    stopper = MaxTotalCostStopper(3.0, meters=[launch.solver_meter, launch.refiner_meter, launch.reflection_meter])
    assert stopper.realised_usd == pytest.approx(total)


def test_every_call_routes_to_bedrock(launch):
    """No call may reach api.anthropic.com."""
    _optimize(launch, budget=3.0, name="r4")
    assert launch.net.models_called, "no LM call was made at all"
    for model in launch.net.models_called:
        assert model.startswith("bedrock/"), f"{model} would not route to Bedrock"


def test_budget_stopper_actually_stops(launch):
    """A tiny budget must terminate the run rather than exhaust the trainset."""
    result = _optimize(launch, budget=0.60, name="r5")
    spent = (
        launch.solver_meter.budgeted_usd
        + launch.refiner_meter.budgeted_usd
        + launch.reflection_meter.budgeted_usd
    )
    assert spent >= 0.60, "stopper fired before the budget was reached"
    assert spent < 0.60 * 4, f"stopper overshot badly: ${spent:.2f} on a $0.60 budget"
    assert result is not None


def test_run_is_resumable_after_the_smoke_run(launch):
    _optimize(launch, budget=1.0, name="r6")
    assert (launch.tmp_path / "r6" / "gepa_state.bin").exists()


def test_seed_val_replay_still_free_and_train_runs_live(launch):
    """Routing rule holds under the real client too.

    Scoped to the BASE candidate: a mutated candidate legitimately grades val
    (that is the full val evaluation after acceptance). The invariant is only
    that the base candidate's val results come from the cache.
    """
    current: dict = {"candidate": None}
    real_evaluate = launch.adapter.evaluate
    real_grade = launch.grader.grade_batch
    base_val_graded: list[str] = []

    def eval_spy(batch, candidate, capture_traces=False):
        current["candidate"] = dict(candidate)
        return real_evaluate(batch, candidate, capture_traces=capture_traces)

    def grade_spy(items):
        if current["candidate"] == SEED_CANDIDATE:
            base_val_graded.extend(t.instance_id for t, _g, _p in items if t.instance_id in set(launch.val_ids))
        return real_grade(items)

    launch.adapter.evaluate = eval_spy
    launch.grader.grade_batch = grade_spy
    _optimize(launch, budget=3.0, name="r7")

    assert launch.grader.graded, "nothing was graded at all"
    assert not base_val_graded, f"base candidate re-graded val instances: {base_val_graded[:3]}"
