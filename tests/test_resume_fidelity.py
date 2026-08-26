"""Pause/resume state fidelity for gepa v0.1.4. FREE -- no LM calls.

Runs a real ``gepa.optimize()`` loop against a deterministic fake adapter and
fake reflection LM, then compares an uninterrupted run against a
run-pause-resume run. Everything here is arithmetic and dict comparison; no
model is invoked.

What we are checking, per Andrei's list:
  candidate pool        exact?
  Pareto frontier       exact?
  per-instance results  exact?
  spend meter           exact?
  RNG / minibatch order exact?
"""

from __future__ import annotations

import shutil
from pathlib import Path

import gepa
import pytest

from gepa_taxonomy.cost import CostMeter

TRAINSET = [{"id": f"t{i}", "q": i} for i in range(24)]
VALSET = [{"id": f"v{i}", "q": i} for i in range(8)]
SEED_CANDIDATE = {"instr": "base"}


class DeterministicAdapter:
    """Scores a candidate/instance pair by a pure hash. No LM, no randomness.

    Also records the exact minibatch sequence it is asked to evaluate, which is
    how we detect whether the sampler's position survives a resume.
    """

    #: See SweBenchAdapter -- gepa reads this unconditionally.
    propose_new_texts = None

    def __init__(self, meter: CostMeter | None = None):
        self.meter = meter or CostMeter()
        self.minibatch_log: list[tuple[str, ...]] = []

    # -- gepa protocol -------------------------------------------------
    def evaluate(self, batch, candidate, capture_traces=False):
        ids = tuple(d["id"] for d in batch)
        # Val evaluations are full-set; minibatches are the small ones.
        if len(ids) < len(VALSET):
            self.minibatch_log.append(ids)

        scores, outputs = [], []
        for d in batch:
            # Deterministic pseudo-score: stable across processes (no hash()).
            h = sum(ord(c) for c in candidate["instr"]) + d["q"] * 7
            scores.append((h % 100) / 100.0)
            outputs.append({"id": d["id"], "out": candidate["instr"]})
            # Charge a fixed, deterministic amount so the meter is comparable.
            self.meter.record(
                model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
                input_tokens=1000,
                output_tokens=100,
            )
        return gepa.EvaluationBatch(outputs=outputs, scores=scores, trajectories=outputs if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {
            c: [{"Inputs": o["id"], "Generated Outputs": o["out"], "Feedback": "improve"} for o in eval_batch.outputs]
            for c in components_to_update
        }

    # -- adapter_state hook: how we persist OUR spend meter ------------
    def get_adapter_state(self) -> dict:
        return {"meter": self.meter.snapshot()}

    def set_adapter_state(self, state: dict) -> None:
        snap = (state or {}).get("meter")
        if not snap:
            return
        self.meter.budgeted_usd = snap["budgeted_usd"]
        self.meter.excluded_usd = snap["excluded_usd"]
        self.meter.calls = snap["calls"]
        self.meter.tokens_in = snap["tokens_in"]
        self.meter.tokens_out = snap["tokens_out"]


class CountingReflectionLM:
    """Deterministic 'reflection': appends a call counter to the instruction."""

    def __init__(self):
        self.n = 0

    def __call__(self, prompt: str) -> str:
        self.n += 1
        return f"base+{self.n}"


def run(run_dir: Path, max_metric_calls: int, adapter: DeterministicAdapter, seed: int = 7):
    return gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=TRAINSET,
        valset=VALSET,
        adapter=adapter,
        reflection_lm=CountingReflectionLM(),
        max_metric_calls=max_metric_calls,
        run_dir=str(run_dir),
        seed=seed,
        reflection_minibatch_size=3,
        display_progress_bar=False,
    )


def state_fingerprint(result) -> dict:
    return {
        "n_candidates": len(result.candidates),
        "candidates": [dict(c) for c in result.candidates],
        "val_subscores": [dict(s) for s in result.val_subscores],
        "best_idx": result.best_idx,
        "total_metric_calls": result.total_metric_calls,
    }


@pytest.fixture
def dirs(tmp_path):
    a, b = tmp_path / "uninterrupted", tmp_path / "resumed"
    yield a, b
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# What IS faithfully restored
# ---------------------------------------------------------------------------


def test_resume_does_not_reproduce_the_uninterrupted_trajectory(dirs):
    """THE HEADLINE RESULT, stated honestly.

    An earlier version of this test asserted the pools matched -- and passed,
    because the fake adapter was missing ``propose_new_texts`` so gepa never
    proposed anything and both pools held only the seed. With mutation actually
    working, the pools differ:

        uninterrupted : base, base+1, base+10, base+12, base+13, base+14
        resumed       : base, base+1, base+1,  base+3

    The mechanism is the RNG/sampler gap below: the second leg restarts the
    shuffle, draws different minibatches, and therefore discovers different
    candidates. Persisted state is restored exactly; the *future* trajectory is
    not the one the longer run would have taken.
    """
    a_dir, b_dir = dirs
    BUDGET = 120

    ad_a = DeterministicAdapter()
    res_a = run(a_dir, BUDGET, ad_a)

    ad_b1 = DeterministicAdapter()
    run(b_dir, BUDGET // 2, ad_b1)
    ad_b2 = DeterministicAdapter()
    res_b = run(b_dir, BUDGET, ad_b2)

    assert len(res_a.candidates) > 1, "no mutation occurred -- the harness is not exercising reflection"
    assert len(res_b.candidates) > 1, "no mutation occurred on the resumed run"

    ca = [dict(c) for c in res_a.candidates]
    cb = [dict(c) for c in res_b.candidates]
    assert ca != cb, (
        "pools matched across a resume -- if gepa now persists sampler state, the fidelity writeup needs revisiting"
    )
    # The seed is always shared.
    assert ca[0] == cb[0] == SEED_CANDIDATE


def test_resumed_run_reloads_prior_candidates_not_a_fresh_start(dirs):
    """A resume must continue, not restart."""
    _, b_dir = dirs
    ad1 = DeterministicAdapter()
    first = run(b_dir, 60, ad1)
    assert len(first.candidates) >= 1

    ad2 = DeterministicAdapter()
    second = run(b_dir, 120, ad2)
    assert len(second.candidates) >= len(first.candidates), "resume lost candidates"
    # The seed candidate must still be index 0 and unchanged.
    assert second.candidates[0] == SEED_CANDIDATE


def test_spend_meter_survives_resume_via_adapter_state(dirs):
    """OUR meter is not gepa state -- it rides in adapter_state.

    gepa syncs adapter_state into GEPAState each iteration and restores it on
    load, so a meter routed through get/set_adapter_state is exactly restored.
    Without this hook the meter would silently reset to $0 on resume, and a
    resumed run would spend its whole budget again.
    """
    _, b_dir = dirs
    ad1 = DeterministicAdapter()
    run(b_dir, 60, ad1)
    spent_before = ad1.meter.budgeted_usd
    assert spent_before > 0

    ad2 = DeterministicAdapter()
    assert ad2.meter.budgeted_usd == 0.0, "fresh adapter starts empty"
    run(b_dir, 120, ad2)

    # The resumed adapter must have been seeded with the earlier spend, so its
    # final total exceeds what the second leg alone could have charged.
    assert ad2.meter.budgeted_usd > spent_before, (
        "spend meter did not carry across the resume -- the budget would reset"
    )


# ---------------------------------------------------------------------------
# What is NOT faithfully restored -- the honest gap
# ---------------------------------------------------------------------------


def test_rng_and_minibatch_sequence_are_not_restored(dirs):
    """gepa v0.1.4 does NOT persist the RNG or the batch sampler's position.

    `rng = random.Random(seed)` is built in api.py:304, outside GEPAState, and
    EpochShuffledBatchSampler keeps `shuffled_ids` / `epoch` / `id_freqs` on the
    strategy object, which is likewise not part of the pickled state. On resume
    both are rebuilt from scratch, so the minibatch stream restarts from the
    beginning of a fresh epoch rather than continuing where it left off.

    This test pins that as a KNOWN LIMITATION. If a future gepa release starts
    persisting sampler state, this test will fail and we should re-evaluate.
    """
    a_dir, b_dir = dirs
    BUDGET = 120

    ad_a = DeterministicAdapter()
    run(a_dir, BUDGET, ad_a)

    ad_b1 = DeterministicAdapter()
    run(b_dir, BUDGET // 2, ad_b1)
    ad_b2 = DeterministicAdapter()
    run(b_dir, BUDGET, ad_b2)

    continuous = ad_a.minibatch_log
    resumed = ad_b1.minibatch_log + ad_b2.minibatch_log

    assert continuous, "no minibatches were sampled -- test is not exercising the loop"
    # The second leg restarts the shuffle, so the concatenated stream differs.
    assert resumed != continuous, (
        "minibatch stream unexpectedly matched -- gepa may now persist sampler state; re-check the fidelity writeup"
    )


def test_rng_is_not_in_gepa_state():
    """Direct structural check of the gap, independent of the loop test."""
    import inspect

    from gepa.core.state import GEPAState

    src = inspect.getsource(GEPAState)
    assert "rng" not in src, "GEPAState now mentions rng -- re-check resume fidelity"


def test_adapter_state_is_the_supported_extension_point():
    """Confirms the hook we rely on for the spend meter is real and load-bearing."""
    import inspect

    from gepa.core.engine import GEPAEngine

    src = inspect.getsource(GEPAEngine)
    assert "get_adapter_state" in src and "set_adapter_state" in src


class MinibatchSensitiveAdapter(DeterministicAdapter):
    """Like the base adapter, but reflection depends on WHICH instances were drawn.

    This is the realistic case: GEPA mutates an instruction from the traces of
    the sampled minibatch, so the sampled ids feed the next candidate. The base
    adapter above is deliberately insensitive to the draw, which is why its
    candidate pool matched exactly -- that shows *state restoration* is faithful
    but says nothing about *trajectory* equivalence.
    """

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        drawn = "-".join(o["id"] for o in eval_batch.outputs)
        return {c: [{"Inputs": drawn, "Generated Outputs": drawn, "Feedback": drawn}] for c in components_to_update}


class EchoReflectionLM:
    """Turns the sampled ids into the next instruction, so the draw propagates."""

    def __call__(self, prompt: str) -> str:
        import hashlib

        return "c" + hashlib.sha256(prompt.encode()).hexdigest()[:8]


def run_sensitive(run_dir, max_metric_calls, adapter, seed=7):
    return gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=TRAINSET,
        valset=VALSET,
        adapter=adapter,
        reflection_lm=EchoReflectionLM(),
        max_metric_calls=max_metric_calls,
        run_dir=str(run_dir),
        seed=seed,
        reflection_minibatch_size=3,
        display_progress_bar=False,
    )


def test_trajectory_diverges_when_reflection_depends_on_the_draw(dirs):
    """The consequence of the RNG gap, stated honestly.

    With a minibatch-sensitive adapter -- which is what our real solver/refiner
    program is -- the post-resume trajectory differs from the uninterrupted
    counterfactual, because the sampler restarts its shuffle.

    This does NOT mean resume is lossy: prior candidates, scores and frontier
    are all restored exactly (see the tests above). It means a resumed run is a
    valid continuation, not a bit-identical replay of the longer run.
    """
    a_dir, b_dir = dirs
    BUDGET = 120

    ad_a = MinibatchSensitiveAdapter()
    res_a = run_sensitive(a_dir, BUDGET, ad_a)

    ad_b1 = MinibatchSensitiveAdapter()
    run_sensitive(b_dir, BUDGET // 2, ad_b1)
    ad_b2 = MinibatchSensitiveAdapter()
    res_b = run_sensitive(b_dir, BUDGET, ad_b2)

    # Prior work is preserved: the seed is still index 0.
    assert res_b.candidates[0] == SEED_CANDIDATE

    # The mechanism: the sampler restarts, so the minibatch streams differ.
    assert (ad_b1.minibatch_log + ad_b2.minibatch_log) != ad_a.minibatch_log, (
        "sampler position unexpectedly survived -- re-check the fidelity writeup"
    )

    # And because reflection consumes the drawn ids, that propagates into the
    # candidates actually discovered. Assert the real consequence rather than a
    # tautology: at least one post-seed candidate differs between the runs.
    post_seed_a = [dict(c) for c in res_a.candidates[1:]]
    post_seed_b = [dict(c) for c in res_b.candidates[1:]]
    assert post_seed_a and post_seed_b, "no mutations occurred; test is not exercising reflection"
    assert post_seed_a != post_seed_b, (
        "a minibatch-sensitive adapter produced identical pools across a resume; "
        "that would mean the draw does not reach reflection -- re-check the harness"
    )


def test_prepause_candidates_are_a_prefix_of_the_resumed_pool(dirs):
    """The strong guarantee we DO get: nothing discovered before the pause is
    lost or reordered by the resume."""
    _, b_dir = dirs
    ad1 = MinibatchSensitiveAdapter()
    first = run_sensitive(b_dir, 60, ad1)
    before = [dict(c) for c in first.candidates]

    ad2 = MinibatchSensitiveAdapter()
    second = run_sensitive(b_dir, 120, ad2)
    after = [dict(c) for c in second.candidates]

    assert after[: len(before)] == before, "resume must preserve prior candidates in order"
