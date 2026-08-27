"""Adapter tests: seed-eval reuse, trace capture, and the gold audit."""

from __future__ import annotations

import pytest

from gepa_taxonomy.adapter import Grader, SweBenchAdapter
from gepa_taxonomy.cost import CostMeter
from gepa_taxonomy.program import COMPONENTS, SEED_CANDIDATE, RetrievedFile, SolverRefinerProgram
from gepa_taxonomy.seed_cache import SeedEvaluationCache, candidate_hash
from gepa_taxonomy.tasks import Instance, Task, split_row
from tests.test_gold_blindness import RAW_ROW  # reuse the realistic fixture

HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET = "us.anthropic.claude-sonnet-5"


class CountingLM:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str, *, max_tokens: int = 4096):
        self.calls += 1
        return "--- a/x.py\n+++ b/x.py\n@@\n-a\n+b\n", 1000, 100


class FakeRetriever:
    def retrieve(self, task: Task, *, k: int):
        return [RetrievedFile(path="x.py", content="a\n")]


class FakeGrader(Grader):
    def __init__(self, score: float = 1.0):
        self.score = score
        self.calls = 0

    def grade(self, task, gold, patch):
        self.calls += 1
        return self.score, {"resolved": bool(self.score)}


@pytest.fixture
def instance() -> Instance:
    return split_row(RAW_ROW)


@pytest.fixture
def adapter(instance):
    solver, refiner = CountingLM(), CountingLM()
    program = SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=solver,
        refiner_lm=refiner,
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    a = SweBenchAdapter(
        program=program,
        grader=FakeGrader(),
        instances={instance.task.instance_id: instance},
    )
    a._solver_lm = solver  # type: ignore[attr-defined]  # test handle
    return a


def test_evaluate_returns_scores_and_outputs(adapter, instance):
    res = adapter.evaluate([instance.task.instance_id], SEED_CANDIDATE)
    assert res.scores == [1.0]
    assert res.outputs[0]["instance_id"] == instance.task.instance_id


def test_evaluate_rejects_incomplete_candidate(adapter, instance):
    with pytest.raises(KeyError):
        adapter.evaluate([instance.task.instance_id], {COMPONENTS[0]: "only one"})


def test_traces_are_captured_for_phase_3(adapter, instance):
    adapter.evaluate([instance.task.instance_id], SEED_CANDIDATE)
    assert len(adapter._traces) == 1
    trace = adapter._traces[0]
    assert trace["instance_id"] == instance.task.instance_id
    assert "score" in trace and "feedback" in trace
    assert "solver_prompt_sha256" in trace


def test_trace_flush_writes_jsonl(adapter, instance, tmp_path):
    adapter.evaluate([instance.task.instance_id], SEED_CANDIDATE)
    out = adapter.flush_traces(tmp_path / "traces.jsonl")
    assert out and out.exists()
    assert len(out.read_text().strip().splitlines()) == 1
    assert adapter._traces == []


def test_reflective_dataset_has_no_gold(adapter, instance):
    res = adapter.evaluate([instance.task.instance_id], SEED_CANDIDATE, capture_traces=True)
    ds = adapter.make_reflective_dataset(SEED_CANDIDATE, res, COMPONENTS)
    from gepa_taxonomy.tasks import assert_gold_free

    assert set(ds) == set(COMPONENTS)
    assert_gold_free(ds, where="reflective dataset", gold=instance.gold)


def test_reflective_dataset_carries_metric_feedback(adapter, instance):
    """Optimizer-level metric feedback is legitimate -- that IS GEPA."""
    res = adapter.evaluate([instance.task.instance_id], SEED_CANDIDATE, capture_traces=True)
    ds = adapter.make_reflective_dataset(SEED_CANDIDATE, res, COMPONENTS)
    assert ds[COMPONENTS[0]][0]["score"] == 1.0


# --------------------------------------------------------------------------
# Seed evaluation reuse -- budget exclusion (a) and identical starting state
# --------------------------------------------------------------------------


def test_seed_cache_replays_without_any_lm_call(instance):
    """The point of the cache: identical starting state AND zero spend."""
    solver, refiner = CountingLM(), CountingLM()
    program = SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=solver,
        refiner_lm=refiner,
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    grader = FakeGrader()
    iid = instance.task.instance_id
    cache = SeedEvaluationCache.build(
        SEED_CANDIDATE,
        {iid: {"score": 0.0, "output": {"patch": "", "instance_id": iid}, "trace": {}}},
    )
    a = SweBenchAdapter(program=program, grader=grader, instances={iid: instance}, seed_cache=cache)

    res = a.evaluate([iid], SEED_CANDIDATE)

    assert res.scores == [0.0], "must return the stored result verbatim"
    assert solver.calls == 0 and refiner.calls == 0, "replay must issue no LM call"
    assert grader.calls == 0, "replay must not re-grade"
    assert program.solver_meter.budgeted_usd == 0.0, "replay must cost nothing"


def test_seed_cache_ignored_for_a_different_candidate(instance):
    """Only the base candidate is replayed; mutated candidates must really run."""
    solver, refiner = CountingLM(), CountingLM()
    program = SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=solver,
        refiner_lm=refiner,
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    iid = instance.task.instance_id
    cache = SeedEvaluationCache.build(SEED_CANDIDATE, {iid: {"score": 0.0, "output": {}, "trace": {}}})
    a = SweBenchAdapter(program=program, grader=FakeGrader(), instances={iid: instance}, seed_cache=cache)

    mutated = {**SEED_CANDIDATE, COMPONENTS[0]: "a mutated instruction"}
    a.evaluate([iid], mutated)
    assert solver.calls == 1, "a mutated candidate must actually be evaluated"


def test_seed_cache_miss_falls_through_instead_of_raising(instance):
    """CORRECTED invariant.

    This test previously asserted that any miss for the base candidate raised.
    That was the bug: it made a train-minibatch parent evaluation fatal. A miss
    is now a fall-through to the live path, and val completeness is enforced
    once at launch by assert_covers -- see the scope tests below.
    """
    cache = SeedEvaluationCache.build(SEED_CANDIDATE, {})
    assert cache.get(SEED_CANDIDATE, instance.task.instance_id) is None


def test_seed_cache_round_trips(tmp_path, instance):
    iid = instance.task.instance_id
    cache = SeedEvaluationCache.build(SEED_CANDIDATE, {iid: {"score": 0.5, "output": {}, "trace": {}}})
    path = cache.save(tmp_path / "seed.json")
    reloaded = SeedEvaluationCache.load(path)
    assert reloaded.candidate_fingerprint == cache.candidate_fingerprint
    assert reloaded.get(SEED_CANDIDATE, iid)["score"] == 0.5


def test_candidate_hash_matches_gepas_scheme():
    """We key the cache exactly as gepa does (state.py:31), so it stays
    interchangeable with gepa's own EvaluationCache if the upstream
    `evaluation_cache` passthrough on optimize() ever lands."""
    from gepa.core.state import _candidate_hash

    assert candidate_hash(SEED_CANDIDATE) == _candidate_hash(SEED_CANDIDATE)


def test_gold_audit_fires_on_a_leak(instance):
    """The adapter is the layer that holds gold, so it runs the value check."""

    class LeakyRetriever:
        def retrieve(self, task, *, k):
            return [RetrievedFile(path="t.py", content=instance.gold.test_patch)]

    program = SolverRefinerProgram(
        retriever=LeakyRetriever(),
        solver_lm=CountingLM(),
        refiner_lm=CountingLM(),
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    iid = instance.task.instance_id
    a = SweBenchAdapter(program=program, grader=FakeGrader(), instances={iid: instance})
    from gepa_taxonomy.tasks import GoldLeakError

    with pytest.raises(GoldLeakError):
        a.evaluate([iid], SEED_CANDIDATE)


def test_adapter_declares_propose_new_texts():
    """gepa v0.1.4 reads adapter.propose_new_texts UNCONDITIONALLY
    (reflective_mutation.py:176). An adapter that omits it raises
    AttributeError, which gepa swallows as "did not propose a new candidate" --
    so a baseline run would burn its entire budget and never leave the seed.

    Found by a resume test that was passing for the wrong reason.
    """
    assert hasattr(SweBenchAdapter, "propose_new_texts")
    assert SweBenchAdapter.propose_new_texts is None, "None = 'use the reflection LM'"


def test_adapter_instance_exposes_propose_new_texts(adapter):
    assert adapter.propose_new_texts is None


class BatchGrader(Grader):
    """Records how many harness invocations the adapter makes."""

    def __init__(self, score: float = 1.0):
        self.score = score
        self.batch_calls = 0
        self.graded_ids: list[str] = []

    def grade_batch(self, items):
        self.batch_calls += 1
        self.graded_ids += [t.instance_id for t, _g, _p in items]
        return {t.instance_id: (self.score, {"resolved": bool(self.score)}) for t, _g, _p in items}

    def grade(self, task, gold, patch):
        return self.grade_batch([(task, gold, patch)])[task.instance_id]


def _multi_instances(n: int):
    from gepa_taxonomy.tasks import split_row

    out = {}
    for i in range(n):
        row = dict(RAW_ROW)
        row["instance_id"] = f"repo__proj-{i}"
        inst = split_row(row)
        out[inst.task.instance_id] = inst
    return out


def test_batch_is_graded_in_a_single_harness_call():
    """Per-instance grading would pay harness startup on every rollout; at a
    ~1.9 min emulated evaluation that overhead is large."""
    insts = _multi_instances(5)
    grader = BatchGrader()
    program = SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=CountingLM(),
        refiner_lm=CountingLM(),
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    ad = SweBenchAdapter(program=program, grader=grader, instances=insts, skip_ungradeable_patches=False)
    res = ad.evaluate(list(insts), SEED_CANDIDATE)

    assert len(res.scores) == 5
    assert grader.batch_calls == 1, f"expected one harness call, got {grader.batch_calls}"
    assert sorted(grader.graded_ids) == sorted(insts)


def test_results_stay_in_batch_order():
    """Deferring grading must not reorder outputs relative to the input batch."""
    insts = _multi_instances(4)
    program = SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=CountingLM(),
        refiner_lm=CountingLM(),
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    ad = SweBenchAdapter(program=program, grader=BatchGrader(), instances=insts, skip_ungradeable_patches=False)
    ids = list(insts)
    res = ad.evaluate(ids, SEED_CANDIDATE)
    assert [o["instance_id"] for o in res.outputs] == ids


def test_skipped_patches_never_reach_the_grader():
    """The whole point of the gate: no container for an ungradeable patch."""

    class EmptyLM:
        calls = 0

        def complete(self, prompt, *, max_tokens=4096):
            return "I could not solve this.", 100, 10

    insts = _multi_instances(3)
    grader = BatchGrader()
    program = SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=EmptyLM(),
        refiner_lm=EmptyLM(),
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    ad = SweBenchAdapter(program=program, grader=grader, instances=insts, skip_ungradeable_patches=True)
    res = ad.evaluate(list(insts), SEED_CANDIDATE)

    assert grader.batch_calls == 0, "ungradeable patches reached the grader"
    assert all(s == 0.0 for s in res.scores)
    assert all(t["grading"]["skipped"] for t in ad._traces)


def test_adapter_consumes_the_shape_gepa_actually_passes():
    """gepa hands adapter.evaluate() the dataset ITEMS, not ids.

    So the datasets we give gepa must be instance-id strings -- which is what
    run_seed.py passes. This test pins the contract; getting it wrong would
    fail only at launch, after the base-val evaluation had been paid for.
    """
    import inspect

    import gepa

    insts = _multi_instances(3)
    ids = list(insts)
    program = SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=CountingLM(),
        refiner_lm=CountingLM(),
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    ad = SweBenchAdapter(program=program, grader=BatchGrader(), instances=insts, skip_ungradeable_patches=False)
    # Exactly what gepa would hand us for a minibatch of ids.
    res = ad.evaluate(ids, SEED_CANDIDATE)
    assert [o["instance_id"] for o in res.outputs] == ids
    assert "gepa" in inspect.getmodule(gepa).__name__


# --------------------------------------------------------------------------
# Seed-cache replay SCOPE: (base candidate) x (val instances) only
# --------------------------------------------------------------------------


def test_seed_cache_does_not_intercept_train_instances(instance):
    """The bug that killed the second launch.

    The base candidate is evaluated on TRAIN minibatches during reflective
    mutation (parent evals). Those are ordinary billed rollouts. Treating the
    absence of a train id as an incomplete cache raised KeyError and killed the
    run at the first parent evaluation.
    """
    from gepa_taxonomy.seed_cache import SeedEvaluationCache

    val_id, train_id = "val__proj-1", instance.task.instance_id
    cache = SeedEvaluationCache.build(SEED_CANDIDATE, {val_id: {"score": 1.0, "output": {}, "trace": {}}})
    assert cache.get(SEED_CANDIDATE, val_id) is not None, "val id must replay"
    assert cache.get(SEED_CANDIDATE, train_id) is None, "train id must fall through to live"


def test_adapter_runs_the_program_for_an_out_of_scope_instance(instance):
    """End-to-end: a train instance must reach the live path, not raise."""
    from gepa_taxonomy.seed_cache import SeedEvaluationCache

    iid = instance.task.instance_id
    solver, refiner = CountingLM(), CountingLM()
    program = SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=solver,
        refiner_lm=refiner,
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    cache = SeedEvaluationCache.build(SEED_CANDIDATE, {"some__other-1": {"score": 1.0, "output": {}, "trace": {}}})
    ad = SweBenchAdapter(
        program=program,
        grader=FakeGrader(),
        instances={iid: instance},
        seed_cache=cache,
        skip_ungradeable_patches=False,
    )
    res = ad.evaluate([iid], SEED_CANDIDATE)
    assert solver.calls == 1, "the out-of-scope instance must actually run"
    assert res.scores == [1.0]


def test_assert_covers_still_enforces_val_completeness():
    """the recorded requirement guarantee survives the scoping fix -- just moved to launch time."""
    from gepa_taxonomy.seed_cache import SeedEvaluationCache

    cache = SeedEvaluationCache.build(SEED_CANDIDATE, {"v1": {"score": 1.0, "output": {}, "trace": {}}})
    cache.assert_covers(["v1"])  # complete -> no raise
    with pytest.raises(KeyError, match="missing 1 of the val instances"):
        cache.assert_covers(["v1", "v2"])


# --------------------------------------------------------------------------
# Reflective dataset: harness substance + train-only gold
# --------------------------------------------------------------------------


class SubstanceGrader(Grader):
    """Returns the enriched detail shape LocalDockerGrader now produces."""

    def grade(self, task, gold, patch):
        return 0.0, {
            "resolved": False,
            "errored": False,
            "empty_patch": False,
            "run_id": "gepa-deadbeef00",
            "failing_tests": ["test_widget_renders_the_placeholder_value"],
            "test_output_tail": "AssertionError: {'field': ['%(value)s']} != {'field': ['a b']}",
        }


def _adapter_for(instance, *, grader=None, reflection_gold_ids=None):
    program = SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=CountingLM(),
        refiner_lm=CountingLM(),
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    return SweBenchAdapter(
        program=program,
        grader=grader or FakeGrader(),
        instances={instance.task.instance_id: instance},
        reflection_gold_ids=reflection_gold_ids,
    )


def test_reflective_dataset_surfaces_harness_substance(instance):
    """Reflection must see WHY the harness failed a patch, not a bare 0 --
    a bare score produced only generic, informationless rewrites."""
    ad = _adapter_for(instance, grader=SubstanceGrader())
    res = ad.evaluate([instance.task.instance_id], SEED_CANDIDATE, capture_traces=True)
    ds = ad.make_reflective_dataset(SEED_CANDIDATE, res, COMPONENTS)

    for component in COMPONENTS:
        hr = ds[component][0]["harness_result"]
        assert hr["resolved"] is False
        assert hr["failing_tests"] == ["test_widget_renders_the_placeholder_value"]
        assert "AssertionError" in hr["test_output_tail"]
        assert "run_id" not in hr, "run ids name harness invocations, not patch behaviour"


def test_skip_gate_reason_reaches_reflection(instance):
    """A gate-skipped rollout must tell reflection it never reached a container."""

    class ProseLM:
        def complete(self, prompt, *, max_tokens=4096):
            return "I could not solve this.", 100, 10

    program = SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=ProseLM(),
        refiner_lm=ProseLM(),
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    ad = SweBenchAdapter(
        program=program,
        grader=FakeGrader(),
        instances={instance.task.instance_id: instance},
        skip_ungradeable_patches=True,
    )
    res = ad.evaluate([instance.task.instance_id], SEED_CANDIDATE, capture_traces=True)
    ds = ad.make_reflective_dataset(SEED_CANDIDATE, res, COMPONENTS)
    hr = ds[COMPONENTS[0]][0]["harness_result"]
    assert hr["skipped"] is True
    assert hr["skipped_reason"]


def test_reference_patch_present_for_reflection_gold_ids(instance):
    """Train-manifest ids get the gold patch, neutrally labeled."""
    iid = instance.task.instance_id
    ad = _adapter_for(instance, reflection_gold_ids={iid})
    res = ad.evaluate([iid], SEED_CANDIDATE, capture_traces=True)
    ds = ad.make_reflective_dataset(SEED_CANDIDATE, res, COMPONENTS)

    for component in COMPONENTS:
        ex = ds[component][0]
        # The fixture gold patch is shorter than the excerpt cap, so verbatim.
        assert ex["reference_patch"] == instance.gold.patch
        assert ex["note"] == "reference_patch is the reference solution for this training example"


def test_reference_patch_absent_for_ids_outside_the_set(instance):
    """The other direction: an id NOT in reflection_gold_ids -- a val or test
    instance -- must receive no gold, and the example must stay gold-free."""
    from gepa_taxonomy.tasks import assert_gold_free

    ad = _adapter_for(instance, reflection_gold_ids={"some__other-99"})
    res = ad.evaluate([instance.task.instance_id], SEED_CANDIDATE, capture_traces=True)
    ds = ad.make_reflective_dataset(SEED_CANDIDATE, res, COMPONENTS)

    ex = ds[COMPONENTS[0]][0]
    assert "reference_patch" not in ex and "note" not in ex
    assert_gold_free(ds, where="reflective dataset (id outside gold set)", gold=instance.gold)


def test_reference_patch_absent_when_no_gold_ids_configured(adapter, instance):
    """reflection_gold_ids=None (the default) means gold enters nowhere."""
    res = adapter.evaluate([instance.task.instance_id], SEED_CANDIDATE, capture_traces=True)
    ds = adapter.make_reflective_dataset(SEED_CANDIDATE, res, COMPONENTS)
    ex = ds[COMPONENTS[0]][0]
    assert "reference_patch" not in ex and "note" not in ex


def test_audit_allows_model_output_matching_gold(instance):
    """A model that solves an instance emits the reference patch. That is a
    correct solve, not a leak -- only inputs are value-checked against gold."""
    from gepa_taxonomy.adapter import _MODEL_OUTPUT_FIELDS
    from gepa_taxonomy.tasks import GoldLeakError

    assert _MODEL_OUTPUT_FIELDS == {"solver_patch", "refiner_patch"}

    gold = instance.gold
    ad = _adapter_for(instance)
    rollout = ad.program.run(instance.task, SEED_CANDIDATE)

    # The model's own output reproducing gold verbatim must NOT trip the audit.
    rollout.solver_patch = gold.patch
    rollout.refiner_patch = gold.patch
    ad._audit(rollout, gold)

    # The same text arriving in a PROMPT is still a hard failure.
    rollout.solver_prompt = "here is the fix:\n" + gold.patch
    with pytest.raises(GoldLeakError):
        ad._audit(rollout, gold)


def test_apply_verdict_reaches_the_refiner(tmp_path):
    """The refiner exists to repair a non-applying patch; before this it was
    never told the patch did not apply (applies_cleanly was None on every
    rollout of the first seed run)."""
    import subprocess

    from gepa_taxonomy.program import static_feedback

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "mod.py").write_text("def f():\n    return 1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=repo, check=True,
    )

    good = "--- a/mod.py\n+++ b/mod.py\n@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n"
    fb = static_feedback(good, repo_dir=repo)
    assert fb.applies_cleanly is True
    assert any("applies cleanly" in m for m in fb.messages)

    bad = "--- a/mod.py\n+++ b/mod.py\n@@ -1,2 +1,2 @@\n def totally_other():\n-    return 99\n+    return 2\n"
    fb = static_feedback(bad, repo_dir=repo)
    assert fb.applies_cleanly is False
    assert any("does not apply" in m for m in fb.messages), fb.messages
    # the reason must name the failure concretely, not just "it failed"
    assert any(("hunk" in m.lower()) or ("error" in m.lower()) for m in fb.messages), fb.messages
    assert any("mod.py" in m for m in fb.messages), fb.messages

    # Without a checkout the verdict is unknown -- never a false accusation.
    assert static_feedback(bad, repo_dir=None).applies_cleanly is None


def test_program_passes_apply_verdict_into_the_refiner_prompt(instance, tmp_path):
    """Integration guard for the wiring itself: the whole point of the apply
    check is that it reaches the REFINER's prompt, not just the feedback object."""
    import subprocess

    from gepa_taxonomy.program import SolverRefinerProgram

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "x.py").write_text("real content\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
        cwd=repo, check=True,
    )

    class RecordingLM(CountingLM):
        def __init__(self):
            super().__init__()
            self.prompts: list[str] = []

        def complete(self, prompt: str, *, max_tokens: int = 4096):
            self.prompts.append(prompt)
            return super().complete(prompt, max_tokens=max_tokens)

    refiner = RecordingLM()
    program = SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=CountingLM(),
        refiner_lm=refiner,
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
        repo_dir_for=lambda _t: repo,
    )
    program.run(instance.task, SEED_CANDIDATE)

    prompt = refiner.prompts[0]
    assert "does not apply" in prompt, "refiner was not told the patch fails to apply"
    assert "Rewrite it so every context line matches" in prompt


def test_gold_in_prompt_rejects_the_candidate_not_the_run(instance):
    """A proposal that memorised a reference patch puts gold in its own prompt.
    That candidate must score 0 (so the acceptance gate rejects it) while the
    run continues -- and the leaked rollout must never be scored or cached."""
    from gepa_taxonomy.program import COMPONENTS

    leaky = dict(SEED_CANDIDATE)
    leaky[COMPONENTS[0]] = SEED_CANDIDATE[COMPONENTS[0]] + "\n\nKnown fix:\n" + instance.gold.patch

    ad = _adapter_for(instance)
    res = ad.evaluate([instance.task.instance_id], leaky, capture_traces=True)

    assert res.scores == [0.0], "a gold-leaking candidate must not be credited"
    assert res.trajectories[0]["grading"]["gold_in_prompt"] is True
    assert not res.outputs[0]["patch"], "the leaked rollout's output must be discarded"

    # a clean candidate still evaluates normally afterwards
    ok = ad.evaluate([instance.task.instance_id], SEED_CANDIDATE, capture_traces=True)
    assert ok.trajectories[0].get("grading", {}).get("gold_in_prompt") is None


def test_stored_trace_keeps_the_produced_patch(instance):
    """Reflection critiques the patch and the judge reads it, so the stored
    trace must carry it. A gold-audit change once stripped it here, and every
    reflective example silently showed an empty produced_patch."""
    ad = _adapter_for(instance)
    res = ad.evaluate([instance.task.instance_id], SEED_CANDIDATE, capture_traces=True)
    traj = res.trajectories[0]
    assert traj.get("solver_patch"), "solver_patch missing from the stored trace"
    assert traj.get("refiner_patch"), "refiner_patch missing from the stored trace"

    ds = ad.make_reflective_dataset(SEED_CANDIDATE, res, COMPONENTS)
    for component in COMPONENTS:
        assert ds[component][0]["produced_patch"], f"{component} reflection saw an empty patch"


def test_reflection_spend_is_written_to_disk(tmp_path):
    """The watchdog enforces the dollar ceiling out of process, so it can only
    count what is on disk. Reflection spend was invisible to it."""
    import json

    from gepa_taxonomy.bedrock import MeteredReflectionLM

    class StubLM:
        def complete(self, prompt, *, max_tokens=8192):
            return "proposed instruction", 1000, 200

    log = tmp_path / "reflection.jsonl"
    lm = MeteredReflectionLM(
        lm=StubLM(), meter=CostMeter(), model=SONNET, spend_log=log
    )
    lm("reflect on this")
    lm("and this")

    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 2, "every reflection call must be logged"
    assert all(r["cost_usd"] > 0 for r in rows), "a logged call must carry its cost"
    assert rows[0]["input_tokens"] == 1000 and rows[0]["output_tokens"] == 200
