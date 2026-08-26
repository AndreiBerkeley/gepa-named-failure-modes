"""Full-launch smoke test for the HotpotQA arm, with ONLY the network faked. FREE.

A fourth launch has now died in a fourth place: ``MeteredReflectionLM`` was
constructed without its ``lm``, raising ``TypeError`` before ``gepa.optimize()``.
That one was loud and cost nothing -- but it is the same class of bug as the
three in ``test_launch_smoke.py``, and the same lesson: the run script's object
graph was never actually constructed by a test.

So this stubs at the **lowest possible layer** -- ``litellm.completion`` -- and
builds everything above it for real: the real ``BedrockLM``, the real
``MeteredReflectionLM``, the real program, the real adapter, the real cost
meters and stopper, real manifests, and gepa's real reflection machinery. The
retriever is stubbed only because the 1.2 GB BM25 index is not a test fixture.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gepa
import pytest

from failure_taxonomy import FAILURE_MODES_KEY
from gepa_taxonomy.bedrock import BedrockLM, MeteredReflectionLM, verify_reflection_lm
from gepa_taxonomy.cost import CostMeter, MaxTotalCostStopper
from gepa_taxonomy.hotpotqa.adapter import HotpotQAAdapter, instances_by_id
from gepa_taxonomy.hotpotqa.program import SEED_CANDIDATE, MultiHopProgram
from gepa_taxonomy.hotpotqa.retrieval import Passage
from gepa_taxonomy.hotpotqa.tasks import instance_from_record

REPO = Path(__file__).resolve().parents[1]

SOLVER = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
REFLECTION = "global.anthropic.claude-sonnet-4-6"

#: Last line of the final_answer template -- an exact tell, rather than guessing
#: from keywords. gepa's reflection prompt embeds our rollout traces verbatim,
#: so any "does it look like a question" heuristic misclassifies it.
ANSWER_TEMPLATE_MARKER = "Respond with the answer only, as briefly as possible."
MUTATION_MARKER = "an improved instruction"


class FakeNetwork:
    """Stands in for ``litellm.completion``."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if ANSWER_TEMPLATE_MARKER in prompt:
            # A rollout driven by a MUTATED instruction answers correctly, so
            # the grader can reward it and gepa can accept a candidate. Without
            # this every candidate scores 0.0, which is indistinguishable from
            # reflection being broken.
            content = "Paris" if MUTATION_MARKER in prompt else "wrong"
        elif "Respond with the search query only." in prompt:
            content = "second hop query"
        elif "Respond with the summary only." in prompt:
            content = "a summary"
        else:
            content = f"```\n{MUTATION_MARKER}\n```"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=1500, completion_tokens=100),
        )

    @property
    def models_called(self) -> set[str]:
        return {c["model"] for c in self.calls}


class StubRetriever:
    def retrieve(self, query, *, k=None):
        return [Passage("Alpha", "alpha body"), Passage("Beta", "beta body")]


def _instances(n=6):
    return [
        instance_from_record(
            {
                "id": f"q{i}",
                "question": f"question {i}?",
                "answer": "Paris",
                "level": "hard",
                "type": "bridge",
                "supporting_facts": {"title": ["Alpha", "Beta"], "sent_id": [0, 0]},
            }
        )
        for i in range(n)
    ]


@pytest.fixture
def network(monkeypatch):
    import litellm

    fake = FakeNetwork()
    monkeypatch.setattr(litellm, "completion", fake)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    return fake


def _build(tmp_path, budget=5.0):
    """Construct exactly what ``scripts/run_hotpotqa_seed.py`` constructs."""
    train, val = _instances(6), _instances(6)
    solver_meter, reflection_meter = CostMeter(), CostMeter()
    instances = instances_by_id([*train, *val])

    program = MultiHopProgram(
        retriever=StubRetriever(),
        lm=BedrockLM(model=SOLVER, max_retries=8),
        meter=solver_meter,
        model=SOLVER,
    )
    adapter = HotpotQAAdapter(
        program=program,
        instances=instances,
        reflection_gold_ids=frozenset(i.task.example_id for i in train),
    )
    reflection_lm = MeteredReflectionLM(
        lm=BedrockLM(model=REFLECTION, max_retries=8),
        meter=reflection_meter,
        model=REFLECTION,
        spend_log=tmp_path / "reflection_spend.jsonl",
    )
    stopper = MaxTotalCostStopper(budget, [solver_meter, reflection_meter])
    return train, val, adapter, reflection_lm, stopper, solver_meter, reflection_meter


def test_the_reflection_lm_is_constructed_correctly(tmp_path, network):
    """The exact bug that killed the first launch: MeteredReflectionLM built
    without its `lm`. Loud and free, but only if something constructs it."""
    *_, reflection_lm, _, _, _ = _build(tmp_path)[2:] + (None,)
    report = verify_reflection_lm(reflection_lm)
    assert report["ok"] and report["callable"]


def test_reflection_lm_preflight_rejects_a_bare_bedrock_lm(network):
    """gepa SWALLOWS this one: it logs 'did not propose a new candidate' and the
    run burns its whole budget without ever leaving the seed."""
    from gepa_taxonomy.bedrock import ReflectionConformanceError

    with pytest.raises(ReflectionConformanceError, match="not callable"):
        verify_reflection_lm(BedrockLM(model=REFLECTION))


def test_a_full_launch_completes_and_proposes(tmp_path, network):
    train, val, adapter, reflection_lm, stopper, solver_meter, reflection_meter = _build(tmp_path)

    result = gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        stop_callbacks=[stopper],
        seed=1,
        display_progress_bar=False,
        run_dir=str(tmp_path / "run"),
    )

    assert len(result.candidates) > 1, "reflection never proposed a candidate"
    assert reflection_meter.calls > 0, "reflection spend was never metered"
    assert solver_meter.budgeted_usd > 0
    assert network.models_called == {f"bedrock/{SOLVER}", f"bedrock/{REFLECTION}"}


def test_the_optimizer_finds_the_better_candidate(tmp_path, network):
    train, val, adapter, reflection_lm, stopper, *_ = _build(tmp_path)
    result = gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        stop_callbacks=[stopper],
        seed=1,
        display_progress_bar=False,
        run_dir=str(tmp_path / "run"),
    )
    assert max(result.val_aggregate_scores) == 1.0


def test_reflection_spend_is_written_through_for_the_watchdog(tmp_path, network):
    """The out-of-process watchdog can only enforce a ceiling on what is on
    disk; unlogged reflection spend makes that ceiling under-count (D030)."""
    train, val, adapter, reflection_lm, stopper, *_ = _build(tmp_path)
    gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        stop_callbacks=[stopper],
        seed=1,
        display_progress_bar=False,
        run_dir=str(tmp_path / "run"),
    )
    log = tmp_path / "reflection_spend.jsonl"
    assert log.exists() and log.read_text(encoding="utf-8").strip()


def test_the_budget_stopper_actually_halts_the_run(tmp_path, network):
    train, val, adapter, reflection_lm, stopper, solver_meter, reflection_meter = _build(tmp_path, budget=0.05)
    gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        stop_callbacks=[stopper],
        seed=1,
        display_progress_bar=False,
        run_dir=str(tmp_path / "run"),
    )
    spent = solver_meter.budgeted_usd + reflection_meter.budgeted_usd
    # Overshoot by up to one iteration is expected and documented; unbounded is not.
    assert spent < 1.0, f"stopper did not halt the run: ${spent:.2f} against a $0.05 budget"


def test_baseline_reflective_dataset_carries_no_failure_modes(tmp_path, network):
    """Without --taxonomy the adapter is never wrapped, so the baseline arm is
    byte-for-byte gepa's own input."""
    _train, val, adapter, *_ = _build(tmp_path)
    batch = adapter.evaluate(val[:2], dict(SEED_CANDIDATE), capture_traces=True)
    dataset = adapter.make_reflective_dataset(dict(SEED_CANDIDATE), batch, ["final_answer"])
    assert all(FAILURE_MODES_KEY not in ex for ex in dataset["final_answer"])
