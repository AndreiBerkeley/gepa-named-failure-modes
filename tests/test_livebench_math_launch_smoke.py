"""Full-launch smoke test for the LiveBench-Math arm. FREE: the network is faked.

Every launch failure this project has had was reachable without a real model, and
each was missed because the tests stubbed too high up -- replacing our own classes
instead of the transport underneath them. So this stubs at the lowest layer only:
``litellm.completion``.

Everything above it is real: the real ``BedrockLM``, the real
``MeteredReflectionLM``, the real solve->review program, the real grader, the
real adapter, the real cost meters and stopper, and gepa's real reflection
machinery.
"""

from __future__ import annotations

from types import SimpleNamespace

import gepa
import pytest

from failure_taxonomy import FAILURE_MODES_KEY
from gepa_taxonomy.bedrock import BedrockLM, MeteredReflectionLM, verify_reflection_lm
from gepa_taxonomy.cost import CostMeter, MaxTotalCostStopper
from gepa_taxonomy.livebench_math.adapter import LiveBenchMathAdapter, instances_by_id
from gepa_taxonomy.livebench_math.program import REVIEW, SEED_CANDIDATE, SOLVE, SolveReviewProgram
from gepa_taxonomy.livebench_math.tasks import Gold, Instance, Task

SOLVER = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
REFLECTION = "global.anthropic.claude-sonnet-4-6"

MUTATION_MARKER = "an improved instruction"

#: gepa's own reflection preamble (``strategies/instruction_proposal.py``).
#: Detecting REFLECTION and treating everything else as a rollout is the safe
#: direction: the opposite test once answered a reflection request with a rollout
#: response, so nothing was ever proposed and it looked like reflection was
#: broken (the AppWorld smoke hit exactly this).
REFLECTION_MARKER = "I provided an assistant with the following instructions"


class FakeLM:
    """Stands in for ``litellm.completion``."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if REFLECTION_MARKER in prompt:
            return _response(f"```\n{MUTATION_MARKER}\n```")
        # A rollout driven by a MUTATED instruction answers correctly, so the
        # grader can reward it and gepa can accept a candidate. Without this every
        # candidate ties the base and nothing is ever accepted -- which is
        # indistinguishable from reflection being broken.
        letter = "C" if MUTATION_MARKER in prompt else "B"
        return _response(f"Reasoning about the problem.\n\\boxed{{{letter}}}")

    @property
    def models_called(self) -> set[str]:
        return {c["model"] for c in self.calls}

    @property
    def rollout_prompts(self) -> list[str]:
        return [c["messages"][0]["content"] for c in self.calls if REFLECTION_MARKER not in c["messages"][0]["content"]]


def _response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=800, completion_tokens=200),
    )


#: Half score C (which the mutated instruction produces), half score A (which
#: nothing produces), so the optimizer sees a non-degenerate distribution. With
#: every instance tied, acceptance looks broken.
def _instances(n: int = 6) -> list[Instance]:
    out = []
    for i in range(n):
        out.append(
            Instance(
                task=Task(
                    example_id=f"q{i}",
                    question=(
                        rf"Problem number {i}. $\textbf{{(A) }}5\qquad\textbf{{(B) }}6"
                        r"\qquad\textbf{(C) }7\qquad\textbf{(D) }8\qquad\textbf{(E) }9$"
                    ),
                    task="math_comp",
                    subtask="amc_12a_2023",
                ),
                gold=Gold(example_id=f"q{i}", ground_truth="C" if i % 2 == 0 else "A"),
            )
        )
    return out


@pytest.fixture
def fake_lm(monkeypatch):
    import litellm

    lm = FakeLM()
    monkeypatch.setattr(litellm, "completion", lm)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    return lm


def _build(tmp_path, budget=5.0, workers=1):
    instances = _instances()
    solver_meter, reflection_meter = CostMeter(), CostMeter()
    program = SolveReviewProgram(
        lm=BedrockLM(model=SOLVER, max_retries=2),
        meter=solver_meter,
        model=SOLVER,
        max_tokens=512,
    )
    adapter = LiveBenchMathAdapter(
        program=program,
        instances=instances_by_id(instances),
        reflection_gold_ids=frozenset(i.task.example_id for i in instances),
        max_workers=workers,
    )
    reflection_lm = MeteredReflectionLM(
        lm=BedrockLM(model=REFLECTION, max_retries=2),
        meter=reflection_meter,
        model=REFLECTION,
        spend_log=tmp_path / "reflection_spend.jsonl",
    )
    stopper = MaxTotalCostStopper(budget, [solver_meter, reflection_meter])
    return instances, adapter, reflection_lm, stopper, solver_meter, reflection_meter


def _optimize(tmp_path, instances, adapter, reflection_lm, stopper, **kw):
    return gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=instances,
        valset=instances,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        stop_callbacks=[stopper],
        seed=1,
        display_progress_bar=False,
        run_dir=str(tmp_path / "run"),
        **kw,
    )


def test_the_reflection_lm_conforms(tmp_path, fake_lm):
    _, _, reflection_lm, _, _, _ = _build(tmp_path)
    assert verify_reflection_lm(reflection_lm)["ok"]


def test_a_full_launch_completes_and_proposes(tmp_path, fake_lm):
    instances, adapter, reflection_lm, stopper, solver_meter, reflection_meter = _build(tmp_path)
    result = _optimize(tmp_path, instances, adapter, reflection_lm, stopper)

    assert len(result.candidates) > 1, "reflection never proposed a candidate"
    assert reflection_meter.calls > 0, "reflection spend was never metered"
    assert solver_meter.budgeted_usd > 0
    assert fake_lm.models_called == {f"bedrock/{SOLVER}", f"bedrock/{REFLECTION}"}


def test_every_rollout_makes_exactly_two_calls(tmp_path, fake_lm):
    """Cost predictability is the property that ruled AppWorld out. If a
    rollout can vary in call count, seeds under one budget stop being comparable."""
    instances, adapter, *_ = _build(tmp_path)
    batch = adapter.evaluate(instances, dict(SEED_CANDIDATE), capture_traces=True)
    assert len(fake_lm.rollout_prompts) == 2 * len(instances)
    for trace in batch.trajectories:
        components = [c["component"] for c in trace["module_calls"]]
        assert components == [SOLVE, REVIEW]


def test_the_review_stage_actually_sees_the_draft(tmp_path, fake_lm):
    instances, adapter, *_ = _build(tmp_path)
    adapter.evaluate(instances[:1], dict(SEED_CANDIDATE), capture_traces=True)
    solve_prompt, review_prompt = fake_lm.rollout_prompts
    assert "draft_answer" in review_prompt
    assert "draft_answer" not in solve_prompt


def test_scores_are_assembled_by_index_not_completion_order(tmp_path, fake_lm):
    """gepa keys val subscores and the Pareto frontier POSITIONALLY, so a
    batch returned in completion order attaches every score to the wrong
    instance -- silently. Run it concurrently, which is when order can drift."""
    instances, adapter, *_ = _build(tmp_path, workers=4)
    batch = adapter.evaluate(instances, dict(SEED_CANDIDATE), capture_traces=True)
    assert [t["example_id"] for t in batch.trajectories] == [i.task.example_id for i in instances]
    # The base instruction answers B; gold alternates C/A, so every score is 0.0.
    assert batch.scores == [0.0] * len(instances)


def test_partial_credit_reaches_the_optimizer(tmp_path, fake_lm):
    """The mutated instruction answers C, which is gold for the even instances."""
    instances, adapter, *_ = _build(tmp_path)
    mutated = {SOLVE: MUTATION_MARKER, REVIEW: MUTATION_MARKER}
    batch = adapter.evaluate(instances, mutated, capture_traces=True)
    assert set(batch.scores) == {0.0, 1.0}
    assert 0 < sum(batch.scores) < len(instances)


def test_solve_feedback_reports_what_review_did(tmp_path, fake_lm):
    """The only signal separating 'solve was wrong' from 'solve was right and
    review broke it'. Both score 0 and are otherwise identical to the optimizer."""
    instances, adapter, *_ = _build(tmp_path)
    batch = adapter.evaluate(instances[:2], dict(SEED_CANDIDATE), capture_traces=True)
    dataset = adapter.make_reflective_dataset(dict(SEED_CANDIDATE), batch, [SOLVE, REVIEW])
    assert all("review stage" in ex["feedback"] for ex in dataset[SOLVE])
    assert all("review stage" not in ex["feedback"] for ex in dataset[REVIEW])


def test_baseline_reflective_dataset_carries_no_failure_modes(tmp_path, fake_lm):
    instances, adapter, *_ = _build(tmp_path)
    batch = adapter.evaluate(instances[:2], dict(SEED_CANDIDATE), capture_traces=True)
    dataset = adapter.make_reflective_dataset(dict(SEED_CANDIDATE), batch, [SOLVE, REVIEW])
    assert all(FAILURE_MODES_KEY not in ex for ex in dataset[SOLVE])
    assert all(FAILURE_MODES_KEY not in ex for ex in dataset[REVIEW])


def test_the_budget_stopper_halts_the_run(tmp_path, fake_lm):
    instances, adapter, reflection_lm, stopper, solver_meter, reflection_meter = _build(tmp_path, budget=0.02)
    _optimize(tmp_path, instances, adapter, reflection_lm, stopper)
    spent = solver_meter.budgeted_usd + reflection_meter.budgeted_usd
    assert spent < 1.0, f"stopper did not halt: ${spent:.2f} against a $0.02 budget"


def test_a_transport_storm_aborts_rather_than_scoring_zero(tmp_path, fake_lm, monkeypatch):
    """A model we cannot reach scores 0.0, which the optimizer cannot tell from a
    bad candidate. Better to stop than to silently corrupt a paid run."""
    import litellm

    def throttled(**kwargs):
        raise RuntimeError("RateLimitError: too many requests")

    monkeypatch.setattr(litellm, "completion", throttled)
    instances, adapter, *_ = _build(tmp_path)
    adapter.max_transport_errors = 2
    with pytest.raises(RuntimeError, match="aborting"):
        adapter.evaluate(instances, dict(SEED_CANDIDATE))
