"""Full-launch smoke test for the IFBench arm. FREE: the network is faked.

Stubs at the lowest layer only -- ``litellm.completion``. Everything above it is
real: the real ``BedrockLM``, the real ``MeteredReflectionLM``, the real
generate->ensure program, the real vendored verifiers, the real adapter, the
real cost meters and stopper, and gepa's real reflection machinery.
"""

from __future__ import annotations

from types import SimpleNamespace

import gepa
import pytest

from failure_taxonomy import FAILURE_MODES_KEY
from gepa_taxonomy.bedrock import BedrockLM, MeteredReflectionLM, verify_reflection_lm
from gepa_taxonomy.cost import CostMeter, MaxTotalCostStopper
from gepa_taxonomy.ifbench.adapter import IFBenchAdapter, instances_by_id
from gepa_taxonomy.ifbench.program import ENSURE, GENERATE, SEED_CANDIDATE, GenerateEnsureProgram
from gepa_taxonomy.ifbench.tasks import Gold, Instance, Task

SOLVER = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
REFLECTION = "global.anthropic.claude-sonnet-4-6"

MUTATION_MARKER = "an improved instruction"

#: gepa's own reflection preamble (``strategies/instruction_proposal.py``).
#: Detecting REFLECTION and treating everything else as a rollout is the safe
#: direction: the opposite once answered a reflection request with a rollout
#: response, so nothing was ever proposed and it looked like reflection was broken.
REFLECTION_MARKER = "I provided an assistant with the following instructions"

COMPLIANT = "alpha beta gamma delta epsilon zeta eta"  # 7 words, satisfies 5-10
TOO_SHORT = "alpha beta"  # 2 words, fails


class FakeLM:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if REFLECTION_MARKER in prompt:
            return _response(f"```\n{MUTATION_MARKER}\n```")
        # A rollout driven by a MUTATED instruction complies; the seed does not.
        # Without this every candidate ties the base and nothing is ever accepted
        # -- indistinguishable from reflection being broken.
        return _response(COMPLIANT if MUTATION_MARKER in prompt else TOO_SHORT)

    @property
    def models_called(self) -> set[str]:
        return {c["model"] for c in self.calls}

    @property
    def rollout_prompts(self) -> list[str]:
        return [c["messages"][0]["content"] for c in self.calls if REFLECTION_MARKER not in c["messages"][0]["content"]]


def _response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=600, completion_tokens=150),
    )


def _instances(n: int = 6) -> list[Instance]:
    """Alternates one- and two-constraint instances, mirroring the real pool's
    mix so partial credit is actually exercised."""
    out = []
    for i in range(n):
        two = i % 2 == 1
        ids = ("count:word_count_range",) + (("format:title_case",) if two else ())
        kwargs = ({"min_words": 5, "max_words": 10},) + (({},) if two else ())
        out.append(
            Instance(
                task=Task(example_id=str(i), prompt=f"Task {i}. Respond using between 5 and 10 words."),
                gold=Gold(example_id=str(i), instruction_ids=ids, kwargs=kwargs),
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
    program = GenerateEnsureProgram(
        lm=BedrockLM(model=SOLVER, max_retries=2), meter=solver_meter, model=SOLVER, max_tokens=512
    )
    adapter = IFBenchAdapter(
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


def test_the_seed_prompts_are_the_published_ones(tmp_path):
    """Transcribed verbatim from GEPA Appendix L, IFBench GPT-4.1 Mini "Base
    Prompt" blocks. Seeding from the optimised or MIPROv2 blocks printed beside
    them would start the baseline from an already-searched point."""
    assert SEED_CANDIDATE[GENERATE] == "Respond to the query"
    assert SEED_CANDIDATE[ENSURE].startswith("Ensure the response is correct and adheres to the given constraints.")


def test_every_rollout_makes_exactly_two_calls(tmp_path, fake_lm):
    """Cost predictability is the property AppWorld failed."""
    instances, adapter, *_ = _build(tmp_path)
    batch = adapter.evaluate(instances, dict(SEED_CANDIDATE), capture_traces=True)
    assert len(fake_lm.rollout_prompts) == 2 * len(instances)
    for trace in batch.trajectories:
        assert [c["component"] for c in trace["module_calls"]] == [GENERATE, ENSURE]


def test_the_ensure_stage_sees_the_draft(tmp_path, fake_lm):
    instances, adapter, *_ = _build(tmp_path)
    adapter.evaluate(instances[:1], dict(SEED_CANDIDATE), capture_traces=True)
    generate_prompt, ensure_prompt = fake_lm.rollout_prompts
    assert "response:" in ensure_prompt
    assert "response:" not in generate_prompt


def test_the_structured_constraint_spec_never_reaches_a_prompt(tmp_path, fake_lm):
    """The natural-language constraint is in the query by design -- that IS the
    task. The verifier id and its kwargs are not: handing those over would let
    the program satisfy the checker rather than the instruction."""
    instances, adapter, *_ = _build(tmp_path)
    adapter.evaluate(instances[:2], dict(SEED_CANDIDATE), capture_traces=True)
    for prompt in fake_lm.rollout_prompts:
        assert "count:word_count_range" not in prompt
        assert "min_words" not in prompt


def test_scores_are_assembled_by_index_not_completion_order(tmp_path, fake_lm):
    """gepa keys val subscores and the Pareto frontier POSITIONALLY. Run
    it concurrently, which is when completion order can drift."""
    instances, adapter, *_ = _build(tmp_path, workers=4)
    batch = adapter.evaluate(instances, dict(SEED_CANDIDATE), capture_traces=True)
    assert [t["example_id"] for t in batch.trajectories] == [i.task.example_id for i in instances]
    assert batch.scores == [0.0] * len(instances), "the seed prompt answers too short, so nothing complies"


def test_partial_credit_reaches_the_optimizer(tmp_path, fake_lm):
    """The mutated instruction satisfies the word count but not title case, so
    two-constraint instances land at 0.5 -- the only source of partial credit."""
    instances, adapter, *_ = _build(tmp_path)
    mutated = {GENERATE: MUTATION_MARKER, ENSURE: MUTATION_MARKER}
    batch = adapter.evaluate(instances, mutated, capture_traces=True)
    assert set(batch.scores) == {0.5, 1.0}
    assert 0 < sum(batch.scores) < len(instances)


def test_generate_feedback_reports_what_ensure_did(tmp_path, fake_lm):
    """The only signal separating 'the draft was non-compliant' from 'the draft
    was fine and ensure broke it'. Both score identically otherwise."""
    instances, adapter, *_ = _build(tmp_path)
    batch = adapter.evaluate(instances[:2], dict(SEED_CANDIDATE), capture_traces=True)
    dataset = adapter.make_reflective_dataset(dict(SEED_CANDIDATE), batch, [GENERATE, ENSURE])
    assert all("ensure stage" in ex["feedback"] for ex in dataset[GENERATE])
    assert all("ensure stage" not in ex["feedback"] for ex in dataset[ENSURE])


def test_baseline_reflective_dataset_carries_no_failure_modes(tmp_path, fake_lm):
    instances, adapter, *_ = _build(tmp_path)
    batch = adapter.evaluate(instances[:2], dict(SEED_CANDIDATE), capture_traces=True)
    dataset = adapter.make_reflective_dataset(dict(SEED_CANDIDATE), batch, [GENERATE, ENSURE])
    assert all(FAILURE_MODES_KEY not in ex for ex in dataset[GENERATE])
    assert all(FAILURE_MODES_KEY not in ex for ex in dataset[ENSURE])


def test_the_budget_stopper_halts_the_run(tmp_path, fake_lm):
    instances, adapter, reflection_lm, stopper, solver_meter, reflection_meter = _build(tmp_path, budget=0.02)
    _optimize(tmp_path, instances, adapter, reflection_lm, stopper)
    spent = solver_meter.budgeted_usd + reflection_meter.budgeted_usd
    assert spent < 1.0, f"stopper did not halt: ${spent:.2f} against a $0.02 budget"


def test_a_transport_storm_aborts_rather_than_scoring_zero(tmp_path, fake_lm, monkeypatch):
    import litellm

    def throttled(**kwargs):
        raise RuntimeError("RateLimitError: too many requests")

    monkeypatch.setattr(litellm, "completion", throttled)
    instances, adapter, *_ = _build(tmp_path)
    adapter.max_transport_errors = 2
    with pytest.raises(RuntimeError, match="aborting"):
        adapter.evaluate(instances, dict(SEED_CANDIDATE))
