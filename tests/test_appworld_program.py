"""Tests for the AppWorld ReAct program and adapter. Offline: no server, no model."""

from __future__ import annotations

import pytest
from gepa.core.adapter import EvaluationBatch

from failure_taxonomy import FAILURE_MODES_KEY, Occurrence, TaxonomyFeedbackEnricher, build_trace, extract_calls
from gepa_taxonomy.appworld.adapter import AppWorldAdapter
from gepa_taxonomy.appworld.client import TaskResult
from gepa_taxonomy.appworld.program import REACT, ReActProgram, extract_code
from gepa_taxonomy.appworld.prompts import DEMONSTRATION, SEED_CANDIDATE, SEED_INSTRUCTION

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_seed_is_the_published_instruction_not_something_we_wrote():
    """A hand-improved seed would confound the baseline; the comparison is about
    what reflection adds to the PUBLISHED starting point."""
    assert "**Key instructions**" in SEED_INSTRUCTION
    assert "apis.supervisor.complete_task()" in SEED_INSTRUCTION
    assert set(SEED_CANDIDATE) == {REACT}


def test_the_demonstration_is_scaffolding_not_the_component():
    """215 of the published file's 267 lines are one worked example. Handing GEPA
    a 7.6 KB component including it would be an order of magnitude larger than
    any component in the paper, and invites rewriting a correct example."""
    assert len(SEED_INSTRUCTION) < len(DEMONSTRATION)
    assert "Marked the active task complete" not in SEED_INSTRUCTION


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("```python\nprint(1)\n```", "print(1)"),
        ("prose\n```\nx = 2\n```\ntail", "x = 2"),
        ("no code at all", ""),
        ("", ""),
    ],
)
def test_code_extraction(text, expected):
    assert extract_code(text) == expected


def test_unfenced_output_yields_no_code_rather_than_executing_prose():
    """Executing model prose because it might be code is how a formatting failure
    becomes an environment mutation."""
    assert extract_code("I will now list the playlists.") == ""


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeClient:
    def __init__(self, complete_after=1, evaluate_result=None):
        self.complete_after = complete_after
        self.executed: list[str] = []
        self.closed = False
        self.evaluate_result = evaluate_result or TaskResult(
            task_id="t1", success=True, score=1.0, num_tests=2, passes=("a", "b"), failures=()
        )

    def initialize(self, task_id):
        return {
            "instruction": "Do the thing.",
            "supervisor": {"first_name": "A", "last_name": "B", "email": "a@b.c", "phone_number": "1"},
        }

    def execute(self, task_id, code):
        self.executed.append(code)
        return f"ran: {code}"

    def task_completed(self, task_id):
        return len(self.executed) >= self.complete_after

    def evaluate(self, task_id):
        return self.evaluate_result

    def close(self, task_id):
        self.closed = True


class FakeLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt, *, max_tokens=2048):
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if self.replies else "```python\npass\n```"
        return (reply, 500, 50)


class FakeMeter:
    def record(self, *, model, input_tokens, output_tokens, phase):
        return 0.002


def _program(replies, client=None, max_steps=30):
    return ReActProgram(
        client=client or FakeClient(),
        lm=FakeLM(replies),
        meter=FakeMeter(),
        model="fake",
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_a_completed_task_stops_the_loop():
    program = _program(["```python\napis.supervisor.complete_task()\n```"])
    rollout = program.run("t1", SEED_CANDIDATE)
    assert rollout.completed and rollout.steps == 1
    assert rollout.score == 1.0
    assert program.client.closed, "the environment must be released"


def test_the_step_budget_bounds_a_runaway_task():
    """A ReAct rollout takes as many steps as it takes; without a cap one
    pathological task could consume a seed's budget."""
    client = FakeClient(complete_after=999)
    program = _program(["```python\nx=1\n```"] * 5, client=client, max_steps=3)
    rollout = program.run("t1", SEED_CANDIDATE)
    assert rollout.steps == 3
    assert rollout.exhausted_steps, "hitting the cap is a distinct failure mode and must be recorded"
    assert not rollout.completed


def test_a_step_with_no_code_is_counted_and_told_so():
    client = FakeClient(complete_after=999)
    program = _program(["I will think about it.", "```python\nx=1\n```"], client=client, max_steps=2)
    rollout = program.run("t1", SEED_CANDIDATE)
    assert rollout.empty_code_steps == 1
    assert "No code block found" in program.lm.prompts[1]
    assert client.executed == ["x=1"], "prose must not be executed"


def test_the_instruction_reaches_the_prompt_and_the_demo_comes_with_it():
    program = _program(["```python\napis.supervisor.complete_task()\n```"])
    program.run("t1", {REACT: "MY EVOLVED INSTRUCTION"})
    prompt = program.lm.prompts[0]
    assert "MY EVOLVED INSTRUCTION" in prompt
    assert "Do the thing." in prompt
    assert DEMONSTRATION[:60] in prompt


def test_environment_output_is_fed_back_to_the_next_step():
    client = FakeClient(complete_after=2)
    program = _program(["```python\nfirst()\n```", "```python\nsecond()\n```"], client=client)
    program.run("t1", SEED_CANDIDATE)
    assert "ran: first()" in program.lm.prompts[1]


def test_an_initialize_failure_returns_a_scored_rollout_rather_than_raising():
    class Broken(FakeClient):
        def initialize(self, task_id):
            raise RuntimeError("server down")

    program = _program([], client=Broken())
    rollout = program.run("t1", SEED_CANDIDATE)
    assert rollout.score == 0.0 and "server down" in rollout.error


def test_the_environment_is_released_even_when_a_step_raises():
    class Boom(FakeClient):
        def execute(self, task_id, code):
            raise RuntimeError("execute failed")

    client = Boom()
    program = _program(["```python\nx=1\n```"], client=client)
    rollout = program.run("t1", SEED_CANDIDATE)
    assert client.closed, "a leaked environment outlives the rollout"
    assert "execute failed" in rollout.error


# ---------------------------------------------------------------------------
# Trace shape
# ---------------------------------------------------------------------------


def test_every_step_is_a_module_call_under_one_component():
    client = FakeClient(complete_after=3)
    program = _program(["```python\na()\n```", "```python\nb()\n```", "```python\nc()\n```"], client=client)
    trace = program.run("t1", SEED_CANDIDATE).to_trace()
    calls = extract_calls(trace)
    assert len(calls) == 3
    assert {c.component for c in calls} == {REACT}


def test_the_judge_sees_one_component_not_three_repeats():
    """Attribution is unary here, which is exactly the ablation against
    HotpotQA's four components."""
    client = FakeClient(complete_after=3)
    program = _program(["```python\na()\n```", "```python\nb()\n```", "```python\nc()\n```"], client=client)
    trace = build_trace(program.run("t1", SEED_CANDIDATE).to_trace(), trace_id="t1")
    assert trace.components == (REACT,)
    assert trace.is_segmented


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def _adapter(replies, client=None, **kw):
    return AppWorldAdapter(program=_program(replies, client=client), **kw)


def test_adapter_returns_a_real_evaluation_batch():
    adapter = _adapter(["```python\napis.supervisor.complete_task()\n```"])
    batch = adapter.evaluate(["t1"], SEED_CANDIDATE, capture_traces=True)
    assert isinstance(batch, EvaluationBatch)
    assert batch.scores == [1.0] and batch.num_metric_calls == 1


def test_adapter_declares_propose_new_texts():
    assert _adapter([]).propose_new_texts is None


def test_feedback_names_the_failed_requirements():
    """AppWorld's own verdict is the baseline arm's feedback -- strong on purpose,
    so the taxonomy arm must beat it rather than a weakened version."""
    failing = TaskResult(
        task_id="t1",
        success=False,
        score=0.5,
        num_tests=2,
        passes=("assert no changes.",),
        failures=("assert answers match.",),
    )
    adapter = _adapter(
        ["```python\napis.supervisor.complete_task()\n```"],
        client=FakeClient(evaluate_result=failing),
    )
    batch = adapter.evaluate(["t1"], SEED_CANDIDATE, capture_traces=True)
    dataset = adapter.make_reflective_dataset(SEED_CANDIDATE, batch, [REACT])
    feedback = dataset[REACT][0]["feedback"]
    assert "assert answers match." in feedback
    assert "Requirements failed (1)" in feedback


def test_step_exhaustion_is_surfaced_to_reflection():
    """'ran out of steps' and 'answered wrongly' are different failures and must
    not look the same in the feedback."""
    client = FakeClient(complete_after=999)
    adapter = _adapter(["```python\nx=1\n```"] * 3, client=client, max_workers=1)
    adapter.program.max_steps = 2
    batch = adapter.evaluate(["t1"], SEED_CANDIDATE, capture_traces=True)
    dataset = adapter.make_reflective_dataset(SEED_CANDIDATE, batch, [REACT])
    assert "without calling complete_task()" in dataset[REACT][0]["feedback"]


def test_summary_reports_step_behaviour():
    adapter = _adapter(["```python\napis.supervisor.complete_task()\n```"])
    adapter.evaluate(["t1"], SEED_CANDIDATE)
    s = adapter.summary()
    assert s["rollouts"] == 1 and s["mean_steps"] == 1.0 and s["transport_errors"] == 0


# ---------------------------------------------------------------------------
# Taxonomy enricher integration
# ---------------------------------------------------------------------------


class ScriptedJudge:
    candidate_key = ""

    def judge(self, traces):
        return {t.trace_id: [Occurrence("A.1", "Never_Verified_Result", "ran: x=1", REACT)] for t in traces}


def test_the_enricher_routes_to_the_single_component():
    adapter = _adapter(["```python\napis.supervisor.complete_task()\n```"])
    batch = adapter.evaluate(["t1"], SEED_CANDIDATE, capture_traces=True)
    baseline = adapter.make_reflective_dataset(SEED_CANDIDATE, batch, [REACT])
    dataset = TaxonomyFeedbackEnricher(judge=ScriptedJudge())(
        candidate=SEED_CANDIDATE,
        eval_batch=batch,
        components_to_update=[REACT],
        reflective_dataset=baseline,
    )
    assert dataset[REACT][0][FAILURE_MODES_KEY] == [{"name": "Never_Verified_Result", "evidence": "ran: x=1"}]


def test_baseline_arm_is_unchanged_without_the_enricher():
    adapter = _adapter(["```python\napis.supervisor.complete_task()\n```"])
    batch = adapter.evaluate(["t1"], SEED_CANDIDATE, capture_traces=True)
    bare = adapter.make_reflective_dataset(SEED_CANDIDATE, batch, [REACT])
    assert FAILURE_MODES_KEY not in bare[REACT][0]
