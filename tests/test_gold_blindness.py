"""The HARD INVARIANT: no program component ever sees gold data, on any split.

Gold = reference patch, test_patch, FAIL_TO_PASS / PASS_TO_PASS ids or their
output, or harness results.

Optimizer-level reflection seeing train-instance metric feedback is fine (that
is GEPA). Program-level gold blindness is absolute, so these tests cover the
program on *every* split, not just held-out ones.
"""

from __future__ import annotations

import dataclasses

import pytest

from gepa_taxonomy.cost import CostMeter
from gepa_taxonomy.program import (
    SEED_CANDIDATE,
    RetrievedFile,
    SolverRefinerProgram,
    static_feedback,
)
from gepa_taxonomy.tasks import (
    GOLD_FIELDS,
    Gold,
    GoldLeakError,
    Task,
    assert_gold_free,
    split_row,
)

HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET = "us.anthropic.claude-sonnet-5"

RAW_ROW = {
    "instance_id": "django__django-11099",
    "repo": "django/django",
    "base_commit": "d26b2424437dabeeca94d7900b37d2df4410da0c",
    "problem_statement": "UsernameValidator allows trailing newline in usernames",
    "environment_setup_commit": "419a78300f7cd27611196e1e464d50fd0385ff27",
    "version": "3.0",
    "patch": "--- a/django/contrib/auth/validators.py\n+++ b/django/contrib/auth/validators.py\n@@\n-regex = r'^[\\w.@+-]+$'\n+regex = r'\\A[\\w.@+-]+\\Z'\n",
    "test_patch": "--- a/tests/auth_tests/test_validators.py\n+++ b/tests/auth_tests/test_validators.py\n@@\n+    def test_trailing_newline(self): ...\n",
    "FAIL_TO_PASS": '["test_ascii_validator_with_trailing_newline_rejected"]',
    "PASS_TO_PASS": '["test_unicode_validator_accepts_valid_names"]',
}


@pytest.fixture
def instance():
    return split_row(RAW_ROW)


class RecordingLM:
    """Captures every prompt it is ever handed, so tests can inspect them."""

    def __init__(self, reply: str = "--- a/x.py\n+++ b/x.py\n@@\n-a\n+b\n"):
        self.prompts: list[str] = []
        self.reply = reply

    def complete(self, prompt: str, *, max_tokens: int = 4096):
        self.prompts.append(prompt)
        return self.reply, 1000, 100


class FakeRetriever:
    def retrieve(self, task: Task, *, k: int):
        return [RetrievedFile(path="django/contrib/auth/validators.py", content="regex = r'^[\\w.@+-]+$'\n")]


def build_program(solver_lm, refiner_lm):
    return SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=solver_lm,
        refiner_lm=refiner_lm,
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )


# --------------------------------------------------------------------------
# Structural enforcement
# --------------------------------------------------------------------------


def test_task_type_has_no_gold_fields():
    """Structural, not conventional: gold cannot travel on a Task."""
    names = {f.name for f in dataclasses.fields(Task)}
    assert not (names & {f.lower() for f in GOLD_FIELDS})
    assert not (names & GOLD_FIELDS)


def test_task_is_frozen_and_slotted(instance):
    """Gold cannot be attached to a Task after construction either.

    frozen+slots raises FrozenInstanceError for an existing field and TypeError
    for a brand-new attribute; both are blocked, which is the property we want.
    """
    with pytest.raises(Exception):
        instance.task.patch = "gold"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        instance.task.instance_id = "other"  # type: ignore[misc]


def test_split_row_routes_gold_away_from_task(instance):
    assert instance.gold.patch == RAW_ROW["patch"]
    assert instance.gold.fail_to_pass == ("test_ascii_validator_with_trailing_newline_rejected",)
    blob = repr(instance.task)
    assert RAW_ROW["patch"] not in blob
    assert RAW_ROW["test_patch"] not in blob
    assert "FAIL_TO_PASS" not in blob


def test_prompt_context_exposes_only_safe_fields(instance):
    assert set(instance.task.to_prompt_context()) == {"instance_id", "repo", "problem_statement"}


# --------------------------------------------------------------------------
# The detector itself
# --------------------------------------------------------------------------


def test_assert_gold_free_catches_field_names(instance):
    with pytest.raises(GoldLeakError):
        assert_gold_free({"patch": "anything"}, where="test")
    with pytest.raises(GoldLeakError):
        assert_gold_free({"FAIL_TO_PASS": ["x"]}, where="test")


def test_assert_gold_free_catches_values_under_innocent_keys(instance):
    """A leak that renames the key must still be caught."""
    sneaky = {"hint": instance.gold.patch}
    with pytest.raises(GoldLeakError, match="gold patch"):
        assert_gold_free(sneaky, where="test", gold=instance.gold)


def test_assert_gold_free_catches_grading_test_ids(instance):
    leak = f"Failing test: {instance.gold.fail_to_pass[0]}"
    with pytest.raises(GoldLeakError, match="grading test id"):
        assert_gold_free(leak, where="test", gold=instance.gold)


def test_assert_gold_free_passes_clean_payloads(instance):
    assert_gold_free(instance.task.to_prompt_context(), where="test", gold=instance.gold)


# --------------------------------------------------------------------------
# End-to-end: the program on every split
# --------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["train", "val", "generation", "test"])
def test_no_gold_reaches_any_lm_prompt(instance, split):
    """The invariant applies on *every* split, including train."""
    solver, refiner = RecordingLM(), RecordingLM()
    build_program(solver, refiner).run(instance.task, SEED_CANDIDATE)

    assert solver.prompts and refiner.prompts
    for prompt in (*solver.prompts, *refiner.prompts):
        assert_gold_free(prompt, where=f"{split} prompt", gold=instance.gold)


def test_refiner_never_sees_fail_to_pass_output(instance):
    """The single biggest integrity risk: grading-signal feedback."""
    solver, refiner = RecordingLM(), RecordingLM()
    build_program(solver, refiner).run(instance.task, SEED_CANDIDATE)
    prompt = refiner.prompts[0]
    assert "FAIL_TO_PASS" not in prompt
    assert "PASS_TO_PASS" not in prompt
    for tid in (*instance.gold.fail_to_pass, *instance.gold.pass_to_pass):
        assert tid not in prompt


def test_static_feedback_is_gold_free_by_construction(instance):
    """Feedback is derived from the patch alone -- it has no route to gold."""
    fb = static_feedback("not a diff at all")
    assert_gold_free(fb.render(), where="feedback", gold=instance.gold)
    assert fb.is_well_formed is False


def test_program_run_raises_if_a_leak_is_injected(instance):
    """The boundary check must fail the run, not merely warn.

    Simulates a retriever that (wrongly) surfaces the gold test file.
    """

    class LeakyRetriever:
        def retrieve(self, task, *, k):
            return [RetrievedFile(path="tests/test_validators.py", content=instance.gold.test_patch)]

    solver = RecordingLM()
    prog = build_program(solver, RecordingLM())
    prog.retriever = LeakyRetriever()
    rollout = prog.run(instance.task, SEED_CANDIDATE)

    # The program itself cannot catch this: it has no Gold to compare against,
    # and giving it one would BE the leak. The value-based check belongs to the
    # harness, which legitimately holds gold in order to grade. It must fire.
    with pytest.raises(GoldLeakError, match="test_patch"):
        assert_gold_free(solver.prompts[0], where="solver prompt", gold=instance.gold)
    # The rollout retains the exact prompts precisely so this audit is possible.
    with pytest.raises(GoldLeakError, match="test_patch"):
        assert_gold_free(rollout.prompts(), where="rollout prompts", gold=instance.gold)


def test_trace_file_does_not_embed_raw_prompts(instance):
    """Traces carry a prompt digest, not the whole retrieved context."""
    prog = build_program(RecordingLM(), RecordingLM())
    trace = prog.run(instance.task, SEED_CANDIDATE).to_trace()
    assert "solver_prompt_sha256" in trace
    assert "solver_prompt" not in trace and "refiner_prompt" not in trace


def test_leak_detection_survives_json_round_trip(instance):
    """Regression: an earlier detector serialized with json.dumps, which escapes
    newlines and silently prevented any multi-line gold patch from matching."""
    payload = {"context": instance.gold.patch}
    with pytest.raises(GoldLeakError, match="gold patch"):
        assert_gold_free(payload, where="test", gold=instance.gold)


def test_leak_detection_survives_reformatting(instance):
    """Whitespace-normalized comparison catches re-indented gold."""
    reflowed = "  ".join(instance.gold.patch.split("\n"))
    with pytest.raises(GoldLeakError, match="gold patch"):
        assert_gold_free({"ctx": reflowed}, where="test", gold=instance.gold)


def test_leak_detection_walks_nested_structures(instance):
    nested = {"a": [{"b": ("c", {"d": instance.gold.patch})}]}
    with pytest.raises(GoldLeakError, match="gold patch"):
        assert_gold_free(nested, where="test", gold=instance.gold)


def test_gold_is_never_an_argument_to_program_run():
    """`run` has no parameter through which a Gold could be passed."""
    import inspect

    sig = inspect.signature(SolverRefinerProgram.run)
    annotations = {p.annotation for p in sig.parameters.values()}
    assert Gold not in annotations
    assert not any("Gold" in str(a) for a in annotations)


def test_program_fields_carry_no_gold_channel():
    """No constructor field could smuggle gold into the program."""
    names = {f.name for f in dataclasses.fields(SolverRefinerProgram)}
    assert not (names & {"gold", "golds", "patch", "test_patch"})


def test_test_id_published_in_the_problem_statement_is_not_a_leak():
    """SWE-Bench issue text often names the failing test. The benchmark hands
    that text to every solver, so finding the name in a prompt is not a leak --
    it killed a test-eval run at $11.50 on django__django-14349."""
    from gepa_taxonomy.tasks import Gold, GoldLeakError, assert_gold_free

    tid = "test_validators (validators.tests.TestValidators)"
    gold = Gold(instance_id="django__django-14349", patch="", test_patch="", fail_to_pass=(tid,), pass_to_pass=())
    statement = f"URLValidator tests failed: {tid} raised ValueError"

    # published with the task -> allowed
    assert_gold_free({"prompt": f"Fix this issue.\n{statement}"}, where="p", gold=gold, public_text=statement)

    # the same id with NO public source -> still a hard failure
    with pytest.raises(GoldLeakError):
        assert_gold_free({"prompt": f"make {tid} pass"}, where="p", gold=gold, public_text="")
