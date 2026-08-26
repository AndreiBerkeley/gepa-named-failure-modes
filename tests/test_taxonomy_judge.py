"""The treatment arm: taxonomy-conditioned reflection, and the baseline it must not disturb.

Two things are load-bearing here and both are asserted on real flowed content
rather than on shapes:

1. With no judge attached, the reflective dataset is byte-identical to what the
   baseline seeds were run against. Every key, its order, and its value.
2. With one attached, codes reach exactly the component whose trace produced
   them -- never the other subagent, never a resolved instance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gepa_taxonomy.adapter import Grader, SweBenchAdapter
from gepa_taxonomy.cost import CostMeter, MaxTotalCostStopper
from gepa_taxonomy.program import (
    COMPONENTS,
    REFINER,
    SEED_CANDIDATE,
    SOLVER,
    RetrievedFile,
    SolverRefinerProgram,
)
from gepa_taxonomy.tasks import split_row
from gepa_taxonomy.taxonomy_judge import (
    MAX_TRACE_CHARS,
    JudgeCache,
    TaxonomyJudge,
    build_role_record,
    taxonomy_fingerprint,
)
from tests.test_gold_blindness import RAW_ROW

HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET = "global.anthropic.claude-sonnet-4-6"

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_TAXONOMY = REPO_ROOT / "results" / "taxonomy" / "base_val_v1" / "taxonomy.pruned.json"
REAL_TRACES = REPO_ROOT / "results" / "traces" / "base_val.adamast.jsonl"

SOLVER_PATCH = "--- a/x.py\n+++ b/x.py\n@@\n-a\n+b"
REFINER_PATCH = "--- a/x.py\n+++ b/x.py\n@@\n-a\n+c"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TwoStageLM:
    """Distinguishable solver and refiner outputs, so role scoping is testable."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int = 4096):
        self.prompts.append(prompt)
        return self.reply + "\n", 1000, 100


class FakeRetriever:
    def retrieve(self, task, *, k: int):
        return [RetrievedFile(path="x.py", content="a\n")]


class ScoreByIdGrader(Grader):
    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def grade(self, task, gold, patch):
        score = self.scores.get(task.instance_id, 0.0)
        return score, {"resolved": bool(score)}


TAXONOMY_DOC = {
    "schema_version": 1,
    "status": "accepted",
    "codes": [
        {
            "id": "A.9",
            "name": "Diff_Context_Mismatch",
            "category": "A",
            "severity": "major",
            "description": "Context lines do not match the file.",
        },
        {
            "id": "B.4",
            "name": "Solver_Malformed_Diff",
            "category": "B",
            "severity": "critical",
            "applies_to_role": "solver",
            "description": "The solver emitted a diff that is not well formed.",
        },
        {
            "id": "C.4",
            "name": "Malformed_Diff_Format",
            "category": "C",
            "severity": "critical",
            "description": "The diff is not well formed.",
        },
    ],
}


@pytest.fixture
def taxonomy_file(tmp_path) -> Path:
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(TAXONOMY_DOC))
    return path


def make_instances(*instance_ids: str) -> dict:
    out = {}
    for instance_id in instance_ids:
        inst = split_row({**RAW_ROW, "instance_id": instance_id})
        out[instance_id] = inst
    return out


def make_adapter(instances, scores, **kwargs) -> SweBenchAdapter:
    program = SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=TwoStageLM(SOLVER_PATCH),
        refiner_lm=TwoStageLM(REFINER_PATCH),
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    return SweBenchAdapter(
        program=program,
        grader=ScoreByIdGrader(scores),
        instances=instances,
        **kwargs,
    )


class RecordingJudge:
    """Stands in for TaxonomyJudge at the adapter boundary."""

    def __init__(self, codes_by_component: dict[str, list[dict]] | None = None):
        self.calls: list[tuple[str, list[str]]] = []
        self.codes = codes_by_component or {}

    def judge(self, candidate, component, subjects):
        self.calls.append((component, sorted(subjects)))
        return {iid: [dict(c) for c in self.codes.get(component, [])] for iid in subjects}


def runner_returning(modes: list[dict], *, prompt_chars: int = 0, response_chars: int = 0, seen=None):
    """A judge transport that answers every trace in the request with ``modes``."""

    def _run(request):
        if seen is not None:
            seen.append(request)
        return {
            "diagnoses": [
                {"trace_id": t["problem_id"], "failure_modes": [dict(m) for m in modes], "none_apply": not modes}
                for t in request["traces"]
            ],
            "usage": {
                "calls": len(request["traces"]),
                "prompt_chars": prompt_chars,
                "response_chars": response_chars,
                "measured": True,
            },
        }

    return _run


def make_rollout(instance):
    """A real rollout, so the judge sees real prompts rather than a mock."""
    program = SolverRefinerProgram(
        retriever=FakeRetriever(),
        solver_lm=TwoStageLM(SOLVER_PATCH),
        refiner_lm=TwoStageLM(REFINER_PATCH),
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )
    return program.run(instance.task, SEED_CANDIDATE)


# ---------------------------------------------------------------------------
# 1. The baseline must not move
# ---------------------------------------------------------------------------


def test_baseline_reflective_example_is_byte_identical():
    """No judge => the exact dict the baseline seeds were run against.

    Pinned by value and by key order, so the treatment arm cannot differ from
    the baseline by anything except the added ``failure_modes`` key.
    """
    instances = make_instances("django__django-11099")
    adapter = make_adapter(instances, {"django__django-11099": 0.0})
    res = adapter.evaluate(["django__django-11099"], SEED_CANDIDATE, capture_traces=True)
    ds = adapter.make_reflective_dataset(SEED_CANDIDATE, res, COMPONENTS)

    expected = {
        "instance_id": "django__django-11099",
        "problem_statement_excerpt": "UsernameValidator allows trailing newline in usernames",
        "retrieved_paths": ["x.py"],
        "produced_patch": "--- a/x.py\n+++ b/x.py\n@@\n-a\n+b",
        "automated_checks": {
            "is_well_formed": True,
            "applies_cleanly": None,
            "syntax_ok": None,
            "messages": [],
        },
        "harness_result": {"resolved": False},
        "score": 0.0,
    }
    # Each component sees ITS OWN output: the solver example carries the
    # solver's patch, the refiner example the refiner's. Both were empty while
    # _finish stripped the model-output fields, which hid this entirely.
    assert json.dumps(ds[SOLVER][0]) == json.dumps(expected)
    refiner_expected = dict(expected)
    refiner_expected["produced_patch"] = "--- a/x.py\n+++ b/x.py\n@@\n-a\n+c"
    assert json.dumps(ds[REFINER][0]) == json.dumps(refiner_expected)


def test_no_judge_means_no_judge_rollouts_retained():
    """The baseline arm does not pay the memory cost of the treatment arm."""
    instances = make_instances("django__django-11099")
    adapter = make_adapter(instances, {"django__django-11099": 0.0})
    adapter.evaluate(["django__django-11099"], SEED_CANDIDATE, capture_traces=True)
    assert adapter._judge_rollouts == {}


# ---------------------------------------------------------------------------
# 2. Codes reach the right example and only the right example
# ---------------------------------------------------------------------------


def test_failure_modes_present_for_failed_absent_for_resolved():
    instances = make_instances("failed__1", "resolved__1")
    judge = RecordingJudge({SOLVER: [{"code": "A.9", "name": "Diff_Context_Mismatch", "evidence": "e", "severity": "major"}]})
    adapter = make_adapter(instances, {"failed__1": 0.0, "resolved__1": 1.0}, taxonomy_judge=judge)

    res = adapter.evaluate(["failed__1", "resolved__1"], SEED_CANDIDATE, capture_traces=True)
    ds = adapter.make_reflective_dataset(SEED_CANDIDATE, res, [SOLVER])

    by_id = {e["instance_id"]: e for e in ds[SOLVER]}
    assert by_id["failed__1"]["failure_modes"] == [
        {"code": "A.9", "name": "Diff_Context_Mismatch", "evidence": "e", "severity": "major"}
    ]
    assert "failure_modes" not in by_id["resolved__1"]
    # The resolved instance was never sent to the judge at all.
    assert judge.calls == [(SOLVER, ["failed__1"])]


def test_only_the_component_under_update_is_judged():
    """GEPA updates one component per iteration; the other must not be judged."""
    instances = make_instances("failed__1")
    judge = RecordingJudge({SOLVER: [], REFINER: []})
    adapter = make_adapter(instances, {"failed__1": 0.0}, taxonomy_judge=judge)

    res = adapter.evaluate(["failed__1"], SEED_CANDIDATE, capture_traces=True)
    adapter.make_reflective_dataset(SEED_CANDIDATE, res, [REFINER])

    assert judge.calls == [(REFINER, ["failed__1"])]


def test_solver_codes_never_appear_on_refiner_examples():
    instances = make_instances("failed__1")
    judge = RecordingJudge(
        {
            SOLVER: [{"code": "B.4", "name": "Solver_Malformed_Diff", "evidence": "s", "severity": "critical"}],
            REFINER: [{"code": "C.4", "name": "Malformed_Diff_Format", "evidence": "r", "severity": "critical"}],
        }
    )
    adapter = make_adapter(instances, {"failed__1": 0.0}, taxonomy_judge=judge)

    res = adapter.evaluate(["failed__1"], SEED_CANDIDATE, capture_traces=True)
    ds = adapter.make_reflective_dataset(SEED_CANDIDATE, res, COMPONENTS)

    assert [m["code"] for m in ds[SOLVER][0]["failure_modes"]] == ["B.4"]
    assert [m["code"] for m in ds[REFINER][0]["failure_modes"]] == ["C.4"]
    assert judge.calls == [(SOLVER, ["failed__1"]), (REFINER, ["failed__1"])]


def test_judged_but_clean_instance_still_gets_the_key():
    """An empty list is a diagnosis ("nothing in this taxonomy fired"), not a miss."""
    instances = make_instances("failed__1")
    adapter = make_adapter(instances, {"failed__1": 0.0}, taxonomy_judge=RecordingJudge({SOLVER: []}))
    res = adapter.evaluate(["failed__1"], SEED_CANDIDATE, capture_traces=True)
    ds = adapter.make_reflective_dataset(SEED_CANDIDATE, res, [SOLVER])
    assert ds[SOLVER][0]["failure_modes"] == []


# ---------------------------------------------------------------------------
# 3. Role-scoped trace records
# ---------------------------------------------------------------------------


def test_role_record_sections_are_scoped_and_ordered():
    instance = make_instances("failed__1")["failed__1"]
    rollout = make_rollout(instance)

    solver_rec = build_role_record(component=SOLVER, rollout=rollout, task=instance.task)
    refiner_rec = build_role_record(component=REFINER, rollout=rollout, task=instance.task)

    for rec, role in ((solver_rec, "SOLVER"), (refiner_rec, "REFINER")):
        body = rec.raw_trajectory
        positions = [
            body.index("[TASK]"),
            body.index(f"[ROLE: {role.lower()}]"),
            body.index(f"[INPUT GIVEN TO THE {role}]"),
            body.index(f"[{role} PROMPT]"),
            body.index(f"[{role} OUTPUT]"),
        ]
        assert positions == sorted(positions)

    # Each record carries its own subagent's output and not the other's.
    assert solver_rec.raw_trajectory.endswith(SOLVER_PATCH)
    assert refiner_rec.raw_trajectory.endswith(REFINER_PATCH)
    # The refiner's input section states what it was actually given.
    assert "Candidate patch produced by the solver" in refiner_rec.raw_trajectory
    assert "Automated checks reported on that candidate patch" in refiner_rec.raw_trajectory
    assert "Candidate patch produced by the solver" not in solver_rec.raw_trajectory
    # Role is machine-readable as well as stated in prose.
    assert solver_rec.metadata["role"] == "solver"
    assert refiner_rec.metadata["role"] == "refiner"
    assert solver_rec.problem_id.endswith("::solver")
    assert refiner_rec.problem_id.endswith("::refiner")


def test_role_record_carries_the_full_untruncated_prompt():
    instance = make_instances("failed__1")["failed__1"]
    rollout = make_rollout(instance)
    solver_prompt, refiner_prompt = rollout.prompts()

    solver_rec = build_role_record(component=SOLVER, rollout=rollout, task=instance.task)
    refiner_rec = build_role_record(component=REFINER, rollout=rollout, task=instance.task)

    assert solver_prompt and solver_prompt in solver_rec.raw_trajectory
    assert refiner_prompt and refiner_prompt in refiner_rec.raw_trajectory
    assert "[truncated]" not in solver_rec.raw_trajectory


def test_outcome_stays_out_of_the_trajectory():
    """AdaMAST's checklist: the judge must not read the oracle outcome."""
    instance = make_instances("failed__1")["failed__1"]
    rollout = make_rollout(instance)
    rec = build_role_record(
        component=SOLVER,
        rollout=rollout,
        task=instance.task,
        grading={"resolved": False, "failing_tests": ["test_trailing_newline"]},
    )
    assert "test_trailing_newline" not in rec.raw_trajectory
    assert rec.metadata["grading"]["failing_tests"] == ["test_trailing_newline"]


# ---------------------------------------------------------------------------
# 4. The judge itself: cache, metering, failure
# ---------------------------------------------------------------------------


def subjects_for(instance, rollout):
    return {instance.task.instance_id: {"rollout": rollout, "task": instance.task, "grading": {}}}


def test_cache_prevents_a_second_judgement(taxonomy_file, tmp_path):
    instance = make_instances("failed__1")["failed__1"]
    rollout = make_rollout(instance)
    seen: list = []
    judge = TaxonomyJudge(
        taxonomy_path=taxonomy_file,
        meter=CostMeter(),
        model=SONNET,
        cache=JudgeCache.open(tmp_path / "judgements.jsonl"),
        runner=runner_returning([{"code": "A.9", "evidence": "hunk header does not match"}], seen=seen),
    )

    first = judge.judge(SEED_CANDIDATE, SOLVER, subjects_for(instance, rollout))
    second = judge.judge(SEED_CANDIDATE, SOLVER, subjects_for(instance, rollout))

    assert first == second
    assert len(seen) == 1, "the same (taxonomy, candidate, component, instance) must be judged once"
    assert judge.cache.hits == 1


def test_cache_survives_a_restart(taxonomy_file, tmp_path):
    instance = make_instances("failed__1")["failed__1"]
    rollout = make_rollout(instance)
    seen: list = []
    path = tmp_path / "judgements.jsonl"

    first_cache = JudgeCache.open(path)
    judge = TaxonomyJudge(
        taxonomy_path=taxonomy_file,
        meter=CostMeter(),
        model=SONNET,
        cache=first_cache,
        runner=runner_returning([{"code": "A.9", "evidence": "e"}], seen=seen),
    )
    judge.judge(SEED_CANDIDATE, SOLVER, subjects_for(instance, rollout))
    first_cache.close()

    resumed = TaxonomyJudge(
        taxonomy_path=taxonomy_file,
        meter=CostMeter(),
        model=SONNET,
        cache=JudgeCache.open(path),
        runner=runner_returning([{"code": "A.9", "evidence": "e"}], seen=seen),
    )
    out = resumed.judge(SEED_CANDIDATE, SOLVER, subjects_for(instance, rollout))

    assert len(seen) == 1, "a resumed run must not re-pay for a judgement"
    assert [m["code"] for m in out["failed__1"]] == ["A.9"]


def test_cache_key_separates_components(taxonomy_file, tmp_path):
    """A solver judgement must never be served for a refiner request."""
    instance = make_instances("failed__1")["failed__1"]
    rollout = make_rollout(instance)
    seen: list = []
    judge = TaxonomyJudge(
        taxonomy_path=taxonomy_file,
        meter=CostMeter(),
        model=SONNET,
        cache=JudgeCache.open(tmp_path / "j.jsonl"),
        runner=runner_returning([{"code": "A.9", "evidence": "e"}], seen=seen),
    )
    judge.judge(SEED_CANDIDATE, SOLVER, subjects_for(instance, rollout))
    judge.judge(SEED_CANDIDATE, REFINER, subjects_for(instance, rollout))
    assert len(seen) == 2
    assert [r["traces"][0]["metadata"]["role"] for r in seen] == ["solver", "refiner"]


def test_cache_key_separates_taxonomies(taxonomy_file, tmp_path):
    instance = make_instances("failed__1")["failed__1"]
    rollout = make_rollout(instance)
    seen: list = []
    cache = JudgeCache.open(tmp_path / "j.jsonl")

    def build(path):
        return TaxonomyJudge(
            taxonomy_path=path,
            meter=CostMeter(),
            model=SONNET,
            cache=cache,
            runner=runner_returning([{"code": "A.9", "evidence": "e"}], seen=seen),
        )

    build(taxonomy_file).judge(SEED_CANDIDATE, SOLVER, subjects_for(instance, rollout))

    edited = tmp_path / "taxonomy2.json"
    doc = json.loads(json.dumps(TAXONOMY_DOC))
    doc["codes"][0]["description"] += " (revised)"
    edited.write_text(json.dumps(doc))
    build(edited).judge(SEED_CANDIDATE, SOLVER, subjects_for(instance, rollout))

    assert len(seen) == 2, "a changed taxonomy invalidates judgements made under the old one"
    assert taxonomy_fingerprint(taxonomy_file) != taxonomy_fingerprint(edited)


def test_judge_spend_lands_in_the_meter_and_the_budget(taxonomy_file):
    instance = make_instances("failed__1")["failed__1"]
    rollout = make_rollout(instance)
    meter = CostMeter()
    stopper = MaxTotalCostStopper(0.01, meters=[meter])
    judge = TaxonomyJudge(
        taxonomy_path=taxonomy_file,
        meter=meter,
        model=SONNET,
        runner=runner_returning(
            [{"code": "A.9", "evidence": "e"}], prompt_chars=35_000, response_chars=700
        ),
    )

    assert stopper() is False
    judge.judge(SEED_CANDIDATE, SOLVER, subjects_for(instance, rollout))

    # 35_000 / 3.5 = 10_000 input tokens; 700 / 3.5 = 200 output tokens.
    # global.anthropic.claude-sonnet-4-6 is $3/$15 per million.
    assert meter.budgeted_usd == pytest.approx(10_000 * 3.0e-6 + 200 * 15.0e-6)
    assert meter.by_model[SONNET] == pytest.approx(meter.budgeted_usd)
    assert judge.spend_usd == pytest.approx(meter.budgeted_usd)
    assert stopper() is True, "judging must be able to exhaust the same budget as rollouts"


def test_judge_failure_is_soft(taxonomy_file, capsys):
    instance = make_instances("failed__1")["failed__1"]
    rollout = make_rollout(instance)

    def explode(request):
        raise RuntimeError("bedrock said no")

    judge = TaxonomyJudge(taxonomy_path=taxonomy_file, meter=CostMeter(), model=SONNET, runner=explode)
    assert judge.judge(SEED_CANDIDATE, SOLVER, subjects_for(instance, rollout)) == {}
    assert judge.failures == 1
    assert "taxonomy-judge" in capsys.readouterr().out

    # A second failure does not repeat the message: the run log is shared with gepa.
    judge.judge(SEED_CANDIDATE, SOLVER, subjects_for(instance, rollout))
    assert capsys.readouterr().out == ""
    assert judge.failures == 2


def test_a_broken_judge_does_not_break_reflection(taxonomy_file):
    """End to end: the exception happens inside a real judge, the dataset survives."""

    def explode(request):
        raise RuntimeError("bedrock said no")

    judge = TaxonomyJudge(taxonomy_path=taxonomy_file, meter=CostMeter(), model=SONNET, runner=explode)
    instances = make_instances("failed__1")
    adapter = make_adapter(instances, {"failed__1": 0.0}, taxonomy_judge=judge)

    res = adapter.evaluate(["failed__1"], SEED_CANDIDATE, capture_traces=True)
    ds = adapter.make_reflective_dataset(SEED_CANDIDATE, res, [SOLVER])

    assert "failure_modes" not in ds[SOLVER][0]
    assert ds[SOLVER][0]["instance_id"] == "failed__1"


def test_a_failed_judgement_is_not_cached(taxonomy_file, tmp_path):
    """Failures must stay retryable; only paid answers are durable."""
    instance = make_instances("failed__1")["failed__1"]
    rollout = make_rollout(instance)
    cache = JudgeCache.open(tmp_path / "j.jsonl")

    def explode(request):
        raise RuntimeError("nope")

    TaxonomyJudge(
        taxonomy_path=taxonomy_file, meter=CostMeter(), model=SONNET, cache=cache, runner=explode
    ).judge(SEED_CANDIDATE, SOLVER, subjects_for(instance, rollout))
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# 5. Normalising the judge's answer against the certified taxonomy
# ---------------------------------------------------------------------------


def test_name_and_severity_come_from_the_taxonomy_not_the_judge(taxonomy_file):
    """AdaMAST's judge catalog drops ``severity``, so a judge-reported one is invented."""
    instance = make_instances("failed__1")["failed__1"]
    rollout = make_rollout(instance)
    judge = TaxonomyJudge(
        taxonomy_path=taxonomy_file,
        meter=CostMeter(),
        model=SONNET,
        runner=runner_returning(
            [{"code": "A.9", "name": "Something The Judge Made Up", "severity": "minor", "evidence": "quoted span"}]
        ),
    )
    out = judge.judge(SEED_CANDIDATE, SOLVER, subjects_for(instance, rollout))
    assert out["failed__1"] == [
        {
            "code": "A.9",
            "name": "Diff_Context_Mismatch",
            "evidence": "quoted span",
            "severity": "major",
        }
    ]


def test_role_scoped_codes_are_dropped_for_the_wrong_subagent(taxonomy_file):
    instance = make_instances("failed__1")["failed__1"]
    rollout = make_rollout(instance)
    modes = [{"code": "B.4", "evidence": "s"}, {"code": "C.4", "evidence": "c"}]

    def build():
        return TaxonomyJudge(
            taxonomy_path=taxonomy_file, meter=CostMeter(), model=SONNET, runner=runner_returning(modes)
        )

    solver_judge = build()
    refiner_judge = build()
    solver_out = solver_judge.judge(SEED_CANDIDATE, SOLVER, subjects_for(instance, rollout))
    refiner_out = refiner_judge.judge(SEED_CANDIDATE, REFINER, subjects_for(instance, rollout))

    assert [m["code"] for m in solver_out["failed__1"]] == ["B.4", "C.4"]
    assert [m["code"] for m in refiner_out["failed__1"]] == ["C.4"]
    assert refiner_judge.role_mismatch_dropped == 1


def test_unknown_codes_are_discarded(taxonomy_file):
    """A code outside the certified taxonomy is not a diagnosis."""
    instance = make_instances("failed__1")["failed__1"]
    rollout = make_rollout(instance)
    judge = TaxonomyJudge(
        taxonomy_path=taxonomy_file,
        meter=CostMeter(),
        model=SONNET,
        runner=runner_returning([{"code": "Z.99", "evidence": "invented"}, {"code": "A.9", "evidence": "real"}]),
    )
    out = judge.judge(SEED_CANDIDATE, SOLVER, subjects_for(instance, rollout))
    assert [m["code"] for m in out["failed__1"]] == ["A.9"]


# ---------------------------------------------------------------------------
# 6. Wiring against the real artifacts
# ---------------------------------------------------------------------------


def test_the_judge_request_pins_mode_and_trace_budget(taxonomy_file):
    instance = make_instances("failed__1")["failed__1"]
    rollout = make_rollout(instance)
    seen: list = []
    TaxonomyJudge(
        taxonomy_path=taxonomy_file, meter=CostMeter(), model=SONNET, runner=runner_returning([], seen=seen)
    ).judge(SEED_CANDIDATE, SOLVER, subjects_for(instance, rollout))

    request = seen[0]
    assert request["mode"] == "default", "single-pass selection judge: one model call per trace"
    assert request["provider"] == "bedrock"
    assert request["max_trace_chars"] == MAX_TRACE_CHARS


@pytest.mark.skipif(not REAL_TRACES.exists(), reason="base-val traces not present")
def test_max_trace_chars_covers_every_real_trace():
    """The whole point of raising it off AdaMAST's 6000 default."""
    longest = 0
    with REAL_TRACES.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                longest = max(longest, len(json.loads(line)["raw_trajectory"]))
    assert longest > 100_000, "sanity: these are the long traces this constant exists for"
    assert MAX_TRACE_CHARS > longest * 2, f"largest real trajectory is {longest} chars"


@pytest.mark.skipif(not REAL_TAXONOMY.exists(), reason="pruned taxonomy not present")
def test_the_real_taxonomy_loads_and_is_role_aware():
    judge = TaxonomyJudge(taxonomy_path=REAL_TAXONOMY, meter=CostMeter(), model=SONNET, runner=runner_returning([]))
    assert len(judge._codes) == 22
    roles = {c.get("applies_to_role") for c in judge._codes.values() if c.get("applies_to_role")}
    assert roles == {"solver"}, "role routing depends on these matching ROLE_BY_COMPONENT"


def test_the_worker_is_standalone():
    """It runs under AdaMAST's interpreter, which has none of this package.

    Checked on the parsed import statements, not on the text: a docstring that
    mentions ``gepa_taxonomy`` is fine, an import of it is fatal at run time and
    would only surface as a failed judgement inside a paid run.
    """
    import ast

    source = (REPO_ROOT / "src" / "gepa_taxonomy" / "_adamast_worker.py").read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"json", "sys", "typing", "adamast", "__future__"}, imported
