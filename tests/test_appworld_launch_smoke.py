"""Full-launch smoke test for the AppWorld arm. FREE: both networks faked.

Every launch failure this project has had was reachable without a real model,
and each was missed because the tests stubbed too high up -- replacing our own
classes instead of the transport underneath them. So this stubs at the two
lowest layers only: ``litellm.completion`` for the LM, and ``urllib.request``
for the AppWorld HTTP client.

Everything above them is real: the real ``BedrockLM``, the real
``MeteredReflectionLM``, the real ``AppWorldClient`` (including its JSON parsing
and no-op-aware scoring), the real ReAct loop, the real adapter, the real cost
meters and stopper, and gepa's real reflection machinery.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import gepa
import pytest

from failure_taxonomy import FAILURE_MODES_KEY
from gepa_taxonomy.appworld.adapter import AppWorldAdapter, client_factory
from gepa_taxonomy.appworld.program import ReActProgram
from gepa_taxonomy.appworld.prompts import REACT, SEED_CANDIDATE
from gepa_taxonomy.bedrock import BedrockLM, MeteredReflectionLM, verify_reflection_lm
from gepa_taxonomy.cost import CostMeter, MaxTotalCostStopper

SOLVER = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
REFLECTION = "global.anthropic.claude-sonnet-4-6"

MUTATION_MARKER = "an improved instruction"

#: Exact tell for a rollout turn. It must live in the TASK TEMPLATE and nowhere
#: else: gepa puts the current instruction into its reflection prompt as
#: ``<curr_param>``, so anything drawn from the instruction (e.g.
#: "**Key instructions**") matches reflection prompts too -- and the fake then
#: answers a reflection request with a code block, so nothing is ever proposed.
#: That happened, and it looked exactly like reflection being broken.
ROLLOUT_MARKER = "Using these APIs, now generate code to solve the actual task:"


class FakeLM:
    """Stands in for ``litellm.completion``."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if ROLLOUT_MARKER in prompt:
            # A rollout driven by a MUTATED instruction answers better, so the
            # grader can reward it and gepa can accept a candidate. Without this
            # every candidate ties the base and nothing is ever accepted -- which
            # is indistinguishable from reflection being broken.
            answer = "improved" if MUTATION_MARKER in prompt else "x"
            return _response(f"```python\napis.supervisor.complete_task(answer='{answer}')\n```")
        return _response(f"```\n{MUTATION_MARKER}\n```")

    @property
    def models_called(self) -> set[str]:
        return {c["model"] for c in self.calls}


def _response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=3000, completion_tokens=120),
    )


class FakeAppWorldHTTP:
    """Stands in for the AppWorld environment server.

    Tasks whose id ends in an even digit succeed; the rest fail one requirement.
    That gives the optimizer a non-degenerate score distribution to select on --
    without it every candidate ties and acceptance looks broken.
    """

    def __init__(self):
        self.requests: list[tuple[str, dict]] = []
        self.last_code: dict[str, str] = {}

    def __call__(self, request, timeout=None):
        path = request.full_url.rsplit("/", 1)[-1]
        payload = json.loads(request.data.decode()) if request.data else {}
        self.requests.append((path, payload))
        task_id = payload.get("task_id", "")
        if path == "execute":
            self.last_code[task_id] = payload.get("code", "")

        if path == "initialize":
            body = {
                "output": {
                    "instruction": f"Do task {task_id}.",
                    "supervisor": {
                        "first_name": "A",
                        "last_name": "B",
                        "email": "a@b.c",
                        "phone_number": "1",
                    },
                }
            }
        elif path == "execute":
            body = {"output": "ok"}
        elif path == "task_completed":
            body = {"output": True}
        elif path == "evaluate":
            # An improved instruction solves everything; the seed solves half.
            good = "improved" in self.last_code.get(task_id, "") or task_id.endswith(("0", "2", "4", "6", "8"))
            body = {
                "output": {
                    "success": good,
                    "difficulty": 1,
                    "num_tests": 2,
                    "passes": ([{"requirement": "assert answers match."}] if good else [])
                    + [{"requirement": "assert no model changes.", "label": "no_op_pass"}],
                    "failures": [] if good else [{"requirement": "assert answers match."}],
                }
            }
        else:
            body = {"output": None}
        return _http(json.dumps(body))


class _http:
    def __init__(self, text):
        self._text = text
        self.status = 200

    def read(self):
        return self._text.encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fakes(monkeypatch):
    import litellm

    import gepa_taxonomy.appworld.client as client_mod

    lm, http = FakeLM(), FakeAppWorldHTTP()
    monkeypatch.setattr(litellm, "completion", lm)
    monkeypatch.setattr(client_mod.urllib.request, "urlopen", http)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    return lm, http


def _build(tmp_path, budget=5.0, workers=1):
    tasks = [f"task_{i}" for i in range(6)]
    solver_meter, reflection_meter = CostMeter(), CostMeter()
    program = ReActProgram(
        client=None,
        lm=BedrockLM(model=SOLVER, max_retries=2),
        meter=solver_meter,
        model=SOLVER,
        max_steps=4,
    )
    adapter = AppWorldAdapter(
        program=program,
        client_factory=client_factory(8123, 1, prefix="smoke"),
        max_workers=workers,
    )
    reflection_lm = MeteredReflectionLM(
        lm=BedrockLM(model=REFLECTION, max_retries=2),
        meter=reflection_meter,
        model=REFLECTION,
        spend_log=tmp_path / "reflection_spend.jsonl",
    )
    stopper = MaxTotalCostStopper(budget, [solver_meter, reflection_meter])
    return tasks, adapter, reflection_lm, stopper, solver_meter, reflection_meter


def _optimize(tmp_path, tasks, adapter, reflection_lm, stopper, **kw):
    return gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=tasks,
        valset=tasks,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        stop_callbacks=[stopper],
        seed=1,
        display_progress_bar=False,
        run_dir=str(tmp_path / "run"),
        **kw,
    )


def test_the_reflection_lm_conforms(tmp_path, fakes):
    *_, reflection_lm, _, _, _ = _build(tmp_path)[1:] + (None,)
    assert verify_reflection_lm(reflection_lm)["ok"]


def test_a_full_appworld_launch_completes_and_proposes(tmp_path, fakes):
    lm, http = fakes
    tasks, adapter, reflection_lm, stopper, solver_meter, reflection_meter = _build(tmp_path)
    result = _optimize(tmp_path, tasks, adapter, reflection_lm, stopper)

    assert len(result.candidates) > 1, "reflection never proposed a candidate"
    assert reflection_meter.calls > 0, "reflection spend was never metered"
    assert solver_meter.budgeted_usd > 0
    assert lm.models_called == {f"bedrock/{SOLVER}", f"bedrock/{REFLECTION}"}
    # The whole environment lifecycle really ran.
    assert {p for p, _ in http.requests} >= {"initialize", "execute", "task_completed", "evaluate", "close"}


def test_partial_credit_reaches_the_optimizer(tmp_path, fakes):
    """Half the tasks fail one substantive requirement, so scores must be a mix
    of 0.0 and 1.0 -- not all-or-nothing, and not all-tied."""
    tasks, adapter, *_ = _build(tmp_path)
    batch = adapter.evaluate(tasks, dict(SEED_CANDIDATE), capture_traces=True)
    assert set(batch.scores) == {0.0, 1.0}
    assert 0 < sum(batch.scores) < len(tasks)


def test_no_op_passes_do_not_inflate_the_failing_tasks(tmp_path, fakes):
    """Every task passes 'assert no model changes' trivially. If that counted,
    a failing task would score 0.5 and the floor would rise."""
    tasks, adapter, *_ = _build(tmp_path)
    batch = adapter.evaluate(tasks, dict(SEED_CANDIDATE), capture_traces=True)
    failing = [s for s in batch.scores if s < 1.0]
    assert failing and all(s == 0.0 for s in failing)


def test_each_worker_is_pinned_to_its_own_server_port():
    """The real isolation boundary is the SERVER, not the experiment name.

    ``appworld/serve/environment.py`` holds ``world`` as a module-level global
    and rejects requests for any other task, so two workers on one server clobber
    each other -- which killed two of three rollouts in the first real smoke
   . The superseded version of this test asserted distinct *experiment
    names* and passed happily against a fake that did not model the constraint.
    """
    from gepa_taxonomy.appworld.adapter import client_factory

    make = client_factory(8123, 3, prefix="t")
    urls = {make(tid).base_url for tid in (101, 202, 303)}
    assert urls == {
        "http://localhost:8123",
        "http://localhost:8124",
        "http://localhost:8125",
    }, f"workers shared a server: {urls}"

    # A thread keeps the port it was first given; reassigning mid-run would move
    # a live rollout to a server holding a different task world.
    assert make(101).base_url == make(101).base_url == "http://localhost:8123"


def test_the_budget_stopper_halts_the_run(tmp_path, fakes):
    tasks, adapter, reflection_lm, stopper, solver_meter, reflection_meter = _build(tmp_path, budget=0.05)
    _optimize(tmp_path, tasks, adapter, reflection_lm, stopper)
    spent = solver_meter.budgeted_usd + reflection_meter.budgeted_usd
    assert spent < 2.0, f"stopper did not halt: ${spent:.2f} against a $0.05 budget"


def test_baseline_reflective_dataset_carries_no_failure_modes(tmp_path, fakes):
    tasks, adapter, *_ = _build(tmp_path)
    batch = adapter.evaluate(tasks[:2], dict(SEED_CANDIDATE), capture_traces=True)
    dataset = adapter.make_reflective_dataset(dict(SEED_CANDIDATE), batch, [REACT])
    assert all(FAILURE_MODES_KEY not in ex for ex in dataset[REACT])
