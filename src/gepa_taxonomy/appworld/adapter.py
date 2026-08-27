"""``GEPAAdapter`` for the single-component AppWorld ReAct agent.

Feedback
--------
AppWorld grades a task against **named requirements**, and the ones that failed
are the baseline arm's feedback -- the direct analogue of HotpotQA's "documents
still to be retrieved". It is a strong signal, and deliberately so: the taxonomy
arm has to beat it rather than a weakened version of it.

What it does *not* say is why the requirement failed, and that is the gap the
taxonomy is supposed to fill. "assert answers match." tells the optimizer the
answer was wrong; it does not say the agent never logged in, paged through only
the first page of results, or called ``complete_task()`` without an answer. On
HotpotQA the equivalent gap is narrower — the feedback names the missing
documents — which is part of why the two benchmarks are worth running together.

Concurrency: one SERVER per worker, not one experiment name per worker
---------------------------------------------------------------------
``appworld/serve/environment.py`` keeps ``world`` as a **module-level global**
and rejects any request for a different task::

    The active task world is 0d8a4ee_3. But you have requested to operate on
    0d8a4ee_1

So a server process holds exactly one live task, and the experiment name does
not isolate anything. An earlier version of this file claimed the opposite and
gave each worker its own experiment name against a single server; two of three
smoke rollouts died on that. Concurrency therefore needs **N server
processes on N ports**, and each worker is pinned to one of them.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from gepa.core.adapter import EvaluationBatch

from gepa_taxonomy.appworld.client import AppWorldClient, TaskResult
from gepa_taxonomy.appworld.program import REACT, ModuleCall, ReActProgram, Rollout

_EXCERPT = 1500


def _excerpt(text: str, limit: int = _EXCERPT) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + " ... [truncated]"


_TRANSPORT_MARKERS = (
    "ratelimit",
    "throttl",
    "timeout",
    "serviceunavailable",
    "internal server",
    "internalserver",
    "apiconnection",
    "connectionerror",
    "overloaded",
    "toomanyrequests",
    "unreachable",
)


def _is_transport_error(exc: BaseException) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in _TRANSPORT_MARKERS)


@dataclass
class AppWorldAdapter:
    """Runs the ReAct agent over AppWorld tasks and builds its reflective dataset."""

    program: ReActProgram
    #: Builds a client per worker. Distinct experiment names keep concurrent
    #: rollouts from sharing an environment.
    client_factory: Any = None
    #: gepa reads this attribute UNCONDITIONALLY (api.py:224); omitting it makes
    #: every reflection silently fail.
    propose_new_texts: None = None

    max_workers: int = 1
    max_transport_errors: int = 25
    #: Replay of the base candidate's val evaluation, shared by every seed and
    #: both arms. AppWorld rollouts are stochastic — a multi-step agent
    #: even more so than a fixed 4-call chain — so without this the baseline and
    #: taxonomy arms at the same seed would differ by an independent draw of the
    #: starting state as well as by the treatment.
    seed_cache: Any | None = None

    rollouts: int = 0
    #: Rollouts served from the shared base-val (no LM call, no environment, no spend).
    replayed: int = 0
    spend_usd: float = 0.0
    transport_errors: int = 0
    program_errors: int = 0
    steps_taken: int = 0
    step_exhaustions: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _local: threading.local = field(default_factory=threading.local, repr=False)

    def _program_for_thread(self) -> ReActProgram:
        """One program (and client) per worker thread.

        The environment is keyed by ``(task_id, experiment_name)``; sharing one
        experiment name across threads would let two rollouts of the same task
        collide in the same environment.
        """
        program = getattr(self._local, "program", None)
        if program is None:
            if self.client_factory is None:
                program = self.program
            else:
                program = ReActProgram(
                    client=self.client_factory(threading.get_ident()),
                    lm=self.program.lm,
                    meter=self.program.meter,
                    model=self.program.model,
                    max_steps=self.program.max_steps,
                    max_tokens=self.program.max_tokens,
                )
            self._local.program = program
        return program

    def evaluate(
        self,
        batch: Sequence[str],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        n = len(batch)
        results: list[Rollout | None] = [None] * n

        def run_one(index: int, task_id: str) -> None:
            replayed = self._replay(candidate, task_id)
            if replayed is not None:
                with self._lock:
                    self.replayed += 1
                    self.rollouts += 1
                results[index] = replayed
                return
            try:
                rollout = self._program_for_thread().run(task_id, candidate)
            except Exception as exc:
                with self._lock:
                    if _is_transport_error(exc):
                        self.transport_errors += 1
                        over = self.transport_errors >= self.max_transport_errors
                    else:
                        self.program_errors += 1
                        over = False
                if over:
                    raise RuntimeError(
                        f"aborting: {self.transport_errors} rollouts could not reach the "
                        f"model or the AppWorld server (last: {type(exc).__name__}: {exc}). "
                        "These score 0.0 and are indistinguishable from a bad candidate."
                    ) from exc
                rollout = Rollout(task_id=task_id, error=f"{type(exc).__name__}: {exc}")

            with self._lock:
                self.rollouts += 1
                self.spend_usd += rollout.cost_usd
                self.steps_taken += rollout.steps
                self.step_exhaustions += int(rollout.exhausted_steps)
            results[index] = rollout

        if self.max_workers <= 1 or n <= 1:
            for i, task_id in enumerate(batch):
                run_one(i, task_id)
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, n)) as pool:
                for future in [pool.submit(run_one, i, t) for i, t in enumerate(batch)]:
                    future.result()

        outputs, scores, trajectories = [], [], []
        for rollout in results:
            assert rollout is not None, "a worker returned no result"
            outputs.append(rollout.result.success if rollout.result else False)
            scores.append(rollout.score)
            if capture_traces:
                trace = rollout.to_trace()
                trace["grading"] = {
                    "score": rollout.score,
                    "success": bool(rollout.result and rollout.result.success),
                    "passes": list(rollout.result.passes) if rollout.result else [],
                    "failures": list(rollout.result.failures) if rollout.result else [],
                }
                trajectories.append(trace)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            num_metric_calls=n,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        dataset: dict[str, list[dict[str, Any]]] = {}
        for component in components_to_update:
            examples = []
            for trace in eval_batch.trajectories or []:
                grading = trace.get("grading") or {}
                calls = trace.get("module_calls") or []
                examples.append(
                    {
                        "task_id": trace.get("task_id"),
                        "task": _excerpt(str(trace.get("task") or "")),
                        # The last turn is where the agent either finished or gave
                        # up, so it is the most informative single excerpt.
                        "final_step": _excerpt(str(calls[-1].get("output") if calls else "")),
                        "steps": trace.get("steps"),
                        "feedback": self._feedback(trace, grading),
                        "score": grading.get("score"),
                    }
                )
            dataset[component] = examples
        return dataset

    def _replay(self, candidate: dict[str, str], task_id: str) -> Rollout | None:
        """Serve the base candidate's val rollout from the shared evaluation.

        Scope is **(base candidate) x (val tasks)**. Both kinds of miss return
        None rather than raising: a mutated candidate is not replayable, and the
        base candidate is also legitimately run on TRAIN tasks during reflective
        mutation, which are ordinary billed rollouts. Treating that second case
        as an incomplete cache killed a run once.

        A replayed rollout starts no environment and makes no LM call, so it
        costs nothing -- which is how the shared-evaluation budget exclusion
        holds by construction rather than by special-casing the stopper.
        """
        if self.seed_cache is None:
            return None
        stored = self.seed_cache.get(candidate, task_id)
        if stored is None:
            return None

        trace = stored.get("trace") or {}
        graded = stored.get("grading") or {}
        return Rollout(
            task_id=task_id,
            instruction_text=trace.get("task", ""),
            steps=int(trace.get("steps") or 0),
            completed=bool(trace.get("completed")),
            exhausted_steps=bool(trace.get("exhausted_steps")),
            empty_code_steps=int(trace.get("empty_code_steps") or 0),
            calls=[
                ModuleCall(
                    component=c["component"],
                    prompt=c.get("prompt", ""),
                    output=c.get("output", ""),
                    input=c.get("input", ""),
                )
                for c in (trace.get("module_calls") or [])
            ],
            result=TaskResult(
                task_id=task_id,
                success=bool(graded.get("success")),
                score=float(stored.get("score") or 0.0),
                num_tests=len(graded.get("passes") or []) + len(graded.get("failures") or []),
                passes=tuple(graded.get("passes") or []),
                failures=tuple(graded.get("failures") or []),
            ),
            error=trace.get("error"),
        )

    def _feedback(self, trace: Mapping[str, Any], grading: Mapping[str, Any]) -> str:
        """AppWorld's own verdict: which named requirements passed and failed."""
        passes = grading.get("passes") or []
        failures = grading.get("failures") or []
        lines = [
            f"Requirements passed ({len(passes)}): " + (", ".join(map(str, passes)) or "none"),
            f"Requirements failed ({len(failures)}): " + (", ".join(map(str, failures)) or "none"),
        ]
        if trace.get("exhausted_steps"):
            lines.append(f"The agent used all {trace.get('steps')} steps without calling complete_task().")
        if trace.get("empty_code_steps"):
            lines.append(f"{trace['empty_code_steps']} step(s) produced no code block.")
        if trace.get("error"):
            lines.append(f"The rollout errored: {trace['error']}")
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        return {
            "rollouts": self.rollouts,
            "replayed": self.replayed,
            "spend_usd": round(self.spend_usd, 4),
            "mean_steps": round(self.steps_taken / self.rollouts, 2) if self.rollouts else 0.0,
            "step_exhaustions": self.step_exhaustions,
            "transport_errors": self.transport_errors,
            "program_errors": self.program_errors,
        }


def client_factory(base_port: int, n_ports: int, prefix: str = "gepa", host: str = "localhost"):
    """Pin each worker thread to its own AppWorld **server**.

    A server holds one live task world globally, so workers must not share one
   . Threads are assigned ports round-robin on first use and keep them,
    which means ``n_ports`` servers must actually be running -- see
    ``ensure_servers`` in the run script.
    """
    assigned: dict[int, int] = {}
    counter = itertools.count()
    lock = threading.Lock()

    def make(thread_id: int) -> AppWorldClient:
        with lock:
            if thread_id not in assigned:
                assigned[thread_id] = base_port + (next(counter) % n_ports)
            port = assigned[thread_id]
        return AppWorldClient(base_url=f"http://{host}:{port}", experiment_name=prefix)

    return make


__all__ = ["REACT", "AppWorldAdapter", "client_factory"]
