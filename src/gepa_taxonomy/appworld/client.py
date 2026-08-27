"""Thin HTTP client for an ``appworld serve environment`` server.

Deliberately depends on nothing but the standard library. AppWorld pins
``pydantic >=1.9,<2.0`` while gepa and litellm require pydantic v2, so the two
cannot share an environment -- and AppWorld's own ``remote_environment_url``
client does not help, because it still imports the package that conflicts
. Talking to the server over plain JSON sidesteps the problem entirely:
the server runs in its own venv (in WSL, where ``signal.SIGALRM`` exists), and
this side imports nothing from it.

The endpoints, read off the server's OpenAPI spec rather than assumed::

    POST /initialize      task_id, experiment_name  -> task instruction, supervisor
    POST /execute         + code                    -> execution output
    POST /task_completed                            -> bool
    POST /evaluate        + suppress_errors         -> {success, passes, failures, ...}
    POST /close

``evaluate`` MUST be called with ``suppress_errors=True``. Without it the
evaluator's own assertion propagates and the server answers HTTP 500 -- which
looks like an outage but is really "the task was failed", and treating it as an
outage would abort healthy runs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE_URL = "http://localhost:8123"


class AppWorldServerError(RuntimeError):
    """The environment server could not service a request."""


#: AppWorld's label for a requirement that passed because nothing happened --
#: typically "assert no model changes". Counting these is actively harmful: an
#: agent that does absolutely nothing passes them all, so a do-nothing rollout
#: scores 0.50 on a 2-requirement task. That inflates the floor, compresses the
#: dynamic range the optimizer selects on, and rewards inaction. Measured on a
#: real task before this exclusion existed.
NO_OP_LABEL = "no_op_pass"


@dataclass(frozen=True, slots=True)
class TaskResult:
    """One graded AppWorld task.

    Two numbers, deliberately:

    * ``success`` -- AppWorld's own Task Goal Completion. This is what the
      leaderboard and the published GEPA/ACE comparisons report, so it is the
      headline figure and must not be redefined.
    * ``score`` -- the fraction of **substantive** requirements passed, used for
      optimization only. Partial credit is why this benchmark was chosen:
      a binary metric makes a minibatch comparison a coin flip, which is what
      left the SWE-Bench round unable to discriminate.

    Selection metric != reporting metric is a deliberate split, and gepa already
    separates the two uses (``sum(scores)`` for acceptance, ``mean`` for
    tracking).
    """

    task_id: str
    success: bool
    score: float
    num_tests: int
    passes: tuple[str, ...]
    failures: tuple[str, ...]
    #: Requirements that passed trivially; excluded from ``score``, kept for audit.
    no_op_passes: tuple[str, ...] = ()
    difficulty: int | None = None

    @classmethod
    def from_payload(cls, task_id: str, payload: dict[str, Any]) -> TaskResult:
        raw_passes = payload.get("passes") or []
        substantive = tuple(_requirement(p) for p in raw_passes if not _is_no_op(p))
        no_ops = tuple(_requirement(p) for p in raw_passes if _is_no_op(p))
        failures = tuple(_requirement(f) for f in payload.get("failures") or [])

        denominator = len(substantive) + len(failures)
        if denominator:
            score = len(substantive) / denominator
        else:
            # Every requirement was a no-op pass, so there is nothing substantive
            # to grade. Fall back to AppWorld's own verdict rather than inventing
            # a score from an empty set.
            score = 1.0 if payload.get("success") else 0.0

        return cls(
            task_id=task_id,
            success=bool(payload.get("success")),
            score=score,
            num_tests=int(payload.get("num_tests") or (len(raw_passes) + len(failures))),
            passes=substantive,
            failures=failures,
            no_op_passes=no_ops,
            difficulty=payload.get("difficulty"),
        )


def _is_no_op(entry: Any) -> bool:
    return isinstance(entry, dict) and str(entry.get("label") or "") == NO_OP_LABEL


def _requirement(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("requirement") or entry.get("label") or entry)
    return str(entry)


@dataclass
class AppWorldClient:
    """Drives one AppWorld task at a time over HTTP."""

    base_url: str = DEFAULT_BASE_URL
    experiment_name: str = "gepa"
    timeout_s: int = 300
    max_retries: int = 3
    calls: int = 0
    _last_error: str | None = field(default=None, repr=False)

    # -- transport ---------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        body = json.dumps(payload).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(
                f"{self.base_url}{path}",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    self.calls += 1
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                # A 500 here is usually the task failing an assertion, not the
                # server falling over; the caller decides. Do not retry it --
                # retrying a deterministic assertion just wastes wall-clock.
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                raise AppWorldServerError(f"{path} -> HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        raise AppWorldServerError(f"{path} unreachable after {self.max_retries} attempts: {last}")

    # -- api ---------------------------------------------------------------

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/", timeout=10) as response:
                return response.status == 200
        except Exception:
            return False

    def initialize(self, task_id: str) -> dict[str, Any]:
        """Start ``task_id`` and return its instruction and supervisor details."""
        payload = self._post("/initialize", self._task(task_id))
        return payload.get("output", payload)

    def execute(self, task_id: str, code: str) -> str:
        """Run ``code`` in the task's environment and return its output."""
        payload = self._post("/execute", {**self._task(task_id), "code": code})
        output = payload.get("output", payload)
        return output if isinstance(output, str) else json.dumps(output, default=str)

    def task_completed(self, task_id: str) -> bool:
        payload = self._post("/task_completed", self._task(task_id))
        return bool(payload.get("output", payload))

    def evaluate(self, task_id: str) -> TaskResult:
        """Grade the task. Always suppresses evaluator assertions -- see module docstring."""
        payload = self._post("/evaluate", {**self._task(task_id), "suppress_errors": True})
        return TaskResult.from_payload(task_id, payload.get("output", payload) or {})

    def close(self, task_id: str) -> None:
        """Release the task's environment.

        Failure is swallowed: an environment we could not close is a leak on the
        server, not a reason to fail a rollout that has already been graded.
        """
        try:
            self._post("/close", self._task(task_id))
        except AppWorldServerError:
            pass

    def _task(self, task_id: str) -> dict[str, Any]:
        return {"task_id": task_id, "experiment_name": self.experiment_name}
