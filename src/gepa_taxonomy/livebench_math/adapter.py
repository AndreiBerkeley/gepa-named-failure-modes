"""``GEPAAdapter`` for the LiveBench-Math solve -> review program.

Structurally the HotpotQA adapter with a different program and grader; the
concurrency, error-classification, index-ordering and replay behaviour are the
same because each of those encodes a bug this project already paid for (F014,
F016, F029, F032).

Per-stage feedback
------------------
``review`` produces the graded answer, so it sees the grading verdict.
``solve`` sees the same verdict *plus* whether review kept or overrode its
draft -- which is the only signal that separates "solve was wrong" from "solve
was right and review broke it". Without it both look identical to the optimizer:
a score of 0 with no indication of which module lost the point.

Gold in reflection
------------------
Correct answers appear in reflective feedback only for ids in
``reflection_gold_ids`` -- the TRAIN manifest. Optimizer-level
supervision on train instances is standard GEPA practice; val and test gold
never enters reflection, and the program is gold-blind on every split.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from gepa.core.adapter import EvaluationBatch

from gepa_taxonomy.livebench_math.grading import Grade, answer_feedback, grade, score_feedback
from gepa_taxonomy.livebench_math.program import (
    COMPONENTS,
    REVIEW,
    SOLVE,
    ModuleCall,
    Rollout,
    SolveReviewProgram,
)
from gepa_taxonomy.livebench_math.tasks import Instance

_EXCERPT = 1500


def _excerpt(text: str, limit: int = _EXCERPT) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + " ... [truncated]"


@dataclass
class LiveBenchMathAdapter:
    """Runs the 2-module program and builds its reflective dataset."""

    program: SolveReviewProgram
    instances: Mapping[str, Instance]
    #: Ids whose ground truth may appear in reflective feedback (the train set).
    reflection_gold_ids: frozenset[str] | None = None
    #: gepa reads this attribute UNCONDITIONALLY (api.py:224). Omitting it makes
    #: every reflection silently fail, so a run burns its budget and never leaves
    #: the seed candidate.
    propose_new_texts: None = None

    max_transport_errors: int = 25
    #: Instances evaluated concurrently. Rollouts are network-bound and
    #: independent, so this is near-linear until the provider's rate limit binds
    #: -- at which point ``transport_errors`` climbs, which is the signal to drop it.
    max_workers: int = 1
    #: Replay of the base candidate's val evaluation, shared by every seed and
    #: both arms, so all runs start from byte-identical state.
    seed_cache: Any | None = None

    rollouts: int = 0
    replayed: int = 0
    spend_usd: float = 0.0
    #: Rollouts that failed because the model could not be reached rather than
    #: because the program misbehaved. Counted apart and hard-capped: they score
    #: 0.0, which the optimizer cannot distinguish from a genuinely bad candidate.
    transport_errors: int = 0
    program_errors: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def evaluate(
        self,
        batch: Sequence[Instance],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Run ``candidate`` over ``batch``, ``max_workers`` instances at a time.

        Results are assembled **by index, not by completion order**: gepa keys
        val subscores and the Pareto frontier positionally, so a batch
        returned in completion order would attach every score to the wrong
        instance -- silently.
        """
        n = len(batch)
        results: list[tuple[Rollout, Grade] | None] = [None] * n

        def run_one(index: int, instance: Instance) -> None:
            replayed = self._replay(candidate, instance)
            if replayed is not None:
                with self._lock:
                    self.replayed += 1
                results[index] = replayed
                return
            try:
                rollout = self.program.run(instance.task, candidate)
            except Exception as exc:
                # Never raise for ONE example: gepa's contract asks for a failure
                # score plus a trajectory explaining it. A transport failure is
                # not the program's fault, so it is counted apart -- and a storm
                # of them aborts rather than quietly scoring the candidate down.
                with self._lock:
                    if _is_transport_error(exc):
                        self.transport_errors += 1
                        over_limit = self.transport_errors >= self.max_transport_errors
                    else:
                        self.program_errors += 1
                        over_limit = False
                if over_limit:
                    raise RuntimeError(
                        f"aborting: {self.transport_errors} rollouts failed to reach the model "
                        f"(last: {type(exc).__name__}: {exc}). These score 0.0 and are "
                        f"indistinguishable from a bad candidate, so continuing would corrupt "
                        f"the run. Reduce --workers or raise --max-retries."
                    ) from exc
                rollout = Rollout(
                    example_id=instance.task.example_id,
                    question=instance.task.question,
                    subtask=instance.task.subtask,
                    error=f"{type(exc).__name__}: {exc}",
                )
            graded = self._grade(rollout, instance)
            with self._lock:
                self.rollouts += 1
                self.spend_usd += rollout.cost_usd
            results[index] = (rollout, graded)

        if self.max_workers <= 1 or n <= 1:
            for index, instance in enumerate(batch):
                run_one(index, instance)
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, n)) as pool:
                futures = [pool.submit(run_one, i, inst) for i, inst in enumerate(batch)]
                for future in futures:
                    # Re-raises the abort above; a systemic failure must not be
                    # swallowed just because it happened on a worker thread.
                    future.result()

        outputs: list[Any] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] = []
        for entry in results:
            assert entry is not None, "a worker returned no result"
            rollout, graded = entry
            outputs.append(rollout.answer)
            scores.append(graded.score)
            if capture_traces:
                trace = rollout.to_trace()
                trace["grading"] = {
                    "score": graded.score,
                    "scorer": graded.scorer,
                    "parsed": graded.parsed,
                    "positions": list(graded.positions),
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
        trajectories = list(eval_batch.trajectories or [])
        dataset: dict[str, list[dict[str, Any]]] = {}

        for component in components_to_update:
            examples: list[dict[str, Any]] = []
            for trace in trajectories:
                example_id = str(trace.get("example_id") or trace.get("instance_id") or "")
                instance = self.instances.get(example_id)
                if instance is None:
                    continue
                grading = trace.get("grading") or {}
                examples.append(
                    {
                        "example_id": example_id,
                        "problem": _excerpt(str(trace.get("task") or "")),
                        "produced": _excerpt(self._produced_by(trace, component)),
                        "feedback": self._feedback_for(component, trace, instance, example_id),
                        "score": grading.get("score"),
                    }
                )
            dataset[component] = examples
        return dataset

    # -- internals ---------------------------------------------------------

    def _grade(self, rollout: Rollout, instance: Instance) -> Grade:
        return grade(
            rollout.answer,
            instance.gold.ground_truth,
            subtask=instance.task.subtask,
            question=instance.task.question,
        )

    def _replay(self, candidate: dict[str, str], instance: Instance) -> tuple[Rollout, Grade] | None:
        """Serve the base candidate's val rollout from the shared evaluation.

        Scope is **(base candidate) x (val instances)**. Both kinds of miss
        return None rather than raising: a different candidate is not replayable,
        and the base candidate is also legitimately evaluated on TRAIN minibatches
        during reflective mutation, which are ordinary billed rollouts. Treating
        that second case as an incomplete cache killed a run once;
        completeness is asserted once at launch instead.
        """
        if self.seed_cache is None:
            return None
        stored = self.seed_cache.get(candidate, instance.task.example_id)
        if stored is None:
            return None

        trace = stored.get("trace") or {}
        rollout = Rollout(
            example_id=instance.task.example_id,
            question=instance.task.question,
            subtask=instance.task.subtask,
            draft_answer=trace.get("draft_answer", ""),
            answer=trace.get("answer", ""),
            calls=[
                ModuleCall(
                    component=c["component"],
                    prompt=c.get("prompt", ""),
                    output=c.get("output", ""),
                    input=c.get("input", ""),
                )
                for c in (trace.get("module_calls") or [])
            ],
            error=trace.get("error"),
        )
        # Re-graded rather than read from the cache: grading is deterministic
        # given the answer, and keeping one implementation means a metric change
        # can never silently disagree with a cached score.
        return rollout, self._grade(rollout, instance)

    def _produced_by(self, trace: Mapping[str, Any], component: str) -> str:
        for call in trace.get("module_calls") or []:
            if call.get("component") == component:
                return str(call.get("output") or "")
        return ""

    def _feedback_for(
        self,
        component: str,
        trace: Mapping[str, Any],
        instance: Instance,
        example_id: str,
    ) -> str:
        """Stage-scoped feedback."""
        grading = trace.get("grading") or {}
        graded = Grade(
            score=float(grading.get("score") or 0.0),
            scorer=str(grading.get("scorer") or ""),
            parsed=str(grading.get("parsed") or ""),
            positions=tuple(grading.get("positions") or (0, 0)),  # type: ignore[arg-type]
        )
        reveal = self.reflection_gold_ids is not None and example_id in self.reflection_gold_ids
        feedback = answer_feedback(graded, instance.gold.ground_truth) if reveal else score_feedback(graded)

        if component == SOLVE:
            # Whether review kept or replaced the draft is the ONLY signal that
            # separates "solve was wrong" from "solve was right and review broke
            # it". Both score 0 and are otherwise identical to the optimizer.
            draft = str(trace.get("draft_answer") or "")
            answer = str(trace.get("answer") or "")
            verdict = "kept your draft's answer" if _same_answer(draft, answer) else "replaced your draft"
            feedback = f"The review stage {verdict}.\n{feedback}"
        return feedback

    def summary(self) -> dict[str, Any]:
        return {
            "rollouts": self.rollouts,
            "spend_usd": round(self.spend_usd, 4),
            "replayed": self.replayed,
            "transport_errors": self.transport_errors,
            "program_errors": self.program_errors,
        }


def _same_answer(draft: str, answer: str) -> bool:
    """Cheap agreement check on the two stages' final lines.

    Compares the last non-empty line rather than the whole turn: review restates
    its reasoning, so full-text equality would report "replaced" every time and
    the signal would be constant -- i.e. no signal at all.
    """

    def tail(text: str) -> str:
        lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
        return lines[-1].lower() if lines else ""

    return tail(draft) == tail(answer)


#: Substrings identifying a failure to reach the model rather than a failure of
#: the program. Matched on type name and message because litellm raises a wide
#: family of provider-specific classes, and importing them to isinstance-check
#: would couple this adapter to litellm's internals.
_TRANSPORT_MARKERS = (
    "ratelimit",
    "throttl",
    "timeout",
    "serviceunavailable",
    "internalserver",
    "internal server",
    "apiconnection",
    "connectionerror",
    "overloaded",
    "toomanyrequests",
    "bedrockexception",
)


def _is_transport_error(exc: BaseException) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in _TRANSPORT_MARKERS)


def instances_by_id(instances: Iterable[Instance]) -> dict[str, Instance]:
    return {i.task.example_id: i for i in instances}


__all__ = ["COMPONENTS", "REVIEW", "SOLVE", "LiveBenchMathAdapter", "instances_by_id"]
