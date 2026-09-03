"""``GEPAAdapter`` for the IFBench generate -> ensure program.

The concurrency, error-classification, index-ordering and replay behaviour are
deliberate; each encodes a failure seen in earlier runs.

Per-stage feedback
------------------
``ensure_correct_response`` produces the graded text, so it sees the verdict.
``generate_response`` sees the same verdict *plus* whether the ensure stage
changed its draft -- the only signal separating "the draft was non-compliant"
from "the draft was fine and the ensure stage broke it". Both score identically
without it.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from gepa.core.adapter import EvaluationBatch

from gepa_taxonomy.ifbench.grading import Grade, constraint_feedback, grade, score_feedback
from gepa_taxonomy.ifbench.program import (
    COMPONENTS,
    ENSURE,
    GENERATE,
    GenerateEnsureProgram,
    ModuleCall,
    Rollout,
)
from gepa_taxonomy.ifbench.tasks import Instance

_EXCERPT = 2000


def _excerpt(text: str, limit: int = _EXCERPT) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + " ... [truncated]"


@dataclass
class IFBenchAdapter:
    """Runs the 2-module program and builds its reflective dataset."""

    program: GenerateEnsureProgram
    instances: Mapping[str, Instance]
    #: Ids whose failed-constraint names may appear in reflective feedback (train).
    reflection_gold_ids: frozenset[str] | None = None
    #: gepa reads this attribute UNCONDITIONALLY (api.py:224). Omitting it makes
    #: every reflection silently fail, so a run burns its budget and never leaves
    #: the seed candidate.
    propose_new_texts: None = None

    max_transport_errors: int = 25
    max_workers: int = 1
    #: Replay of the base candidate's val evaluation, shared by every seed and
    #: both arms, so all runs start from byte-identical state.
    seed_cache: Any | None = None

    rollouts: int = 0
    replayed: int = 0
    spend_usd: float = 0.0
    transport_errors: int = 0
    program_errors: int = 0
    #: Verifiers that raised. Distinct from program errors: a broken verifier
    #: mis-scores a whole constraint class in BOTH arms without failing loudly.
    verifier_errors: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def evaluate(
        self,
        batch: Sequence[Instance],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Run ``candidate`` over ``batch``, ``max_workers`` instances at a time.

        Results are assembled **by index, not completion order**: gepa keys val
        subscores and the Pareto frontier positionally.
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
                    query=instance.task.prompt,
                    error=f"{type(exc).__name__}: {exc}",
                )
            graded = self._grade(rollout, instance)
            with self._lock:
                self.rollouts += 1
                self.spend_usd += rollout.cost_usd
                self.verifier_errors += len(graded.errors)
            results[index] = (rollout, graded)

        if self.max_workers <= 1 or n <= 1:
            for index, instance in enumerate(batch):
                run_one(index, instance)
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, n)) as pool:
                futures = [pool.submit(run_one, i, inst) for i, inst in enumerate(batch)]
                for future in futures:
                    future.result()

        outputs: list[Any] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] = []
        for entry in results:
            assert entry is not None, "a worker returned no result"
            rollout, graded = entry
            outputs.append(rollout.response)
            scores.append(graded.score)
            if capture_traces:
                trace = rollout.to_trace()
                trace["grading"] = {
                    "score": graded.score,
                    "all_followed": graded.all_followed,
                    "followed": list(graded.followed),
                    "failed_ids": list(graded.failed_ids),
                    "loose_score": graded.loose_score,
                    "errors": list(graded.errors),
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
                        "query": _excerpt(str(trace.get("task") or "")),
                        "produced": _excerpt(self._produced_by(trace, component)),
                        "feedback": self._feedback_for(component, trace, instance, example_id),
                        "score": grading.get("score"),
                    }
                )
            dataset[component] = examples
        return dataset

    # -- internals ---------------------------------------------------------

    def _grade(self, rollout: Rollout, instance: Instance) -> Grade:
        return grade(rollout.response, instance.gold, prompt=instance.task.prompt)

    def _replay(self, candidate: dict[str, str], instance: Instance) -> tuple[Rollout, Grade] | None:
        """Serve the base candidate's val rollout from the shared evaluation.

        Both kinds of miss return None rather than raising: a different candidate
        is not replayable, and the base candidate is legitimately re-evaluated on
        TRAIN minibatches during reflective mutation. Treating that second case
        as an incomplete cache killed a run once.
        """
        if self.seed_cache is None:
            return None
        stored = self.seed_cache.get(candidate, instance.task.example_id)
        if stored is None:
            return None

        trace = stored.get("trace") or {}
        rollout = Rollout(
            example_id=instance.task.example_id,
            query=instance.task.prompt,
            draft=trace.get("draft", ""),
            response=trace.get("response", ""),
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
        # Re-graded rather than read from cache: grading is deterministic given
        # the response, so one implementation means a metric change can never
        # silently disagree with a cached score.
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
        grading = trace.get("grading") or {}
        graded = Grade(
            score=float(grading.get("score") or 0.0),
            all_followed=bool(grading.get("all_followed")),
            followed=tuple(grading.get("followed") or ()),
            failed_ids=tuple(grading.get("failed_ids") or ()),
            loose_score=float(grading.get("loose_score") or 0.0),
        )
        reveal = self.reflection_gold_ids is not None and example_id in self.reflection_gold_ids
        feedback = constraint_feedback(graded, instance.gold) if reveal else score_feedback(graded, instance.gold)

        if component == GENERATE:
            # Whether the ensure stage rewrote the draft is the ONLY signal
            # separating "the draft was non-compliant" from "the draft was fine
            # and ensure broke it". Both score identically otherwise.
            draft = str(trace.get("draft") or "").strip()
            response = str(trace.get("response") or "").strip()
            verdict = "kept your draft unchanged" if draft == response else "rewrote your draft"
            feedback = f"The ensure stage {verdict}.\n{feedback}"
        return feedback

    def summary(self) -> dict[str, Any]:
        return {
            "rollouts": self.rollouts,
            "spend_usd": round(self.spend_usd, 4),
            "replayed": self.replayed,
            "transport_errors": self.transport_errors,
            "program_errors": self.program_errors,
            "verifier_errors": self.verifier_errors,
        }


#: Substrings identifying a failure to reach the model rather than a failure of
#: the program. Matched on type name and message because litellm raises a wide
#: family of provider-specific classes.
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


__all__ = ["COMPONENTS", "ENSURE", "GENERATE", "IFBenchAdapter", "instances_by_id"]
