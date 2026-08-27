"""``GEPAAdapter`` for the HotpotQA multi-hop program.

Two responsibilities, per gepa's protocol: run the program on a batch and score
it, then build the per-component reflective dataset GEPA mutates instructions
from.

Per-stage feedback
------------------
The published setup says the feedback module "identifies the set of relevant
documents remaining to be retrieved **at each stage** of the program, and
provides that as feedback to the modules at that stage". So the retrieval
modules see retrieval feedback scoped to their own hop, and ``final_answer``
sees answer feedback. Giving every module the same undifferentiated blob would
be a weaker baseline than the published one, and beating a weakened baseline
proves nothing.

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

from gepa_taxonomy.hotpotqa.grading import Grade, answer_feedback, grade, retrieval_feedback
from gepa_taxonomy.hotpotqa.program import (
    COMPONENTS,
    CREATE_QUERY_HOP2,
    FINAL_ANSWER,
    SUMMARIZE1,
    SUMMARIZE2,
    ModuleCall,
    MultiHopProgram,
    Rollout,
)
from gepa_taxonomy.hotpotqa.retrieval import Passage
from gepa_taxonomy.hotpotqa.tasks import Instance

#: Which retrieval hop each component is responsible for. ``final_answer`` is
#: absent: it is scored on the answer, not on retrieval.
_HOP_FOR_COMPONENT: dict[str, int] = {SUMMARIZE1: 1, CREATE_QUERY_HOP2: 2, SUMMARIZE2: 2}

_EXCERPT = 1500


def _excerpt(text: str, limit: int = _EXCERPT) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + " ... [truncated]"


@dataclass
class HotpotQAAdapter:
    """Runs the 4-module program and builds its reflective dataset."""

    program: MultiHopProgram
    instances: Mapping[str, Instance]
    #: Ids whose gold answer may appear in reflective feedback (the train set).
    reflection_gold_ids: frozenset[str] | None = None
    #: gepa reads this attribute UNCONDITIONALLY (api.py:224). Omitting it makes
    #: every reflection silently fail, so a run burns its budget and never
    #: leaves the seed candidate.
    propose_new_texts: None = None

    #: Abort once this many rollouts have died of transport errors rather than
    #: of anything the program did. See ``transport_errors``.
    max_transport_errors: int = 25
    #: Instances evaluated concurrently. Rollouts are network-bound and
    #: independent, so this is close to a linear speedup until the provider's
    #: rate limit becomes the constraint -- at which point ``transport_errors``
    #: starts climbing, which is the signal to lower it.
    max_workers: int = 1
    #: Replay of the base candidate's val evaluation, shared by every seed and
    #: both arms. Without it each run re-samples the starting state, so
    #: the baseline and taxonomy arms at the same seed would differ by an
    #: independent 300-rollout draw as well as by the treatment -- noise added
    #: to precisely the comparison the experiment exists to make.
    seed_cache: Any | None = None

    rollouts: int = 0
    #: Rollouts served from the shared base-val evaluation (no LM call, no spend).
    replayed: int = 0
    spend_usd: float = 0.0
    #: Rollouts that failed because the model could not be reached (throttling,
    #: timeout, connection) rather than because the program misbehaved.
    #:
    #: These are counted separately and hard-capped because they are scored 0.0
    #: like any other failure, and a 0.0 that means "Bedrock throttled us" is
    #: indistinguishable to the optimizer from "this candidate is bad". Under
    #: concurrent seeds that is a live risk, and silently degrading a paid run
    #: is far worse than stopping it.
    transport_errors: int = 0
    program_errors: int = 0
    _last_grades: dict[str, Grade] = field(default_factory=dict, repr=False)
    #: Guards the counters and the grade map, which workers mutate concurrently.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def evaluate(
        self,
        batch: Sequence[Instance],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Run ``candidate`` over ``batch``, ``max_workers`` instances at a time.

        Instances are independent, and a rollout spends essentially all of its
        wall-clock blocked on the network -- measured at 0.0 CPU-seconds per 20
        seconds of elapsed time. Serially that made a 300-instance val
        evaluation take ~30 minutes and a $60 seed ~15 hours, with the machine
        idle throughout.

        The four calls *within* a rollout stay sequential: each one consumes the
        previous one's output, so there is nothing to overlap there.

        Results are assembled **by index, not by completion order**. gepa keys
        val subscores and the Pareto frontier positionally, so a batch
        returned in completion order would silently attach every score to the
        wrong instance.
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
                # score plus a trajectory explaining it. But a transport failure
                # is not the program's fault, so it is counted apart -- and a
                # storm of them aborts rather than quietly scoring the candidate
                # down (gepa reserves exceptions for systemic failures, and a
                # model we cannot reach is exactly that).
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
                    error=f"{type(exc).__name__}: {exc}",
                )
            graded = grade(rollout.answer, rollout.retrieved_titles, instance.gold)
            with self._lock:
                self.rollouts += 1
                self.spend_usd += rollout.cost_usd
                self._last_grades[instance.task.example_id] = graded
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
                    "f1": graded.f1,
                    "em": graded.em,
                    "retrieval_recall": graded.retrieval_recall,
                    "missing_titles": list(graded.missing_titles),
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
                produced = self._produced_by(trace, component)
                example: dict[str, Any] = {
                    "example_id": example_id,
                    "question": _excerpt(str(trace.get("task") or "")),
                    "produced": _excerpt(produced),
                    "feedback": self._feedback_for(component, trace, instance, example_id),
                    "score": grading.get("score"),
                }
                examples.append(example)
            dataset[component] = examples
        return dataset

    # -- internals ---------------------------------------------------------

    def _replay(self, candidate: dict[str, str], instance: Instance) -> tuple[Rollout, Grade] | None:
        """Serve the base candidate's val rollout from the shared evaluation.

        Scope is **(base candidate) x (val instances)**, and both kinds of miss
        return None rather than raising: a different candidate is not replayable,
        and the base candidate is also legitimately evaluated on TRAIN
        minibatches during reflective mutation, which are ordinary billed
        rollouts. Treating that second case as an incomplete cache killed a run
        once; completeness is asserted once at launch instead.

        A replayed rollout issues no LM call, so it contributes no spend -- which
        satisfies the budget exclusion for the shared seed evaluation by
        construction rather than by special-casing the stopper.
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
            summary_1=trace.get("summary_1", ""),
            query_hop2=trace.get("query_hop2", ""),
            summary_2=trace.get("summary_2", ""),
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
            # Passage bodies are not stored -- they are already embedded verbatim
            # in the module prompts above, so re-storing them would double the
            # cache for nothing. Titles are what grading and feedback need, and
            # rebuilding them as passages keeps ``retrieved_titles`` working
            # without adding a replay-only field to Rollout.
            passages_hop1=[Passage(title=t, text="") for t in (trace.get("retrieved_titles") or [])],
        )
        # Re-graded rather than stored: grading is deterministic given the answer
        # and the retrieved titles, and keeping one implementation means a metric
        # change can never silently disagree with a cached score.
        graded = grade(rollout.answer, rollout.retrieved_titles, instance.gold)
        return rollout, graded

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
        """Stage-scoped feedback, matching the published setup."""
        if component == FINAL_ANSWER:
            if self.reflection_gold_ids is not None and example_id in self.reflection_gold_ids:
                return answer_feedback(str(trace.get("answer") or ""), instance.gold)
            grading = trace.get("grading") or {}
            return f"Answer F1: {float(grading.get('f1') or 0.0):.2f}"

        retrieved = [str(t) for t in (trace.get("retrieved_titles") or [])]
        feedback = retrieval_feedback(retrieved, instance.gold)
        if component == CREATE_QUERY_HOP2:
            feedback = f"Second-hop search query issued: {trace.get('query_hop2') or '(empty)'}\n{feedback}"
        return feedback

    def summary(self) -> dict[str, Any]:
        return {
            "rollouts": self.rollouts,
            "spend_usd": round(self.spend_usd, 4),
            "replayed": self.replayed,
            "transport_errors": self.transport_errors,
            "program_errors": self.program_errors,
        }


#: Substrings identifying a failure to reach the model rather than a failure of
#: the program. Matched on the exception's type name and message because
#: litellm raises a wide family of provider-specific classes, and importing them
#: to isinstance-check would couple this adapter to litellm's internals.
_TRANSPORT_MARKERS = (
    "ratelimit",
    "throttl",
    "timeout",
    "serviceunavailable",
    "internalserver",
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


__all__ = ["COMPONENTS", "HotpotQAAdapter", "instances_by_id"]
