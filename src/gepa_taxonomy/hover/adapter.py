"""``GEPAAdapter`` for the HoVer 3-hop retrieval program.

Structurally the HotpotQA and IFBench adapters -- the concurrency, error
classification, index-ordering and replay behaviour are identical because each
encodes a bug this project has already paid for (F014, F016, F029, F032).

Two deliberate departures from the older adapters
-------------------------------------------------
1. **Failure classification uses ``gepa_taxonomy.failures``** rather than a
   local allow-list of transport substrings. The older adapters classify an
   unmatched exception as a *program* error, which does not count toward the
   abort threshold; IFBench seed 2 accumulated 273 of them and the guard never
   fired. ``failures`` inverts the default -- anything unrecognised is
   TRANSPORT, so it counts toward the abort -- and it keeps bounded error
   *samples*, which is what F053 asked for: the older adapters record a count
   with no cause, leaving 273 failures in a paid run permanently undiagnosable.

   HoVer is a new arm with no completed runs to stay comparable with, so this is
   the right place to adopt it.

2. **Per-stage feedback is hop-scoped.** Each query-writing component is told
   what its own hop retrieved and which documents are still missing, because on
   a strictly all-or-nothing metric every module gets the same 0.0 and there is
   otherwise nothing to tell them apart.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from gepa.core.adapter import EvaluationBatch

from gepa_taxonomy.failures import FailureLog
from gepa_taxonomy.hover.grading import Grade, grade, retrieval_feedback, score_feedback
from gepa_taxonomy.hover.program import (
    COMPONENTS,
    CREATE_QUERY_HOP2,
    CREATE_QUERY_HOP3,
    SUMMARIZE1,
    HoverMultiHopProgram,
    ModuleCall,
    Rollout,
)
from gepa_taxonomy.hover.tasks import Instance

_EXCERPT = 2000


def _excerpt(text: str, limit: int = _EXCERPT) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + " ... [truncated]"


@dataclass
class HoverAdapter:
    """Runs the 4-module program and builds its reflective dataset."""

    program: HoverMultiHopProgram
    instances: Mapping[str, Instance]
    #: Ids whose gold titles may be named in reflective feedback (train only).
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
    failures: FailureLog = field(default_factory=FailureLog)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Kept as properties so the summary shape matches the other adapters and
    # chain_status keeps parsing it.
    @property
    def transport_errors(self) -> int:
        return self.failures.summary()["transport_errors"]

    @property
    def program_errors(self) -> int:
        return self.failures.summary()["program_errors"]

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
                self.failures.record(exc)
                if self.failures.aborting_count >= self.max_transport_errors:
                    raise RuntimeError(
                        f"aborting: {self.failures.aborting_count} rollouts failed to reach the model "
                        f"(last: {type(exc).__name__}: {exc}). These score 0.0 and are "
                        f"indistinguishable from a bad candidate, so continuing would corrupt "
                        f"the run. Reduce --workers or raise --max-retries."
                    ) from exc
                rollout = Rollout(
                    example_id=instance.task.example_id,
                    claim=instance.task.claim,
                    error=f"{type(exc).__name__}: {exc}",
                )
            graded = grade(instance.gold, rollout.hop_titles)
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
                    future.result()

        outputs: list[Any] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] = []
        for entry in results:
            assert entry is not None, "a worker returned no result"
            rollout, graded = entry
            outputs.append(list(rollout.retrieved_titles))
            scores.append(graded.score)
            if capture_traces:
                trace = rollout.to_trace()
                trace["grading"] = {
                    "score": graded.score,
                    "loose_recall": graded.loose_recall,
                    "found": list(graded.found),
                    "missing": list(graded.missing),
                    "per_hop_found": [list(h) for h in graded.per_hop_found],
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
                        "claim": _excerpt(str(trace.get("task") or "")),
                        "produced": _excerpt(self._produced_by(trace, component)),
                        "feedback": self._feedback_for(component, trace, instance, example_id),
                        "score": grading.get("score"),
                    }
                )
            dataset[component] = examples
        return dataset

    # -- internals ---------------------------------------------------------

    def _replay(self, candidate: dict[str, str], instance: Instance) -> tuple[Rollout, Grade] | None:
        """Serve the base candidate's val rollout from the shared evaluation.

        Both kinds of miss return None rather than raising: a different
        candidate is not replayable, and the base candidate is also legitimately
        evaluated on TRAIN minibatches, which are ordinary billed rollouts.
        Treating that second case as an incomplete cache killed a run once
       ; completeness is asserted once at launch instead.
        """
        if self.seed_cache is None:
            return None
        stored = self.seed_cache.get(candidate, instance.task.example_id)
        if stored is None:
            return None

        trace = stored.get("trace") or {}
        rollout = Rollout(
            example_id=instance.task.example_id,
            claim=instance.task.claim,
            summary_1=trace.get("summary_1", ""),
            query_hop2=trace.get("query_hop2", ""),
            summary_2=trace.get("summary_2", ""),
            query_hop3=trace.get("query_hop3", ""),
            calls=[
                ModuleCall(
                    component=str(c.get("component") or ""),
                    prompt=str(c.get("prompt") or ""),
                    output=str(c.get("output") or ""),
                    input=str(c.get("input") or ""),
                )
                for c in (trace.get("module_calls") or [])
            ],
            error=trace.get("error"),
        )
        # Grades are recomputed from the stored per-hop titles rather than
        # trusted from the cache: the metric is ours and may be corrected, and a
        # stale cached score would silently outlive the fix.
        #
        # The titles come off the TRACE, which is the only thing the shared
        # base-val cache stores per instance. A cache written before
        # ``hop_titles`` was added to the trace would regrade every replayed
        # rollout against nothing and read as total retrieval failure, so an
        # empty list is refused rather than scored.
        hop_titles = [list(h) for h in (trace.get("hop_titles") or [])]
        if not hop_titles:
            raise ValueError(
                f"replayed trace for {instance.task.example_id} has no 'hop_titles'. "
                "The base-val cache predates the field; rebuild it with "
                "scripts/build_hover_base_val.py --force rather than scoring these as 0.0."
            )
        return rollout, grade(instance.gold, hop_titles)

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
        """Hop-scoped feedback.

        Every component receives the same 0.0 on a failed rollout, so the score
        alone cannot tell them apart. What separates them is which hop was
        supposed to find the missing document -- so each query writer is shown
        the query it actually issued alongside the outstanding gold.
        """
        grading = trace.get("grading") or {}
        graded = Grade(
            score=float(grading.get("score") or 0.0),
            loose_recall=float(grading.get("loose_recall") or 0.0),
            found=tuple(grading.get("found") or ()),
            missing=tuple(grading.get("missing") or ()),
            per_hop_found=tuple(tuple(h) for h in (grading.get("per_hop_found") or ())),
        )

        # Gold titles may only be named for train instances -- naming them on a
        # val or test rollout would hand the program its answer.
        if self.reflection_gold_ids is not None and example_id in self.reflection_gold_ids:
            feedback = retrieval_feedback(graded, component)
        else:
            feedback = score_feedback(graded)

        issued = {
            CREATE_QUERY_HOP2: ("Second-hop", trace.get("query_hop2")),
            CREATE_QUERY_HOP3: ("Third-hop", trace.get("query_hop3")),
        }.get(component)
        if issued:
            label, query = issued
            feedback = f"{label} search query issued: {query or '(empty)'}\n{feedback}"
        elif component == SUMMARIZE1:
            feedback = (
                "This summary is the only input the second-hop query writer sees; "
                f"a summary that drops an entity makes it unfindable later.\n{feedback}"
            )
        return feedback

    def summary(self) -> dict[str, Any]:
        out = {
            "rollouts": self.rollouts,
            "spend_usd": round(self.spend_usd, 4),
            "replayed": self.replayed,
        }
        out.update(self.failures.summary())
        return out


def instances_by_id(instances: Iterable[Instance]) -> dict[str, Instance]:
    return {i.task.example_id: i for i in instances}


__all__ = ["COMPONENTS", "HoverAdapter", "instances_by_id"]
