"""The SWE-Bench ``GEPAAdapter``: the seam between our program and gepa.

Implements gepa's adapter protocol directly rather than going through the DSPy
adapter, so we control exactly what lands in the reflective dataset -- which is
precisely the surface Phase 5 will condition on the taxonomy.

Responsibilities, in order:

1. Run the program (2 LM calls) for each task in the batch.
2. Serve the base candidate's val results from the replay cache, so every run
   starts from identical state and the reused evaluation costs nothing.
3. Grade the resulting patch via the (host-agnostic) evaluation backend.
4. Audit each rollout for gold leakage -- this is the layer that *has* gold, so
   this is where the value-based check belongs.
5. Capture a structured trace per rollout, so Phase 3's harvest is a filter over
   data we already have.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gepa.core.adapter import EvaluationBatch

from gepa_taxonomy.cost import Phase
from gepa_taxonomy.patch_gate import gate
from gepa_taxonomy.program import COMPONENTS, Rollout, SolverRefinerProgram
from gepa_taxonomy.rollout_cache import RolloutCache
from gepa_taxonomy.seed_cache import SeedEvaluationCache, candidate_hash
from gepa_taxonomy.tasks import Gold, GoldLeakError, Instance, Task, assert_gold_free


class Grader:
    """Grades a patch. The only component that may hold a :class:`Gold`."""

    def grade(self, task: Task, gold: Gold, patch: str) -> tuple[float, dict[str, Any]]:
        raise NotImplementedError


#: Use gepa's OWN result type rather than mirroring its shape.
#:
#: We previously defined a look-alike with three fields. gepa's contract has
#: five, and it reads the two we omitted **unconditionally**:
#:   engine.py:129                 eval_result.objective_scores
#:   reflective_mutation.py:329    e.num_metric_calls
#: The first killed a launch at the seed valset replay; the second would have
#: fired later, during reflective mutation, i.e. after paid LM calls. Importing
#: the real dataclass makes that class of drift impossible.
EvaluationBatchResult = EvaluationBatch

#: Grading-detail keys surfaced to reflection, verbatim. ``run_id`` is
#: deliberately absent: it names a harness invocation, not anything about the
#: patch, and would only add noise to the reflection prompt.
#: Trace fields that are the model's own output. Excluded from value-based
#: gold matching only; field-name checks still apply to them.
_MODEL_OUTPUT_FIELDS = frozenset({"solver_patch", "refiner_patch"})

_HARNESS_RESULT_KEYS = (
    "resolved",
    "errored",
    "empty_patch",
    "skipped",
    "skipped_reason",
    "failing_tests",
    "test_output_tail",
)

#: How many rollouts stay available to the taxonomy judge. Only the parent's
#: current minibatch is ever judged (reflective_mutation.py:420), so this needs
#: to cover one minibatch plus the rollouts an interleaved child evaluation
#: pushes in between -- not the run. A rollout retains both prompts (~120 KB),
#: and a $100 run produces thousands of them.
_JUDGE_ROLLOUT_CAPACITY = 256


def _candidate_carries_gold(candidate: dict[str, str], gold: Gold) -> bool:
    """True if the candidate's own text contains non-trivial gold values.

    Same normalisation as the guard, so the two agree on what "contains" means.
    """
    hay = " ".join(" ".join(t.split()) for t in candidate.values())
    for value in (gold.patch, gold.test_patch):
        needle = " ".join(value.split())
        if len(needle) >= 20 and needle in hay:
            return True
    return False


@dataclass
class SweBenchAdapter:
    """gepa adapter for the solver->refiner SWE-Bench program."""

    program: SolverRefinerProgram
    grader: Grader
    instances: dict[str, Instance]
    seed_cache: SeedEvaluationCache | None = None
    #: Durable, rollout-granularity cache (O011). Without it an interruption
    #: loses the whole in-flight iteration's paid work.
    rollout_cache: RolloutCache | None = None
    #: Skip container evaluation for patches the harness would score 0 anyway
    #: (O012). See patch_gate for the equivalence argument.
    skip_ungradeable_patches: bool = True
    #: Checkout root used for the tier-2 "does it apply" check.
    repo_cache_dir: Path | None = None
    #: When set, every rollout is ALSO emitted as an AdaMAST-native record with
    #: the full trajectory. Required when this evaluation doubles as the
    #: taxonomy-generation source (D025) -- the ordinary trace keeps only prompt
    #: digests, which fail AdaMAST validation.
    adamast_path: Path | None = None
    difficulty: dict[str, str] = field(default_factory=dict)
    _adamast: list[Any] = field(default_factory=list, repr=False)
    trace_path: Path | None = None
    phase: Phase = "optimization"
    #: Fail the run on a detected gold leak (default) rather than warning.
    strict_gold_check: bool = True
    #: Ids whose gold patch may appear in the reflective dataset -- the TRAIN
    #: manifest set, and nothing else (D028). This is optimizer-level, train-only
    #: supervision: the program never sees it, and val/test ids must never be
    #: placed in this set.
    reflection_gold_ids: set[str] | None = None
    #: TREATMENT arm only. When set, each failed rollout in the parent's
    #: reflection minibatch is diagnosed against a certified AdaMAST taxonomy
    #: and the codes ride along in the reflective dataset. ``None`` is the
    #: BASELINE arm and the emitted examples are byte-identical to the
    #: pre-feature ones -- pinned by
    #: ``test_taxonomy_judge.py::test_baseline_reflective_example_is_byte_identical``.
    taxonomy_judge: Any = None
    _traces: list[dict[str, Any]] = field(default_factory=list, repr=False)
    #: Rollouts retained for the judge, keyed as gepa keys candidates.
    #: The judge needs the subagent's PROMPT and OUTPUT, and neither survives
    #: anywhere else: ``Rollout.to_trace`` stores prompt digests, and ``_finish``
    #: strips the patches. So a rollout served from the rollout cache cannot be
    #: judged -- the judge cache is what keeps that from mattering, since the
    #: first evaluation of a (candidate, instance) pair is the one that fills
    #: both. Bounded because a full run's prompts are hundreds of megabytes.
    _judge_rollouts: dict[tuple[str, str], Any] = field(default_factory=dict, repr=False)

    #: REQUIRED by gepa v0.1.4. ``ReflectiveMutationProposer._propose_texts_batch``
    #: reads ``self.adapter.propose_new_texts`` *unconditionally*
    #: (reflective_mutation.py:176). An adapter that simply omits the attribute
    #: raises AttributeError, which gepa catches and logs as "Reflective mutation
    #: did not propose a new candidate" -- so the run burns its whole budget and
    #: never leaves the seed candidate. ``None`` means "I do not own proposal;
    #: use the reflection LM", which is what we want.
    #: Pinned by test_adapter.py::test_adapter_declares_propose_new_texts.
    propose_new_texts: Any = None

    # -- gepa protocol ----------------------------------------------------

    def evaluate(
        self,
        batch: Sequence[str],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        missing = set(COMPONENTS) - set(candidate)
        if missing:
            raise KeyError(f"candidate is missing components: {sorted(missing)}")

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] = []

        # Pass 1 -- run the program, serve cache hits, and apply the skip gate.
        # Container grading is deferred so it can be done in ONE harness call:
        # per-instance invocation would pay harness startup on every rollout.
        executed = 0  # rollouts where the program actually ran (not replayed)
        pending: list[tuple[str, Any, Any]] = []  # (instance_id, rollout, task/gold)
        staged: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        for instance_id in batch:
            inst = self.instances[instance_id]
            order.append(instance_id)

            replayed = self._replay(candidate, instance_id)
            if replayed is not None:
                staged[instance_id] = {
                    "output": replayed["output"],
                    "score": replayed["score"],
                    "trace": replayed.get("trace", {}),
                    "store": False,
                }
                continue

            cached = self.rollout_cache.get(candidate, instance_id) if self.rollout_cache else None
            if cached is not None:
                staged[instance_id] = {
                    "output": cached["output"],
                    "score": cached["score"],
                    "trace": cached.get("trace", {}),
                    "store": False,
                }
                continue

            rollout = self.program.run(inst.task, candidate, phase=self.phase)
            executed += 1
            try:
                self._audit(rollout, inst.gold)
            except GoldLeakError:
                # Two very different causes surface identically here. Gold that
                # arrived through the CANDIDATE's own instruction is a memorised
                # proposal: that candidate is invalid, so reject it and carry on.
                # Gold arriving any other way (retrieval, scaffolding) is a
                # pipeline defect and must still fail the run.
                if not _candidate_carries_gold(candidate, inst.gold):
                    raise
                staged[instance_id] = {
                    "output": {"patch": "", "instance_id": instance_id},
                    "score": 0.0,
                    "trace": {
                        "instance_id": instance_id,
                        "grading": {"resolved": False, "gold_in_prompt": True},
                    },
                    "store": False,
                }
                print(f"  [gold-in-prompt] candidate rejected on {instance_id}", flush=True)
                continue

            decision = gate(
                rollout.final_patch,
                self._repo_dir(inst.task),
                enabled=self.skip_ungradeable_patches,
            )
            if decision.skip:
                staged[instance_id] = self._finish(
                    candidate,
                    instance_id,
                    rollout,
                    score=decision.score,
                    detail={"resolved": False, "skipped": True, "skipped_reason": decision.reason},
                )
            else:
                pending.append((instance_id, rollout, inst))

        # Pass 2 -- one batched harness call for everything that needs a container.
        if pending:
            graded = self._grade_many([(i.task, i.gold, r.final_patch) for _id, r, i in pending])
            for instance_id, rollout, _inst in pending:
                score, detail = graded[instance_id]
                staged[instance_id] = self._finish(candidate, instance_id, rollout, score=score, detail=detail)

        for instance_id in order:
            item = staged[instance_id]
            outputs.append(item["output"])
            scores.append(item["score"])
            if capture_traces:
                trajectories.append(item["trace"])

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            # We expose no multi-objective metrics; None is the documented value.
            objective_scores=None,
            # The honest count: replayed and cached rollouts cost nothing, so
            # reporting len(batch) would overstate the work actually done.
            num_metric_calls=executed,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: Iterable[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build the reflective dataset GEPA mutates instructions from.

        This is the surface Phase 5's taxonomy conditioning will attach to, so
        it is kept explicit and small: what the component was asked to do, what
        it produced, how the harness judged it, and -- for ids in
        ``reflection_gold_ids``, i.e. train instances only -- the reference
        patch (D028). That is optimizer-level supervision; the program itself
        stays gold-blind, and val/test instances never receive gold here.

        With a ``taxonomy_judge`` attached this also carries ``failure_modes``:
        the taxonomy codes the judge found in THAT component's own trace, for
        failed instances only. Codes are per component, never per rollout --
        reflection rewriting ``solver_instruction`` must not be shown the
        refiner's failures.
        """
        trajectories = eval_batch.trajectories or []
        dataset: dict[str, list[dict[str, Any]]] = {}

        for component in components_to_update:
            failure_modes = self._failure_modes(candidate, component, trajectories)
            examples: list[dict[str, Any]] = []
            for traj in trajectories:
                if not traj:
                    continue
                instance_id = traj["instance_id"]
                grading = traj.get("grading") or {}
                example: dict[str, Any] = {
                    "instance_id": instance_id,
                    "problem_statement_excerpt": _excerpt(
                        self.instances[instance_id].task.problem_statement
                    ),
                    "retrieved_paths": traj.get("retrieved_paths", []),
                    "produced_patch": _excerpt(
                        traj.get("solver_patch" if component == COMPONENTS[0] else "refiner_patch", ""),
                        limit=2000,
                    ),
                    "automated_checks": traj.get("feedback", {}),
                    "harness_result": {k: grading[k] for k in _HARNESS_RESULT_KEYS if k in grading},
                    "score": traj.get("score"),
                }
                if instance_id in failure_modes:
                    example["failure_modes"] = failure_modes[instance_id]
                if self.reflection_gold_ids is not None and instance_id in self.reflection_gold_ids:
                    # reflection_gold_ids is the train-manifest set; val/test
                    # ids must never receive gold here (D028).
                    example["reference_patch"] = _excerpt(
                        self.instances[instance_id].gold.patch, limit=2000
                    )
                    example["note"] = "reference_patch is the reference solution for this training example"
                examples.append(example)
            dataset[component] = examples
        return dataset

    # -- internals --------------------------------------------------------

    def _failure_modes(
        self,
        candidate: dict[str, str],
        component: str,
        trajectories: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Taxonomy codes for ``component`` on the failed rollouts in this batch.

        Empty dict when no judge is attached -- the baseline arm never reaches
        the judge and never pays for it. Instances whose rollout was served from
        a cache are skipped here rather than judged on a reconstruction: the
        judge is shown the real prompt and the real output or nothing.
        """
        if self.taxonomy_judge is None:
            return {}

        subjects: dict[str, dict[str, Any]] = {}
        for traj in trajectories:
            if not traj:
                continue
            # Failures only. A missing score is treated as a failure because the
            # only trace that omits it is the rejected gold-in-prompt candidate.
            if (traj.get("score") or 0.0) > 0:
                continue
            instance_id = traj["instance_id"]
            rollout = self._judge_rollouts.get((candidate_hash(candidate), instance_id))
            if rollout is None:
                continue
            subjects[instance_id] = {
                "rollout": rollout,
                "task": self.instances[instance_id].task,
                "grading": traj.get("grading") or {},
            }
        if not subjects:
            return {}
        return self.taxonomy_judge.judge(candidate, component, subjects)

    def _retain_for_judge(self, candidate: dict[str, str], instance_id: str, rollout: Rollout) -> None:
        self._judge_rollouts[(candidate_hash(candidate), instance_id)] = rollout
        while len(self._judge_rollouts) > _JUDGE_ROLLOUT_CAPACITY:
            self._judge_rollouts.pop(next(iter(self._judge_rollouts)))

    def _grade_many(self, items: list[tuple[Task, Gold, str]]) -> dict[str, tuple[float, dict[str, Any]]]:
        """Prefer the grader's batch API; fall back to per-instance."""
        batch_fn = getattr(self.grader, "grade_batch", None)
        if batch_fn is not None:
            return batch_fn(items)
        return {t.instance_id: self.grader.grade(t, g, p) for t, g, p in items}

    def _finish(
        self,
        candidate: dict[str, str],
        instance_id: str,
        rollout: Rollout,
        *,
        score: float,
        detail: dict[str, Any],
    ) -> dict[str, Any]:
        """Record the trace, persist to the durable cache, and stage the result."""
        # The STORED trace keeps the produced patches: reflection critiques the
        # patch, the judge reads it, and the rollout cache replays it. Only the
        # gold audit filters them out, and it does so on its own copy.
        trace = rollout.to_trace()
        trace["score"] = score
        trace["grading"] = detail
        self._traces.append(trace)

        if self.taxonomy_judge is not None:
            self._retain_for_judge(candidate, instance_id, rollout)

        if self.adamast_path is not None:
            from gepa_taxonomy.adamast_trace import build_record

            self._adamast.append(
                build_record(
                    rollout=rollout,
                    task=self.instances[instance_id].task,
                    score=score,
                    grading=detail,
                    difficulty=self.difficulty.get(instance_id),
                )
            )

        out = {"patch": rollout.final_patch, "instance_id": instance_id}
        if self.rollout_cache is not None:
            self.rollout_cache.put(
                candidate, instance_id, score=score, output=out, trace=trace, cost_usd=rollout.cost_usd
            )
        return {"output": out, "score": score, "trace": trace, "store": True}

    def _repo_dir(self, task: Task) -> Path | None:
        """Checkout for the tier-2 apply check; None disables that tier."""
        if self.repo_cache_dir is None:
            return None
        d = Path(self.repo_cache_dir) / task.repo.replace("/", "__")
        return d if d.is_dir() else None

    def _replay(self, candidate: dict[str, str], instance_id: str) -> dict[str, Any] | None:
        """Serve the base candidate's val result verbatim, if we have it."""
        if self.seed_cache is None:
            return None
        return self.seed_cache.get(candidate, instance_id)

    def _audit(self, rollout: Rollout, gold: Gold) -> None:
        """Value-based gold-leak check at the true program boundary.

        The program cannot do this itself: it has no Gold, and giving it one
        would *be* the leak. The adapter holds gold in order to grade, so the
        check belongs here.
        """
        try:
            # Inputs: strict. Nothing gold-derived may reach the program.
            assert_gold_free(
                rollout.prompts(),
                where=f"prompts[{rollout.instance_id}]",
                gold=gold,
                public_text=self.instances[rollout.instance_id].task.problem_statement,
            )
            # Outputs: a model that solves an instance correctly *produces* the
            # reference patch, so value-matching its own patches reports a leak
            # where there is none. The prompt check above is what makes this
            # safe: the model can only emit what its inputs allowed.
            trace = {k: v for k, v in rollout.to_trace().items() if k not in _MODEL_OUTPUT_FIELDS}
            assert_gold_free(trace, where=f"trace[{rollout.instance_id}]", gold=gold)
        except Exception:
            if self.strict_gold_check:
                raise

    def flush_adamast(self) -> Path | None:
        """Append generation-grade records. Written incrementally so an
        interruption cannot lose traces we have already paid for."""
        if self.adamast_path is None or not self._adamast:
            return None
        import json as _json

        self.adamast_path.parent.mkdir(parents=True, exist_ok=True)
        with self.adamast_path.open("a") as fh:
            for rec in self._adamast:
                fh.write(_json.dumps(rec.to_dict()) + "\n")
        self._adamast = []
        return self.adamast_path

    def flush_traces(self, path: Path | None = None) -> Path | None:
        """Write captured traces as JSONL -- the Phase 3 stage artifact."""
        target = path or self.trace_path
        if target is None or not self._traces:
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as fh:
            for trace in self._traces:
                fh.write(json.dumps(trace) + "\n")
        written, self._traces = len(self._traces), []
        _ = written
        return target


def _excerpt(text: str, limit: int = 1200) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"
