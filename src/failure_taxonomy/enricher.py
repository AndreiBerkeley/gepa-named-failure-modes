"""Optimizer-side taxonomy feedback for GEPA reflective mutation.

The whole intervention is one sentence: *enrich the reflective dataset with a
structured diagnosis immediately before proposal, and change nothing else.*
The task adapter runs normally and remains the sole owner of evaluation and its
ordinary reflective dataset. GEPA then invokes this optional enricher with the
captured evaluation batch, so it can review the full trajectories without
wrapping DSPy, LangChain, OpenAI, or any other adapter.

Usage::

    enricher = TaxonomyFeedbackEnricher(judge=LLMFailureJudge(taxonomy, lm))
    gepa.optimize(
        seed_candidate=...,
        adapter=my_adapter,
        reflection_lm=reflection_lm,
        reflective_dataset_enricher=enricher,
        ...,
    )

With no enricher configured, GEPA does not enter this code. That keeps baseline
adapters and proposal behavior byte-identical to ordinary GEPA.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from failure_taxonomy.cache import candidate_key
from failure_taxonomy.judge import FailureJudge, Occurrence
from failure_taxonomy.reduce import assert_generation_disjoint
from failure_taxonomy.trace import build_trace

#: Key added to reflective-dataset examples. The single point of difference
#: between a baseline run and a taxonomy-conditioned one.
FAILURE_MODES_KEY = "failure_modes"


class TaxonomyFeedbackEnricher:
    """Adds taxonomy diagnoses between dataset construction and reflection."""

    def __init__(
        self,
        judge: FailureJudge,
        *,
        instance_id_keys: Sequence[str] = ("instance_id", "id", "example_id"),
        log: Callable[[str], None] = print,
    ) -> None:
        self.judge = judge
        self.instance_id_keys = tuple(instance_id_keys)
        self.log = log
        self.injected_examples = 0
        self.injected_occurrences = 0
        self.skipped_batches = 0
        self._warned = False

    # -- the intervention --------------------------------------------------

    def __call__(
        self,
        *,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        trajectories = list(getattr(eval_batch, "trajectories", None) or [])
        if not trajectories:
            return reflective_dataset

        try:
            by_component = self._diagnose(candidate, trajectories)
        except Exception as exc:
            # Fail soft: a lost diagnosis must never cost a paid run.
            self._warn(f"diagnosis failed: {type(exc).__name__}: {exc}")
            return reflective_dataset

        return self._inject(reflective_dataset, trajectories, by_component)

    # -- internals ---------------------------------------------------------

    def _diagnose(self, candidate: dict[str, str], trajectories: list[Any]) -> list[dict[str | None, list[Occurrence]]]:
        """Judge every rollout, returning per-trajectory occurrences grouped by component.

        Every instance is judged, not only failures. Submitting only failures
        would leak the outcome through selection -- the judge is deliberately
        not shown scores, but if the only traces it ever receives are failures,
        "no failure mode applies" becomes an answer it is never able to give.
        """
        traces = [build_trace(traj, trace_id=self._trace_id(traj, index)) for index, traj in enumerate(trajectories)]
        generation_ids = getattr(self.judge, "taxonomy", None)
        generation_ids = getattr(generation_ids, "generation_trace_ids", frozenset())
        if generation_ids:
            assert_generation_disjoint(generation_ids, [t.trace_id for t in traces], context="minibatch")
        if not any(t.is_segmented for t in traces):
            self._warn(
                "no trajectory exposes 'module_calls'; judging whole trajectories and "
                "routing every occurrence to all components"
            )

        key = candidate_key(candidate)
        if hasattr(self.judge, "candidate_key"):
            self.judge.candidate_key = key

        results = self.judge.judge(traces)
        return [_group_by_component(results.get(t.trace_id) or []) for t in traces]

    def _inject(
        self,
        dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        trajectories: list[Any],
        by_component: list[dict[str | None, list[Occurrence]]],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Attach each component's occurrences to its reflective examples.

        Examples are matched to trajectories positionally. GEPA guarantees
        trajectories align one-to-one with outputs and scores, and adapters
        build reflective examples by walking that same list, so index alignment
        is the contract in practice. It is still *checked*: a length mismatch
        means the inner adapter filtered or reordered, and attaching a
        diagnosis to the wrong rollout is worse than attaching none.
        """
        out: dict[str, list[dict[str, Any]]] = {}
        for component, examples in dataset.items():
            examples = list(examples)
            if len(examples) != len(trajectories):
                self.skipped_batches += 1
                self._warn(
                    f"component {component!r}: {len(examples)} reflective examples for "
                    f"{len(trajectories)} trajectories; cannot align, leaving undiagnosed"
                )
                out[component] = [dict(e) for e in examples]
                continue

            rebuilt: list[dict[str, Any]] = []
            for example, grouped in zip(examples, by_component, strict=True):
                enriched = dict(example)
                occurrences = grouped.get(component, []) + grouped.get(None, [])
                if occurrences:
                    enriched[FAILURE_MODES_KEY] = [o.for_reflection() for o in occurrences]
                    self.injected_examples += 1
                    self.injected_occurrences += len(occurrences)
                rebuilt.append(enriched)
            out[component] = rebuilt
        return out

    def _trace_id(self, trajectory: Any, index: int) -> str:
        for key in self.instance_id_keys:
            value = trajectory.get(key) if isinstance(trajectory, Mapping) else getattr(trajectory, key, None)
            if value:
                return str(value)
        return f"index-{index}"

    def _warn(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        self.log(f"  [taxonomy-feedback] {message} (logged once)")

    def summary(self) -> dict[str, Any]:
        """Counters for the additional review stage."""
        summary: dict[str, Any] = {
            "examples_diagnosed": self.injected_examples,
            "occurrences_injected": self.injected_occurrences,
            "unalignable_batches": self.skipped_batches,
        }
        judge_summary = getattr(self.judge, "summary", None)
        if callable(judge_summary):
            summary["judge"] = judge_summary()
        return summary


def _group_by_component(occurrences: Sequence[Occurrence]) -> dict[str | None, list[Occurrence]]:
    """Bucket occurrences by attributed component, preserving order and repeats."""
    grouped: dict[str | None, list[Occurrence]] = {}
    for occurrence in occurrences:
        grouped.setdefault(occurrence.component, []).append(occurrence)
    return grouped
