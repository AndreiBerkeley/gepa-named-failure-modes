"""Harvesting traces for the taxonomy-generation stage.

This is the *first* half of the stage boundary. A taxonomy generator reads
traces and emits ``taxonomy.json``; this package reads ``taxonomy.json`` and
conditions reflection. Neither half needs to know which tool produced or
consumes the other, which is what makes a bring-your-own-taxonomy workflow
possible: skip generation entirely, hand over a JSON file, and the judge works.

Why the export is segmented
---------------------------
A generator that has to recover component structure from trace prose will
recover whatever the prose contains. On a run whose prompts embedded retrieved
source files, one generator's agent discovery returned ``["any", "name", "str",
"rel_obj_id", "key", "col_pos", "alias", "truncated"]`` -- Python identifiers
scraped out of the embedded code -- and described the "agent" as implementing
array iteration. It found one role where the program had two, so an entire
component received no role-specific codes at all.

GEPA knows the component names exactly. Writing them into the export removes
the guess, and costs nothing.

Outcomes stay out of the trajectory
-----------------------------------
Scores and pass/fail land in ``metadata``, never in ``raw_trajectory``. A
generator shown the outcome can write codes that describe *being wrong* rather
than describing observable behaviour, and those codes are then unjudgeable at
optimization time, when no outcome is available to the judge.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from failure_taxonomy.trace import SegmentedTrace, build_trace


def harvest_traces(
    eval_batch: Any,
    *,
    instance_ids: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    instance_id_keys: Sequence[str] = ("instance_id", "id", "example_id"),
) -> list[SegmentedTrace]:
    """Turn one ``EvaluationBatch`` into segmented traces.

    ``scores`` are attached as metadata when present so a later analysis can
    partition traces by outcome without the generator having seen it.
    """
    trajectories = list(getattr(eval_batch, "trajectories", None) or [])
    scores = list(getattr(eval_batch, "scores", None) or [])

    traces: list[SegmentedTrace] = []
    for index, trajectory in enumerate(trajectories):
        trace_id = _trace_id(trajectory, index, instance_ids, instance_id_keys)
        meta: dict[str, Any] = dict(metadata or {})
        if index < len(scores):
            meta["score"] = scores[index]
        traces.append(
            build_trace(
                trajectory,
                trace_id=trace_id,
                task=tasks[index] if tasks and index < len(tasks) else "",
                metadata=meta,
            )
        )
    return traces


def write_generation_traces(traces: Iterable[SegmentedTrace], path: str | Path) -> Path:
    """Write traces as JSONL for a taxonomy generator.

    Returns the path written. Raises on an empty trajectory rather than writing
    it: a record with no trajectory text fails generator-side validation anyway,
    and discovering that after paying for the rollouts is the expensive way to
    find out.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for trace in traces:
        record = trace.to_generation_record()
        if not record["raw_trajectory"].strip():
            raise ValueError(f"trace {trace.trace_id!r} has an empty trajectory; refusing to write it")
        records.append(record)
    if not records:
        raise ValueError("no traces to write")
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out


def trace_report(traces: Sequence[SegmentedTrace]) -> dict[str, Any]:
    """Summarise a trace bundle before it is handed to a generator.

    The counts worth checking before spending: how many traces carry component
    structure at all, and which component names the generator will see. A bundle
    that is mostly unsegmented will produce a taxonomy whose codes cannot be
    routed, and it is far cheaper to notice that here.
    """
    components: dict[str, int] = {}
    segmented = 0
    for trace in traces:
        if trace.is_segmented:
            segmented += 1
        for name in trace.components:
            components[name] = components.get(name, 0) + 1
    return {
        "traces": len(traces),
        "segmented": segmented,
        "unsegmented": len(traces) - segmented,
        "components": dict(sorted(components.items())),
        "chars": {
            "total": sum(len(t.render()) for t in traces),
            "max": max((len(t.render()) for t in traces), default=0),
        },
    }


def _trace_id(
    trajectory: Any,
    index: int,
    instance_ids: Sequence[str] | None,
    instance_id_keys: Sequence[str],
) -> str:
    if instance_ids and index < len(instance_ids):
        return str(instance_ids[index])
    for key in instance_id_keys:
        value = trajectory.get(key) if isinstance(trajectory, Mapping) else getattr(trajectory, key, None)
        if value:
            return str(value)
    return f"index-{index}"
