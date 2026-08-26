"""Emit generation-grade traces in the format AdaMAST actually ingests.

Verified against ``multi-agent-systems-failure-taxonomy/AdaMAST``
(``docs/TRACE_FORMATS.md``, read 2026-08-07) rather than assumed.

The canonical record
--------------------
``problem_id`` and ``raw_trajectory`` are the required fields; ``task`` may be
empty and ``metadata`` may hold any JSON object::

    {"problem_id": "...", "task": "...", "raw_trajectory": "...", "metadata": {}}

If none of ``raw_trajectory`` / ``trajectory`` / ``messages`` / ``trace.trajectory``
is present, **validation fails rather than guessing**.

Why this module had to exist
----------------------------
Our run traces stored ``solver_prompt_sha256`` / ``refiner_prompt_sha256`` --
digests, not content -- because at the time traces were only for scores and
debugging, and full prompts looked like bloat. Under the new plan the base-val
evaluation *is* the taxonomy-generation source, so a digest is useless: AdaMAST
needs the trajectory text to diagnose a failure. Emitting digests would have
meant re-paying for all 60 rollouts.

The size worry was also misjudged: 60 rollouts x ~60 KB of prompt is ~4 MB.

Gold blindness
--------------
AdaMAST's own checklist says to "keep benchmark labels or oracle outcomes out of
the trajectory when they would leak the answer to the judge". That matches our
hard invariant: the trajectory is assembled only from what the program itself
saw. Outcome information (score, resolved) goes in ``metadata``, never in
``raw_trajectory``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Fields AdaMAST requires on a native record.
REQUIRED_FIELDS = ("problem_id", "raw_trajectory")


def render_trajectory(
    *,
    solver_prompt: str,
    solver_output: str,
    feedback: str,
    refiner_prompt: str,
    refiner_output: str,
) -> str:
    """Render the two-stage rollout as a readable trajectory.

    The ``[ROLE]`` labels mirror how AdaMAST renders chat messages. They are a
    convention for readability, not required markup -- ``raw_trajectory`` is
    free-form text.
    """
    parts = [
        "[SYSTEM]\nTwo-stage SWE-Bench program: BM25 retrieval -> solver -> "
        "automated checks -> refiner. Exactly two model calls.",
        f"[USER: solver prompt]\n{solver_prompt}",
        f"[ASSISTANT: solver output]\n{solver_output or '(no output)'}",
        f"[TOOL: automated checks on the solver patch]\n{feedback or '(none)'}",
        f"[USER: refiner prompt]\n{refiner_prompt}",
        f"[ASSISTANT: refiner output]\n{refiner_output or '(no output)'}",
    ]
    return "\n\n".join(parts)


@dataclass(frozen=True)
class AdamastRecord:
    problem_id: str
    task: str
    raw_trajectory: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "task": self.task,
            "raw_trajectory": self.raw_trajectory,
            "metadata": self.metadata,
        }


def build_record(
    *,
    rollout: Any,
    task: Any,
    score: float,
    grading: dict[str, Any],
    difficulty: str | None = None,
    split: str = "val",
    system: str = "gepa-swebench-solver-refiner",
    extra_metadata: dict[str, Any] | None = None,
) -> AdamastRecord:
    """Build one AdaMAST-native record from a completed rollout.

    ``rollout`` must retain its prompts; a rollout whose prompts were dropped
    cannot produce a usable trajectory and is rejected rather than silently
    emitting an empty one.
    """
    solver_prompt, refiner_prompt = rollout.prompts()
    if not solver_prompt:
        raise ValueError(
            f"rollout for {rollout.instance_id} has no solver prompt; a trace without a "
            "trajectory fails AdaMAST validation. Re-run with prompt retention enabled."
        )

    trajectory = render_trajectory(
        solver_prompt=solver_prompt,
        solver_output=rollout.solver_patch,
        feedback=rollout.feedback.render(),
        refiner_prompt=refiner_prompt,
        refiner_output=rollout.refiner_patch,
    )

    metadata: dict[str, Any] = {
        "system": system,
        "split": split,
        "benchmark": "SWE-bench_Verified",
        "instance_id": rollout.instance_id,
        "repo": getattr(task, "repo", None),
        "base_commit": getattr(task, "base_commit", None),
        "difficulty": difficulty,
        "retrieved_paths": list(rollout.retrieved_paths),
        # Outcome lives here, NOT in the trajectory -- AdaMAST's checklist warns
        # against leaking oracle outcomes into what the judge reads.
        "score": score,
        "resolved": bool(score),
        "grading": grading,
        "solver_tokens_in": rollout.solver_tokens[0],
        "solver_tokens_out": rollout.solver_tokens[1],
        "refiner_tokens_in": rollout.refiner_tokens[0],
        "refiner_tokens_out": rollout.refiner_tokens[1],
        "cost_usd": rollout.cost_usd,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return AdamastRecord(
        problem_id=rollout.instance_id,
        task=getattr(task, "problem_statement", "") or "",
        raw_trajectory=trajectory,
        metadata=metadata,
    )


def write_jsonl(records: list[AdamastRecord], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict()) + "\n")
    return p


def validate(path: str | Path) -> dict[str, Any]:
    """Local pre-flight mirroring ``adamast validate``. No model calls.

    Checks the two conditions AdaMAST fails on: a missing identifier and an
    empty trajectory.
    """
    p = Path(path)
    n, empty, missing = 0, 0, 0
    ids: set[str] = set()
    dupes = 0
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        n += 1
        if not all(rec.get(f) for f in REQUIRED_FIELDS):
            missing += 1
            continue
        if not (rec.get("raw_trajectory") or "").strip():
            empty += 1
        pid = rec["problem_id"]
        if pid in ids:
            dupes += 1
        ids.add(pid)
    return {
        "trace_count": n,
        "empty_trajectories": empty,
        "missing_required_fields": missing,
        "duplicate_problem_ids": dupes,
        "ok": n > 0 and empty == 0 and missing == 0 and dupes == 0,
    }
