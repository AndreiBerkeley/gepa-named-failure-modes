"""The trace record AdaMAST ingests, plus a local pre-flight validator.

``problem_id`` and ``raw_trajectory`` are the required fields; ``task`` may be
empty and ``metadata`` may hold any JSON object::

    {"problem_id": "...", "task": "...", "raw_trajectory": "...", "metadata": {}}

Outcome information (score, feedback) belongs in ``metadata``, never in
``raw_trajectory``: AdaMAST's checklist warns against leaking oracle outcomes
into what its judge reads, and the harvest code holds the same line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Fields AdaMAST requires on a native record.
REQUIRED_FIELDS = ("problem_id", "raw_trajectory")


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
