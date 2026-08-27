"""Observability for reflective proposals.

Appends every post-enrichment reflective dataset to a JSONL file, exactly as
reflection consumed it: gepa fires ``on_reflective_dataset_built`` after the
enricher runs and after concretisation, so each record is the judge round's
actual output to the optimizer, not a reconstruction.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _field(event: Any, name: str) -> Any:
    # gepa's events are TypedDicts (plain dicts at runtime); tolerate attribute
    # style too, so a future dataclass event keeps logging.
    if isinstance(event, Mapping):
        return event.get(name)
    return getattr(event, name, None)


class ReflectionDatasetLogger:
    """gepa callback: one JSONL record per reflective dataset built."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append: records survive a killed run.
        self._fh = self._path.open("a", buffering=1, encoding="utf-8")

    def on_reflective_dataset_built(self, event: Any) -> None:
        record = {
            "iteration": _field(event, "iteration"),
            "candidate_idx": _field(event, "candidate_idx"),
            "components": list(_field(event, "components") or []),
            "dataset": _field(event, "dataset"),
        }
        self._fh.write(json.dumps(record, default=str) + "\n")

    def close(self) -> None:
        self._fh.close()
