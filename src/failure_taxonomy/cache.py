"""Write-through cache of judgements.

Durability is the whole point. Judging happens inside a paid optimization loop,
so an interruption must re-pay nothing: each judgement is appended and flushed
to disk the instant it completes, rather than held until some later checkpoint.
A truncated final record -- the signature of a process killed mid-append -- is
dropped at load rather than treated as corruption, because refusing to start is
a worse failure than losing one judgement.

Key
---
``(taxonomy fingerprint, candidate key, trace id)``. The component is *not* part
of the key any more: one judgement now covers a whole rollout and carries its
own per-occurrence attribution, so there is nothing left to scope by. The
taxonomy fingerprint is present so that editing or re-pruning a taxonomy
invalidates every judgement made under the old one instead of silently mixing
two code sets inside one run.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from failure_taxonomy.judge import Occurrence


def candidate_key(candidate: Mapping[str, str]) -> str:
    """Stable hash of a candidate program.

    Keyed identically to GEPA's own evaluation cache
    (``sha256`` over the sorted items) so the two stay interchangeable.
    """
    payload = json.dumps(sorted(candidate.items()), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class JudgeCache:
    """Append-only JSONL cache of judgements."""

    path: Path
    _entries: dict[tuple[str, str, str], list[Occurrence]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _fh: Any = field(default=None, repr=False)
    #: Judgements served from cache this session, i.e. not paid for twice.
    hits: int = 0
    #: Malformed trailing records discarded at load (an interrupted append).
    truncated_records: int = 0

    @classmethod
    def open(cls, path: str | Path) -> JudgeCache:
        cache = cls(path=Path(path))
        cache.load()
        cache.path.parent.mkdir(parents=True, exist_ok=True)
        cache._fh = cache.path.open("a", buffering=1)
        return cache

    @staticmethod
    def _key(taxonomy: str, candidate_key: str, trace_id: str) -> tuple[str, str, str]:
        return (taxonomy, candidate_key, trace_id)

    def load(self) -> int:
        if not self.path.exists():
            return 0
        loaded = 0
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = self._key(rec["taxonomy"], rec["candidate_key"], rec["trace_id"])
                    occurrences = [_occurrence_from(o) for o in rec["occurrences"]]
                except (json.JSONDecodeError, KeyError, TypeError):
                    self.truncated_records += 1
                    continue
                self._entries[key] = occurrences
                loaded += 1
        return loaded

    def get(self, *, taxonomy: str, candidate_key: str, trace_id: str) -> list[Occurrence] | None:
        with self._lock:
            hit = self._entries.get(self._key(taxonomy, candidate_key, trace_id))
            if hit is None:
                return None
            self.hits += 1
            return list(hit)

    def put(
        self,
        *,
        taxonomy: str,
        candidate_key: str,
        trace_id: str,
        occurrences: Iterable[Occurrence],
    ) -> None:
        items = list(occurrences)
        rec = {
            "taxonomy": taxonomy,
            "candidate_key": candidate_key,
            "trace_id": trace_id,
            "occurrences": [
                {"code": o.code, "name": o.name, "evidence": o.evidence, "component": o.component} for o in items
            ],
        }
        with self._lock:
            self._entries[self._key(taxonomy, candidate_key, trace_id)] = items
            if self._fh is not None:
                self._fh.write(json.dumps(rec) + "\n")
                self._fh.flush()
                os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __len__(self) -> int:
        return len(self._entries)


def _occurrence_from(raw: Mapping[str, Any]) -> Occurrence:
    return Occurrence(
        code=str(raw["code"]),
        name=str(raw.get("name") or raw["code"]),
        evidence=str(raw.get("evidence") or ""),
        component=raw.get("component"),
    )
