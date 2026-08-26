"""Write-through rollout cache: durability at rollout granularity.

Problem
-------
gepa saves its state once per **iteration** (``engine.py:742``). An interruption
mid-iteration therefore discards that iteration's paid work -- at our measured
rates roughly $2.02 and ~28 rollouts, which locally is ~50 minutes of emulated
container time.

Approach
--------
Append every graded rollout to a JSONL the instant it completes, keyed exactly
as gepa keys its own evaluation cache: ``sha256(sorted(candidate.items()))``
plus the instance id. On restart the file is replayed into memory and consulted
before any LM call or container run, so completed work is never paid for twice.

Why JSONL and not a pickle: an append-only text file is durable under an abrupt
kill. A truncated final line loses at most the single rollout that was in
flight, and :meth:`load` discards it rather than failing to start.

This is the seed-cache mechanism generalised. Unlike ``SeedEvaluationCache``,
which is a frozen read-only replay of one candidate, this cache is written
continuously across all candidates for the lifetime of a run directory.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gepa_taxonomy.seed_cache import candidate_hash


@dataclass
class RolloutCache:
    """Append-only, crash-durable cache of graded rollouts."""

    path: Path
    _entries: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _fh: Any = field(default=None, repr=False)
    #: Rollouts served from cache this session -- i.e. work not paid for twice.
    hits: int = 0
    #: Malformed trailing records discarded at load (an interrupted append).
    truncated_records: int = 0

    @classmethod
    def open(cls, path: str | Path) -> RolloutCache:
        cache = cls(path=Path(path))
        cache.load()
        cache._fh = cache.path.open("a", buffering=1)  # line-buffered
        return cache

    # -- persistence ---------------------------------------------------

    def load(self) -> int:
        """Replay the log. A truncated final line is dropped, not fatal."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            return 0
        loaded = 0
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (rec["candidate_hash"], rec["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    # Only the last line can legitimately be partial; count and move on.
                    self.truncated_records += 1
                    continue
                self._entries[key] = rec
                loaded += 1
        return loaded

    def _write(self, rec: dict[str, Any]) -> None:
        """Append and force to disk, so a kill cannot lose an acknowledged rollout."""
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    # -- API ------------------------------------------------------------

    def get(self, candidate: dict[str, str], instance_id: str) -> dict[str, Any] | None:
        key = (candidate_hash(candidate), instance_id)
        with self._lock:
            rec = self._entries.get(key)
            if rec is not None:
                self.hits += 1
            return rec

    def put(
        self,
        candidate: dict[str, str],
        instance_id: str,
        *,
        score: float,
        output: dict[str, Any],
        trace: dict[str, Any] | None = None,
        cost_usd: float = 0.0,
    ) -> None:
        rec = {
            "candidate_hash": candidate_hash(candidate),
            "instance_id": instance_id,
            "score": score,
            "output": output,
            "trace": trace or {},
            "cost_usd": cost_usd,
        }
        with self._lock:
            self._entries[(rec["candidate_hash"], instance_id)] = rec
            self._write(rec)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def recovered_usd(self) -> float:
        """Spend represented by cached rollouts -- what a restart does not re-pay."""
        return sum(r.get("cost_usd", 0.0) for r in self._entries.values())
