"""Reuse of the base candidate's val evaluation across every run.

Requirement (D009)
-------------------------------
The base candidate's initial val evaluation is run **once** and its results are
reused by every optimization run -- all 3 seeds and later both arms -- so every
run starts from literally identical state, not a re-sampled approximation.

Why not gepa's own ``EvaluationCache``
--------------------------------------
gepa v0.1.4 *does* have a first-class ``EvaluationCache`` in
``gepa/core/state.py``, keyed on ``sha256(sorted(candidate.items()))`` plus the
example id, and ``GEPAEngine`` accepts a pre-built one via ``evaluation_cache=``.
That is exactly the mechanism we want -- but ``gepa.optimize()`` does not expose
it: at ``api.py:396-398`` it unconditionally constructs a *fresh empty* cache
when ``cache_evaluation=True``, with no injection point.

Baselines must run unmodified gepa v0.1.4 (baseline purity), so we cannot
patch that. We therefore implement reuse at the **adapter** level, using the
same key shape gepa uses. This is behaviour-neutral by construction: on a hit
the adapter returns the stored result verbatim and issues no LM call, which is
indistinguishable from having just computed it -- and, usefully, means the
reused seed evaluation contributes no spend, satisfying budget exclusion (a)
without any special-casing in the stopper.

Adding an ``evaluation_cache`` passthrough to ``optimize()`` is a clean,
self-contained upstream PR.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def candidate_hash(candidate: dict[str, str]) -> str:
    """Hash a candidate exactly as gepa v0.1.4 does (``state.py:31``).

    Matching gepa's key shape keeps our cache interchangeable with theirs if the
    upstream passthrough lands.
    """
    return hashlib.sha256(json.dumps(sorted(candidate.items())).encode()).hexdigest()


@dataclass
class SeedEvaluationCache:
    """Replay store for the base candidate's val evaluation.

    Read-through only: a miss is a hard error rather than a silent
    recomputation, because a silent recompute is precisely the failure mode
    this component exists to prevent -- it would make one seed start from a
    different state than the others.
    """

    entries: dict[str, dict[str, Any]]
    candidate_fingerprint: str
    frozen: bool = True

    @classmethod
    def build(cls, candidate: dict[str, str], results: dict[str, dict[str, Any]]) -> SeedEvaluationCache:
        return cls(entries=dict(results), candidate_fingerprint=candidate_hash(candidate))

    @classmethod
    def load(cls, path: str | Path) -> SeedEvaluationCache:
        data = json.loads(Path(path).read_text())
        return cls(
            entries=data["entries"],
            candidate_fingerprint=data["candidate_fingerprint"],
            frozen=True,
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "candidate_fingerprint": self.candidate_fingerprint,
                    "n": len(self.entries),
                    "entries": self.entries,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return p

    def matches(self, candidate: dict[str, str]) -> bool:
        """True if ``candidate`` is the base candidate this cache was built for."""
        return candidate_hash(candidate) == self.candidate_fingerprint

    def covers(self, instance_id: str) -> bool:
        """Whether this instance is inside the cache's scope (the val split)."""
        return instance_id in self.entries

    def missing(self, instance_ids: Iterable[str]) -> list[str]:
        """Which of ``instance_ids`` this cache does not cover."""
        return sorted(i for i in instance_ids if i not in self.entries)

    def assert_covers(self, instance_ids: Iterable[str]) -> None:
        """Launch-time completeness check for the val split.

        This is where D009's guarantee belongs -- checked ONCE against the val
        manifest, not inferred from a per-lookup miss. Doing it per-lookup was
        the bug: a train instance is legitimately absent, and treating that as
        an incomplete cache crashed the run.
        """
        gaps = self.missing(instance_ids)
        if gaps:
            raise KeyError(
                f"seed cache is missing {len(gaps)} of the val instances it must cover "
                f"(e.g. {gaps[:3]}). Every run must start from identical state, so the "
                "base-candidate val evaluation has to cover the whole val split. "
                "Rebuild it with scripts/build_base_val.py."
            )

    def get(self, candidate: dict[str, str], instance_id: str) -> dict[str, Any] | None:
        """Return the stored result verbatim, or None if this lookup is out of scope.

        The cache's scope is **(base candidate) x (val instances)**. Two kinds of
        miss are both legitimate and both return None:

        * a *different* candidate -- only the base candidate is replayed;
        * an instance outside the val split -- the base candidate is also
          evaluated on TRAIN minibatches during reflective mutation, and those
          are ordinary billed rollouts that must run live.

        The second case used to raise, on the mistaken assumption that any miss
        for the base candidate meant an incomplete cache. It killed a run at the
        first parent evaluation on a train minibatch. Completeness is now
        asserted once at launch by :meth:`assert_covers`.
        """
        if not self.matches(candidate):
            return None
        return self.entries.get(instance_id)
