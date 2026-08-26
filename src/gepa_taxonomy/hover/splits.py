"""Deterministic splits over HoVer, mirroring the HotpotQA setup.

Sizes are **150 train / 300 val / 300 test** -- the same as HotpotQA, which
takes them from GEPA appendix E.1. Matching them keeps the two benchmarks
directly comparable to each other and to the published numbers, and the prior
HoVer pilot also measured on a 300-instance test split.

Pool
----
HoVer's **dev** release (4,000 labelled claims), not train (18,171). Three
reasons, in order of weight: the official test split is unlabelled so it cannot
be graded locally; drawing all three subsets from one pool keeps them
identically distributed, which is what makes a val-selected candidate's test
score interpretable; and dev is the analogue of the HotpotQA validation pool we
already draw from. 750 of 4,000 leaves comfortable headroom.

Stratification
--------------
By ``num_hops`` (2, 3 or 4) -- HoVer's own difficulty axis, and a steep one:
a 4-hop claim needs four specific articles found before it scores at all, on a
strictly all-or-nothing metric. The pool is 50% / 34% / 17% across 2/3/4 hops,
so an unstratified draw could easily hand val and test materially different
difficulty and make their scores incomparable. That is the same failure the
SWE-Bench round hit from the other direction.

Guarantees, each covered by a test: disjoint, correctly sized, stratified, and
byte-identical across runs at a fixed seed.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SEED = 20260818
DATASET_NAME = "hover"
DATASET_SPLIT = "dev_release_v1.1"

#: Matched to HotpotQA. Order matters: subsets are carved in this order.
DEFAULT_SIZES: dict[str, int] = {"train": 150, "val": 300, "test": 300}


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """A committed stage artifact (D005): reproducible from seed + revision."""

    name: str
    seed: int
    dataset: str
    dataset_split: str
    n: int
    example_ids: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "seed": self.seed,
            "dataset": self.dataset,
            "dataset_split": self.dataset_split,
            "n": self.n,
            # Sorted so the manifest is stable under upstream re-ordering, and so
            # positional indices into it are meaningful: gepa keys val subscores
            # and the Pareto frontier by POSITION, not by id (F014).
            "example_ids": sorted(self.example_ids),
        }

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")
        return out


def stratum_of(record: Mapping[str, object]) -> str:
    """The stratification key: hop count, as a string for stable sorting."""
    return str(record.get("num_hops") or "unknown")


def build_splits(
    records: Sequence[Mapping[str, object]],
    *,
    seed: int = DEFAULT_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, SplitManifest]:
    """Partition ``records`` into disjoint, stratified subsets.

    Allocation is largest-remainder within each stratum, which holds the subset
    proportions without the drift a naive round-robin accumulates. Any shortfall
    from rounding is filled from the unallocated remainder, so requested sizes
    are met exactly.
    """
    sizes = dict(sizes or DEFAULT_SIZES)
    total_needed = sum(sizes.values())
    if len(records) < total_needed:
        raise ValueError(f"pool has {len(records)} records, need {total_needed}")

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_stratum[stratum_of(record)].append(str(record["uid"]))

    rng = random.Random(seed)
    for ids in by_stratum.values():
        ids.sort()  # deterministic starting order before shuffling
        rng.shuffle(ids)

    pool_size = len(records)
    assigned: dict[str, list[str]] = {name: [] for name in sizes}
    cursor: dict[str, int] = dict.fromkeys(by_stratum, 0)

    for name, want in sizes.items():
        exact = {s: want * len(ids) / pool_size for s, ids in by_stratum.items()}
        quota = {s: int(v) for s, v in exact.items()}
        shortfall = want - sum(quota.values())
        for stratum in sorted(exact, key=lambda s: (-(exact[s] - quota[s]), s))[:shortfall]:
            quota[stratum] += 1

        for stratum, take in quota.items():
            ids, start = by_stratum[stratum], cursor[stratum]
            chunk = ids[start : start + take]
            assigned[name].extend(chunk)
            cursor[stratum] = start + len(chunk)

        # A stratum can run dry when its quota exceeds what remains in it; top up
        # from anything still unassigned so the requested size is exact.
        if len(assigned[name]) < want:
            leftovers = [
                example_id for stratum, ids in sorted(by_stratum.items()) for example_id in ids[cursor[stratum] :]
            ]
            needed = want - len(assigned[name])
            assigned[name].extend(leftovers[:needed])
            taken = set(leftovers[:needed])
            for stratum, ids in by_stratum.items():
                cursor[stratum] += sum(1 for i in ids[cursor[stratum] :] if i in taken)

    return {
        name: SplitManifest(
            name=name,
            seed=seed,
            dataset=DATASET_NAME,
            dataset_split=DATASET_SPLIT,
            n=len(ids),
            example_ids=tuple(ids),
        )
        for name, ids in assigned.items()
    }
