"""Deterministic splits over HotpotQA, mirroring the published GEPA setup.

Sizes are the paper's: **150 train / 300 val / 300 test** (GEPA, appendix E.1 --
"We use 150 examples for training, 300 for validation, and 300 for testing").
Matching them keeps our numbers comparable to the published HotpotQA results
rather than merely adjacent to them.

Pool
----
HotpotQA's ``fullwiki`` **validation** split (7,405 labelled examples). The
official ``test`` split is unlabelled -- it is the leaderboard holdout -- so it
cannot be graded locally, and the ``train`` split's questions were used to build
the distractor sets. Drawing all three subsets from one labelled pool keeps them
identically distributed, which is what makes a val-selected candidate's test
score interpretable.

Stratification
--------------
Proportional by ``(level, type)`` -- HotpotQA's own difficulty label
(easy/medium/hard) crossed with its reasoning type (bridge/comparison). Those
two interact: comparison questions are structurally easier to retrieve for
(both entities are named in the question) while bridge questions require the
second hop to be *derived* from the first. A split that is unbalanced on that
axis would make val and test measure different things, which is exactly the
failure the SWE-Bench round ran into from the other direction.

Guarantees, each covered by a test: disjoint, correctly sized, stratified,
and byte-identical across runs at a fixed seed.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SEED = 20260812
DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "fullwiki"
DATASET_SPLIT = "validation"

#: The paper's sizes. Order matters: subsets are carved in this order.
DEFAULT_SIZES: dict[str, int] = {"train": 150, "val": 300, "test": 300}


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """A committed stage artifact (D005): reproducible from seed + revision."""

    name: str
    seed: int
    dataset: str
    dataset_config: str
    dataset_split: str
    n: int
    example_ids: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "seed": self.seed,
            "dataset": self.dataset,
            "dataset_config": self.dataset_config,
            "dataset_split": self.dataset_split,
            "n": self.n,
            # Sorted so a manifest is stable under any re-ordering upstream, and
            # so positional indices into it are meaningful: gepa keys val
            # subscores and the Pareto frontier by POSITION, not by id (F014).
            "example_ids": sorted(self.example_ids),
        }

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")
        return out


def stratum_of(record: Mapping[str, object]) -> tuple[str, str]:
    """The stratification key: (difficulty level, reasoning type)."""
    return (str(record.get("level") or "unknown"), str(record.get("type") or "unknown"))


def build_splits(
    records: Sequence[Mapping[str, object]],
    *,
    seed: int = DEFAULT_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, SplitManifest]:
    """Partition ``records`` into disjoint, stratified subsets.

    Allocation is largest-remainder within each stratum, which keeps the subset
    proportions right without the drift a naive round-robin accumulates. Any
    shortfall from rounding is filled from the unallocated remainder, so the
    requested sizes are always met exactly.
    """
    sizes = dict(sizes or DEFAULT_SIZES)
    total_needed = sum(sizes.values())
    if len(records) < total_needed:
        raise ValueError(f"pool has {len(records)} records, need {total_needed}")

    by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        by_stratum[stratum_of(record)].append(str(record["id"]))

    rng = random.Random(seed)
    for ids in by_stratum.values():
        ids.sort()  # deterministic starting order before shuffling
        rng.shuffle(ids)

    pool_size = len(records)
    assigned: dict[str, list[str]] = {name: [] for name in sizes}
    cursor: dict[tuple[str, str], int] = dict.fromkeys(by_stratum, 0)

    for name, want in sizes.items():
        # Proportional quota per stratum, largest-remainder for the leftovers.
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

        # A stratum can run dry when its quota exceeds what is left in it; top
        # up from anything still unassigned so the requested size is exact.
        if len(assigned[name]) < want:
            leftovers = [
                example_id for stratum, ids in sorted(by_stratum.items()) for example_id in ids[cursor[stratum] :]
            ]
            needed = want - len(assigned[name])
            assigned[name].extend(leftovers[:needed])
            taken = set(leftovers[:needed])
            for stratum, ids in by_stratum.items():
                consumed = sum(1 for i in ids[cursor[stratum] :] if i in taken)
                cursor[stratum] += consumed

    return {
        name: SplitManifest(
            name=name,
            seed=seed,
            dataset=DATASET_NAME,
            dataset_config=DATASET_CONFIG,
            dataset_split=DATASET_SPLIT,
            n=len(ids),
            example_ids=tuple(ids),
        )
        for name, ids in assigned.items()
    }
