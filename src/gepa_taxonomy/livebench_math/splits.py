"""Deterministic splits over LiveBench-Math (math_comp + olympiad).

Pool
----
``livebench/math``, ``test`` split, filtered to the two tasks this arm uses:
**math_comp 146 + olympiad 72 = 218**. ``AMPS_Hard`` (150) is excluded.

Retired questions are KEPT. LiveBench stamps ``livebench_removal_date`` on
questions it retires to keep its public leaderboard contamination-free, which
would leave only 82 of these 218. That matters for a leaderboard; it does not
for a paired A/B, where contamination raises both arms identically and the
comparison is internal. Dropping them would leave splits smaller than
Terminal-Bench's 89, which was rejected outright on exactly that ground.
The trade is recorded because it is real: it inflates the absolute base rate,
so our numbers are not comparable to a published LiveBench score.

Sizes
-----
**40 train / 90 val / 88 test.** val and test are near-equal because both are
load-bearing -- val drives Pareto selection, test is the headline -- and both
must stay well clear of the val=60 that proved noise-dominated on SWE-Bench.
train is small by design: with a minibatch of 5 and ~40 iterations, 40
instances is ~200 draws, and every instance is seen repeatedly regardless.

Stratification
--------------
Proportional by ``(task, subtask-family)``. The axis matters more here than on
HotpotQA because the three scorers are not interchangeable: olympiad is the only
one that returns partial credit, and AIME's exact-match is much harder than
AMC's five-way multiple choice. A split unbalanced on that axis would make val
and test measure different things -- and would make val's partial-credit share,
which is what keeps minibatch comparisons informative, a matter of luck.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SEED = 20260812
DATASET_NAME = "livebench/math"
DATASET_CONFIG = "default"
DATASET_SPLIT = "test"

#: 218 instances split 40/90/88. Order matters: subsets are carved in this order.
DEFAULT_SIZES: dict[str, int] = {"train": 40, "val": 90, "test": 88}


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """A committed stage artifact: reproducible from seed + revision."""

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
            # Sorted so a manifest is stable under any upstream re-ordering, and
            # so positional indices into it are meaningful: gepa keys val
            # subscores and the Pareto frontier by POSITION, not by id.
            "example_ids": sorted(self.example_ids),
        }

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")
        return out


def stratum_of(record: Mapping[str, object]) -> tuple[str, str]:
    """(task, scorer family). Groups the AMC variants, which share a scorer."""
    task = str(record.get("task") or "unknown")
    sub = str(record.get("subtask") or "unknown").lower()
    if sub.startswith("aime"):
        family = "aime"
    elif sub in {"imo", "usamo"}:
        family = "proof_reorder"
    else:
        family = "multiple_choice"
    return (task, family)


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

    by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        by_stratum[stratum_of(record)].append(str(record["question_id"]))

    rng = random.Random(seed)
    for ids in by_stratum.values():
        ids.sort()  # deterministic starting order before shuffling
        rng.shuffle(ids)

    pool_size = len(records)
    assigned: dict[str, list[str]] = {name: [] for name in sizes}
    cursor: dict[tuple[str, str], int] = dict.fromkeys(by_stratum, 0)

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

        # A stratum can run dry when its quota exceeds what is left in it; top up
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
            dataset_config=DATASET_CONFIG,
            dataset_split=DATASET_SPLIT,
            n=len(ids),
            example_ids=tuple(ids),
        )
        for name, ids in assigned.items()
    }
