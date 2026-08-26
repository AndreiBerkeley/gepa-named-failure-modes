"""Deterministic, repo-stratified splits over the gradeable SWE-Bench pool.

Only SWE-Bench's ``test`` split (2,294 instances) has validated environment
images, so that is the pool we partition -- not the nominal 21,527 rows. See
``docs/findings/phase1-swebench-feasibility.md`` finding A.

Four disjoint subsets, sized by their cost profile rather than uniformly:

======  =====  ============================================================
subset      n  why this size
======  =====  ============================================================
val       100  Fully re-evaluated per promoted candidate -- the dominant
               cost. 100 (not 60) because per-candidate grading SE at 60 is
               ~+/-5pp at realistic resolve rates, the same order as the
               effects we care about, so selection would be noise-dominated.
gen       150  Trace harvest for taxonomy generation.
test      400  Final held-out comparison; 0.96 power at +8pp (paired).
train    1644  Remainder. Only sampled minibatches are ever run, so size is
               ~free and larger means more reflection diversity.
======  =====  ============================================================

Guarantees, each covered by a test:

* **Disjoint** -- no instance appears in two subsets.
* **Exhaustive** -- the four subsets partition the pool exactly.
* **Stratified** -- proportional by repo, with a floor of >=1 per repo in
  val/generation/test so no repo is invisible in an evaluated subset.
* **Deterministic** -- same seed and dataset revision reproduce byte-identical
  manifests.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SEED = 20260807
DATASET_NAME = "SWE-bench/SWE-bench_Verified"
DATASET_SPLIT = "test"

#: Verified's own difficulty labels, verified against the dataset (not assumed):
#:   "<15 min fix"      194
#:   "15 min - 1 hour"  261
#:   "1-4 hours"         42
#:   ">4 hours"           3
HARD_LABELS: frozenset[str] = frozenset({"1-4 hours", ">4 hours"})

# Order matters: subsets are carved in this order, so the floor guarantee is
# applied to the evaluated subsets before train absorbs the remainder.
DEFAULT_SIZES: dict[str, int] = {
    "val": 60,
    "test": 300,
    # "train" is implicit: whatever remains (~140).
}

#: How many of val's 60 come from the hard pool; the rest are drawn at random
#: from everything else.
#:
#: The hard pool is only 45 of 500 (9%), so this number trades directly against
#: hard representation in *test*. At 30 the test set held 10 hard instances
#: (3.3%) and zero ">4 hours" -- under-representing the band where a taxonomy is
#: most likely to help. At 15, test keeps 21 hard (7.0%), close to the pool's
#: 9%, while val is still 25% hard versus the 9% a random draw would give.
#: Resolved as O013 / D027.
VAL_HARD = 15

# Subsets that must contain at least one instance from every repo.
FLOOR_SUBSETS: frozenset[str] = frozenset({"test"})


@dataclass(frozen=True)
class SplitManifest:
    """A committed stage-boundary artifact."""

    name: str
    seed: int
    dataset: str
    dataset_split: str
    instance_ids: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "seed": self.seed,
            "dataset": self.dataset,
            "dataset_split": self.dataset_split,
            "n": len(self.instance_ids),
            # Sorted so the file is stable regardless of draw order.
            "instance_ids": sorted(self.instance_ids),
        }

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.json"
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        return path


def _largest_remainder(counts: Mapping[str, int], total_pool: int, target: int, *, floor: int) -> dict[str, int]:
    """Allocate ``target`` slots across repos proportionally to ``counts``.

    Uses largest-remainder (Hare quota) so the allocation sums to exactly
    ``target`` without drift, then applies ``floor`` per repo. Repos are
    processed in a deterministic order throughout.
    """
    repos = sorted(counts)
    exact = {r: counts[r] / total_pool * target for r in repos}
    alloc = {r: max(floor, int(exact[r])) for r in repos}

    # Fix up to hit `target` exactly. Never allocate more than the repo has.
    def _remainder(r: str) -> float:
        return exact[r] - int(exact[r])

    over = sum(alloc.values()) - target
    if over > 0:
        # Floors pushed us over: shave from the repos with the least claim,
        # but never below the floor.
        for r in sorted(repos, key=lambda r: (_remainder(r), -counts[r])):
            while over > 0 and alloc[r] > floor:
                alloc[r] -= 1
                over -= 1
            if over == 0:
                break
    elif over < 0:
        need = -over
        for r in sorted(repos, key=lambda r: (-_remainder(r), -counts[r])):
            if need == 0:
                break
            if alloc[r] < counts[r]:
                alloc[r] += 1
                need -= 1
        # If a single pass was not enough (heavily floored allocations), keep
        # topping up the largest repos until we reach the target.
        while need > 0:
            progressed = False
            for r in sorted(repos, key=lambda r: -counts[r]):
                if need == 0:
                    break
                if alloc[r] < counts[r]:
                    alloc[r] += 1
                    need -= 1
                    progressed = True
            if not progressed:  # pragma: no cover - pool smaller than target
                raise ValueError("cannot allocate: target exceeds available instances")

    return alloc


def build_splits(
    instance_ids_by_repo: Mapping[str, Iterable[str]],
    *,
    sizes: Mapping[str, int] = DEFAULT_SIZES,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[str]]:
    """Carve disjoint, repo-stratified subsets. Pure function -- easy to test."""
    pool: dict[str, list[str]] = {r: sorted(ids) for r, ids in instance_ids_by_repo.items()}
    counts = {r: len(ids) for r, ids in pool.items()}
    total = sum(counts.values())

    requested = sum(sizes.values())
    if requested > total:
        raise ValueError(f"requested {requested} instances but pool holds {total}")

    rng = random.Random(seed)
    # Shuffle once per repo, then draw prefixes. Drawing from a fixed shuffled
    # order (rather than re-sampling per subset) is what makes the subsets
    # disjoint by construction.
    for ids in pool.values():
        rng.shuffle(ids)

    cursors: dict[str, int] = defaultdict(int)
    result: dict[str, list[str]] = {}

    for subset in sizes:
        floor = 1 if subset in FLOOR_SUBSETS else 0
        remaining = {r: counts[r] - cursors[r] for r in pool}
        alloc = _largest_remainder(remaining, sum(remaining.values()), sizes[subset], floor=floor)
        picked: list[str] = []
        for repo in sorted(pool):
            take = alloc[repo]
            start = cursors[repo]
            picked.extend(pool[repo][start : start + take])
            cursors[repo] = start + take
        result[subset] = sorted(picked)

    # train takes everything left over.
    leftover: list[str] = []
    for repo in sorted(pool):
        leftover.extend(pool[repo][cursors[repo] :])
    result["train"] = sorted(leftover)

    return result


def load_manifest(path: Path) -> list[str]:
    """Read a committed manifest. This is the stage boundary for Phases 2-5."""
    data = json.loads(Path(path).read_text())
    return list(data["instance_ids"])


def build_verified_splits(
    instance_ids: list[str],
    difficulty: dict[str, str],
    repo: dict[str, str],
    *,
    sizes: Mapping[str, int] = DEFAULT_SIZES,
    val_hard: int = VAL_HARD,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[str]]:
    """Three-subset split over SWE-bench Verified, stratified on difficulty.

    val is drawn deliberately: ``val_hard`` instances from Verified's own hard
    labels (>1 hour) plus the remainder at random from everything else. val does
    double duty -- it selects candidates AND supplies the taxonomy-generation
    traces -- so it is weighted toward instances that actually fail.

    test is then drawn repo-stratified from what remains, and train takes the
    rest. train size matters little: only sampled minibatches are ever run.
    """
    rng = random.Random(seed)
    ids = sorted(instance_ids)

    hard = sorted(i for i in ids if difficulty.get(i) in HARD_LABELS)
    rest = sorted(i for i in ids if i not in set(hard))
    if val_hard > len(hard):
        raise ValueError(f"asked for {val_hard} hard instances but only {len(hard)} exist")

    rng.shuffle(hard)
    rng.shuffle(rest)

    n_val = sizes["val"]
    val = sorted(hard[:val_hard] + rest[: n_val - val_hard])

    remaining = sorted(set(ids) - set(val))
    by_repo: dict[str, list[str]] = defaultdict(list)
    for i in remaining:
        by_repo[repo[i]].append(i)

    counts = {r: len(v) for r, v in by_repo.items()}
    alloc = _largest_remainder(counts, sum(counts.values()), sizes["test"], floor=1)
    for v in by_repo.values():
        rng.shuffle(v)

    test: list[str] = []
    for r in sorted(by_repo):
        test.extend(by_repo[r][: alloc[r]])
    test = sorted(test)

    train = sorted(set(remaining) - set(test))
    return {"val": val, "test": test, "train": train}
