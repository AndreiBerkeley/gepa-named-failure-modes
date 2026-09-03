"""Deterministic splits over IFBench, following GEPA's published setup.

Two pools, not one
------------------
GEPA does **not** split the IFBench test set. From the paper:

    We split the IF-RLVR Train into our train/val sets, and IFBench as our test
    set, in order to ensure that the optimizers do not access the new, unseen
    constraints being tested in IFBench. Our splits contain 150 training
    examples, 300 for validation, and 294 for testing.

So:

* **train (150) + val (300)** are sampled from ``allenai/IF_multi_constraints_upto5``
  (IF-RLVR Train, 95,373 rows, IFEval-style constraints).
* **test (300)** is all of ``allenai/IFBench_test``, whose 58 constraints are new
  and out-of-distribution.

The two constraint vocabularies are **disjoint** -- 54 ids on the train side, 58
on the test side, zero shared. That separation is the benchmark. An earlier
version of this file split the 300-instance test set 60/120/120, which trained
the optimizer on the very constraints it was then scored against; both arms
would have shared the leak, so the A/B would not have been *biased*, but it would
have measured constraint memorisation rather than the generalisation IFBench
exists to test -- and handed the baseline a shortcut a taxonomy cannot beat.

We use all **300** test instances where the paper reports 294; it does not say
which six were dropped, and inventing an exclusion rule is worse than a
documented six-instance difference.

Two consequences worth naming
-----------------------------
**Partial credit is much better than the test set suggests.** IF-RLVR carries 1-5
constraints per instance (25/25/24/19/7 %), so ~75% of train and val instances
can score strictly between 0 and 1. Selection happens on val, so the acceptance
gate sees real granularity -- the "85% binary" limitation applies only to
the final test measurement.

**val=300** keeps the number of iterations a fixed budget buys per validation
instance low enough that GEPA cannot overfit the validation split.

Stratification
--------------
Proportional by ``n_constraints`` -- the partial-credit axis -- on the train/val
pool. Test is taken whole, so it needs none.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SEED = 20260813

TRAIN_DATASET = "allenai/IF_multi_constraints_upto5"
TRAIN_SPLIT = "train"
TEST_DATASET = "allenai/IFBench_test"
TEST_SPLIT = "train"  # upstream names it "train"; the dataset IS the test set

#: The paper's sizes. test is taken whole (300; the paper reports 294).
TRAIN_SIZE = 150
VAL_SIZE = 300


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """A committed stage artifact: reproducible from seed + revision."""

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
            # gepa keys val subscores POSITIONALLY, so this order IS the
            # run's order. It only has to be deterministic; sorting numerically
            # where the ids allow keeps it legible too.
            "example_ids": sorted(self.example_ids, key=_sort_key),
        }

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")
        return out


def _sort_key(example_id: str) -> tuple[int, str]:
    return (int(example_id), "") if example_id.isdigit() else (10**9, example_id)


def n_constraints_of(record: Mapping[str, object]) -> int:
    """Constraint count for an IF-RLVR row -- the stratification axis."""
    import ast

    raw = record.get("ground_truth")
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return 0
    if isinstance(raw, list) and raw and isinstance(raw[0], Mapping):
        return len(raw[0].get("instruction_id") or [])
    return 0


def build_train_val(
    records: Sequence[Mapping[str, object]],
    *,
    seed: int = DEFAULT_SEED,
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
) -> dict[str, SplitManifest]:
    """Carve disjoint train/val subsets out of the IF-RLVR pool.

    Largest-remainder allocation within each constraint-count stratum, with any
    rounding shortfall filled from the unallocated remainder so the requested
    sizes are met exactly.
    """
    sizes = {"train": train_size, "val": val_size}
    if len(records) < sum(sizes.values()):
        raise ValueError(f"pool has {len(records)} records, need {sum(sizes.values())}")

    by_stratum: dict[int, list[str]] = defaultdict(list)
    for record in records:
        by_stratum[n_constraints_of(record)].append(str(record["key"]))

    rng = random.Random(seed)
    for ids in by_stratum.values():
        ids.sort(key=_sort_key)  # deterministic starting order before shuffling
        rng.shuffle(ids)

    pool_size = len(records)
    assigned: dict[str, list[str]] = {name: [] for name in sizes}
    cursor: dict[int, int] = dict.fromkeys(by_stratum, 0)

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
            dataset=TRAIN_DATASET,
            dataset_split=TRAIN_SPLIT,
            n=len(ids),
            example_ids=tuple(ids),
        )
        for name, ids in assigned.items()
    }


def build_test(records: Sequence[Mapping[str, object]], *, seed: int = DEFAULT_SEED) -> SplitManifest:
    """The test manifest: all of IFBench, untouched by any optimizer."""
    return SplitManifest(
        name="test",
        seed=seed,
        dataset=TEST_DATASET,
        dataset_split=TEST_SPLIT,
        n=len(records),
        example_ids=tuple(str(r["key"]) for r in records),
    )
