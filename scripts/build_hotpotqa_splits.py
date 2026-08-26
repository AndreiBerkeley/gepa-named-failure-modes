#!/usr/bin/env python
"""Write the committed HotpotQA split manifests.

Free and offline apart from the dataset download. Sizes are the published GEPA
ones (150 train / 300 val / 300 test); the manifests are committed stage
artifacts (D005), so a third party can reproduce our exact subsets from the
seed and the dataset revision alone.

    uv run python scripts/build_hotpotqa_splits.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from gepa_taxonomy.hotpotqa.splits import (
    DATASET_CONFIG,
    DATASET_NAME,
    DATASET_SPLIT,
    DEFAULT_SEED,
    DEFAULT_SIZES,
    build_splits,
    stratum_of,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "manifests" / "hotpotqa"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"loading {DATASET_NAME} [{DATASET_CONFIG}/{DATASET_SPLIT}] ...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    records = [
        {"id": r["id"], "level": r.get("level"), "type": r.get("type")}
        for r in ds.select_columns(["id", "level", "type"])
    ]
    print(f"  pool: {len(records):,} labelled examples")

    manifests = build_splits(records, seed=args.seed, sizes=DEFAULT_SIZES)

    # Disjointness is a guarantee, so assert it here too rather than trusting
    # the tests alone -- this is the artifact everything downstream reads.
    seen: set[str] = set()
    for manifest in manifests.values():
        overlap = seen & set(manifest.example_ids)
        if overlap:
            raise SystemExit(f"FATAL: {manifest.name} overlaps a previous split: {sorted(overlap)[:5]}")
        seen |= set(manifest.example_ids)

    by_id = {str(r["id"]): r for r in records}
    for name, manifest in manifests.items():
        path = manifest.write(args.out / f"{name}.json")
        strata = Counter(stratum_of(by_id[i]) for i in manifest.example_ids)
        print(f"\n{name}: n={manifest.n} -> {path.relative_to(REPO)}")
        for (level, qtype), count in sorted(strata.items()):
            print(f"    {level:8} {qtype:12} {count:4}  ({count / manifest.n:.1%})")

    print(f"\nseed={args.seed}  total={sum(m.n for m in manifests.values())}  disjoint=yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
