#!/usr/bin/env python
"""Write the IFBench split manifests. FREE: dataset downloads only.

Follows GEPA's published setup: train/val from IF-RLVR Train, test is the
whole IFBench set, and the two constraint vocabularies are disjoint.

    uv run python demo/build_ifbench_splits.py

Deterministic given the seed. Refuses to overwrite differing manifests without
--force: val subscores are keyed positionally against these ids.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "demo" / "manifests"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite manifests that differ")
    args = parser.parse_args()

    from datasets import load_dataset

    from gepa_taxonomy.ifbench._vendor.ifbench import instructions_registry as ifbench_reg
    from gepa_taxonomy.ifbench._vendor.ifevalg import instructions_registry as ifevalg_reg
    from gepa_taxonomy.ifbench.splits import (
        DEFAULT_SEED,
        TEST_DATASET,
        TEST_SPLIT,
        TRAIN_DATASET,
        TRAIN_SPLIT,
        build_test,
        build_train_val,
        n_constraints_of,
    )

    seed = args.seed if args.seed is not None else DEFAULT_SEED

    pool = list(load_dataset(TRAIN_DATASET, split=TRAIN_SPLIT))
    test_pool = list(load_dataset(TEST_DATASET, split=TEST_SPLIT))
    print(f"train/val pool : {len(pool):,} rows  ({TRAIN_DATASET})")
    print(f"test pool      : {len(test_pool)} rows  ({TEST_DATASET})")

    counts = collections.Counter(n_constraints_of(r) for r in pool)
    multi = sum(v for k, v in counts.items() if k > 1)
    print(f"  train/val constraints per instance: {dict(sorted(counts.items()))}")
    print(f"  partial credit possible on {multi:,}/{len(pool):,} ({multi / len(pool):.0%})")

    # The disjointness that makes this split the benchmark, asserted not assumed.
    overlap = set(ifbench_reg.INSTRUCTION_DICT) & set(ifevalg_reg.INSTRUCTION_DICT)
    assert not overlap, f"constraint vocabularies overlap, so test is no longer OOD: {sorted(overlap)[:5]}"
    print(
        f"  vocabularies: {len(ifevalg_reg.INSTRUCTION_DICT)} train ids, "
        f"{len(ifbench_reg.INSTRUCTION_DICT)} test ids, {len(overlap)} shared"
    )

    manifests = build_train_val(pool, seed=seed)
    manifests["test"] = build_test(test_pool, seed=seed)

    train_ids, val_ids = set(manifests["train"].example_ids), set(manifests["val"].example_ids)
    assert not (train_ids & val_ids), "train and val overlap"

    args.out.mkdir(parents=True, exist_ok=True)
    by_id = {str(r["key"]): r for r in pool}
    for name in ("train", "val", "test"):
        manifest = manifests[name]
        path = args.out / f"{name}.json"
        payload = json.dumps(manifest.to_json(), indent=2) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != payload and not args.force:
            raise SystemExit(
                f"REFUSING TO OVERWRITE: {path} exists and differs.\n"
                "Seeds already run against the old manifest would become incomparable\n"
                "-- gepa keys val subscores POSITIONALLY against these ids.\n"
                "  overwrite deliberately:  --force"
            )
        path.write_text(payload, encoding="utf-8")

        if name == "test":
            per = collections.Counter(len(r["instruction_id_list"]) for r in test_pool)
            share = sum(v for k, v in per.items() if k > 1) / len(test_pool)
            print(f"  {name:<5} n={manifest.n:<4} multi-constraint {share:.0%}   (OOD vocabulary, never optimised on)")
        else:
            per = collections.Counter(n_constraints_of(by_id[i]) for i in manifest.example_ids)
            share = sum(v for k, v in per.items() if k > 1) / manifest.n
            print(f"  {name:<5} n={manifest.n:<4} multi-constraint {share:.0%}   {dict(sorted(per.items()))}")

    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
