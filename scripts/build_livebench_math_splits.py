#!/usr/bin/env python
"""Write the LiveBench-Math split manifests. FREE: dataset download only.

    uv run python scripts/build_livebench_math_splits.py

Deterministic given the seed, so re-running overwrites with identical files.
Refuses to overwrite differing manifests without --force: every run's val
subscores are keyed positionally against these ids, so silently changing
a manifest between seeds would make the seeds incomparable.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "manifests" / "livebench_math"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite manifests that differ")
    args = parser.parse_args()

    from datasets import load_dataset

    from gepa_taxonomy.livebench_math.splits import (
        DATASET_NAME,
        DATASET_SPLIT,
        DEFAULT_SEED,
        DEFAULT_SIZES,
        build_splits,
        stratum_of,
    )
    from gepa_taxonomy.livebench_math.tasks import is_included

    seed = args.seed if args.seed is not None else DEFAULT_SEED
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    pool = [r for r in ds if is_included(r)]
    print(f"pool: {len(pool)} of {len(ds)} rows (math_comp + olympiad; AMPS_Hard excluded per D050)")
    print(f"  by stratum: {dict(collections.Counter(stratum_of(r) for r in pool))}")
    retired = sum(1 for r in pool if str(r.get("livebench_removal_date") or "").strip())
    print(f"  retired-but-kept: {retired} of {len(pool)} (see splits.py -- inflates the base rate)")

    manifests = build_splits(pool, seed=seed, sizes=DEFAULT_SIZES)

    ids_seen: set[str] = set()
    for name, manifest in manifests.items():
        overlap = ids_seen & set(manifest.example_ids)
        assert not overlap, f"{name} overlaps an earlier split: {sorted(overlap)[:3]}"
        ids_seen |= set(manifest.example_ids)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, manifest in manifests.items():
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

        by_stratum = collections.Counter(
            stratum_of(r) for r in pool if str(r["question_id"]) in set(manifest.example_ids)
        )
        print(f"  {name:<5} n={manifest.n:<4} {dict(by_stratum)}")

    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
