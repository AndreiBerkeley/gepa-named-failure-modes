#!/usr/bin/env python
"""Build the committed split manifests. Free to run -- no API calls.

    uv run python scripts/build_splits.py

Deterministic: same seed + dataset revision reproduce byte-identical files.
Re-running is safe and should produce no git diff.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gepa_taxonomy.splits import (
    DATASET_NAME,
    DATASET_SPLIT,
    DEFAULT_SEED,
    DEFAULT_SIZES,
    HARD_LABELS,
    VAL_HARD,
    SplitManifest,
    build_verified_splits,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "manifests" / "swebench_verified"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=MANIFEST_DIR)
    ap.add_argument("--val", type=int, default=DEFAULT_SIZES["val"])
    ap.add_argument("--test", type=int, default=DEFAULT_SIZES["test"])
    ap.add_argument("--val-hard", type=int, default=VAL_HARD)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    ids = list(ds["instance_id"])
    difficulty = dict(zip(ids, ds["difficulty"], strict=True))
    repo_of = dict(zip(ids, ds["repo"], strict=True))

    sizes = {"val": args.val, "test": args.test}
    splits = build_verified_splits(ids, difficulty, repo_of, sizes=sizes, val_hard=args.val_hard, seed=args.seed)

    print(f"pool: {len(ids)} instances ({DATASET_NAME}, split={DATASET_SPLIT}, seed={args.seed})")
    hard_pool = [i for i in ids if difficulty[i] in HARD_LABELS]
    print(f"hard pool (>1 hour): {len(hard_pool)}\n")

    for name, sids in splits.items():
        manifest = SplitManifest(
            name=name,
            seed=args.seed,
            dataset=DATASET_NAME,
            dataset_split=DATASET_SPLIT,
            instance_ids=tuple(sids),
        )
        path = manifest.write(args.out)
        nh = sum(1 for i in sids if difficulty[i] in HARD_LABELS)
        nr = len({repo_of[i] for i in sids})
        print(
            f"  {name:6s} n={len(sids):4d}  hard={nh:3d} ({nh / len(sids):5.1%})  "
            f"repos={nr:2d}  -> {path.relative_to(REPO_ROOT)}"
        )

    total = sum(len(v) for v in splits.values())
    assert total == len(ids), f"partition not exhaustive: {total} != {len(ids)}"
    seen = set()
    for sids in splits.values():
        assert not (seen & set(sids)), "splits overlap"
        seen |= set(sids)
    print(f"\nverified: disjoint and exhaustive ({total})")

    print("\ndifficulty composition:")
    labels = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]
    print(f"  {'split':7}" + "".join(f"{lab:>18}" for lab in labels))
    for name, sids in splits.items():
        row = f"  {name:7}"
        for lab in labels:
            row += f"{sum(1 for i in sids if difficulty[i] == lab):18}"
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
