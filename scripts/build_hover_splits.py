#!/usr/bin/env python
"""Write the HoVer train/val/test split manifests. FREE: local JSON only.

    PYTHONUTF8=1 uv run python scripts/build_hover_splits.py

Manifests are a committed stage artifact (D005): reproducible from the seed plus
the dataset revision, and readable by any stage without re-deriving them. Writing
them is idempotent -- rerunning at the same seed produces byte-identical files,
so a rebuild cannot silently move the splits underneath a finished run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_POOL = REPO / "data" / "hover" / "hover_dev_release_v1.1.json"
DEFAULT_OUT = REPO / "manifests" / "hover"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite existing manifests")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(REPO / "src"))
    from gepa_taxonomy.hover.splits import DEFAULT_SEED, DEFAULT_SIZES, build_splits
    from gepa_taxonomy.hover.tasks import instance_from_record

    if not args.pool.exists():
        raise SystemExit(f"HoVer pool not found: {args.pool}")

    existing = [n for n in DEFAULT_SIZES if (args.out / f"{n}.json").exists()]
    if existing and not args.force:
        raise SystemExit(
            f"REFUSING: manifests already exist ({', '.join(existing)}).\n"
            "Rebuilding would move the splits underneath every finished run keyed to them.\n"
            "  --force  to overwrite deliberately"
        )

    records = json.loads(args.pool.read_text(encoding="utf-8"))
    print(f"pool     : {args.pool.name}  ({len(records)} claims)")
    print(f"hop mix  : {dict(sorted(Counter(r.get('num_hops') for r in records).items()))}")
    print(f"labels   : {dict(sorted(Counter(r.get('label') for r in records).items()))}")

    seed = args.seed if args.seed is not None else DEFAULT_SEED
    splits = build_splits(records, seed=seed)

    by_id = {str(r["uid"]): r for r in records}
    print(f"\nseed     : {seed}")
    for name, manifest in splits.items():
        hops = Counter(by_id[i].get("num_hops") for i in manifest.example_ids)
        path = manifest.write(args.out / f"{name}.json")
        share = {h: f"{c / manifest.n:.0%}" for h, c in sorted(hops.items())}
        print(f"  {name:<6} n={manifest.n:<4} hops={share}  -> {path.relative_to(REPO)}")

    # A silent gold-extraction failure here would score every rollout against
    # nothing and look like a broken program, so it is checked once, loudly.
    # Sample ACROSS strata, not the first 50. Manifest example_ids are in
    # stratum-allocation order, so a head slice sees a single hop count and the
    # check would pass while telling you nothing.
    val_ids = sorted(splits["val"].example_ids)
    sample = [instance_from_record(by_id[i]) for i in val_ids[:: max(1, len(val_ids) // 50)]]
    empty = [s.task.example_id for s in sample if not s.gold.titles]
    if empty:
        raise SystemExit(
            f"REFUSING: {len(empty)} of {len(sample)} sampled val instances have no gold titles, "
            f"e.g. {empty[:3]}. HoVer supporting_facts is a list of [title, index] pairs; "
            "HotpotQA's dict-of-parallel-lists extractor returns () on it SILENTLY."
        )
    # HoVer guarantees num_hops == number of distinct supporting articles --
    # verified across all 4,000 dev and 18,171 train records, no exceptions.
    # That makes hop count exactly the number of documents the all-or-nothing
    # metric demands, so it is the difficulty axis rather than a proxy for one.
    mismatched = [s.task.example_id for s in sample if len(s.gold.titles) != s.task.num_hops]
    if mismatched:
        raise SystemExit(
            f"REFUSING: num_hops disagrees with the gold title count for "
            f"{len(mismatched)} of {len(sample)} sampled instances, e.g. {mismatched[:3]}. "
            "Stratifying on num_hops assumes they match."
        )
    sizes = Counter(len(s.gold.titles) for s in sample)
    print(f"\ngold titles per claim ({len(sample)} val sampled across strata): {dict(sorted(sizes.items()))}")
    print("num_hops == gold title count for every sampled instance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
