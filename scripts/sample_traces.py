#!/usr/bin/env python
"""Sample a trace bundle down to a size worth generating a taxonomy from. FREE.

    uv run python scripts/sample_traces.py \
        --in results/runs/cloudcast-stock/traces.jsonl \
        --out results/runs/cloudcast-stock/traces.sample.jsonl --n 300

Why sample at all
-----------------
AdaMAST runs four annotators over up to five agreement rounds, so generation
cost scales with the bundle. The other benchmarks fed it 300 traces and that
produced taxonomies passing its gate at kappa 1.00, so 300 is the size we know
works. An ``optimize_anything`` run emits one trace per *evaluation*, which
reaches many hundreds -- mostly repetition, since a handful of candidates are
each evaluated repeatedly.

Why NOT just take the base candidate's traces
---------------------------------------------
On HotpotQA / IFBench / HoVer the taxonomy came from the base candidate over 300
validation instances: one program, 300 different inputs. That shape does not
exist here. Circle packing is single-task -- the seed candidate produces exactly
ONE trace -- and CloudCast's dataset is 5 configurations. There is no
base-candidate bundle to draw 300 from, so the sample necessarily spans the
search.

That is a real difference in what the taxonomy describes, and it should be
stated rather than glossed: these codes characterise the failure modes the
search *encounters*, not the failure modes of the unoptimized program.

Stratification
--------------
Two axes, because either alone biases the result:

* **Position.** Early traces are near-seed behaviour, late ones are evolved. A
  sample from only the tail would describe a program that no longer resembles
  what the arms start from.
* **Failure.** Hard failures are rare (2% on CloudCast) but are the most
  informative traces a *failure* taxonomy can see. Proportional sampling would
  keep ~6 of them; they are kept in full instead.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: Below this, a score means "the candidate did not produce a usable result"
#: rather than "it produced a poor one" -- CloudCast uses a large negative
#: sentinel for syntax and topology failures.
FAILURE_SENTINEL = -1000.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", dest="dst", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300, help="target size; 300 matches the other benchmarks")
    ap.add_argument("--strata", type=int, default=10, help="position bands sampled evenly")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"not found: {args.src}")
    if args.dst.exists() and not args.force:
        raise SystemExit(
            f"REFUSING: {args.dst} exists. A taxonomy generated from a different sample is a "
            "different taxonomy, and every judged run is keyed to its fingerprint.\n  --force to overwrite"
        )

    rows = [json.loads(l) for l in args.src.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) <= args.n:
        print(f"{len(rows)} traces <= target {args.n}; copying unchanged")
        args.dst.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return 0

    def score(r: dict) -> float | None:
        s = r.get("metadata", {}).get("score")
        return s if isinstance(s, (int, float)) else None

    failures = [r for r in rows if (s := score(r)) is not None and s <= FAILURE_SENTINEL]
    rest = [r for r in rows if r not in failures]

    # Keep every hard failure: they are rare and are the traces a FAILURE
    # taxonomy most needs. Proportional sampling would discard most of them.
    keep = list(failures)
    remaining = max(0, args.n - len(keep))

    rng = random.Random(args.seed)
    # Even-sized bands, so the tail band is not a short remainder that cannot
    # fill its quota. Slicing every `len//strata` leaves a stub band, and a stub
    # under-delivers silently -- that is how a request for 300 returned 282.
    bands: list[list[dict]] = [[] for _ in range(args.strata)]
    for i, r in enumerate(rest):
        bands[min(i * args.strata // len(rest), args.strata - 1)].append(r)
    bands = [b for b in bands if b]

    pools = []
    for b in bands:
        pool = list(b)
        rng.shuffle(pool)
        pools.append(pool)

    # Round-robin draw. Any band that runs dry simply stops contributing and the
    # others make up the difference, so the target is met exactly whenever the
    # corpus is large enough.
    taken = 0
    cursor = [0] * len(pools)
    while taken < remaining and any(cursor[i] < len(pools[i]) for i in range(len(pools))):
        for i, pool in enumerate(pools):
            if taken >= remaining:
                break
            if cursor[i] < len(pool):
                keep.append(pool[cursor[i]])
                cursor[i] += 1
                taken += 1

    # Restore original order so position still means something to a reader.
    order = {id(r): i for i, r in enumerate(rows)}
    keep.sort(key=lambda r: order[id(r)])

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text("\n".join(json.dumps(r) for r in keep) + "\n", encoding="utf-8")

    empty = sum(1 for r in keep if not str(r.get("raw_trajectory", "")).strip())
    kept_scores = [s for r in keep if (s := score(r)) is not None]
    print(f"in  : {len(rows)} traces")
    print(f"out : {len(keep)} traces -> {args.dst}")
    print(f"  hard failures kept in full : {len(failures)}")
    print(f"  sampled evenly across      : {len(bands)} position bands (early -> late)")
    if kept_scores:
        print(f"  score range preserved      : {min(kept_scores):.6f} .. {max(kept_scores):.6f}")
    print(f"  empty trajectories         : {empty}   (AdaMAST refuses the bundle if any)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
