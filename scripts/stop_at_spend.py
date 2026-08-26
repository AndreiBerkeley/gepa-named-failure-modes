#!/usr/bin/env python
"""Stop runs once cumulative spend reaches a ceiling. FREE: watches files.

    uv run python scripts/stop_at_spend.py --prefix circlepack-stock --usd 10
    uv run python scripts/stop_at_spend.py --prefix circlepack-stock --usd 10 --loop

Why an EXTERNAL ceiling
-----------------------
A run's own ``MaxTotalCostStopper`` is fixed at launch. These seeds were resumed
with ``--budget`` set to *original minus already spent*, so their internal
stoppers fire at a cumulative $12. Lowering that to $10 without restarting them
-- and without losing the search they have already paid for -- has to come from
outside.

Spend is read through ``track.spend_of``, deliberately imported rather than
reimplemented. Spend accounting here is subtle enough to have been wrong three
separate times (a meter that resets on resume, an append-only log that does not,
and a magnitude comparison that silently flipped once a resumed segment outgrew
the original total). A second copy of that logic would drift from the first, and
the copy that drifts is the one holding the purse strings.

How it stops
------------
By writing ``<run_dir>/gepa.stop`` -- gepa's own ``FileStopper``. The run
finishes its current iteration and exits through the normal path, writing
``summary.json``. Nothing is killed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track import _is_finished, spend_of  # noqa: E402  single source of truth


def matching(prefix: str) -> list[Path]:
    out = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        if any(m in d.name for m in (".netfail_", "smoke-", ".bak", ".old")):
            continue
        if not (d / "run_log.txt").exists() or _is_finished(d):
            continue
        out.append(d)
    return out


def check(prefix: str, ceiling: float, announced: set) -> list[str]:
    events = []
    for run in matching(prefix):
        usd, basis = spend_of(run)
        stop = run / "gepa.stop"
        if usd >= ceiling:
            if not stop.exists():
                stop.touch()
                events.append(f"CEILING {run.name}: ${usd:.2f} >= ${ceiling:.2f} -- gepa.stop written ({basis})")
            continue
        # Announce once per whole dollar so the climb is visible without
        # emitting on every poll.
        band = int(usd)
        key = (run.name, band)
        if key not in announced and band >= int(ceiling) - 2:
            announced.add(key)
            events.append(f"{run.name}: ${usd:.2f} of ${ceiling:.2f}")
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--usd", type=float, required=True, help="cumulative dollar ceiling per run")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()

    if not args.loop:
        runs = matching(args.prefix)
        if not runs:
            print(f"no live runs matching {args.prefix!r}")
            return 0
        for run in runs:
            usd, basis = spend_of(run)
            flag = "   <-- AT/OVER CEILING" if usd >= args.usd else ""
            print(f"{run.name:<28}${usd:>7.2f} of ${args.usd:.2f}   {basis}{flag}")
        return 0

    announced: set = set()
    while True:
        try:
            for line in check(args.prefix, args.usd, announced):
                print(line, flush=True)
            if not matching(args.prefix):
                print(f"no live runs matching {args.prefix!r} -- spend watcher exiting", flush=True)
                return 0
        except Exception as exc:
            print(f"spend watcher error (continuing): {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
