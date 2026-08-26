#!/usr/bin/env python
"""Stop runs once they reach a target candidate count. FREE: watches files.

    uv run python scripts/stop_at_depth.py --prefix hover-baseline --candidates 22
    uv run python scripts/stop_at_depth.py --prefix hover-baseline --candidates 22 --loop

Why depth and not spend
-----------------------
Baseline seed 1 explored 22 candidates for $49.60. Seeds that stop on dollars
alone end at whatever depth their own luck bought -- and depth, not spend, is
what the taxonomy arms are later matched against (D059). A seed that stops at 16
candidates cannot anchor a 22-candidate comparison.

So the rule is a floor on depth, and the dollar budget becomes the backstop it
was always meant to be rather than the thing that decides when search ends.

How it stops
------------
By writing ``<run_dir>/gepa.stop``, which is gepa's own ``FileStopper`` -- wired
automatically at api.py:261. The run finishes its current iteration and exits
through the normal path, writing ``summary.json``. Nothing is killed and no
state is lost, which is the difference between this and a kill.

``candidates = accepted + 1``: gepa counts the seed candidate, the log does not.
Verified against finished seed 1 (21 accepts, 22 candidates).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs"


def candidates_of(run: Path) -> int:
    log = run / "run_log.txt"
    if not log.exists():
        return 0
    return log.read_text(encoding="utf-8", errors="replace").count("Accepted candidate") + 1


def _finished(run: Path) -> bool:
    """summary.json is the LAST thing gepa writes, so if nothing is newer than
    it the run has exited. One rule covers both awkward cases: a freshly resumed
    run has a resume log newer than its stale summary (live), while a just
    finished run has nothing newer (finished). Keying this to "a resume log
    exists and is newer or equal" instead pinned COMPLETED runs as live forever,
    because tee's final write lands in the same second as the summary -- which
    is how a finished CloudCast seed got reported as hung."""
    summary = run / "summary.json"
    if not summary.exists():
        return False
    newest = 0.0
    for p in (run / "run_log.txt", run / "run_log_stderr.txt", run / "gepa_state.bin",
              run / "reflection_spend.jsonl", run.parent / f"{run.name}.resume.log"):
        if p.exists():
            newest = max(newest, p.stat().st_mtime)
    return newest <= summary.stat().st_mtime + 2.0


def matching(prefix: str) -> list[Path]:
    """Live run dirs whose name starts with prefix and that have no summary yet."""
    out = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        if any(m in d.name for m in (".netfail_", "smoke-", ".bak")):
            continue
        log, summary = d / "run_log.txt", d / "summary.json"
        if not log.exists():
            continue
        if _finished(d):
            continue
        out.append(d)
    return out


def check(prefix: str, target: int, announced: set) -> list[str]:
    events = []
    for run in matching(prefix):
        n = candidates_of(run)
        stop = run / "gepa.stop"
        if n >= target and not stop.exists():
            stop.touch()
            events.append(f"DEPTH REACHED {run.name}: {n} candidates >= {target} -- gepa.stop written")
        elif n >= target:
            continue
        else:
            # Progress notice at each new candidate, so a stalled depth climb is
            # visible rather than only the final stop.
            key = (run.name, n)
            if n and key not in announced:
                announced.add(key)
                if n >= target - 3:
                    events.append(f"{run.name}: {n}/{target} candidates")
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True, help="run-directory prefix, e.g. hover-baseline")
    ap.add_argument("--candidates", type=int, required=True, help="stop at this many candidates (incl. base)")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()

    if not args.loop:
        runs = matching(args.prefix)
        if not runs:
            print(f"no live runs matching {args.prefix!r}")
            return 0
        for run in runs:
            n = candidates_of(run)
            print(f"{run.name:<28}{n:>3}/{args.candidates} candidates" + ("   <-- at target" if n >= args.candidates else ""))
        return 0

    announced: set = set()
    while True:
        try:
            for line in check(args.prefix, args.candidates, announced):
                print(line, flush=True)
            if not matching(args.prefix):
                print(f"no live runs matching {args.prefix!r} -- depth watcher exiting", flush=True)
                return 0
        except Exception as exc:
            print(f"depth watcher error (continuing): {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
