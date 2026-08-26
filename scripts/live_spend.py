#!/usr/bin/env python
"""True spend for RUNNING runs. FREE: reads local artifacts only.

    uv run python scripts/live_spend.py

Why this is needed at all
-------------------------
``CostMeter`` accumulates in memory and is written to disk exactly once, into
``summary.json``, when a run ends. The only thing persisted *during* a run is
``reflection_spend.jsonl``, because ``MeteredReflectionLM`` appends per call.

So the naive live figure -- sum that file -- reports **reflection only**. On a
finished HoVer seed that was $1.26 of $49.60: **2.5% of the true cost**. Solver
spend, which is the other 97.5%, is invisible until the run is over.

The two families differ, and conflating them is the trap:

* **GEPAAdapter runs** (hotpotqa / ifbench / hover) call a solver model per
  rollout. Solver dominates; it must be reconstructed from the rollout count.
* **optimize_anything runs** (circle packing / cloudcast) have no solver -- the
  candidate is executed locally, as code or a simulator. The only money is the
  reflection LM, and the refiner shares the same metered object, so the
  reflection log genuinely IS the total there.

Reconstruction uses a per-benchmark $/rollout measured from a FINISHED run of
the same arm, not the base-val rate: prompts grow as candidates evolve, and on
HoVer the finished rate was 1.58x the seed-prompt rate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs"

#: $/rollout measured from finished runs of the same arm. Falls back to the
#: base-val rate only when no finished run exists, and says so.
MEASURED_RATE = {
    "hover": 0.00674,      # hover-baseline-seed1: $48.34 solver / 7170 rollouts
    "hotpotqa": 0.00556,   # hotpotqa-baseline-seed1
    "ifbench": 0.00782,    # ifbench-baseline-seed1
}
#: Runs whose evaluation is local compute, so the reflection log is the total.
NO_SOLVER = ("circlepack", "cloudcast")
#: Rollouts served from the shared base-val cache. No LM call, no cost.
REPLAYED = 300


def live_snapshots(run: Path) -> dict[str, float] | None:
    """Read the per-stream snapshots a running meter now writes.

    Present only for runs STARTED after CostMeter gained ``spend_log``. Older
    runs fall back to reconstruction below, which is why that path is kept.
    """
    found: dict[str, float] = {}
    for stream in ("solver", "reflection", "judge"):
        path = run / f"spend.{stream}.json"
        if path.exists():
            try:
                found[stream] = json.loads(path.read_text(encoding="utf-8"))["budgeted_usd"]
            except Exception:
                pass
    return found or None


def rollouts_of(run: Path) -> int:
    """Highest rollout count in the progress bar, across BOTH bar formats.

    Uncapped runs print ``7150rollouts [..]``; capped ones print ``7150/20000``.
    Matching only one silently reads a stale early number from a file that
    contains both, which is exactly how a spend reconstruction once came out
    negative.
    """
    path = run / "run_log_stderr.txt"
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    counts = [int(m) for m in re.findall(r"(\d+)rollouts \[", text)]
    counts += [int(m) for m in re.findall(r"\| (\d+)/\d+", text)]
    return max(counts + [0])


def reflection_of(run: Path) -> float:
    path = run / "reflection_spend.jsonl"
    if not path.exists():
        return 0.0
    return sum(json.loads(line)["cost_usd"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    live = sorted(d for d in RUNS.iterdir() if d.is_dir() and not (d / "summary.json").exists())
    # A run that was resumed has a summary from an earlier segment; include any
    # directory whose logs are still being written.
    resumed = [
        d for d in RUNS.iterdir()
        if d.is_dir() and (d / "summary.json").exists() and (d / "run_log.txt").exists()
        and (d / "run_log.txt").stat().st_mtime > (d / "summary.json").stat().st_mtime
    ]
    for d in resumed:
        if d not in live:
            live.append(d)

    if not live:
        print("no runs in progress")
        return 0

    print(f"{'run':<28}{'rollouts':>10}{'solver':>10}{'reflect':>9}{'TOTAL':>10}   basis")
    print("-" * 82)
    for run in sorted(live):
        snaps = live_snapshots(run)
        if snaps is not None:
            total = sum(snaps.values())
            print(
                f"{run.name:<28}{'-':>10}{snaps.get('solver', 0.0):>10.2f}"
                f"{snaps.get('reflection', 0.0):>9.2f}{total:>10.2f}   METERED live"
                + (f" (judge ${snaps['judge']:.2f})" if snaps.get("judge") else "")
            )
            continue
        refl = reflection_of(run)
        if any(k in run.name for k in NO_SOLVER):
            print(f"{run.name:<28}{'-':>10}{'-':>10}{refl:>9.2f}{refl:>10.2f}   exact (no solver: local compute)")
            continue
        family = next((k for k in MEASURED_RATE if run.name.startswith(k)), None)
        n = rollouts_of(run)
        if family is None:
            print(f"{run.name:<28}{n:>10}{'?':>10}{refl:>9.2f}{'?':>10}   NO RATE for this benchmark")
            continue
        rate = MEASURED_RATE[family]
        solver = max(0, n - REPLAYED) * rate
        print(
            f"{run.name:<28}{n:>10}{solver:>10.2f}{refl:>9.2f}{solver + refl:>10.2f}"
            f"   solver reconstructed @ ${rate:.5f}/rollout"
        )
    print("\nreflection is exact (written through per call); solver is reconstructed")
    print("because CostMeter only reaches disk when a run ends.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
