#!/usr/bin/env python
"""Print the exact commands to resume every paused run. FREE: reads disk only.

    uv run python scripts/resume.py                # show state + commands
    uv run python scripts/resume.py --clear-stops  # ALSO delete gepa.stop files

Survives a reboot: everything it needs is on disk, nothing depends on a live
process, a shell variable, or this session.

Three things make resume non-obvious, and all three have bitten this project:

1. **``gepa.stop`` must be deleted first.** It is how the pause was performed.
   Left in place, a resumed run reads it on its first iteration and exits
   immediately -- looking like a crash, having spent a little money. Run this
   with ``--clear-stops`` and it removes them.

2. **Resume is NOT cost-continuous.** ``CostMeter`` lives in the process,
   so a resumed run starts counting from zero. Passing the ORIGINAL budget would
   authorise the full amount a second time. The commands below therefore carry a
   REDUCED budget: original minus what was already spent.

3. **``--resume`` is mandatory.** The run scripts refuse to start when
   ``gepa_state.bin`` exists, because gepa would otherwise silently resume and
   inherit that state without anyone deciding to.

Spend is read from the per-stream ``spend.*.json`` snapshots where present, and
otherwise reconstructed from the rollout count at a rate measured on a finished
run of the same arm. ``results/PAUSE_SNAPSHOT.json`` holds the figures as of the
pause and is the tie-breaker if anything disagrees.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs"
SNAPSHOT = REPO / "results" / "PAUSE_SNAPSHOT.json"

#: Original per-seed budgets. The resume command gets original-minus-spent.
BUDGET = {
    "hover-baseline-seed2": 60.0, "hover-baseline-seed3": 60.0,
    "cloudcast-stock-seed1": 15.0, "cloudcast-stock-seed2": 15.0, "cloudcast-stock-seed3": 15.0,
    "circlepack-stock-seed1": 12.0, "circlepack-stock-seed2": 12.0, "circlepack-stock-seed3": 12.0,
}
MEASURED_RATE = {"hover": 0.00674}
REPLAYED = 300


def reflection_log_total(run: Path) -> float:
    """Cumulative reflection spend from the APPEND-ONLY log.

    ``MeteredReflectionLM`` appends one record per call and never rewrites the
    file, so this figure spans every pause/resume segment. ``spend.*.json`` does
    not: it is a snapshot of a process-local meter that restarts at zero, so a
    resumed run's file describes only its own segment. Trusting the snapshot
    alone silently forgets everything spent before the last resume.
    """
    f = run / "reflection_spend.jsonl"
    if not f.exists():
        return 0.0
    try:
        return sum(json.loads(l)["cost_usd"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip())
    except Exception:
        return 0.0


def spend_of(run: Path) -> tuple[float, str]:
    streams = {}
    for s in ("solver", "reflection", "judge"):
        p = run / f"spend.{s}.json"
        if p.exists():
            try:
                streams[s] = json.loads(p.read_text(encoding="utf-8"))["budgeted_usd"]
            except Exception:
                pass
    if streams:
        metered = sum(streams.values())
        # For the no-solver benchmarks the append-only log IS the total cost, and
        # it outlives meter resets. Take whichever is larger so a resumed run
        # cannot under-report and re-authorise money already spent.
        if any(k in run.name for k in ("cloudcast", "circlepack")):
            # Append-only log already spans every segment; adding the snapshot
            # would double-count.
            return max(metered, reflection_log_total(run)), "metered+log"
        # Solver spend has no append-only log, so a resumed run's meter covers
        # only its newest segment and the earlier ones must be added. Keyed to
        # "was this resumed", never to comparing the two magnitudes: that
        # comparison silently flips once the new segment outgrows the old total,
        # handing back budget that was already spent.
        if (run.parent / f"{run.name}.resume.log").exists() and snapshot_spend(run.name) > 0:
            return snapshot_spend(run.name) + metered, "metered+prior"
        return metered, "metered"

    refl = reflection_log_total(run)
    fam = next((k for k in MEASURED_RATE if run.name.startswith(k)), None)
    if fam is None:
        return refl, "exact (no solver)"
    t = (run / "run_log_stderr.txt")
    text = t.read_text(encoding="utf-8", errors="replace") if t.exists() else ""
    n = max([int(m) for m in re.findall(r"\| (\d+)/\d+", text)]
            + [int(m) for m in re.findall(r"(\d+)rollouts \[", text)] + [0])
    return max(0, n - REPLAYED) * MEASURED_RATE[fam] + refl, "reconstructed"


def snapshot_spend(name: str) -> float:
    """Spend recorded at the pause. Used as a FLOOR, never as the only source.

    A resumed-then-paused-again run writes fresh ``spend.*.json`` files that
    start from zero, so trusting the live files alone would forget every earlier
    segment and re-authorise money that was already spent."""
    if not SNAPSHOT.exists():
        return 0.0
    try:
        return float(json.loads(SNAPSHOT.read_text(encoding="utf-8"))["runs"][name]["spent_usd"])
    except Exception:
        return 0.0


def command_for(name: str, remaining: float) -> str:
    seed = name[-1]
    if name.startswith("hover"):
        return (f"PYTHONUTF8=1 uv run python scripts/run_hover_seed.py --seed {seed} "
                f"--budget {remaining:.2f} --workers 8 --minibatch-size 6 --resume")
    bench = "cloudcast" if name.startswith("cloudcast") else "circlepack"
    script = "run_cloudcast.py" if bench == "cloudcast" else "run_circle_packing.py"
    calls = 1000 if bench == "cloudcast" else 150
    return (f"PYTHONUTF8=1 ../gepa-v0.1.4/.venv/Scripts/python.exe scripts/{script} "
            f"--arm stock --budget {remaining:.2f} --seed {seed} "
            f"--max-metric-calls {calls} --out results/runs/{name} --resume")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clear-stops", action="store_true", help="delete gepa.stop so resumed runs do not exit at once")
    ap.add_argument(
        "--emit",
        choices=["hover", "cloudcast", "circlepack"],
        help="print ONE shell command per resumable seed of this benchmark, nothing else. "
        "Consumed by resume_all.sh so budgets are computed from disk at launch time "
        "rather than baked into a command that goes stale.",
    )
    args = ap.parse_args()

    if args.emit:
        prefix = {"hover": "hover-baseline-seed", "cloudcast": "cloudcast-stock-seed",
                  "circlepack": "circlepack-stock-seed"}[args.emit]
        for name, budget in BUDGET.items():
            if not name.startswith(prefix):
                continue
            run = RUNS / name
            if not run.is_dir() or not (run / "gepa_state.bin").exists():
                continue
            spent, _ = spend_of(run)
            spent = max(spent, snapshot_spend(name))
            remaining = max(0.0, budget - spent)
            if remaining < 0.5:
                continue  # nothing meaningful left to buy
            if (run / "gepa.stop").exists():
                (run / "gepa.stop").unlink()  # else the resumed run exits on iteration 1
            print(command_for(name, remaining))
        return 0

    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8")) if SNAPSHOT.exists() else {"runs": {}}
    print(f"pause snapshot: {snap.get('captured_at', 'MISSING')}\n")

    print(f"{'run':<26}{'cand':>5}{'best':>9}{'spent':>8}{'left':>8}  state  stop")
    print("-" * 72)
    rows = []
    for name, budget in BUDGET.items():
        run = RUNS / name
        if not run.is_dir():
            continue
        spent, basis = spend_of(run)
        rec = snap.get("runs", {}).get(name, {})
        # Never authorise less than the snapshot implies: take the LARGER spend.
        spent = max(spent, rec.get("spent_usd", 0.0))
        remaining = max(0.0, budget - spent)
        has_state = (run / "gepa_state.bin").exists()
        has_stop = (run / "gepa.stop").exists()
        best = rec.get("best_score")
        b = f"{best:.4f}" if isinstance(best, (int, float)) else "-"
        print(f"{name:<26}{rec.get('candidates', '?'):>5}{b:>9}{spent:>8.2f}{remaining:>8.2f}"
              f"  {'yes' if has_state else 'NONE':<6} {'yes' if has_stop else 'no'}")
        rows.append((name, remaining, has_state, has_stop))

    stops = [r for r in rows if r[3]]
    if args.clear_stops:
        for name, _, _, has_stop in rows:
            if has_stop:
                (RUNS / name / "gepa.stop").unlink()
        print(f"\ncleared {len(stops)} gepa.stop file(s) -- runs are now resumable")
    elif stops:
        print(f"\n{len(stops)} run(s) still carry gepa.stop. Resuming now would exit immediately.")
        print("Clear them first:  uv run python scripts/resume.py --clear-stops")

    missing = [r[0] for r in rows if not r[2]]
    if missing:
        print(f"\nNO gepa_state.bin, cannot resume (would restart from zero): {', '.join(missing)}")

    print("\n" + "=" * 72)
    print("RESUME COMMANDS -- run from the repo root, after `source ~/.bashrc`")
    print("Budgets are REDUCED by spend already incurred (the meter restarts).")
    print("=" * 72)
    for name, remaining, has_state, _ in rows:
        if not has_state:
            continue
        if remaining < 0.5:
            print(f"\n# {name}: only ${remaining:.2f} left -- treat as finished")
            continue
        print(f"\n# {name}  (${remaining:.2f} of its original budget remains)")
        print(command_for(name, remaining))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
