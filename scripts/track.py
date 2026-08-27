#!/usr/bin/env python
"""One line per live run: score, accepts, corrected spend, budget headroom.

    uv run python scripts/track.py           # table
    uv run python scripts/track.py --oneline # for a monitor heartbeat

Spend, corrected
----------------
``reflection_spend.jsonl`` alone is NOT the cost of a run. On a finished HoVer
seed it was $1.26 of $49.60 -- 2.5%. Solver spend dominates and reaches disk
only when the run ends.

Runs started after ``CostMeter`` gained ``spend_log`` write a live snapshot per
stream and are read exactly. Older runs are reconstructed from the rollout count
times a rate measured on a FINISHED run of the same arm -- not the base-val
rate, which understates by ~1.6x because prompts grow as candidates evolve.

``optimize_anything`` runs have no solver at all: the candidate is executed
locally, so the LM spend IS the total.

Stagnation
----------
``moved`` is how many iterations since the best-so-far last improved. It is the
number worth acting on: HoVer seed 1 reached its peak at accepted candidate 5 of
22 and spent ~$25 more finding nothing better. A large ``moved`` with budget
remaining means the run is paying for search that has stopped paying back.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs"

#: $/rollout measured from FINISHED runs of the same arm.
MEASURED_RATE = {"hover": 0.00674, "hotpotqa": 0.00556, "ifbench": 0.00782}
#: Evaluation is local compute; LM spend is the whole cost.
NO_SOLVER = ("circlepack", "cloudcast")
#: Served from the shared base-val cache: no LM call, no cost.
REPLAYED = 300
#: Relative size below which a score change is noise, not progress. 1e-6 keeps
#: real HoVer steps (smallest observed 0.0167 absolute, ~3e-2 relative) while
#: rejecting circle packing's 1e-8 float wobble.
MEANINGFUL_REL = 1e-6


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def rollouts_of(run: Path) -> int:
    """Both progress-bar formats. Matching only one reads a stale early number
    from a file that contains both."""
    t = _text(run / "run_log_stderr.txt")
    return max(
        [int(m) for m in re.findall(r"\| (\d+)/\d+", t)]
        + [int(m) for m in re.findall(r"(\d+)rollouts \[", t)]
        + [0]
    )


#: Spend recorded when the runs were paused. A RESUMED run's ``spend.*.json``
#: describes only its newest segment, because ``CostMeter`` is process-local and
#: restarts at zero -- so the live file alone under-reports a resumed run
#: by everything it spent before the resume.
_SNAPSHOT = REPO / "results" / "PAUSE_SNAPSHOT.json"


def prior_spend(name: str) -> float:
    if not _SNAPSHOT.exists():
        return 0.0
    try:
        return float(json.loads(_SNAPSHOT.read_text(encoding="utf-8"))["runs"][name]["spent_usd"])
    except Exception:
        return 0.0


def reflection_log_total(run: Path) -> float:
    """Cumulative reflection spend. Append-only, so it spans every segment."""
    f = run / "reflection_spend.jsonl"
    if not f.exists():
        return 0.0
    try:
        return sum(json.loads(l)["cost_usd"] for l in _text(f).splitlines() if l.strip())
    except Exception:
        return 0.0


def was_resumed(run: Path) -> bool:
    """True when this run is a continuation of an earlier, already-paid segment.

    resume_all.sh writes ``<name>.resume.log`` at launch, so its presence marks
    the run as resumed regardless of how much the new segment has since spent.
    """
    return (run.parent / f"{run.name}.resume.log").exists() and prior_spend(run.name) > 0


def spend_of(run: Path) -> tuple[float, str]:
    """(usd, how it was obtained). Correct across pause/resume cycles."""
    snaps = {}
    for stream in ("solver", "reflection", "judge"):
        p = run / f"spend.{stream}.json"
        if p.exists():
            try:
                snaps[stream] = json.loads(p.read_text(encoding="utf-8"))["budgeted_usd"]
            except Exception:
                pass

    prior = prior_spend(run.name)
    if snaps:
        metered = sum(snaps.values())
        if any(k in run.name for k in NO_SOLVER):
            # No solver, so the cost is reflection + judge.
            #
            # The reflection LOG is append-only and survives resumes, while the
            # reflection METER lags (flushes every 25 calls) -- so take the
            # larger of those two for the reflection component. But the judge is
            # a SEPARATE stream and must be ADDED, never max()'d in: on
            # circlepack-taxonomy-seed1 the meters read reflection $5.90 + judge
            # $0.90 = $6.81 while the log alone read $9.69, so a plain
            # max() reported $9.69 and silently dropped the judge's $0.90 --
            # under-reporting a run that was already past its ceiling.
            judge = 0.0
            p = run / "spend.judge.json"
            if p.exists():
                try:
                    judge = json.loads(p.read_text(encoding="utf-8"))["budgeted_usd"]
                except Exception:
                    pass
            reflection = max(snaps.get("reflection", 0.0), reflection_log_total(run))
            return max(reflection + judge, prior), "metered+log"

        # Hover has solver spend, which has NO append-only log -- only
        # spend.solver.json, which restarts at zero on resume. So the segments
        # must be added, and whether to add is decided by whether this run was
        # RESUMED, not by comparing magnitudes.
        #
        # The previous rule was "metered < prior means the meter restarted". That
        # holds only while the new segment is smaller than everything before it.
        # The moment seed 3's segment outgrew its $12.48 of prior spend, its
        # reported total fell from $26.63 to $14.14 -- a run silently gaining
        # $12 of headroom it had already spent.
        if was_resumed(run):
            return prior + metered, "metered+prior"
        return metered, "metered"

    refl = max(reflection_log_total(run), prior)
    if any(k in run.name for k in NO_SOLVER):
        return refl, "exact"
    fam = next((k for k in MEASURED_RATE if run.name.startswith(k)), None)
    if fam is None:
        return refl, "reflection-only"
    return max(0, rollouts_of(run) - REPLAYED) * MEASURED_RATE[fam] + refl, "reconstructed"


#: Turns a raw objective into the number a reader can act on. GEPA's own
#: score is whatever the evaluator returns, which for CloudCast is
#: ``1/(1+cost)`` -- a value like 0.0089 that looks like a failure and is
#: actually a 39% cost reduction. Reported alongside, never instead of, the raw
#: score.
REFERENCE = {
    "circlepack": ("sum of radii", 2.6358, "AlphaEvolve"),
}


def interpretable(run: Path, best: float | None) -> str:
    """A human-readable reading of ``best``, or '' when the raw score is already
    interpretable."""
    log = _text(run / "run_log.txt")

    if "cloudcast" in run.name:
        # The objective is 1/(1+cost) per config, aggregated over the 5 configs,
        # so cost is recovered as 1/score - 1 and the reduction is measured
        # BASE-to-BEST on that.
        #
        # Do NOT compute this from the logged ``raw_cost`` values. Those are
        # PER-CONFIG, and taking min() against costs[0] compares the cheapest
        # single config to whichever config happened to log first -- which
        # differs per seed. That is how one run read -0.9% off a $219 baseline
        # while its identical sibling read -4.0% off $224, and how the pilot's
        # true -41.9% was reported as -38.9%.
        m = re.search(r"Base program full valset score: ([\d.]+)", log)
        best = [float(v) for v in re.findall(r"Best valset aggregate score so far: ([\d.]+)", log)]
        if not m or not best:
            return ""
        base_s, best_s = float(m.group(1)), max(best)
        if base_s <= 0 or best_s <= 0:
            return ""
        base_c, best_c = 1 / base_s - 1, 1 / best_s - 1
        return (f"cost {base_c:.0f} -> {best_c:.0f} = "
                f"{-(base_c - best_c) / base_c * 100:+.1f}% (published -40.2%)")

    for key, (label, ref, who) in REFERENCE.items():
        if run.name.startswith(key) and best is not None:
            pct = (best - ref) / ref * 100
            # AlphaEvolve's figure is a REFERENCE, not a ceiling. The benchmark
            # states no target -- "sum of all circle radii (higher is better!)"
            # -- and the true optimum for 26 circles is unknown, so exceeding it
            # is a normal outcome to aim for rather than an anomaly.
            mark = "  *** EXCEEDS ***" if best > ref else ""
            return f"{label} {best:.4f} vs {ref} ({who}) = {pct:+.2f}%{mark}"
    return ""


def score_of(run: Path) -> tuple[float | None, int, int]:
    """(best-so-far, accepts, iterations since it last improved)."""
    log = _text(run / "run_log.txt")
    accepts = log.count("Accepted candidate")
    series = [float(v) for v in re.findall(r"Best valset aggregate score so far: ([\d.]+)", log)]
    if not series:
        # optimize_anything logs differently; fall back to its own best line.
        series = [float(v) for v in re.findall(r"[Bb]est score[: ]+([\d.]+)", log)]
    if not series:
        return None, accepts, 0
    best = max(series)
    # "Improvement" must mean a MEANINGFUL one. Circle packing accepted three
    # candidates in a row that moved the objective in the 8th-14th decimal place
    # (2.6255144511111106 -> ...113): float noise from a reshuffled solver, not
    # progress. Counting those as improvements reported flat=1 on a run whose
    # score had not really moved for nine iterations -- the same class of
    # under-reporting as the entries-vs-iterations bug below.
    floor = best * MEANINGFUL_REL if best > 0 else MEANINGFUL_REL
    last_improved = max(i for i, v in enumerate(series) if v >= best - floor)

    # Flat-ness must be counted in ITERATIONS, not in entries of this series.
    # gepa writes "Best valset aggregate score so far" only on iterations that
    # ran a full evaluation, so a run that spends 25 iterations rejecting
    # proposals contributes almost nothing to `series` -- and the naive
    # len(series) - last_improved reports a handful of flat steps instead of 25.
    # That under-reporting is why a plateau watcher keyed to it never fired.
    iters = [int(m) for m in re.findall(r"^Iteration (\d+)", log, re.MULTILINE)]
    pairs = re.findall(r"^Iteration (\d+): Best valset aggregate score so far: ([\d.]+)", log, re.MULTILINE)
    improved_at = [int(i) for i, v in pairs if float(v) >= best - floor]
    if iters and improved_at:
        return best, accepts, max(iters) - min(improved_at)
    return best, accepts, len(series) - 1 - last_improved


def _is_finished(run: Path) -> bool:
    """A run is finished when its summary is the LAST thing written about it.

    gepa writes summary.json on the way out, after everything else, so once a run
    exits nothing in its directory changes again. Comparing the summary against
    the NEWEST of every artifact therefore settles both awkward cases with one
    rule:

    * A freshly RESUMED run has an old summary from the paused segment, but its
      resume log was just created -- newer than the summary, so: live. Keying
      this to run_log.txt alone hid resumed runs for minutes, because the log is
      not touched until the first candidate lands.
    * A run that has just FINISHED wrote its summary last, so nothing is newer:
      finished. An earlier attempt keyed this to "a resume log exists and is
      newer or equal", which pinned completed runs as live forever, since tee's
      final write lands in the same second as the summary.

    The grace window absorbs filesystem timestamp coarseness; without it a
    same-second write reads as newer and a finished run never settles.
    """
    summary = run / "summary.json"
    if not summary.exists():
        return False
    newest = 0.0
    for p in (run / "run_log.txt", run / "run_log_stderr.txt", run / "gepa_state.bin",
              run / "reflection_spend.jsonl", run.parent / f"{run.name}.resume.log"):
        if p.exists():
            newest = max(newest, p.stat().st_mtime)
    return newest <= summary.stat().st_mtime + 2.0


def rows(budgets: dict[str, float]) -> list[dict]:
    out = []
    for name, budget in budgets.items():
        run = RUNS / name
        if not run.exists():
            continue
        if _is_finished(run):
            continue
        best, accepts, stale = score_of(run)
        usd, basis = spend_of(run)
        out.append(
            {"run": name, "best": best, "accepts": accepts, "stale": stale,
             "usd": usd, "budget": budget, "left": budget - usd, "basis": basis,
             "reading": interpretable(run, best)}
        )
    return out


DEFAULT_BUDGETS = {
    "hover-baseline-seed1": 60.0, "hover-baseline-seed2": 60.0, "hover-baseline-seed3": 60.0,
    # Taxonomy arm gets MORE than the baseline so judge spend does not eat
    # search depth -- the HotpotQA/IFBench precedent. Stopping rule is
    # DEPTH (22 candidates), not this number; the budget is the backstop.
    **{f"hover-taxonomy-seed{s}": 60.0 for s in (1, 2, 3)},
    # The pilots. circlepack's $20 is an externally enforced ceiling, below the
    # $40 the run itself was started with -- headroom here is what is left before
    # the watcher stops it, not before gepa would.
    "circlepack-stock": 20.0,
    "cloudcast-stock": 30.0,
    # The 3-seed chains. Listed ahead of launch so a seed appears the moment its
    # directory exists rather than when someone remembers to add it.
    **{f"cloudcast-stock-seed{s}": 15.0 for s in (1, 2, 3)},
    # Taxonomy arm: SAME $15 as stock. Judge spend competes with
    # reflection for it, so this arm buys fewer candidates -- that trade is
    # what the comparison measures.
    **{f"cloudcast-taxonomy-seed{s}": 15.0 for s in (1, 2, 3)},
    # $10 is an EXTERNAL ceiling (stop_at_spend.py); their own stoppers still
    # sit at a cumulative $12. Headroom shown is to the ceiling that will
    # actually fire, not the one gepa knows about.
    **{f"circlepack-stock-seed{s}": 10.0 for s in (1, 2, 3)},
    # Taxonomy arm at the same $10 effective ceiling the stock seeds ran
    # under, so the arms compare at equal spend AND (circle packing being
    # cheap to judge) at comparable depth.
    **{f"circlepack-taxonomy-seed{s}": 10.0 for s in (1, 2, 3)},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oneline", action="store_true")
    args = ap.parse_args()

    data = rows(DEFAULT_BUDGETS)
    if not data:
        print("no runs in progress")
        return 0

    if args.oneline:
        parts = []
        for r in data:
            b = f"{r['best']:.4f}" if r["best"] is not None else "-"
            warn = "!" if r["left"] < 0.1 * r["budget"] else ""
            # Prefer the interpretable reading; the raw score is meaningless
            # at a glance for objectives like 1/(1+cost).
            shown = r["reading"].split(" (published")[0] if r["reading"] else f"best={b}"
            parts.append(f"{r['run'].replace('-baseline','').replace('-stock','')} {shown} acc={r['accepts']} "
                         f"flat={r['stale']} ${r['usd']:.0f}/{r['budget']:.0f}{warn}")
        print(" | ".join(parts))
        return 0

    print(f"{'run':<24}{'best':>9}{'acc':>5}{'flat':>6}{'spent':>9}{'left':>8}  basis")
    print("-" * 74)
    for r in data:
        b = f"{r['best']:.4f}" if r["best"] is not None else "-"
        print(f"{r['run']:<24}{b:>9}{r['accepts']:>5}{r['stale']:>6}{r['usd']:>9.2f}{r['left']:>8.2f}  {r['basis']}")
        if r["reading"]:
            print(f"{'':>24}  {r['reading']}")
        if r["left"] < 0.1 * r["budget"]:
            print(f"{'':>24}  ^ under 10% of budget remaining")
        if r["stale"] >= 8:
            print(f"{'':>24}  ^ best-so-far flat for {r['stale']} iterations -- paying for search that stopped paying back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
