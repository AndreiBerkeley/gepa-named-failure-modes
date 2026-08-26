#!/usr/bin/env python
"""Depth-matched taxonomy-vs-baseline comparison. FREE: reads summaries only.

    uv run python scripts/depth_matched.py
    uv run python scripts/depth_matched.py --benchmark ifbench

Why depth-matching, and why per seed
------------------------------------
The taxonomy arm buys judge calls out of the same dollar budget as rollouts, so
at equal spend it explores fewer candidates. Comparing best-val at equal *spend*
therefore penalises it for the judge, and comparing at equal *wall-clock* is
meaningless. Depth -- candidates explored -- is the axis on which the two arms
do comparable work, so each taxonomy seed is stopped at the accept count its own
baseline seed reached (D059).

Pairing is by seed, not by mean. The three baseline seeds span 1.9pp on HotpotQA
and 3.9pp on IFBench purely from GEPA's search variance, which is the same order
as the effect being measured. Seed-paired differences cancel that; best-vs-best
across arms does not.

The truncation column is the one to read sceptically. If a taxonomy seed's best
candidate sits near the end of its run, truncating to baseline depth will drop
its score, and the untruncated figure was flattering it. ``delta_trunc`` shows
exactly how much was given back.

Not a significance test. Three seeds cannot carry one; ``compare_arms.py`` does
the paired Wilcoxon over per-instance TEST scores, which is the real claim.
These are val numbers and val is what selection optimises.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs"

#: Pre-resume spend the meter never saw, because it is process-local and
#: restarts at zero on resume (F064). Reconstructed, not measured -- read the
#: totals for these two as approximate.
PRE_RESUME_USD = {
    "hotpotqa-taxonomy-seed2": 31.20,
    "ifbench-taxonomy-seed1": 25.00,
    # Died without writing a summary during the crash-pause, so this half is
    # reconstructed (11,220 rollouts, 741 judge calls, $1.17 reflection); the
    # resumed run's own summary covers only the final accept.
    "hotpotqa-taxonomy-seed3": 86.93,
    # THREE segments, not two -- the only run in the set that was interrupted
    # twice:
    #   1. original    -> graceful stop at 11 candidates, metered exactly $31.69
    #                     (preserved in summary.presume.json)
    #   2. resume A    -> aborted on the credential failure (F067). Ran ~1h and
    #                     NEVER wrote a summary, so its spend is invisible to
    #                     every metered artifact: ~$10.87, reconstructed.
    #   3. resume B    -> completed; its summary.json covers only itself.
    # Omitting segment 2 understated this seed by 14%.
    "ifbench-taxonomy-seed3": 42.56,
}


def load(name: str) -> dict | None:
    path = RUNS / name / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def spend_of(summary: dict, name: str) -> tuple[float, bool]:
    """Realised spend and whether it is exact. Resumed runs under-report."""
    total = sum(v.get("budgeted_usd", 0) for v in summary.get("spend", {}).values() if isinstance(v, dict))
    extra = PRE_RESUME_USD.get(name, 0.0)
    return total + extra, extra == 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=["hotpotqa", "ifbench"], help="default: both")
    args = parser.parse_args()

    benchmarks = [args.benchmark] if args.benchmark else ["hotpotqa", "ifbench"]

    for bench in benchmarks:
        print(f"\n{'=' * 78}\n{bench}\n{'=' * 78}")
        header = f"{'seed':<5} {'base':>7} {'tax':>7} {'delta':>8} {'d_trunc':>8} {'cands b/t':>11} {'$ base':>8} {'$ tax':>9}"
        print(header)
        print("-" * len(header))

        deltas = []
        for seed in (1, 2, 3):
            b_name = f"{bench}-baseline-seed{seed}"
            t_name = f"{bench}-taxonomy-seed{seed}"
            b, t = load(b_name), load(t_name)
            if b is None or t is None:
                missing = b_name if b is None else t_name
                print(f"{seed:<5} {'--':>7} {'--':>7} {'':>8} {'':>8} {'':>11}   waiting on {missing}")
                continue

            n = b["candidates"]
            scores = t.get("val_aggregate_scores")

            # A summary.json only means the run EXITED, not that it reached
            # depth: a manual gepa.stop writes one too. Comparing a taxonomy run
            # that never got near baseline depth would read as a real delta when
            # it is just an unfinished run, so it is excluded from the mean and
            # labelled rather than quietly averaged in.
            if t["candidates"] < n:
                print(
                    f"{seed:<5} {b['best_val_score']:>7.4f} {t['best_val_score']:>7.4f} "
                    f"{'PARTIAL':>9} {'':>8} {n:>5}/{t['candidates']:<5}   "
                    f"stopped short of depth -- excluded"
                )
                continue

            if scores:
                trunc = scores[:n]
                tax_best = max(trunc)
                given_back = max(trunc) - max(scores)
            else:
                # Pre-dates the val_aggregate_scores field; fall back to the
                # recorded best and say so rather than silently comparing
                # different depths.
                tax_best = t["best_val_score"]
                given_back = float("nan")

            delta = tax_best - b["best_val_score"]
            deltas.append(delta)
            b_spend, b_exact = spend_of(b, b_name)
            t_spend, t_exact = spend_of(t, t_name)
            print(
                f"{seed:<5} {b['best_val_score']:>7.4f} {tax_best:>7.4f} {delta * 100:>+7.2f}pp "
                f"{given_back * 100:>+7.2f}pp {n:>5}/{t['candidates']:<5} "
                f"{b_spend:>7.2f}{'' if b_exact else '~'} {t_spend:>8.2f}{'' if t_exact else '~'}"
            )

        if deltas:
            mean = sum(deltas) / len(deltas)
            print("-" * len(header))
            print(f"{'mean':<5} {'':>7} {'':>7} {mean * 100:>+7.2f}pp   over {len(deltas)} seed pair(s)")
            if len(deltas) < 3:
                print("      (partial -- not the headline until all three land)")

    print("\ndelta   = taxonomy - baseline, both at baseline depth")
    print("d_trunc = how much truncating to baseline depth cost the taxonomy arm")
    print("~       = includes a reconstructed pre-resume component (F064)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
