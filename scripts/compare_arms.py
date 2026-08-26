#!/usr/bin/env python
"""Paired baseline-vs-taxonomy comparison from saved test evaluations. FREE.

Reads only artifacts already on disk, so it can be re-run, re-argued and
re-checked without spending a dollar. The per-instance vectors are what make
that possible -- both eval scripts write them for exactly this reason.

    uv run python scripts/compare_arms.py --benchmark ifbench
    uv run python scripts/compare_arms.py --benchmark hotpotqa

Why this exists separately
--------------------------
``eval_hotpotqa_test.py`` runs a paired Wilcoxon for exactly two candidates in
one invocation, and ``eval_ifbench_test.py`` runs none. Neither aggregates
**across seeds**, which is the actual experimental design: 3 seeds per arm, and
the question is whether the treatment helps *in general*, not whether it helped
on seed 1.

The statistics, and their limits
--------------------------------
Per seed, the arms are paired **by instance** -- both arms score the same test
instance, so the difference is within-pair and a Wilcoxon signed-rank over those
differences is the right test. Signed-rank rather than a t-test because the
metric is bounded and not remotely normal; signed-rank rather than McNemar
because the metric is continuous, and McNemar would first collapse it to
win/lose and discard most of the information (the SWE-Bench round used McNemar
because its metric really was binary).

Across seeds, this script deliberately does **not** report a pooled p-value over
3xN instance differences. Those are not independent -- each test instance appears
once per seed -- and pooling them would shrink the p-value by roughly sqrt(3)
for no real gain in evidence. What is reported instead is the per-seed effect
and the mean of per-seed effects, with the seed-level spread. If a defensible
single number is wanted later, the right tool is a mixed-effects model with seed
as a random effect, not a bigger Wilcoxon.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_per_instance(path: Path, label: str | None = None) -> dict[str, float] | None:
    """Read a test evaluation in either script's format.

    ``eval_ifbench_test.py`` writes ``{"per_instance": {id: score}}``;
    ``eval_hotpotqa_test.py`` writes ``{"instances": [{example_id, <label>: score}]}``
    with one column per evaluated candidate. Supporting both means the analysis
    does not care which script produced the numbers.
    """
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))

    if "per_instance" in payload:
        return {str(k): float(v) for k, v in payload["per_instance"].items()}

    rows = payload.get("instances")
    if not rows:
        return None
    if label is None:
        # Pick the single score column, if there is exactly one.
        keys = [k for k in rows[0] if k != "example_id"]
        if len(keys) != 1:
            raise SystemExit(f"{path} holds {len(keys)} candidates {keys}; pass --label to choose one")
        label = keys[0]
    return {str(r["example_id"]): float(r[label]) for r in rows}


def paired(baseline: dict[str, float], treatment: dict[str, float]) -> dict[str, object]:
    """Within-instance paired comparison. Only instances present in both count."""
    shared = sorted(set(baseline) & set(treatment))
    if not shared:
        return {"n": 0}
    diffs = [treatment[i] - baseline[i] for i in shared]
    nonzero = [d for d in diffs if d != 0.0]

    out: dict[str, object] = {
        "n": len(shared),
        "baseline_mean": statistics.mean(baseline[i] for i in shared),
        "treatment_mean": statistics.mean(treatment[i] for i in shared),
        "mean_difference": statistics.mean(diffs),
        "instances_differing": len(nonzero),
        "treatment_better": sum(1 for d in nonzero if d > 0),
        "baseline_better": sum(1 for d in nonzero if d < 0),
        "wilcoxon_p": None,
    }
    if nonzero:
        try:
            from scipy.stats import wilcoxon

            _stat, p = wilcoxon(diffs, zero_method="wilcox")
            out["wilcoxon_p"] = float(p)
        except ImportError:
            pass
    return out


def find(benchmark: str, arm: str, seed: int) -> Path:
    return REPO / "results" / "runs" / f"{benchmark}-{arm}-seed{seed}" / "test_eval.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, help="run-dir prefix, e.g. ifbench or hotpotqa-baseline's stem")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--label", default=None, help="score column, for multi-candidate hotpotqa outputs")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    per_seed: dict[int, dict[str, object]] = {}
    missing: list[str] = []

    for seed in args.seeds:
        b_path, t_path = find(args.benchmark, "baseline", seed), find(args.benchmark, "taxonomy", seed)
        baseline = load_per_instance(b_path, args.label)
        treatment = load_per_instance(t_path, args.label)
        if baseline is None:
            missing.append(str(b_path.relative_to(REPO)))
        if treatment is None:
            missing.append(str(t_path.relative_to(REPO)))
        if baseline and treatment:
            per_seed[seed] = paired(baseline, treatment)

    # The base-on-test reference, if it has been computed. Without it the arms
    # can be compared to each other but neither can be compared to "no
    # optimisation at all" -- and a val gain with a flat test score is exactly
    # what the SWE-Bench round produced.
    base_ref = load_per_instance(
        REPO / "results" / "runs" / f"{args.benchmark}-baseline-seed{args.seeds[0]}" / "test_eval_cand0.json"
    )

    if missing:
        print("missing test evaluations (run the eval script first):")
        for m in missing:
            print(f"  {m}")
        print()
    if not per_seed:
        print("nothing to compare yet.")
        return 1

    print(f"{'seed':>4} {'baseline':>9} {'taxonomy':>9} {'diff':>8} {'better':>7} {'worse':>6} {'p':>8}")
    for seed, r in sorted(per_seed.items()):
        p = r["wilcoxon_p"]
        print(
            f"{seed:>4} {r['baseline_mean']:>9.4f} {r['treatment_mean']:>9.4f} "
            f"{r['mean_difference']:>+8.4f} {r['treatment_better']:>7} {r['baseline_better']:>6} "
            f"{(f'{p:.4f}' if p is not None else 'n/a'):>8}"
        )

    diffs = [float(r["mean_difference"]) for r in per_seed.values()]
    print()
    print(f"  mean of per-seed differences : {statistics.mean(diffs):+.4f}")
    if len(diffs) > 1:
        print(f"  spread across seeds          : {max(diffs) - min(diffs):.4f}")
        print(f"  seeds favouring taxonomy     : {sum(1 for d in diffs if d > 0)}/{len(diffs)}")
    if base_ref is not None:
        print(f"  base (unoptimised) on test   : {statistics.mean(base_ref.values()):.4f}")
    else:
        print("  base (unoptimised) on test   : NOT COMPUTED -- run the eval script with --candidate-index 0")
    print()
    print("  No pooled p-value across seeds: the same test instance appears once per")
    print("  seed, so those differences are not independent. Per-seed effects and their")
    print("  spread are the honest summary; a mixed-effects model is the tool if a")
    print("  single number is needed.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "benchmark": args.benchmark,
                    "per_seed": {str(k): v for k, v in per_seed.items()},
                    "mean_of_per_seed_differences": statistics.mean(diffs),
                    "base_on_test": statistics.mean(base_ref.values()) if base_ref else None,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
