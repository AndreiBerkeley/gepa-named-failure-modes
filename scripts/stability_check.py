#!/usr/bin/env python
"""Repeated-instance emulation stability check. FREE -- no LM calls.

Runs the same gold-patch evaluations several times and checks the verdicts are
identical every round. Gold patches must always resolve, so any round that
disagrees is flakiness introduced by the environment -- which on this arm64 Mac
means amd64 emulation perturbing timing-sensitive tests.

Why this matters: a flaky verdict is not merely noise in the final number. It
corrupts the reward signal GEPA optimizes against, so a candidate can be
promoted or rejected for reasons unrelated to its patch.

    uv run python scripts/stability_check.py --instances 3 --repeats 3

Writes results/stability/stability.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "stability"


def run_round(ids: list[str], run_id: str, workers: int, timeout: int) -> dict[str, bool]:
    """One harness round. Returns {instance_id: resolved}."""
    from datasets import load_dataset

    ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
    gold = dict(zip(ds["instance_id"], ds["patch"], strict=True))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = OUT_DIR / f"{run_id}-predictions.json"
    preds_path.write_text(
        json.dumps(
            [{"instance_id": i, "model_name_or_path": "gold", "model_patch": gold[i]} for i in ids],
            indent=2,
        )
    )

    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        "SWE-bench/SWE-bench_Verified",
        "--split",
        "test",
        "--predictions_path",
        str(preds_path),
        "--run_id",
        run_id,
        "--max_workers",
        str(workers),
        "--cache_level",
        "env",
        "--timeout",
        str(timeout),
        "--instance_ids",
        *ids,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True)

    report = REPO_ROOT / f"gold.{run_id}.json"
    if not report.exists():
        return dict.fromkeys(ids, False)
    data = json.loads(report.read_text())
    resolved = set(data.get("resolved_ids", []))
    return {i: (i in resolved) for i in ids}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--ids", nargs="*", default=None, help="explicit instance ids (overrides --instances)")
    ap.add_argument(
        "--spread",
        action="store_true",
        help="pick instances from DISTINCT repos -- a single-repo sample is weak evidence",
    )
    ap.add_argument("--run-prefix", default="stab")
    args = ap.parse_args()

    from gepa_taxonomy.splits import load_manifest

    val_ids = load_manifest(REPO_ROOT / "manifests" / "swebench_verified" / "val.json")
    if args.ids:
        ids = list(args.ids)
    elif args.spread:
        from datasets import load_dataset

        ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
        repo_of = dict(zip(ds["instance_id"], ds["repo"], strict=True))
        ids, seen = [], set()
        for i in val_ids:
            r = repo_of[i]
            if r in seen:
                continue
            seen.add(r)
            ids.append(i)
            if len(ids) >= args.instances:
                break
    else:
        ids = val_ids[: args.instances]
    print(f"stability check: {len(ids)} instances x {args.repeats} repeats (gold patches, $0)")
    for i in ids:
        print(f"  {i}")
    print()

    rounds: list[dict[str, bool]] = []
    for r in range(args.repeats):
        t0 = time.time()
        verdicts = run_round(ids, f"{args.run_prefix}-r{r}", args.max_workers, args.timeout)
        rounds.append(verdicts)
        dt = time.time() - t0
        got = sum(verdicts.values())
        print(f"  round {r + 1}/{args.repeats}: {got}/{len(ids)} resolved  ({dt / 60:.1f} min)")

    print("\nper-instance verdicts across rounds:")
    unstable: list[str] = []
    for i in ids:
        seq = [rounds[r][i] for r in range(args.repeats)]
        stable = len(set(seq)) == 1
        flag = "" if stable else "   <-- UNSTABLE"
        if not stable:
            unstable.append(i)
        print(f"  {i:42} {['PASS' if v else 'FAIL' for v in seq]}{flag}")

    all_resolved = all(all(r.values()) for r in rounds)
    result = {
        "instances": ids,
        "repeats": args.repeats,
        "rounds": rounds,
        "unstable_instances": unstable,
        "all_rounds_fully_resolved": all_resolved,
    }
    (OUT_DIR / "stability.json").write_text(json.dumps(result, indent=2) + "\n")

    print()
    if unstable:
        print(f"UNSTABLE: {len(unstable)} instance(s) disagreed across rounds.")
        print("Emulated execution is perturbing verdicts. Local numbers cannot be")
        print("trusted as a reward signal; run evaluation on the x86_64 box.")
        return 1
    if not all_resolved:
        print("STABLE BUT WRONG: verdicts were consistent, yet some gold patch did")
        print("not resolve. That is a harness/environment problem, not flakiness.")
        return 1
    print(f"STABLE: all {len(ids)} instances resolved identically in all {args.repeats} rounds.")
    print("No emulation-induced flakiness detected at this sample size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
