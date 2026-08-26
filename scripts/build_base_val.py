#!/usr/bin/env python
"""Evaluate the BASE candidate on the val split, ONCE. **THIS SPENDS API TOKENS.**

Every optimization run -- all 3 seeds, both arms -- replays this result instead
of recomputing it, so all runs start from literally identical state rather than
a re-sampled approximation (D009). It is also a shared one-time pipeline cost,
excluded from every per-seed dollar budget (D013).

Cost: ~$7.14 (100 rollouts at the measured mean of $0.0714).
Wall clock: ~0.8 h at 4 workers with the val images pre-pulled.

    zsh -c 'source ~/.zshrc >/dev/null 2>&1; \
      caffeinate -dimsu uv run python scripts/build_base_val.py'

Interruptible: results are written through a durable rollout cache after every
rollout, so a re-run resumes instead of re-paying. Progress and spend are
printed as it goes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = REPO_ROOT / "manifests" / "swebench_verified"
OUT_DIR = REPO_ROOT / "results" / "seed_cache"
CACHE_DIR = REPO_ROOT / ".cache" / "repos"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFESTS / "val.json")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "base_val.json")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--cache-level", default="env")
    ap.add_argument("--profile-prefix", default=None)
    ap.add_argument("--batch-size", type=int, default=25, help="instances per harness invocation")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; free")
    args = ap.parse_args()

    from gepa_taxonomy.cost import (
        REFINER_BASE,
        SOLVER_BASE,
        CostMeter,
        price_call,
        with_profile,
    )
    from gepa_taxonomy.cost import REFINER_MODEL as _DEF_REF
    from gepa_taxonomy.cost import SOLVER_MODEL as _DEF_SOL

    solver_model = with_profile(SOLVER_BASE, args.profile_prefix) if args.profile_prefix else _DEF_SOL
    refiner_model = with_profile(REFINER_BASE, args.profile_prefix) if args.profile_prefix else _DEF_REF

    from gepa_taxonomy.program import SEED_CANDIDATE
    from gepa_taxonomy.splits import DATASET_NAME, DATASET_SPLIT, load_manifest

    ids = load_manifest(args.manifest)
    if args.limit:
        ids = ids[: args.limit]

    # Measured mean rollout, for the up-front estimate.
    est = len(ids) * (price_call(solver_model, 16_369, 444) + price_call(refiner_model, 16_592, 200))

    print("=" * 72)
    print("BASE-CANDIDATE VAL EVALUATION (one-time, shared by all runs)")
    print("=" * 72)
    print(f"  instances     {len(ids)}")
    print(f"  solver        {solver_model}")
    print(f"  refiner       {refiner_model}")
    print(f"  workers       {args.max_workers}   cache_level {args.cache_level}")
    print(f"  batch size    {args.batch_size} instances per harness call")
    print(f"  estimated     ${est:.2f}")
    print(f"  output        {args.out}")

    if args.out.exists():
        print(f"\n{args.out.name} already exists. Delete it to rebuild.")
        return 0

    if args.dry_run:
        print("\n--dry-run: nothing evaluated, nothing spent.")
        return 0

    from gepa_taxonomy.bedrock import BedrockLM, require_credentials

    require_credentials()

    from datasets import load_dataset

    from gepa_taxonomy.adapter import SweBenchAdapter
    from gepa_taxonomy.grading import LocalDockerGrader
    from gepa_taxonomy.program import SolverRefinerProgram
    from gepa_taxonomy.retrieval import BM25Retriever
    from gepa_taxonomy.rollout_cache import RolloutCache
    from gepa_taxonomy.tasks import split_row

    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    difficulty = dict(zip(ds["instance_id"], ds["difficulty"], strict=True))
    wanted = set(ids)
    instances = {r["instance_id"]: split_row(r) for r in ds if r["instance_id"] in wanted}
    missing = wanted - set(instances)
    if missing:
        print(f"missing instances: {sorted(missing)[:5]}", file=sys.stderr)
        return 2

    solver_meter, refiner_meter = CostMeter(), CostMeter()
    program = SolverRefinerProgram(
        retriever=BM25Retriever(cache_dir=CACHE_DIR),
        # Gives the refiner a real apply verdict; the checkout is already at
        # this task's base_commit because retrieval just placed it there.
        repo_dir_for=lambda t: CACHE_DIR / t.repo.replace("/", "__"),
        solver_lm=BedrockLM(model=solver_model),
        refiner_lm=BedrockLM(model=refiner_model),
        solver_meter=solver_meter,
        refiner_meter=refiner_meter,
        solver_model=solver_model,
        refiner_model=refiner_model,
    )
    work = REPO_ROOT / "results" / "base_val_work"
    cache = RolloutCache.open(work / "rollouts.jsonl")
    if len(cache):
        print(f"\nresuming: {len(cache)} rollouts already done (${cache.recovered_usd:.2f} not re-paid)")

    adapter = SweBenchAdapter(
        program=program,
        grader=LocalDockerGrader(
            work_dir=work,
            max_workers=args.max_workers,
            cache_level=args.cache_level,
            run_id_prefix="baseval",
        ),
        instances=instances,
        rollout_cache=cache,
        repo_cache_dir=CACHE_DIR,
        # This IS the shared base evaluation: book it out of the per-seed budget.
        phase="seed_val",
        trace_path=work / "traces.jsonl",
        # D025: this evaluation IS the taxonomy-generation source.
        adamast_path=REPO_ROOT / "results" / "traces" / "base_val.adamast.jsonl",
        difficulty=difficulty,
    )

    results: dict[str, dict] = {}
    try:
        for start in range(0, len(ids), args.batch_size):
            chunk = ids[start : start + args.batch_size]
            print(f"\n--- batch {start // args.batch_size + 1}: instances {start + 1}-{start + len(chunk)} ---")
            out = adapter.evaluate(chunk, SEED_CANDIDATE, capture_traces=True)
            for iid, score, output, traj in zip(
                chunk, out.scores, out.outputs, out.trajectories or [{}] * len(chunk), strict=True
            ):
                results[iid] = {"score": score, "output": output, "trace": traj}
            spent = solver_meter.excluded_usd + refiner_meter.excluded_usd
            done = len(results)
            print(
                f"    {done}/{len(ids)} done | resolved {sum(1 for r in results.values() if r['score'] > 0)}"
                f" | spent ${spent:.2f}"
            )
            adapter.flush_traces()
            adapter.flush_adamast()
    except KeyboardInterrupt:
        print("\ninterrupted -- completed rollouts are in the durable cache; re-run to resume.")
        adapter.flush_adamast()
        cache.close()
        return 130

    from gepa_taxonomy.seed_cache import SeedEvaluationCache

    SeedEvaluationCache.build(SEED_CANDIDATE, results).save(args.out)
    adapter.flush_traces()
    adapter.flush_adamast()
    cache.close()

    spent = solver_meter.excluded_usd + refiner_meter.excluded_usd
    resolved = sum(1 for r in results.values() if r["score"] > 0)
    print("\n" + "=" * 72)
    print(f"  base candidate resolved {resolved}/{len(results)} = {resolved / len(results):.1%}")
    print(f"  spent ${spent:.2f} (estimated ${est:.2f})")
    print(f"  wrote {args.out}")

    from gepa_taxonomy.adamast_trace import validate as _validate

    apath = REPO_ROOT / "results" / "traces" / "base_val.adamast.jsonl"
    if apath.exists():
        rep = _validate(apath)
        print(f"\n  AdaMAST traces: {apath}")
        print(f"    {rep}")
        failures = sum(1 for r in results.values() if r["score"] == 0)
        print(f"    failure traces available for taxonomy generation: {failures}")
        if not rep["ok"]:
            print("    WARNING: local validation failed -- fix before `adamast generate`.")
    print("\n  This is booked to the 'seed_val' phase and does NOT count against")
    print("  any per-seed dollar budget.")
    (args.out.parent / "base_val_summary.json").write_text(
        json.dumps(
            {
                "n": len(results),
                "resolved": resolved,
                "spent_usd": round(spent, 4),
                "estimated_usd": round(est, 4),
                "solver_model": solver_model,
                "refiner_model": refiner_model,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
