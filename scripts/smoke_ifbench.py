#!/usr/bin/env python
"""Live pre-flight for the IFBench arm. **Spends ~$0.10** (8 rollouts).

Everything the offline tests cannot check: that Bedrock is reachable, that the
real model's responses run clean through the vendored verifiers, and what a
rollout actually costs. The cost number is the point -- the AppWorld estimate was
87% low and the LiveBench-Math one 38% high, both from extrapolating the
wrong shape.

    PYTHONUTF8=1 uv run python scripts/smoke_ifbench.py

Reads TRAIN ids only, so it cannot contaminate the shared base-val state.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8, help="rollouts, biased to cover multi-constraint instances")
    parser.add_argument("--manifests", type=Path, default=REPO / "manifests" / "ifbench")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    args = parser.parse_args()

    from gepa_taxonomy.bedrock import BedrockLM, require_credentials
    from gepa_taxonomy.cost import CostMeter
    from gepa_taxonomy.ifbench.adapter import IFBenchAdapter, instances_by_id
    from gepa_taxonomy.ifbench.program import ENSURE, GENERATE, SEED_CANDIDATE, GenerateEnsureProgram
    from gepa_taxonomy.ifbench.tasks import constraint_family

    require_credentials()

    import importlib.util

    spec = importlib.util.spec_from_file_location("_ifb_runner", REPO / "scripts" / "run_ifbench_seed.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    train = runner.load_instances(args.manifests / "train.json")

    # Multi-constraint instances first: they are 76% of train and they are
    # the ones that can score strictly between 0 and 1, so a smoke that misses
    # them proves nothing about partial credit. Then spread across families.
    multi = [i for i in train if i.gold.n_constraints > 1]
    single, seen = [], set()
    for inst in train:
        if inst.gold.n_constraints > 1:
            continue
        family = constraint_family(inst.gold.instruction_ids[0]) if inst.gold.instruction_ids else ""
        if family not in seen:
            seen.add(family)
            single.append(inst)
    picked = (multi[:3] + single)[: args.n]
    print(f"smoke: {len(picked)} rollouts | {sum(1 for i in picked if i.gold.n_constraints > 1)} multi-constraint")

    meter = CostMeter()
    program = GenerateEnsureProgram(
        lm=BedrockLM(model=args.solver_model, max_retries=4),
        meter=meter,
        model=args.solver_model,
        max_tokens=args.max_tokens,
    )
    adapter = IFBenchAdapter(
        program=program,
        instances=instances_by_id(picked),
        reflection_gold_ids=frozenset(i.task.example_id for i in picked),
        max_workers=args.workers,
    )

    started = time.time()
    batch = adapter.evaluate(picked, dict(SEED_CANDIDATE), capture_traces=True)
    elapsed = time.time() - started

    print(f"\n{'id':<6} {'constraints':<38} {'strict':>7} {'loose':>7} {'$':>8}")
    for trace, score, inst in zip(batch.trajectories, batch.scores, picked, strict=True):
        g = trace["grading"]
        ids = ",".join(inst.gold.instruction_ids)
        print(
            f"{trace['example_id']:<6} {ids[:36]:<38} {score:>7.2f} {g['loose_score']:>7.2f} {trace['cost_usd']:>8.4f}"
        )

    per = meter.total_usd / max(1, len(picked))
    changed = sum(1 for t in batch.trajectories if (t.get("draft") or "").strip() != (t.get("response") or "").strip())
    loose = statistics.mean(t["grading"]["loose_score"] for t in batch.trajectories)
    strict = statistics.mean(batch.scores)

    # Built here too: a failure in feedback construction would otherwise surface
    # only after the base val was paid for.
    dataset = adapter.make_reflective_dataset(dict(SEED_CANDIDATE), batch, [GENERATE, ENSURE])

    print(f"\n  strict (selection): {strict:.3f}")
    print(f"  loose             : {loose:.3f}   gap {loose - strict:+.3f} = formatting, not compliance")
    print(f"  ensure changed    : {changed}/{len(picked)} drafts")
    print(
        f"  verifier errors   : {adapter.verifier_errors}   transport: {adapter.transport_errors}   program: {adapter.program_errors}"
    )
    print(f"  elapsed           : {elapsed:.1f}s  ({elapsed / len(picked):.1f}s/rollout)")
    print(f"  spend             : ${meter.total_usd:.4f}  (${per:.5f}/rollout)")
    print(f"  reflective set    : {[(k, len(v)) for k, v in dataset.items()]}")

    val_n = 300
    print("\n  --- projected, from THIS measurement ---")
    print(f"  base val ({val_n})   : ${per * val_n:.2f}")
    for mb in (3, 5):
        # 2*mb minibatch rollouts + reflection + accept_rate * full val.
        # 0.54 is the accept rate measured on HotpotQA seeds 1-2, not a guess.
        per_iter = 2 * mb * per + 0.05 + 0.54 * val_n * per
        for budget in (60,):
            print(
                f"  minibatch {mb:>2}      : ${per_iter:.2f}/iteration -> {budget / per_iter:.0f} iterations at ${budget}"
            )

    if adapter.transport_errors or adapter.program_errors or adapter.verifier_errors:
        print("\n  NOT CLEAN -- do not launch until this is understood.")
        return 1
    print("\n  clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
