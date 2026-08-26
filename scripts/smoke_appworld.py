#!/usr/bin/env python
"""Paid pre-flight for AppWorld: run the seed agent on a handful of real tasks.

**Spends money — a few dollars at the default n=3.** It exists because the
alternative is discovering the same thing at $60.

HotpotQA passed every offline check and then returned a 1.5% base val on its
first real run, because a gold-blindness audit was firing on legitimate
retrieval. No free test could catch it: the failure needed a real model over a
real environment. Ten real rollouts would have. AppWorld has had **zero** real
LM rollouts, so this runs first.

Deliberately low concurrency: this is designed to be safe to run *alongside* the
HotpotQA seeds, which are already using 8 workers against the same Bedrock quota.

What to expect
--------------
ACE reports ReAct+GEPA at 46.4% on AppWorld with a smaller model; we run Haiku
4.5 from the published seed prompt, so a **non-zero** score with sensible step
counts is the bar. Near-zero everywhere, or every task exhausting its steps,
means stop.

    PYTHONUTF8=1 uv run python scripts/smoke_appworld.py --n 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=2, help="kept low: HotpotQA may be running")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--base-url", default="http://localhost:8123")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "smoke" / "appworld_smoke.json")
    args = parser.parse_args()

    import importlib.util

    from gepa_taxonomy.appworld.adapter import AppWorldAdapter, client_factory
    from gepa_taxonomy.appworld.program import ReActProgram
    from gepa_taxonomy.appworld.prompts import SEED_CANDIDATE
    from gepa_taxonomy.bedrock import BedrockLM, require_credentials
    from gepa_taxonomy.cost import CostMeter

    require_credentials()

    spec = importlib.util.spec_from_file_location("_runner", REPO / "scripts" / "run_appworld_seed.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.ensure_servers(args.port, args.workers)

    tasks = runner.load_task_ids(REPO / "manifests" / "appworld" / f"{args.split}.json")[: args.n]
    print(f"running the SEED agent on {len(tasks)} {args.split} tasks, {args.workers} workers")

    meter = CostMeter()
    adapter = AppWorldAdapter(
        program=ReActProgram(
            client=None,
            lm=BedrockLM(model=args.solver_model, max_retries=args.max_retries),
            meter=meter,
            model=args.solver_model,
            max_steps=args.max_steps,
        ),
        client_factory=client_factory(args.port, args.workers, prefix="smoke"),
        max_workers=args.workers,
    )

    started = time.time()
    batch = adapter.evaluate(tasks, dict(SEED_CANDIDATE), capture_traces=True)
    elapsed = time.time() - started

    print("\n=== per task ===")
    for trace, score in zip(batch.trajectories, batch.scores, strict=True):
        g = trace["grading"]
        flag = " STEPS-EXHAUSTED" if trace.get("exhausted_steps") else ""
        print(f"  {trace['task_id']:14} score {score:.2f}  tgc {int(g['success'])}  steps {trace['steps']:2}{flag}")
        if g["failures"]:
            print(f"       failed: {', '.join(map(str, g['failures']))[:110]}")
        if trace.get("error"):
            print(f"       ERROR: {trace['error'][:130]}")

    mean = statistics.mean(batch.scores)
    tgc = statistics.mean(1.0 if t["grading"]["success"] else 0.0 for t in batch.trajectories)
    summary = adapter.summary()

    # The first version of this check reported healthy=True on a run where two of
    # three tasks died with HTTP 500 and the mean score was 0.0, because it only
    # looked at transport errors and step exhaustion. A pre-flight that passes a
    # run like that is worse than no pre-flight: it launders a broken arm as a
    # verified one. Any rollout that errored is now disqualifying on its own.
    errored = sum(1 for t in batch.trajectories if t.get("error"))
    reasons: list[str] = []
    if errored:
        reasons.append(f"{errored}/{len(tasks)} rollouts errored")
    if adapter.transport_errors:
        reasons.append(f"{adapter.transport_errors} transport errors")
    if summary["step_exhaustions"] == len(tasks):
        reasons.append("every task exhausted its steps")
    if mean == 0.0:
        reasons.append("mean score is 0.0")
    healthy = not reasons

    result = {
        "n": len(tasks),
        "mean_score": mean,
        "task_goal_completion": tgc,
        "elapsed_minutes": round(elapsed / 60, 2),
        "spend_usd": meter.total_usd,
        "adapter": summary,
        "healthy": healthy,
        "unhealthy_reasons": reasons,
        "tasks": [
            {
                "task_id": t["task_id"],
                "score": s,
                "success": t["grading"]["success"],
                "steps": t["steps"],
                "exhausted_steps": t.get("exhausted_steps"),
                "failures": t["grading"]["failures"],
                "error": t.get("error"),
            }
            for t, s in zip(batch.trajectories, batch.scores, strict=True)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== summary ===")
    print(f"  mean score (selection) : {mean:.1%}")
    print(f"  TGC (reported)         : {tgc:.1%}")
    print(f"  mean steps             : {summary['mean_steps']}  exhausted {summary['step_exhaustions']}")
    print(f"  errors                 : transport={adapter.transport_errors} program={adapter.program_errors}")
    print(f"  spend                  : ${meter.total_usd:.3f}   elapsed {elapsed / 60:.1f} min")
    print(f"  written                : {args.out}")

    if not healthy:
        print("\n  STOP: " + "; ".join(reasons))
        return 1
    print("\n  Healthy. Safe to build the base val and launch seeds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
