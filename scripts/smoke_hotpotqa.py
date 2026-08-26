#!/usr/bin/env python
"""Paid pre-flight: score the seed candidate on a handful of val instances.

**This spends money -- about $0.06 at the default n=10.** It exists because the
alternative is discovering the same thing at $60.

Seed 1 was launched with every offline check green and returned a base val score
of 1.5%, because a gold-blindness audit was firing on legitimate retrieval. No
free test could catch it: the failure needed a real model producing real
summaries over real retrieved passages. Ten rollouts would have.

What to expect
--------------
Published GEPA baseline on HotpotQA is 38% with GPT-4.1-mini. We run Haiku 4.5
from the same seed prompts, so roughly **20-45%** is healthy. **Near zero means
stop** -- that is the signature of a systemic fault, not a weak prompt.

    uv run python scripts/smoke_hotpotqa.py --n 10
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "results" / "smoke" / "hotpotqa_smoke.json",
        help="where the full result is written. A pre-flight whose only output is "
        "stdout is unreadable the moment the terminal scrolls or the session ends.",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    from gepa_taxonomy.bedrock import BedrockLM, require_credentials
    from gepa_taxonomy.cost import CostMeter
    from gepa_taxonomy.hotpotqa.adapter import HotpotQAAdapter, instances_by_id
    from gepa_taxonomy.hotpotqa.program import SEED_CANDIDATE, MultiHopProgram
    from gepa_taxonomy.hotpotqa.retrieval import WikiRetriever
    from gepa_taxonomy.hotpotqa.tasks import instance_from_record

    require_credentials()

    manifest = json.loads((REPO / "manifests" / "hotpotqa" / f"{args.split}.json").read_text())
    wanted = set(manifest["example_ids"][: args.n])
    ds = load_dataset(manifest["dataset"], manifest["dataset_config"], split=manifest["dataset_split"])
    instances = [instance_from_record(r) for r in ds if r["id"] in wanted]

    meter = CostMeter()
    program = MultiHopProgram(
        retriever=WikiRetriever(k=args.k).load(),
        lm=BedrockLM(model=args.solver_model, max_retries=args.max_retries),
        meter=meter,
        model=args.solver_model,
        k=args.k,
    )
    adapter = HotpotQAAdapter(program=program, instances=instances_by_id(instances))

    print(f"running the SEED candidate on {len(instances)} {args.split} instances ...", flush=True)
    batch = adapter.evaluate(instances, dict(SEED_CANDIDATE), capture_traces=True)

    print("\n=== per instance ===")
    for trace, score in zip(batch.trajectories, batch.scores, strict=True):
        gold = adapter.instances[trace["example_id"]].gold
        recall = trace["grading"]["retrieval_recall"]
        print(f"  {score:.2f}  recall {recall:.0%}  pred={trace['answer'][:44]!r:48} gold={gold.answer[:28]!r}")
        if trace.get("error"):
            print(f"        ERROR: {trace['error'][:150]}")

    mean = statistics.mean(batch.scores)
    recall = statistics.mean(t["grading"]["retrieval_recall"] for t in batch.trajectories)
    healthy = mean >= 0.05

    result = {
        "n": len(instances),
        "split": args.split,
        "solver_model": args.solver_model,
        "mean_answer_f1": mean,
        "mean_retrieval_recall": recall,
        "transport_errors": adapter.transport_errors,
        "program_errors": adapter.program_errors,
        "spend_usd": meter.total_usd,
        "healthy": healthy,
        "instances": [
            {
                "example_id": t["example_id"],
                "question": t["task"],
                "predicted": t["answer"],
                "gold": adapter.instances[t["example_id"]].gold.answer,
                "f1": s,
                "retrieval_recall": t["grading"]["retrieval_recall"],
                "missing_titles": t["grading"]["missing_titles"],
                "query_hop2": t.get("query_hop2", ""),
                "error": t.get("error"),
            }
            for t, s in zip(batch.trajectories, batch.scores, strict=True)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== summary ===")
    print(f"  mean answer F1   : {mean:.1%}")
    print(f"  retrieval recall : {recall:.1%}")
    print(f"  errors           : transport={adapter.transport_errors} program={adapter.program_errors}")
    print(f"  spend            : ${meter.total_usd:.4f}")
    print(f"  written          : {args.out}")

    if not healthy:
        print("\n  STOP. Near-zero mean F1 is a systemic fault, not a weak prompt.")
        return 1
    print("\n  Healthy. Safe to launch the full seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
