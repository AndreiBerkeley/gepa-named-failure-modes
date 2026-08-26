#!/usr/bin/env python
"""Measure real HotpotQA prompt sizes and price a seed. Free: no model is called.

Prompts are constructed exactly as the program constructs them, with real
retrieval, and then priced with the same table the budget stopper uses. Only
the two summarize stages are measurable this way -- the later stages depend on
model output that does not exist yet -- so those are modelled from a stated
output-length assumption, which is printed rather than hidden.

    uv run python scripts/calibrate_hotpotqa.py --n 20
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from gepa_taxonomy.cost import price_call
from gepa_taxonomy.hotpotqa.program import (
    CREATE_QUERY_HOP2,
    CREATE_QUERY_HOP2_PROMPT,
    FINAL_ANSWER,
    FINAL_ANSWER_PROMPT,
    SEED_CANDIDATE,
    SUMMARIZE1,
    SUMMARIZE1_PROMPT,
    SUMMARIZE2,
    SUMMARIZE2_PROMPT,
)
from gepa_taxonomy.hotpotqa.retrieval import WikiRetriever, render_passages
from gepa_taxonomy.hotpotqa.tasks import instance_from_record

REPO = Path(__file__).resolve().parents[1]

#: Chars per token. Deliberately LOW so token counts err HIGH: under-estimating
#: spend would let the budget stopper fire late, which costs real money.
CHARS_PER_TOKEN = 3.5

#: Stated assumptions for the stages whose inputs do not exist before a run.
ASSUMED_SUMMARY_CHARS = 700
ASSUMED_QUERY_CHARS = 120
OUT_TOKENS = {SUMMARIZE1: 200, CREATE_QUERY_HOP2: 40, SUMMARIZE2: 200, FINAL_ANSWER: 25}

SOLVER = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
REFLECTION = "us.anthropic.claude-sonnet-4-6"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--minibatch", type=int, default=15)
    parser.add_argument("--accept-rate", type=float, default=0.30)
    args = parser.parse_args()

    manifest = json.loads((REPO / "manifests" / "hotpotqa" / "val.json").read_text())
    wanted = set(manifest["example_ids"][: args.n])

    from datasets import load_dataset

    ds = load_dataset(manifest["dataset"], manifest["dataset_config"], split=manifest["dataset_split"])
    instances = [instance_from_record(r) for r in ds if r["id"] in wanted]
    retriever = WikiRetriever(k=args.k).load()

    tokens: dict[str, list[int]] = {c: [] for c in OUT_TOKENS}
    for instance in instances:
        question = instance.task.question
        hop1 = retriever.retrieve(question, k=args.k)
        p1 = SUMMARIZE1_PROMPT.format(
            instruction=SEED_CANDIDATE[SUMMARIZE1], question=question, passages=render_passages(hop1)
        )
        tokens[SUMMARIZE1].append(len(p1))

        summary = "x" * ASSUMED_SUMMARY_CHARS
        p2 = CREATE_QUERY_HOP2_PROMPT.format(
            instruction=SEED_CANDIDATE[CREATE_QUERY_HOP2], question=question, summary_1=summary
        )
        tokens[CREATE_QUERY_HOP2].append(len(p2))

        hop2 = retriever.retrieve(question, k=args.k)  # stand-in for the generated query
        p3 = SUMMARIZE2_PROMPT.format(
            instruction=SEED_CANDIDATE[SUMMARIZE2],
            question=question,
            context=summary,
            passages=render_passages(hop2),
        )
        tokens[SUMMARIZE2].append(len(p3))

        p4 = FINAL_ANSWER_PROMPT.format(
            instruction=SEED_CANDIDATE[FINAL_ANSWER],
            question=question,
            summary_1=summary,
            summary_2=summary,
        )
        tokens[FINAL_ANSWER].append(len(p4))

    print(f"=== measured prompt sizes (n={len(instances)}, k={args.k}) ===")
    rollout_usd = 0.0
    for component, chars in tokens.items():
        mean_chars = statistics.mean(chars)
        tin = int(mean_chars / CHARS_PER_TOKEN)
        tout = OUT_TOKENS[component]
        cost = price_call(SOLVER, tin, tout)
        rollout_usd += cost
        print(f"  {component:20} {mean_chars:8,.0f} chars  ~{tin:6,} tok in  {tout:4} out  ${cost:.5f}")

    val_eval = rollout_usd * args.val_size
    reflection = price_call(REFLECTION, 12000, 1000)
    per_iter = rollout_usd * args.minibatch * 2 + reflection + args.accept_rate * val_eval

    print(f"\n=== per rollout ===\n  ${rollout_usd:.5f}  (4 LM calls, fixed)")
    print(f"\n=== per full val evaluation (n={args.val_size}) ===\n  ${val_eval:.2f}")
    print(f"\n=== per iteration (minibatch {args.minibatch}, accept rate {args.accept_rate:.0%}) ===")
    print(f"  parent+child minibatch : ${rollout_usd * args.minibatch * 2:.3f}")
    print(f"  reflection call        : ${reflection:.3f}")
    print(f"  expected val re-eval   : ${args.accept_rate * val_eval:.3f}")
    print(f"  total                  : ${per_iter:.3f}")

    print("\n=== budget -> iterations ===")
    for budget in (10, 15, 20, 25, 40):
        iters = max(0.0, budget - val_eval) / per_iter
        print(f"  ${budget:>3}/seed  ->  ~{iters:5.0f} iterations  (~{iters * args.accept_rate:.0f} accepted)")
    print(
        f"\nAssumptions stated, not hidden: summary ~{ASSUMED_SUMMARY_CHARS} chars, "
        f"query ~{ASSUMED_QUERY_CHARS} chars, {CHARS_PER_TOKEN} chars/token."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
