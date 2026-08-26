#!/usr/bin/env python
"""Measure the retrieval ceiling on HotpotQA. Free: no model is called.

This bounds everything downstream. The program can only answer from what
retrieval surfaces, so if BM25 cannot reach the gold documents, no amount of
prompt optimization -- taxonomy-conditioned or otherwise -- can recover the
answer. Measuring it before spending is the cheap way to find that out.

Two numbers matter:

* **hop-1 recall** -- what a single retrieval on the raw question achieves.
  This is the floor the second hop has to improve on.
* **oracle two-hop recall** -- retrieval on the question PLUS a query built from
  the gold titles. This is the ceiling a perfect ``create_query_hop2`` could
  reach, and therefore the headroom the optimizer is actually competing for.

    uv run python scripts/check_hotpotqa_retrieval.py --split val --n 100
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from gepa_taxonomy.hotpotqa.retrieval import WikiRetriever
from gepa_taxonomy.hotpotqa.tasks import instance_from_record

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    manifest = json.loads((REPO / "manifests" / "hotpotqa" / f"{args.split}.json").read_text())
    wanted = set(manifest["example_ids"][: args.n])

    from datasets import load_dataset

    print(f"loading dataset and selecting {len(wanted)} {args.split} examples ...", flush=True)
    ds = load_dataset(manifest["dataset"], manifest["dataset_config"], split=manifest["dataset_split"])
    instances = [instance_from_record(r) for r in ds if r["id"] in wanted]

    print(f"loading BM25 index (k={args.k}) ...", flush=True)
    retriever = WikiRetriever(k=args.k).load()

    hop1, oracle, n_gold = [], [], []
    for i, instance in enumerate(instances, start=1):
        gold_titles = {t.lower() for t in instance.gold.titles}
        n_gold.append(len(gold_titles))

        first = retriever.retrieve(instance.task.question, k=args.k)
        found1 = {p.title.lower() for p in first} & gold_titles
        hop1.append(len(found1) / len(gold_titles) if gold_titles else 1.0)

        # Oracle second hop: query built from the gold titles retrieval missed.
        missing = [t for t in instance.gold.titles if t.lower() not in {p.title.lower() for p in first}]
        found2 = set(found1)
        if missing:
            second = retriever.retrieve(" ".join(missing), k=args.k)
            found2 |= {p.title.lower() for p in second} & gold_titles
        oracle.append(len(found2) / len(gold_titles) if gold_titles else 1.0)

        if i % 25 == 0:
            print(f"  {i}/{len(instances)}", flush=True)

    def pct(xs):
        return f"{statistics.mean(xs):.1%}"

    print("\n=== retrieval ceiling ===")
    print(f"examples            : {len(instances)}")
    print(f"gold docs / question: {statistics.mean(n_gold):.2f}")
    print(f"hop-1 recall        : {pct(hop1)}   (single retrieval on the raw question)")
    print(f"oracle 2-hop recall : {pct(oracle)}   (ceiling for a perfect create_query_hop2)")
    print(f"fully covered       : {sum(1 for r in oracle if r == 1.0) / len(oracle):.1%} of questions")
    print(f"\nheadroom the optimizer competes for: {statistics.mean(oracle) - statistics.mean(hop1):+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
