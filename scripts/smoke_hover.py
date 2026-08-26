#!/usr/bin/env python
"""Live pre-flight for the HoVer arm. **Spends ~$0.05** (8 rollouts).

Everything the offline tests cannot check: that the BM25 index actually contains
the articles HoVer names, that retrieval finds them, and what a rollout costs.
The cost number is the point -- the AppWorld estimate was 87% low (F042) and the
LiveBench-Math one 38% high, both from extrapolating the wrong shape.

    PYTHONUTF8=1 uv run python scripts/smoke_hover.py

Reads TRAIN ids only, so it cannot contaminate the shared base-val state.

The retrieval-coverage number below is the one to read first. HoVer's gold
titles come from the 2017 Wikipedia dump; if our index is a different snapshot,
some gold articles are simply absent and the metric is capped below 1.0 through
no fault of any candidate. That would look like a hard benchmark rather than a
corpus mismatch, and it is worth knowing before paying for a base val.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=8, help="rollouts, spread across hop counts")
    parser.add_argument("--manifests", type=Path, default=REPO / "manifests" / "hover")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    # A smoke whose only output is stdout is a smoke whose result dies with
    # the terminal. The measured $/rollout is the whole reason it is run and
    # is what every budget downstream is set from, so it gets written down.
    parser.add_argument("--out", type=Path, default=REPO / "results" / "smoke" / "hover_smoke.json")
    args = parser.parse_args()

    import importlib.util
    import sys

    sys.path.insert(0, str(REPO / "src"))

    from gepa_taxonomy.bedrock import BedrockLM, require_credentials
    from gepa_taxonomy.cost import CostMeter
    from gepa_taxonomy.hotpotqa.retrieval import WikiRetriever
    from gepa_taxonomy.hover.adapter import HoverAdapter, instances_by_id
    from gepa_taxonomy.hover.grading import normalize_title
    from gepa_taxonomy.hover.program import COMPONENTS, SEED_CANDIDATE, HoverMultiHopProgram

    require_credentials()

    spec = importlib.util.spec_from_file_location("_hover_runner", REPO / "scripts" / "run_hover_seed.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    train = runner.load_instances(args.manifests / "train.json")

    # Spread across hop counts: a smoke that only saw 2-hop claims would prove
    # nothing about the 4-hop case, which is 26% of the split and where the
    # all-or-nothing metric bites hardest.
    by_hop: dict[int, list] = {}
    for inst in train:
        by_hop.setdefault(inst.task.num_hops, []).append(inst)
    picked: list = []
    while len(picked) < args.n and any(by_hop.values()):
        for hop in sorted(by_hop):
            if by_hop[hop] and len(picked) < args.n:
                picked.append(by_hop[hop].pop(0))
    print(f"smoke: {len(picked)} rollouts | hop mix {dict(sorted(Counter(i.task.num_hops for i in picked).items()))}")

    retriever = WikiRetriever(k=args.k).load()

    # Corpus reachability, free, and a LOWER BOUND -- not a ceiling.
    #
    # It queries each gold title on its own and checks the top k. BM25 on a bare
    # title like "Gillette" or "Animal House" ranks the exact article below
    # everything else sharing those words, so this systematically UNDER-reports.
    # Measured: this probe said 82% on a run whose loose recall was 85.4%, which
    # is arithmetically impossible if only 82% were reachable -- the real
    # rollout has three hops, k=7 each, and the claim as context.
    #
    # Kept because a genuinely missing corpus would still show up as a floor
    # collapse, but it is only alarming when it falls FAR below the loose recall
    # reported afterwards. That comparison is the actual test, so it is printed
    # rather than a fixed threshold.
    wanted = {normalize_title(t) for i in picked for t in i.gold.titles}
    found_in_corpus = set()
    for inst in picked:
        for title in inst.gold.titles:
            for passage in retriever.retrieve(title, k=20):
                if normalize_title(passage.title) == normalize_title(title):
                    found_in_corpus.add(normalize_title(title))
                    break
    coverage = len(found_in_corpus) / max(1, len(wanted))
    print(
        f"corpus reachability (LOWER BOUND): {len(found_in_corpus)}/{len(wanted)} gold titles "
        f"found by bare-title query ({coverage:.0%})"
    )
    print("  compare against loose recall below -- if that is higher, this probe is the underestimate.")

    meter = CostMeter()
    program = HoverMultiHopProgram(
        retriever=retriever,
        lm=BedrockLM(model=args.solver_model, max_retries=4),
        meter=meter,
        model=args.solver_model,
        k=args.k,
    )
    adapter = HoverAdapter(
        program=program,
        instances=instances_by_id(picked),
        reflection_gold_ids=frozenset(i.task.example_id for i in picked),
        max_workers=args.workers,
    )

    started = time.time()
    batch = adapter.evaluate(picked, dict(SEED_CANDIDATE), capture_traces=True)
    elapsed = time.time() - started

    print(f"\n{'id':<10} {'hops':>5} {'strict':>7} {'loose':>7} {'missing':>8} {'$':>8}")
    for trace, score, inst in zip(batch.trajectories, batch.scores, picked, strict=True):
        g = trace["grading"]
        print(
            f"{trace['example_id'][:8]:<10} {inst.task.num_hops:>5} {score:>7.1f} "
            f"{g['loose_recall']:>7.2f} {len(g['missing']):>8} {trace['cost_usd']:>8.4f}"
        )

    strict = statistics.mean(batch.scores)
    loose = statistics.mean(t["grading"]["loose_recall"] for t in batch.trajectories)
    per = meter.budgeted_usd / max(1, len(picked))

    dataset = adapter.make_reflective_dataset(dict(SEED_CANDIDATE), batch, list(COMPONENTS))

    print(f"\n  strict (selection): {strict:.3f}   pilot measured ~0.467 on an easier pool")
    print(f"  loose recall      : {loose:.3f}   gap {loose - strict:+.3f} = partial retrieval the metric ignores")
    print(f"  errors            : transport {adapter.transport_errors}  program {adapter.program_errors}")
    print(f"  elapsed           : {elapsed:.1f}s  ({elapsed / len(picked):.1f}s/rollout)")
    print(f"  spend             : ${meter.budgeted_usd:.4f}  (${per:.5f}/rollout)")
    print(f"  reflective set    : {[(k, len(v)) for k, v in dataset.items()]}")

    val_n = 300
    print("\n  --- projected, from THIS measurement ---")
    print(f"  base val ({val_n})   : ${per * val_n:.2f}")
    for mb in (3, 6):
        # 2*mb minibatch rollouts + reflection + accept_rate * full val.
        per_iter = 2 * mb * per + 0.05 + 0.5 * val_n * per
        print(f"  minibatch {mb:>2}      : ${per_iter:.2f}/iteration -> {60 / per_iter:.0f} iterations at $60")

    payload = {
        "benchmark": "hover",
        "n": len(picked),
        "hop_mix": {str(k): v for k, v in sorted(Counter(i.task.num_hops for i in picked).items())},
        "corpus_coverage": round(coverage, 4),
        "gold_titles_missing_from_index": sorted(wanted - found_in_corpus),
        "strict_retrieval": round(strict, 4),
        "loose_recall": round(loose, 4),
        "usd_per_rollout": round(per, 6),
        "seconds_per_rollout": round(elapsed / len(picked), 2),
        "spend_usd": round(meter.budgeted_usd, 6),
        "projected_base_val_usd": round(per * val_n, 2),
        "solver_model": args.solver_model,
        "adapter": adapter.summary(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  written           : {args.out}")

    if adapter.transport_errors or adapter.program_errors:
        print("\n  NOT CLEAN -- do not launch until this is understood.")
        print(f"  {adapter.failures.summary().get('error_samples')}")
        return 1
    print("\n  clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
