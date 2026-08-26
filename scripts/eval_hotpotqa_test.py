#!/usr/bin/env python
"""Evaluate frozen candidates on the held-out HotpotQA test split. **Spends money.**

This produces the headline number, and it is the one the SWE-Bench round taught
us to distrust val for: both arms there scored 21.7% on val and 14-15% on test,
because selection had overfit the val set. Val is what the optimizer optimises;
test is the only number that answers the question.

Paired statistics
-----------------
HotpotQA scores are **continuous** (answer F1), so the paired test is a
Wilcoxon signed-rank over per-instance differences -- not McNemar, which the
SWE-Bench round used because its metric was binary. McNemar on a continuous
metric would throw away almost all the information by first collapsing scores
to win/lose. The per-instance scores are written out so any other test can be
run over them afterwards without re-spending.

    PYTHONUTF8=1 uv run python scripts/eval_hotpotqa_test.py \
        --candidate baseline-s1=results/runs/hotpotqa-baseline-seed1 \
        --candidate taxonomy-s1=results/runs/hotpotqa-taxonomy-seed1
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class CandidateSelectionError(ValueError):
    """The candidate spec could not be resolved to a single program."""


def parse_candidate_spec(spec: str) -> tuple[str, Path, int | None]:
    """``label=path`` or ``label=path#index``.

    Without an index the BEST candidate by val score is used, read from the
    run's ``summary.json``. An explicit index pins a specific one, which is how
    a re-analysis reproduces an earlier choice exactly.
    """
    if "=" not in spec:
        raise CandidateSelectionError(f"candidate spec needs a label: {spec!r} (want 'label=path')")
    label, _, rest = spec.partition("=")
    index: int | None = None
    if "#" in rest:
        rest, _, raw = rest.partition("#")
        index = int(raw)
    return label.strip(), Path(rest.strip()), index


#: gepa logs a candidate's full-val score each time it picks it as a parent.
_SELECTED = re.compile(r"Selected program (\d+) score: ([0-9.]+)")


def _best_from_log(run_dir: Path) -> tuple[int | None, float | None]:
    """Highest-scoring candidate according to the run log.

    Only candidates that were *selected as parents* appear, so this can miss a
    high scorer that was never picked. It is a fallback for runs predating
    ``best_candidate_index``, not the primary path -- and it is honest about
    what it saw rather than assuming the maximum it found is global.
    """
    log = run_dir / "run_log.txt"
    if not log.exists():
        return None, None
    seen: dict[int, float] = {}
    for idx, score in _SELECTED.findall(log.read_text(encoding="utf-8", errors="replace")):
        seen[int(idx)] = float(score)
    if not seen:
        return None, None
    best = max(seen, key=seen.__getitem__)
    return best, seen[best]


def resolve_candidate(run_dir: Path, index: int | None) -> tuple[dict[str, str], int, float | None]:
    """Return (candidate text, its index, its val score)."""
    candidates_path = run_dir / "candidates.json"
    if not candidates_path.exists():
        raise CandidateSelectionError(f"no candidates.json in {run_dir}")
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not candidates:
        raise CandidateSelectionError(f"{candidates_path} is empty")

    val_score: float | None = None
    if index is None:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise CandidateSelectionError(
                f"{run_dir} has no summary.json, so the best candidate is unknown. "
                "Pass an explicit #index, or wait for the run to finish."
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        val_score = summary.get("best_val_score")
        index = summary.get("best_candidate_index")
        if index is None:
            # Runs launched before best_candidate_index was recorded (seed 1) can
            # still be resolved: gepa logs every candidate's full-val score when
            # it selects it as a parent. Falling back to "the last candidate"
            # would silently pick the NEWEST rather than the best, so derive it
            # or refuse -- never guess.
            index, val_score = _best_from_log(run_dir)
            if index is None:
                raise CandidateSelectionError(
                    f"{summary_path} records no best_candidate_index and {run_dir}/run_log.txt "
                    "shows no candidate scores. Pass #index explicitly."
                )
    if not (0 <= index < len(candidates)):
        raise CandidateSelectionError(f"index {index} out of range for {len(candidates)} candidates")
    return candidates[index], index, val_score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, metavar="LABEL=RUN_DIR[#IDX]")
    parser.add_argument("--manifests", type=Path, default=REPO / "manifests" / "hotpotqa")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N test instances")
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--max-retries", type=int, default=8)
    args = parser.parse_args()

    import importlib.util

    from gepa_taxonomy.bedrock import BedrockLM, require_credentials
    from gepa_taxonomy.cost import CostMeter
    from gepa_taxonomy.hotpotqa.adapter import HotpotQAAdapter, instances_by_id
    from gepa_taxonomy.hotpotqa.program import MultiHopProgram
    from gepa_taxonomy.hotpotqa.retrieval import WikiRetriever

    specs = [parse_candidate_spec(s) for s in args.candidate]
    resolved = [(label, *resolve_candidate(d, i)) for label, d, i in specs]

    require_credentials()

    spec_mod = importlib.util.spec_from_file_location("_runner", REPO / "scripts" / "run_hotpotqa_seed.py")
    runner = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(runner)

    test = runner.load_instances(args.manifests / "test.json")
    if args.limit:
        test = test[: args.limit]

    print(f"test set: {len(test)} instances | candidates: {len(resolved)}")
    for label, _cand, index, val in resolved:
        print(f"  {label:16} index={index} val={val if val is None else f'{val:.4f}'}")

    meter = CostMeter()
    program = MultiHopProgram(
        retriever=WikiRetriever(k=args.k).load(),
        lm=BedrockLM(model=args.solver_model, max_retries=args.max_retries),
        meter=meter,
        model=args.solver_model,
        k=args.k,
    )
    adapter = HotpotQAAdapter(program=program, instances=instances_by_id(test), max_workers=args.workers)

    started = time.time()
    per_label: dict[str, list[float]] = {}
    for label, candidate, _index, _val in resolved:
        print(f"\nevaluating {label} ...", flush=True)
        batch = adapter.evaluate(test, candidate, capture_traces=False)
        per_label[label] = list(batch.scores)
        print(f"  mean F1 {statistics.mean(batch.scores):.4f}")

    elapsed = time.time() - started
    labels = list(per_label)

    results = {
        "n": len(test),
        "elapsed_hours": round(elapsed / 3600, 3),
        "spend_usd": meter.total_usd,
        "transport_errors": adapter.transport_errors,
        "program_errors": adapter.program_errors,
        "candidates": [
            {"label": label, "index": index, "val_score": val, "test_mean_f1": statistics.mean(per_label[label])}
            for label, _c, index, val in resolved
        ],
        "instances": [
            {"example_id": inst.task.example_id, **{label: per_label[label][i] for label in labels}}
            for i, inst in enumerate(test)
        ],
    }

    # Paired comparison, only meaningful for exactly two arms.
    if len(labels) == 2:
        a, b = labels
        diffs = [per_label[b][i] - per_label[a][i] for i in range(len(test))]
        nonzero = [d for d in diffs if d != 0.0]
        results["paired"] = {
            "arms": [a, b],
            "mean_difference": statistics.mean(diffs),
            "instances_differing": len(nonzero),
            "b_better": sum(1 for d in nonzero if d > 0),
            "a_better": sum(1 for d in nonzero if d < 0),
        }
        try:
            from scipy.stats import wilcoxon

            if nonzero:
                _stat, p = wilcoxon(diffs, zero_method="wilcox")
                results["paired"]["wilcoxon_p"] = float(p)
        except ImportError:
            # Not a dependency; the per-instance scores are written out, so the
            # test can be run afterwards without re-spending a dollar.
            results["paired"]["wilcoxon_p"] = None

    out = args.out or REPO / "results" / "test_eval" / f"hotpotqa_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("\n=== test results ===")
    for c in results["candidates"]:
        drop = (
            ""
            if c["val_score"] is None
            else f"   (val {c['val_score']:.4f} -> test, {c['test_mean_f1'] - c['val_score']:+.4f})"
        )
        print(f"  {c['label']:16} test mean F1 {c['test_mean_f1']:.4f}{drop}")
    if "paired" in results:
        p = results["paired"]
        print(f"\n  paired {p['arms'][1]} - {p['arms'][0]}: mean {p['mean_difference']:+.4f}")
        print(f"  differing on {p['instances_differing']}/{len(test)}  ({p['b_better']} vs {p['a_better']})")
        if p.get("wilcoxon_p") is not None:
            print(f"  Wilcoxon signed-rank p = {p['wilcoxon_p']:.4f}")
    print(f"\n  spend ${meter.total_usd:.2f}   written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
