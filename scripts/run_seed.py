#!/usr/bin/env python
"""Launch ONE baseline GEPA seed on local Docker. **THIS SPENDS API TOKENS.**

    zsh -c 'source ~/.zshrc >/dev/null 2>&1; \
      caffeinate -dimsu uv run python scripts/run_seed.py --seed 1 --budget 100'

Baseline purity (CLAUDE.md hard rule 2): gepa v0.1.4 runs unmodified. The only
addition is the total-cost stopper, which is a pure loop-exit observer -- it is
consulted at exactly one call site, the main loop condition, and cannot touch
candidate selection, reflection, sampling or scheduling.

Arms
----
Without ``--taxonomy`` this is the BASELINE arm and behaves exactly as it did
before the taxonomy work: no judge is constructed, no judge module is imported,
and the reflective dataset is byte-identical. With ``--taxonomy`` it is the
TREATMENT arm: failed rollouts in the parent's reflection minibatch are
diagnosed against that taxonomy and the codes ride along in the reflection
prompt. Judge spend is metered into the SAME dollar budget, so the arms are
compared at equal dollars and judging competes with rollouts.

Interruption tolerance
----------------------
Resumable in two layers:

* **gepa state** -- pass the same ``--run-dir``; candidate pool, per-instance
  scores, Pareto frontier and evaluation cache come back exactly.
* **rollout cache** -- every graded rollout is fsync'd to JSONL as it completes,
  so an interruption mid-iteration re-pays nothing. Without this, gepa's
  once-per-iteration save would discard the in-flight iteration (~$2, ~50 min).

Known limitation: the RNG and batch-sampler position are NOT restored, so a
resumed run explores differently from the trajectory an uninterrupted longer run
would have taken. Prior work is preserved in full. See
``docs/findings/phase1-resume-fidelity.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import re
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = REPO_ROOT / "manifests" / "swebench_verified"
DEFAULT_RUNS = REPO_ROOT / "results" / "runs"
CACHE_DIR = REPO_ROOT / ".cache" / "repos"



class QuietLogger:
    """gepa logger, rewritten for readability.

    gepa prints each proposed instruction in full (thousands of characters) and
    reports minibatch scores as bare sums with no denominator. This keeps one
    readable line per event; proposal text is still written to candidates.json.
    """

    _SELECTED = re.compile(r"^(Iteration \d+): Selected program (\d+) score: ([\d.]+)")
    _PROPOSAL = re.compile(r"^(Iteration \d+): Proposed new text for (\w+):\s*(.*)$", re.S)
    _WORSE = re.compile(r"^(Iteration \d+): New subsample score ([\d.]+) is not better than old score ([\d.]+), skipping")
    _REJECT = re.compile(r"^(Iteration \d+): Candidate rejected by acceptance criterion \(old_sum=([\d.]+), new_sum=([\d.]+)\), skipping")
    _ACCEPT = re.compile(r"^(Iteration \d+): Accepted candidate \(subsample score ([\d.]+) -> ([\d.]+)\)")

    def __init__(self, path, minibatch_size: int, valset_size: int):
        self._fh = open(path, "a", buffering=1)
        self._n = minibatch_size
        self._val = valset_size

    def _fmt(self, text: str) -> str:
        n = self._n
        m = self._SELECTED.match(text)
        if m:
            return f"{m.group(1)}: candidate {m.group(2)} selected (val {round(float(m.group(3)) * self._val):d}/{self._val})"
        m = self._PROPOSAL.match(text)
        if m:
            return f"{m.group(1)}: proposed for {m.group(2)}"
        m = self._WORSE.match(text)
        if m:
            return f"{m.group(1)}: minibatch {float(m.group(3)):.0f}/{n} -> {float(m.group(2)):.0f}/{n}  NOT BETTER, skipped"
        m = self._REJECT.match(text)
        if m:
            return f"{m.group(1)}: minibatch {float(m.group(2)):.0f}/{n} -> {float(m.group(3)):.0f}/{n}  REJECTED, skipped"
        m = self._ACCEPT.match(text)
        if m:
            return f"{m.group(1)}: minibatch {float(m.group(2)):.0f}/{n} -> {float(m.group(3)):.0f}/{n}  BETTER, accepted -- running full val eval"
        return text

    def log(self, message: str) -> None:
        text = self._fmt(str(message))
        print(text, flush=True)
        self._fh.write(text + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True, help="per-seed dollar budget (optimization loop only)")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--profile-prefix", default=None)
    ap.add_argument("--val-manifest", type=Path, default=MANIFESTS / "val.json")
    ap.add_argument("--train-manifest", type=Path, default=MANIFESTS / "train.json")
    ap.add_argument("--seed-cache", type=Path, default=REPO_ROOT / "results" / "seed_cache" / "base_val.json")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--cache-level", default="env")
    ap.add_argument("--minibatch-size", type=int, default=3)
    # Absent => BASELINE arm, byte-for-byte the behaviour this script had before
    # the taxonomy work landed. Present => TREATMENT arm.
    ap.add_argument("--taxonomy", type=Path, default=None, help="certified AdaMAST taxonomy; enables the treatment arm")
    ap.add_argument("--judge-model", default=None, help="Bedrock model for judging (default: the refiner model)")
    ap.add_argument("--wall-clock-hours", type=float, default=72.0, help="safety net beside the dollar budget")
    ap.add_argument("--no-skip-gate", action="store_true", help="disable the ungradeable-patch skip")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved config and exit; free")
    args = ap.parse_args()

    # Arm-specific default. Sharing a directory between arms would resume the
    # other arm's gepa state and rollout cache -- i.e. silently start the
    # treatment run from the baseline's candidate pool.
    default_name = f"taxonomy-seed{args.seed}" if args.taxonomy else f"baseline-seed{args.seed}"
    run_dir = args.run_dir or (DEFAULT_RUNS / default_name)
    # Create the run directory FIRST, before any output. A launch wrapped in
    # `tee -a results/runs/<run>/console.log` fails on a missing directory
    # before the script even starts, which is how the first launch attempt was
    # lost. Cheap, idempotent, and safe to do before the dry-run branch.
    run_dir.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)

    from gepa_taxonomy.cost import (
        REFINER_BASE,
        SOLVER_BASE,
        CostMeter,
        MaxTotalCostStopper,
        with_profile,
    )
    from gepa_taxonomy.cost import REFINER_MODEL as _DEF_REF
    from gepa_taxonomy.cost import SOLVER_MODEL as _DEF_SOL

    solver_model = with_profile(SOLVER_BASE, args.profile_prefix) if args.profile_prefix else _DEF_SOL
    refiner_model = with_profile(REFINER_BASE, args.profile_prefix) if args.profile_prefix else _DEF_REF
    judge_model = args.judge_model or refiner_model

    from gepa_taxonomy.splits import load_manifest

    train_ids = load_manifest(args.train_manifest)
    val_ids = load_manifest(args.val_manifest)

    arm = "TAXONOMY" if args.taxonomy else "BASELINE"
    print("=" * 72)
    print(f"{arm} SEED {args.seed}  (local Docker)")
    print("=" * 72)
    print(f"  budget        ${args.budget:.2f}   (optimization loop only)")
    print(f"  solver        {solver_model}")
    print(f"  refiner       {refiner_model}")
    print(f"  taxonomy      {args.taxonomy or 'none (baseline arm)'}")
    if args.taxonomy:
        print(f"  judge model   {judge_model}   (judge spend shares the same ${args.budget:.2f})")
    print(f"  train / val   {len(train_ids)} / {len(val_ids)}")
    print(f"  workers       {args.max_workers}   cache_level {args.cache_level}")
    print(f"  skip gate     {'OFF' if args.no_skip_gate else 'on'}")
    print(f"  run_dir       {run_dir}")
    # A resume is defined by gepa's saved state, not by the directory existing:
    # an empty run_dir is a fresh run.
    state_file = run_dir / "gepa_state.bin"
    print(f"  resuming      {'YES -- ' + str(state_file.name) + ' found' if state_file.exists() else 'no (fresh run)'}")
    print(f"  seed cache    {'present' if args.seed_cache.exists() else 'ABSENT'}")

    if args.dry_run:
        print(f"  log dir       {REPO_ROOT / 'results' / 'logs'}  (created)")
        # Exercise the same contract the engine will, so a mismatch shows up
        # here rather than at launch. Free: no adapter, no LM, no container.
        import dataclasses as _dc
        import typing as _t

        from gepa.core.adapter import EvaluationBatch as _EB

        from gepa_taxonomy.adapter import SweBenchAdapter as _A

        # `from __future__ import annotations` makes __annotations__ a STRING,
        # so compare resolved hints -- an identity check against the raw
        # annotation silently fails.
        ret = _t.get_type_hints(_A.evaluate).get("return")
        print(f"  engine contract fields  : {sorted(f.name for f in _dc.fields(_EB))}")
        print(f"  adapter.evaluate returns: {getattr(ret, '__name__', ret)}")
        if ret is not _EB:
            print("  contract check: FAILED -- adapter must return gepa's EvaluationBatch", file=sys.stderr)
            return 3
        print("  contract check: OK (this is what crashed the first launch)")

        from gepa_taxonomy.bedrock import (
            MeteredReflectionLM as _M,
        )
        from gepa_taxonomy.bedrock import (
            ReflectionConformanceError as _RCE,
        )
        from gepa_taxonomy.bedrock import (
            verify_reflection_lm as _V,
        )

        class _NoNetLM:
            def complete(self, prompt, *, max_tokens=4096):
                raise AssertionError("dry run must not call out")

        try:
            rep = _V(_M(lm=_NoNetLM(), meter=CostMeter(), model=refiner_model))
            print(f"  reflection conformance: OK ({rep['type']}, wrapped as {rep['wrapped_as']})")
        except _RCE as exc:
            print(f"  reflection conformance: FAILED -- {exc}", file=sys.stderr)
            return 4

        if args.taxonomy:
            # A treatment run whose judge cannot start is a baseline run wearing
            # the wrong label, discovered only after the money is gone. Free to
            # check here: it imports adamast and reads the taxonomy, nothing more.
            from gepa_taxonomy.taxonomy_judge import JudgeError as _JErr
            from gepa_taxonomy.taxonomy_judge import TaxonomyJudge as _TJ

            try:
                rep = _TJ(taxonomy_path=args.taxonomy, meter=CostMeter(), model=judge_model).preflight()
                print(
                    f"  taxonomy judge: OK ({rep['codes']} codes, fingerprint {rep['taxonomy']}, "
                    f"adamast {rep.get('adamast', '?')} via {rep['transport']})"
                )
            except (_JErr, OSError, ValueError) as exc:
                print(f"  taxonomy judge: FAILED -- {exc}", file=sys.stderr)
                return 5

        if not args.seed_cache.exists():
            print("\n  (a real launch would refuse: base-candidate val evaluation missing)")
        print("\n--dry-run: nothing launched, nothing spent.")
        return 0

    if not args.seed_cache.exists():
        print(
            "\nRefusing to launch: the base-candidate val evaluation is missing.\n"
            "Every seed must start from identical state, so it is computed once\n"
            "and replayed. Build it first:\n"
            "  uv run python scripts/build_base_val.py",
            file=sys.stderr,
        )
        return 2

    from gepa_taxonomy.bedrock import BedrockLM, MeteredReflectionLM, require_credentials, verify_reflection_lm

    region = require_credentials()

    import gepa
    from datasets import load_dataset
    from gepa.utils.stop_condition import CompositeStopper, TimeoutStopCondition

    from gepa_taxonomy.adapter import SweBenchAdapter
    from gepa_taxonomy.grading import LocalDockerGrader
    from gepa_taxonomy.program import SEED_CANDIDATE, SolverRefinerProgram
    from gepa_taxonomy.retrieval import BM25Retriever
    from gepa_taxonomy.rollout_cache import RolloutCache
    from gepa_taxonomy.seed_cache import SeedEvaluationCache
    from gepa_taxonomy.tasks import split_row

    ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
    wanted = set(train_ids) | set(val_ids)
    instances = {r["instance_id"]: split_row(r) for r in ds if r["instance_id"] in wanted}

    solver_meter, refiner_meter, reflection_meter, judge_meter = CostMeter(), CostMeter(), CostMeter(), CostMeter()
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

    cache = RolloutCache.open(run_dir / "rollouts.jsonl")
    if len(cache):
        print(f"  rollout cache {len(cache)} entries (${cache.recovered_usd:.2f} will not be re-paid)")

    # D009's completeness guarantee, checked ONCE here rather than inferred
    # from a per-lookup miss. The replay scope is (base candidate) x (val);
    # train minibatch evaluations of the base candidate run live and billed.
    seed_cache = SeedEvaluationCache.load(args.seed_cache)
    seed_cache.assert_covers(val_ids)
    print(f"  seed cache covers all {len(val_ids)} val instances")

    judge = judge_cache = None
    if args.taxonomy:
        from gepa_taxonomy.taxonomy_judge import JudgeCache, TaxonomyJudge

        judge_cache = JudgeCache.open(run_dir / "judgements.jsonl")
        judge = TaxonomyJudge(
            taxonomy_path=args.taxonomy,
            meter=judge_meter,
            model=judge_model,
            cache=judge_cache,
            aws_region=region,
        )
        report = judge.preflight()  # raises rather than silently degrading to baseline
        print(
            f"  taxonomy      {report['codes']} codes, fingerprint {report['taxonomy']}, "
            f"adamast {report.get('adamast', '?')}"
        )
        if len(judge_cache):
            print(f"  judge cache   {len(judge_cache)} judgements (${judge_cache.recovered_usd:.2f} not re-paid)")

    adapter = SweBenchAdapter(
        program=program,
        grader=LocalDockerGrader(
            work_dir=run_dir / "harness",
            max_workers=args.max_workers,
            cache_level=args.cache_level,
            run_id_prefix=f"seed{args.seed}",
        ),
        instances=instances,
        seed_cache=seed_cache,
        rollout_cache=cache,
        repo_cache_dir=CACHE_DIR,
        trace_path=run_dir / "traces.jsonl",
        skip_ungradeable_patches=not args.no_skip_gate,
        # D028: the reflective dataset may carry gold for TRAIN instances only.
        reflection_gold_ids=set(train_ids),
        phase="optimization",
        taxonomy_judge=judge,
    )

    # Reflection is metered too. cost.py says the budget covers "minibatch
    # rollouts, reflection calls, and val evaluations"; before this it silently
    # covered only the first and third.
    reflection_lm = MeteredReflectionLM(
        lm=BedrockLM(model=refiner_model), meter=reflection_meter, model=refiner_model,
        spend_log=run_dir / "reflection.jsonl"
    )

    # Preflight: conformance is checked the way the engine will invoke it.
    # A non-callable reflection LM raises TypeError inside gepa, which gepa
    # SWALLOWS -- the run then burns its budget proposing nothing.
    report = verify_reflection_lm(reflection_lm)
    print(f"  reflection LM : {report['type']} callable={report['callable']} "
          f"wrapped_as={report['wrapped_as']} batched={report['batched_path']}")

    # Equal dollars across arms: judging competes with rollouts for the same
    # budget rather than being funded on the side.
    stopper = CompositeStopper(
        MaxTotalCostStopper(args.budget, meters=[solver_meter, refiner_meter, reflection_meter, judge_meter]),
        TimeoutStopCondition(timeout_seconds=args.wall_clock_hours * 3600),
        mode="any",
    )

    # gepa passes dataset ITEMS straight through to adapter.evaluate(), and our
    # adapter is keyed by instance id -- so the datasets are id lists, not Task
    # objects.
    #
    # gepa still keys its own per-instance structures (val_subscores, Pareto
    # frontier) by POSITIONAL INDEX, not by these strings. Mapping a score back
    # to an instance is therefore positional, which is only stable because the
    # manifests are sorted and we never reorder them.
    trainset = list(train_ids)
    valset = list(val_ids)

    print(f"\nlaunching at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    t0 = time.time()
    try:
        result = gepa.optimize(
            seed_candidate=dict(SEED_CANDIDATE),
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm,
            stop_callbacks=stopper,
            run_dir=str(run_dir),
            seed=args.seed,
            reflection_minibatch_size=args.minibatch_size,
            display_progress_bar=False,
            logger=QuietLogger(run_dir / "gepa.log", args.minibatch_size, len(valset)),
        )
    except KeyboardInterrupt:
        print("\ninterrupted -- gepa state and the rollout cache are on disk.")
        print(f"resume with the same command; {len(cache)} rollouts will not be re-paid.")
        adapter.flush_traces()
        cache.close()
        if judge_cache is not None:
            judge_cache.close()
        return 130

    adapter.flush_traces()
    elapsed = (time.time() - t0) / 3600
    spent = (
        solver_meter.budgeted_usd
        + refiner_meter.budgeted_usd
        + reflection_meter.budgeted_usd
        + judge_meter.budgeted_usd
    )

    summary = {
        "seed": args.seed,
        "arm": "taxonomy" if args.taxonomy else "baseline",
        "budget_usd": args.budget,
        "realised_usd": round(spent, 4),
        "elapsed_hours": round(elapsed, 2),
        "candidates": len(result.candidates),
        "best_idx": result.best_idx,
        "total_metric_calls": result.total_metric_calls,
        "reflection_calls": reflection_lm.calls,
        "reflection_usd": round(reflection_meter.budgeted_usd, 4),
        "rollout_cache_hits": cache.hits,
        "harness": adapter.grader.summary(),
        "solver_model": solver_model,
        "refiner_model": refiner_model,
    }
    if judge is not None:
        summary["taxonomy_judge"] = judge.summary()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cache.close()
    if judge_cache is not None:
        judge_cache.close()

    print("\n" + "=" * 72)
    print(f"  candidates discovered : {len(result.candidates)}")
    print(f"  realised spend        : ${spent:.2f}  (budget ${args.budget:.2f})")
    if judge is not None:
        print(
            f"  of which judging      : ${judge_meter.budgeted_usd:.2f}  "
            f"({judge.judged} instances, {judge.calls} batches, {judge_cache.hits} cache hits)"
        )
    print(f"  elapsed               : {elapsed:.1f} h")
    print(f"  harness invocations   : {adapter.grader.calls}")
    print(f"  wrote {run_dir / 'summary.json'}")
    print("\n  NOTE: realised spend overshoots the budget by up to one iteration --")
    print("  the stopper is consulted between iterations. Seeds are compared on")
    print("  realised cost, not nominal budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
