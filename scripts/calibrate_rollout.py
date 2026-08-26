#!/usr/bin/env python
"""ONE calibrated measurement rollout. **THIS SPENDS API TOKENS.**

Runs a single real rollout on one TRAIN instance -- real BM25 retrieval over a
real checkout, real solver call, real refiner call -- and records the ACTUAL
token counts and cost. Those measured numbers replace the estimates in
``scripts/cost_model.py``, collapsing its +/-40% error bar.

Train split is used deliberately: val, generation and test stay untouched by
calibration, so no held-out subset is contaminated.

Cost: ~$0.07 estimated, and bounded below $0.50 by --max-spend regardless.

    zsh -c 'source ~/.zshrc >/dev/null 2>&1; \
      uv run python scripts/calibrate_rollout.py'

Dry-run first (FREE -- does retrieval and prints exact prompt sizes, no LM call):

    uv run python scripts/calibrate_rollout.py --dry-run

Writes results/calibration/calibration.json, consumed by
`scripts/cost_model.py --measured`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "calibration"
CACHE_DIR = REPO_ROOT / ".cache" / "repos"

#: Hard ceiling. The stopper logic is reused here so a runaway prompt cannot
#: turn a calibration run into a real bill.
DEFAULT_MAX_SPEND_USD = 0.50


def _show(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise.

    `Path.relative_to` raises when the target is outside the repo, and this is
    used inside the failure handler -- so a naive call would crash the very
    code path whose job is to tell you your paid data was retained.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default=None, help="instance_id (default: first in the train manifest)")
    ap.add_argument("--dry-run", action="store_true", help="retrieval + prompt sizing only; NO LM call, free")
    ap.add_argument("--max-spend", type=float, default=DEFAULT_MAX_SPEND_USD)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--profile-prefix",
        default=None,
        help="inference-profile prefix, e.g. 'global.' or 'us.' (default: cost.py INFERENCE_PROFILE_PREFIX)",
    )
    ap.add_argument(
        "--sample", type=int, default=0, help="FREE: size prompts for N train instances (implies --dry-run)"
    )
    args = ap.parse_args()

    from datasets import load_dataset

    from gepa_taxonomy.cost import (
        REFINER_BASE,
        SOLVER_BASE,
        CostMeter,
        price_call,
        with_profile,
    )
    from gepa_taxonomy.cost import REFINER_MODEL as _DEFAULT_REFINER
    from gepa_taxonomy.cost import SOLVER_MODEL as _DEFAULT_SOLVER

    if args.profile_prefix:
        SOLVER_MODEL = with_profile(SOLVER_BASE, args.profile_prefix)
        REFINER_MODEL = with_profile(REFINER_BASE, args.profile_prefix)
    else:
        SOLVER_MODEL, REFINER_MODEL = _DEFAULT_SOLVER, _DEFAULT_REFINER
    from gepa_taxonomy.program import (
        REFINER,
        REFINER_PROMPT,
        SEED_CANDIDATE,
        SOLVER,
        SOLVER_PROMPT,
        extract_patch,
        render_context,
        static_feedback,
    )
    from gepa_taxonomy.retrieval import BM25Retriever
    from gepa_taxonomy.splits import load_manifest
    from gepa_taxonomy.tasks import assert_gold_free, split_row

    train_ids = load_manifest(REPO_ROOT / "manifests" / "swebench_verified" / "train.json")

    if args.sample:
        return _sample_sizes(args, train_ids)

    instance_id = args.instance or train_ids[0]
    if instance_id not in set(train_ids):
        print(
            f"{instance_id} is not in the TRAIN split. Calibration must not touch val/generation/test.", file=sys.stderr
        )
        return 2

    ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
    row = next(r for r in ds if r["instance_id"] == instance_id)
    inst = split_row(row)
    task = inst.task

    print(f"instance : {task.instance_id}  ({task.repo} @ {task.base_commit[:10]})")
    print("split    : train (val/generation/test untouched)\n")

    print("retrieving (clones the repo on first use; free)...")
    retriever = BM25Retriever(cache_dir=CACHE_DIR)
    files = retriever.retrieve(task, k=args.top_k)
    context = render_context(files, max_chars=60_000)

    print(f"  {len(files)} files, {len(context):,} context chars")
    for f in files:
        print(f"    {len(f.content):>8,} chars  {f.path}")

    solver_prompt = SOLVER_PROMPT.format(
        instruction=SEED_CANDIDATE[SOLVER],
        repo=task.repo,
        problem_statement=task.problem_statement,
        context=context,
    )
    assert_gold_free(solver_prompt, where="solver prompt", gold=inst.gold)

    print(f"\nsolver prompt : {len(solver_prompt):,} chars  (~{len(solver_prompt) / 3.6:,.0f} tok est)")
    est_rollout = price_call(SOLVER_MODEL, int(len(solver_prompt) / 3.6), 800) + price_call(
        REFINER_MODEL, int(len(solver_prompt) / 3.6) + 860, 800
    )
    print(f"estimated cost of this rollout: ${est_rollout:.4f}  (ceiling ${args.max_spend:.2f})")

    if args.dry_run:
        print("\n--dry-run: no LM call made, nothing spent.")
        print("Prompt sizes above are already useful -- they replace the CONTEXT_TOKENS")
        print("assumption even before any billed call.")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "dryrun.json").write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "n_files": len(files),
                    "context_chars": len(context),
                    "solver_prompt_chars": len(solver_prompt),
                    "problem_statement_chars": len(task.problem_statement),
                },
                indent=2,
            )
            + "\n"
        )
        return 0

    # ---- billed from here ----
    from gepa_taxonomy.bedrock import BedrockLM, require_credentials

    require_credentials()  # fail fast before any work

    # NOTE: deliberately NOT constructing SolverRefinerProgram here. Building it
    # would instantiate the refiner client up front, so an authorization failure
    # would fire BEFORE the solver call and outside the try -- which is exactly
    # how the previous run lost its paid solver measurement. The refiner client
    # is constructed lazily, inside the guarded block below.

    # Staged, and each stage is persisted the moment it completes.
    #
    # A previous run lost a PAID solver measurement when the refiner 403'd:
    # everything was written only at the very end, so the exception discarded
    # data we had already been billed for. Paid data must never be lost, so the
    # solver result is flushed to disk before the refiner is even attempted.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "calibration.json"

    partial: dict = {
        "instance_id": instance_id,
        "repo": task.repo,
        "n_files_retrieved": len(files),
        "context_chars": len(context),
        "solver_model": SOLVER_MODEL,
        "refiner_model": REFINER_MODEL,
        "estimated_usd": round(est_rollout, 6),
        "complete": False,
    }

    def flush(**updates) -> None:
        partial.update(updates)
        out.write_text(json.dumps(partial, indent=2) + "\n")

    flush()  # provenance on disk before a cent is spent

    solver_meter, refiner_meter = CostMeter(), CostMeter()
    print("\ncalling solver...")
    solver_lm = BedrockLM(model=SOLVER_MODEL)
    raw, s_in, s_out = solver_lm.complete(solver_prompt, max_tokens=4096)
    solver_meter.record(model=SOLVER_MODEL, input_tokens=s_in, output_tokens=s_out)
    solver_patch = extract_patch(raw)
    flush(
        solver_tokens_in=s_in,
        solver_tokens_out=s_out,
        solver_usd=round(solver_meter.budgeted_usd, 6),
        solver_patch_chars=len(solver_patch),
    )
    print(f"  solver  {s_in:,} in / {s_out:,} out  ${solver_meter.budgeted_usd:.4f}  [PERSISTED]")

    feedback = static_feedback(solver_patch)
    refiner_prompt = REFINER_PROMPT.format(
        instruction=SEED_CANDIDATE[REFINER],
        repo=task.repo,
        problem_statement=task.problem_statement,
        context=context,
        patch=solver_patch or "(the solver produced no patch)",
        feedback=feedback.render(),
    )
    assert_gold_free(refiner_prompt, where="refiner prompt", gold=inst.gold)

    print("calling refiner...")
    try:
        raw2, r_in, r_out = BedrockLM(model=REFINER_MODEL).complete(refiner_prompt, max_tokens=4096)
    except Exception as exc:
        flush(refiner_error=f"{type(exc).__name__}: {str(exc)[:300]}")
        print(f"\n  REFINER FAILED: {type(exc).__name__}", file=sys.stderr)
        print(f"  {str(exc)[:300]}", file=sys.stderr)
        print(f"\n  The solver measurement WAS retained: {_show(out)}", file=sys.stderr)
        print(f"  solver {s_in:,} in / {s_out:,} out, ${solver_meter.budgeted_usd:.4f} -- not lost.", file=sys.stderr)
        return 1

    refiner_meter.record(model=REFINER_MODEL, input_tokens=r_in, output_tokens=r_out)
    refiner_patch = extract_patch(raw2)
    total = solver_meter.budgeted_usd + refiner_meter.budgeted_usd
    if total > args.max_spend:
        print(f"WARNING: spend ${total:.4f} exceeded the ${args.max_spend:.2f} ceiling.", file=sys.stderr)

    print("\n" + "=" * 66)
    print("MEASURED")
    print("=" * 66)
    print(f"  solver   {s_in:>8,} in  {s_out:>6,} out   ${solver_meter.budgeted_usd:.4f}")
    print(f"  refiner  {r_in:>8,} in  {r_out:>6,} out   ${refiner_meter.budgeted_usd:.4f}")
    print(f"  ROLLOUT  {s_in + r_in:>8,} in  {s_out + r_out:>6,} out   ${total:.4f}")
    print(f"\n  estimate was ${est_rollout:.4f}  ->  measured ${total:.4f} ({(total / est_rollout - 1) * 100:+.0f}%)")
    print(f"  patch produced: {len(refiner_patch or solver_patch):,} chars, well-formed={feedback.is_well_formed}")

    flush(
        refiner_tokens_in=r_in,
        refiner_tokens_out=r_out,
        refiner_usd=round(refiner_meter.budgeted_usd, 6),
        rollout_usd=round(total, 6),
        refiner_patch_chars=len(refiner_patch),
        complete=True,
    )
    print(f"\nwrote {_show(out)}")
    print("next:  uv run python scripts/cost_model.py --measured")
    return 0


def _sample_sizes(args, train_ids: list[str]) -> int:
    """FREE: measure real prompt sizes across N train instances, no LM calls.

    Spreads the sample across distinct repos so one giant codebase does not
    dominate the picture.
    """
    import statistics as st

    from datasets import load_dataset

    from gepa_taxonomy.program import SEED_CANDIDATE, SOLVER, SOLVER_PROMPT, render_context
    from gepa_taxonomy.retrieval import BM25Retriever
    from gepa_taxonomy.tasks import split_row

    ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
    by_id = {r["instance_id"]: r for r in ds if r["instance_id"] in set(train_ids)}

    picked, seen_repos = [], set()
    for iid in train_ids:
        repo = by_id[iid]["repo"]
        if repo in seen_repos:
            continue
        seen_repos.add(repo)
        picked.append(iid)
        if len(picked) >= args.sample:
            break

    retriever = BM25Retriever(cache_dir=CACHE_DIR)
    rows = []
    print(f"sizing {len(picked)} train instances across {len(seen_repos)} repos (FREE)\n")
    for iid in picked:
        task = split_row(by_id[iid]).task
        try:
            files = retriever.retrieve(task, k=args.top_k)
        except Exception as exc:
            print(f"  {iid:38} SKIPPED ({type(exc).__name__})")
            continue
        context = render_context(files, max_chars=60_000)
        prompt = SOLVER_PROMPT.format(
            instruction=SEED_CANDIDATE[SOLVER],
            repo=task.repo,
            problem_statement=task.problem_statement,
            context=context,
        )
        rows.append(
            {
                "instance_id": iid,
                "repo": task.repo,
                "context_chars": len(context),
                "solver_prompt_chars": len(prompt),
                "problem_chars": len(task.problem_statement),
            }
        )
        print(f"  {iid:38} ctx {len(context):>7,}  prompt {len(prompt):>7,} chars")

    if not rows:
        print("no instances sized")
        return 1
    ctx = [r["context_chars"] for r in rows]
    pr = [r["solver_prompt_chars"] for r in rows]
    print(
        f"\ncontext chars : mean {st.mean(ctx):,.0f}  median {st.median(ctx):,.0f}  min {min(ctx):,}  max {max(ctx):,}"
    )
    print(f"solver prompt : mean {st.mean(pr):,.0f}  median {st.median(pr):,.0f} chars")
    print(f"              ~{st.mean(pr) / 3.6:,.0f} tokens at 3.6 chars/token")
    saturated = sum(1 for c in ctx if c >= 59_000)
    print(f"\n{saturated}/{len(ctx)} instances SATURATE the 60k-char context cap.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "prompt_sizes.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "mean_context_chars": st.mean(ctx),
                "mean_solver_prompt_chars": st.mean(pr),
                "saturated": saturated,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {_show(OUT_DIR / 'prompt_sizes.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
