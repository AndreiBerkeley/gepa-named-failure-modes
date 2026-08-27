#!/usr/bin/env python
"""Run one AppWorld seed. **This spends money.** Launch it deliberately.

Single-component ReAct agent, seeded with the published instruction. The
baseline arm is unmodified gepa v0.1.4 plus the dollar-budget stopper; the
treatment arm adds ``--taxonomy PATH`` and differs by exactly one key in the
reflective dataset.

The AppWorld server
-------------------
AppWorld cannot run in this process: it pins ``pydantic <2`` against gepa's and
litellm's v2, and its executor calls ``signal.SIGALRM``, which does not exist on
Windows. It therefore runs in WSL, in its own venv, and we talk to it over
HTTP. This script starts it if it is not already up, and leaves it running
afterwards so a chain of seeds pays the startup once.

    PYTHONUTF8=1 uv run python scripts/run_appworld_seed.py --seed 1 --budget 60
"""

from __future__ import annotations

import argparse
import atexit
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

WSL_DISTRO = "Ubuntu-24.04"
DEFAULT_BASE_URL = "http://localhost:8123"


def ensure_servers(base_port: int, n: int, wait_s: int = 120) -> int:
    """Start ``n`` AppWorld servers, one per worker, on consecutive ports.

    One server per worker is mandatory, not an optimisation: the server keeps its
    task world in a module-level global and rejects requests for any other task
   , so two workers sharing a server clobber each other and two of three
    rollouts die. Idempotent -- already-running ports are left alone, so a chain
    of seeds pays startup once.
    """
    started = 0
    for offset in range(n):
        if ensure_server(f"http://localhost:{base_port + offset}", base_port + offset, wait_s):
            started += 1
    return started


def ensure_server(base_url: str, port: int, wait_s: int = 120) -> bool:
    """Start one AppWorld environment server in WSL unless that port is up."""
    from gepa_taxonomy.appworld.client import AppWorldClient

    probe = AppWorldClient(base_url=base_url)
    if probe.health():
        print(f"AppWorld server already up at {base_url}")
        return False

    print(f"starting AppWorld server in WSL ({WSL_DISTRO}) on port {port} ...", flush=True)
    # ``setsid`` plus a lingering ``sleep`` are both required, and neither is
    # decoration. WSL tears the session down when the launching command exits,
    # which kills a plain ``nohup ... &`` child mid-startup: the redirect creates
    # an empty log and the server never binds. ``setsid`` detaches it into its own
    # session; the ``sleep`` holds the WSL session open long enough for that to
    # finish. Diagnosed from a zero-byte /tmp/aw_env.log and a clean exit code.
    subprocess.Popen(
        [
            "wsl",
            "-d",
            WSL_DISTRO,
            "--",
            "bash",
            "-lc",
            f"cd ~/appworld && setsid nohup ./.venv/bin/appworld serve environment "
            f"--port {port} > /tmp/aw_env.log 2>&1 < /dev/null & sleep 8",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + wait_s
    while time.time() < deadline:
        if probe.health():
            print(f"  server up after {wait_s - int(deadline - time.time())}s")
            return True
        time.sleep(2)
    raise SystemExit(
        f"AppWorld server did not come up at {base_url} within {wait_s}s.\n"
        f"  check:  wsl -d {WSL_DISTRO} -- tail -20 /tmp/aw_env.log"
    )


def load_task_ids(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Sorted in the manifest already; gepa keys val subscores and the Pareto
    # frontier POSITIONALLY, so the order is load-bearing.
    return list(manifest["task_ids"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--log-reflection-datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="append every post-enrichment reflective dataset to "
        "reflection_datasets.jsonl in the run dir, exactly as reflection "
        "consumed it (observability; disable with --no-log-reflection-datasets)",
    )
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--taxonomy", type=Path, default=None, help="enable the treatment arm")
    parser.add_argument("--minibatch-size", type=int, default=10)
    parser.add_argument("--manifests", type=Path, default=REPO / "manifests" / "appworld")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--reflection-model", default="us.anthropic.claude-sonnet-4-6")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="tasks evaluated concurrently. Lower than HotpotQA's 8: each AppWorld "
        "rollout holds a live environment on the server and makes many sequential "
        "LM calls, so concurrency multiplies server load as well as API load.",
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--max-transport-errors", type=int, default=25)
    parser.add_argument(
        "--base-val-cache",
        type=Path,
        default=REPO / "results" / "appworld_base_val" / "base_val_cache.json",
        help="shared base-candidate val evaluation, replayed so every seed and both "
        "arms start byte-identically. Build it with "
        "scripts/build_appworld_base_val.py. Pass 'none' to disable.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import gepa

    from gepa_taxonomy.appworld.adapter import AppWorldAdapter, client_factory
    from gepa_taxonomy.appworld.program import ReActProgram
    from gepa_taxonomy.appworld.prompts import COMPONENTS, SEED_CANDIDATE
    from gepa_taxonomy.bedrock import (
        BedrockLM,
        MeteredReflectionLM,
        require_credentials,
        verify_reflection_lm,
    )
    from gepa_taxonomy.cost import CostMeter, MaxTotalCostStopper
    from gepa_taxonomy.seed_cache import SeedEvaluationCache

    arm = "taxonomy" if args.taxonomy else "baseline"
    out = args.out or REPO / "results" / "runs" / f"appworld-{arm}-seed{args.seed}"

    # gepa RESUMES silently from an existing run_dir (api.py:176), inheriting the
    # old run's scores and candidate pool -- invisibly, because the base
    # candidate is never re-evaluated.
    state = out / "gepa_state.bin"
    if state.exists() and not args.resume:
        raise SystemExit(
            f"REFUSING TO START: {state} already exists.\n"
            f"gepa would silently RESUME from it. Archive the directory, or pass --resume."
        )
    out.mkdir(parents=True, exist_ok=True)

    # gepa's logger writes the run log with the platform default encoding, and a
    # proposed prompt containing an emoji kills the run on Windows.
    import locale

    if (locale.getpreferredencoding(False) or "").lower() not in {"utf-8", "utf8"}:
        raise SystemExit(
            f"REFUSING TO START: default encoding is {locale.getpreferredencoding(False)!r}.\n"
            f"  relaunch with:  PYTHONUTF8=1 uv run python {Path(__file__).name} ..."
        )

    require_credentials()
    ensure_servers(args.port, args.workers)

    train = load_task_ids(args.manifests / "train.json")
    val = load_task_ids(args.manifests / "val.json")
    print(f"arm={arm} seed={args.seed} budget=${args.budget:.2f} train={len(train)} val={len(val)}")

    solver_meter = CostMeter(spend_log=out / "spend.solver.json")
    reflection_meter = CostMeter(spend_log=out / "spend.reflection.json")
    judge_meter = CostMeter(spend_log=out / "spend.judge.json")

    # Meters snapshot to disk every 25 records; a short run can end before the
    # first snapshot, leaving spend files missing. Exit-time flush makes the
    # on-disk record exhaustive, and the heartbeat keeps quiet phases visible.
    for _meter in (solver_meter, reflection_meter, judge_meter):
        atexit.register(_meter.flush)
    from gepa_taxonomy.progress import report_spend

    report_spend((solver_meter, reflection_meter, judge_meter))

    program = ReActProgram(
        client=None,  # replaced per worker by the factory below
        lm=BedrockLM(model=args.solver_model, max_retries=args.max_retries),
        meter=solver_meter,
        model=args.solver_model,
        max_steps=args.max_steps,
    )
    seed_cache = None
    if str(args.base_val_cache).lower() != "none":
        if not args.base_val_cache.exists():
            raise SystemExit(
                f"base-val cache not found: {args.base_val_cache}\n"
                "Every seed and both arms must start from the SAME base-candidate\n"
                "evaluation, or the paired comparison carries an extra draw of noise\n"
                "on top of the treatment.\n\n"
                "  build it:  PYTHONUTF8=1 uv run python scripts/build_appworld_base_val.py\n"
                "  or opt out deliberately:  --base-val-cache none"
            )
        seed_cache = SeedEvaluationCache.load(args.base_val_cache)
        if not seed_cache.matches(dict(SEED_CANDIDATE)):
            raise SystemExit(
                "base-val cache was built for a DIFFERENT seed candidate.\n"
                "Rebuild: scripts/build_appworld_base_val.py --force"
            )
        # Checked ONCE against the val manifest, not per lookup: a TRAIN-task
        # miss is legitimate and crashed a run when treated as an error.
        seed_cache.assert_covers(val)
        print(f"base-val cache: {len(seed_cache.entries)} val tasks will be replayed (no spend)")

    adapter = AppWorldAdapter(
        program=program,
        client_factory=client_factory(args.port, args.workers, prefix=f"{arm}-s{args.seed}"),
        max_workers=args.workers,
        max_transport_errors=args.max_transport_errors,
        seed_cache=seed_cache,
    )

    meters = [solver_meter, reflection_meter]
    taxonomy_feedback = None
    if args.taxonomy:
        import inspect

        from failure_taxonomy import JudgeCache, LLMFailureJudge, TaxonomyFeedbackEnricher, load_taxonomy

        if "reflective_dataset_enricher" not in inspect.signature(gepa.optimize).parameters:
            raise SystemExit(
                "taxonomy runs require GEPA's optimizer-side reflective_dataset_enricher hook. "
                "Apply patches/gepa-reflective-dataset-enricher.patch to the pinned GEPA checkout."
            )

        judge_lm = BedrockLM(model=args.reflection_model, max_retries=args.max_retries)
        taxonomy = load_taxonomy(args.taxonomy)

        def judge_call(prompt: str) -> str:
            text, tin, tout = judge_lm.complete(prompt, max_tokens=4096)
            judge_meter.record(
                model=args.reflection_model, input_tokens=tin, output_tokens=tout, phase="optimization"
            )
            return text

        taxonomy_feedback = TaxonomyFeedbackEnricher(
            judge=LLMFailureJudge(taxonomy=taxonomy, lm=judge_call, cache=JudgeCache.open(out / "judge_cache.jsonl")),
        )
        # Judge spend competes for the SAME budget as rollouts and reflection;
        # that trade is what the comparison measures.
        meters.append(judge_meter)
        print(f"taxonomy: {args.taxonomy} ({len(taxonomy)} codes, {taxonomy.fingerprint})")

    reflection_lm = MeteredReflectionLM(
        lm=BedrockLM(model=args.reflection_model, max_retries=args.max_retries),
        meter=reflection_meter,
        model=args.reflection_model,
        spend_log=out / "reflection_spend.jsonl",
    )
    print(f"reflection LM preflight: {verify_reflection_lm(reflection_lm)}")

    stopper = MaxTotalCostStopper(args.budget, meters)

    optimize_kwargs = {}
    if args.log_reflection_datasets:
        from gepa_taxonomy.reflection_log import ReflectionDatasetLogger

        optimize_kwargs["callbacks"] = [ReflectionDatasetLogger(out / "reflection_datasets.jsonl")]
    if taxonomy_feedback is not None:
        optimize_kwargs["reflective_dataset_enricher"] = taxonomy_feedback

    started = time.time()
    result = gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=args.minibatch_size,
        stop_callbacks=[stopper],
        seed=args.seed,
        display_progress_bar=True,
        run_dir=str(out),
        **optimize_kwargs,
    )
    elapsed = time.time() - started

    summary = {
        "arm": arm,
        "seed": args.seed,
        "benchmark": "appworld",
        "budget_usd": args.budget,
        "minibatch_size": args.minibatch_size,
        "max_steps": args.max_steps,
        "components": list(COMPONENTS),
        "elapsed_hours": round(elapsed / 3600, 3),
        "best_val_score": max(result.val_aggregate_scores) if result.val_aggregate_scores else None,
        "best_candidate_index": (
            max(range(len(result.val_aggregate_scores)), key=result.val_aggregate_scores.__getitem__)
            if result.val_aggregate_scores
            else None
        ),
        "val_aggregate_scores": list(result.val_aggregate_scores or []),
        "candidates": len(result.candidates),
        "spend": {
            "solver": solver_meter.snapshot(),
            "reflection": reflection_meter.snapshot(),
            "judge": judge_meter.snapshot(),
        },
        "adapter": adapter.summary(),
        "taxonomy_feedback": taxonomy_feedback.summary() if taxonomy_feedback is not None else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "candidates.json").write_text(json.dumps(result.candidates, indent=2) + "\n", encoding="utf-8")

    print(f"\nbest val: {summary['best_val_score']}  candidates: {summary['candidates']}")
    print(f"realised: ${sum(m.budgeted_usd for m in (solver_meter, reflection_meter, judge_meter)):.2f}")
    print(f"written : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
