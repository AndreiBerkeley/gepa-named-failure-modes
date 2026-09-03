"""One-command preparation and launch for taxonomy-conditioned GEPA.

The command is unified for the user but deliberately split on disk:

1. evaluate the base candidate and persist its trace bundle;
2. generate and persist a frozen taxonomy;
3. launch GEPA with that taxonomy loaded by the reflection-stage enricher.

Existing artifacts are reused. Supplying ``--taxonomy`` skips the first two
phases, which preserves the package's swappable stage boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from gepa_taxonomy.cost import assert_priced, load_price_overrides

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BenchmarkPipeline:
    build_script: str
    run_script: str
    base_dir_name: str


#: Stage scripts live where the benchmark does: generic stages in scripts/,
#: each benchmark's harvest and run in its own demo/template directory.
BENCHMARKS = {
    "ifbench": BenchmarkPipeline("demo/build_ifbench_base_val.py", "demo/run_ifbench_seed.py", "ifbench_base_val"),
}


def _command_text(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def _run(command: list[str], *, dry_run: bool, pythonpath: str | None = None) -> None:
    print(f"\n$ {_command_text(command)}", flush=True)
    if dry_run:
        return
    environment = {**os.environ, "PYTHONUTF8": "1"}
    if pythonpath:
        environment["PYTHONPATH"] = pythonpath
    subprocess.run(command, cwd=REPO, env=environment, check=True)


ADAMAST_PROVIDER_BY_PREFIX = {
    "bedrock": "bedrock",
    "gemini": "google",
    "openai": "openai",
    "anthropic": "anthropic",
}


def _adamast_model(reflection_model: str) -> tuple[str, str]:
    """Map a litellm reflection-model id to AdaMAST's (provider, model)."""
    if "/" in reflection_model:
        prefix, bare = reflection_model.split("/", 1)
        provider = ADAMAST_PROVIDER_BY_PREFIX.get(prefix)
        if provider is None:
            known = ", ".join(sorted(ADAMAST_PROVIDER_BY_PREFIX))
            raise SystemExit(
                f"cannot map provider prefix {prefix!r} to an AdaMAST provider; known prefixes: {known}"
            )
        return provider, bare
    return "openai", reflection_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=sorted(BENCHMARKS))
    parser.add_argument("--seed", type=int, action="append", required=True, help="repeat to run more than one seed")
    parser.add_argument("--budget", type=float, required=True, help="optimization budget in dollars for each seed")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        help="existing taxonomy.json; skips trace harvest and taxonomy generation",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=2,
        help="reduction: drop codes cited in fewer distinct generation traces than this",
    )
    parser.add_argument(
        "--max-codes",
        type=int,
        default=25,
        help="reduction safety net applied after --min-support, by support ranking",
    )
    parser.add_argument(
        "--taxonomy-version",
        default="auto-v1",
        help="artifact directory name for a newly generated taxonomy",
    )
    parser.add_argument("--results", type=Path, default=REPO / "results")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--solver-model", default="gpt-5-mini")
    parser.add_argument("--reflection-model", default="gpt-5-mini")
    parser.add_argument(
        "--price",
        action="append",
        default=[],
        metavar="MODEL=IN,OUT",
        help="price for a model litellm's table does not know, in USD per million input,output tokens; repeatable",
    )
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--adamast-python", type=Path, default=None)
    parser.add_argument(
        "--gepa-root",
        type=Path,
        default=None,
        help="GEPA checkout containing the reflective_dataset_enricher hook",
    )
    parser.add_argument("--prepare-only", action="store_true", help="stop after producing taxonomy.json")
    parser.add_argument("--dry-run", action="store_true", help="print every phase without spending or writing")
    parser.add_argument(
        "--run-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra argument forwarded to every benchmark run; repeat for multiple arguments",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_price_overrides(args.price)
    assert_priced(args.solver_model, args.reflection_model)
    price_args = [x for spec in args.price for x in ("--price", spec)]
    pipeline = BENCHMARKS[args.benchmark]
    python = sys.executable
    results = args.results.resolve()
    base_dir = results / pipeline.base_dir_name
    base_cache = base_dir / "base_val_cache.json"
    traces = base_dir / "base_val.traces.jsonl"
    pythonpath = None
    if args.gepa_root:
        gepa_source = args.gepa_root.resolve() / "src"
        api_source = gepa_source / "gepa" / "api.py"
        if not api_source.exists() and not args.dry_run:
            raise SystemExit(f"GEPA source not found under {gepa_source}")
        if api_source.exists() and "reflective_dataset_enricher" not in api_source.read_text(encoding="utf-8"):
            raise SystemExit(
                f"{args.gepa_root} does not contain the optimizer-side hook. "
                "Apply patches/gepa-reflective-dataset-enricher.patch first."
            )
        inherited = os.environ.get("PYTHONPATH")
        pythonpath = os.pathsep.join(part for part in (str(gepa_source), str(REPO / "src"), inherited) if part)

    taxonomy_path = args.taxonomy.resolve() if args.taxonomy else None
    if taxonomy_path is None:
        taxonomy_dir = results / "taxonomy" / f"{args.benchmark}-{args.taxonomy_version}"
        taxonomy_path = taxonomy_dir / "taxonomy.json"

        if not traces.exists():
            if base_cache.exists() and not args.dry_run:
                raise SystemExit(
                    f"incomplete base artifact: {base_cache} exists but {traces} does not. "
                    "Move the incomplete directory or rebuild it explicitly with --force."
                )
            _run(
                [
                    python,
                    str(REPO / pipeline.build_script),
                    "--out",
                    str(base_dir),
                    "--workers",
                    str(args.workers),
                    "--solver-model",
                    args.solver_model,
                    "--max-retries",
                    str(args.max_retries),
                    *price_args,
                ],
                dry_run=args.dry_run,
                pythonpath=pythonpath,
            )
        else:
            print(f"\nreuse traces: {traces}")

        provider, bare_model = _adamast_model(args.reflection_model)
        if not taxonomy_path.exists():
            command = [
                python,
                str(REPO / "scripts" / "generate_taxonomy.py"),
                "--traces",
                str(traces),
                "--out",
                str(taxonomy_dir),
                "--model",
                bare_model,
                "--provider",
                provider,
            ]
            if args.adamast_python:
                command.extend(["--adamast-python", str(args.adamast_python.resolve())])
            _run(command, dry_run=args.dry_run, pythonpath=pythonpath)

        # Reduction: generation ran stock, so the frozen artifact is produced
        # here, from measured support. Idempotent: a reduced taxonomy carries a
        # "reduction" marker and is reused as-is.
        if taxonomy_path.exists():
            needs_reduction = "reduction" not in json.loads(taxonomy_path.read_text(encoding="utf-8-sig"))
        else:
            needs_reduction = True  # dry run: generation wrote nothing; show the full plan
        if needs_reduction:
            judgements = taxonomy_dir / "judgements.jsonl"
            if not judgements.exists():
                command = [
                    python,
                    str(REPO / "scripts" / "judge_corpus.py"),
                    "--taxonomy",
                    str(taxonomy_path),
                    "--traces",
                    str(traces),
                    "--out",
                    str(judgements),
                    "--model",
                    bare_model,
                    "--provider",
                    provider,
                    "--workers",
                    str(args.workers),
                    *price_args,
                ]
                if args.adamast_python:
                    command.extend(["--adamast-python", str(args.adamast_python.resolve())])
                _run(command, dry_run=args.dry_run, pythonpath=pythonpath)
            _run(
                [
                    python,
                    str(REPO / "scripts" / "reduce_taxonomy.py"),
                    "--taxonomy",
                    str(taxonomy_path),
                    "--judgements",
                    str(judgements),
                    "--min-support",
                    str(args.min_support),
                    "--max-codes",
                    str(args.max_codes),
                ],
                dry_run=args.dry_run,
                pythonpath=pythonpath,
            )
        else:
            print(f"reuse taxonomy: {taxonomy_path}")
    elif not taxonomy_path.exists() and not args.dry_run:
        raise SystemExit(f"taxonomy not found: {taxonomy_path}")

    if args.prepare_only:
        print(f"\nprepared taxonomy: {taxonomy_path}")
        return 0

    for seed in args.seed:
        run_dir = results / "runs" / f"{args.benchmark}-taxonomy-seed{seed}"
        command = [
            python,
            str(REPO / pipeline.run_script),
            "--seed",
            str(seed),
            "--budget",
            str(args.budget),
            "--taxonomy",
            str(taxonomy_path),
            "--base-val-cache",
            # The shared base evaluation exists only when this pipeline ran the
            # harvest. A user bringing their own taxonomy has no cache to
            # replay, and that is fine: the run evaluates the base program
            # itself instead of replaying a shared measurement.
            str(base_cache) if base_cache.exists() else "none",
            "--out",
            str(run_dir),
            "--workers",
            str(args.workers),
            "--solver-model",
            args.solver_model,
            "--reflection-model",
            args.reflection_model,
            "--max-retries",
            str(args.max_retries),
            *price_args,
            *args.run_arg,
        ]
        _run(command, dry_run=args.dry_run, pythonpath=pythonpath)

    if not args.dry_run:
        manifest = {
            "benchmark": args.benchmark,
            "seeds": args.seed,
            "budget_usd_per_seed": args.budget,
            "base_trace_bundle": str(traces),
            "taxonomy": str(taxonomy_path),
            "solver_model": args.solver_model,
            "reflection_and_judge_model": args.reflection_model,
        }
        manifest_path = results / "runs" / f"{args.benchmark}-taxonomy-pipeline.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"\npipeline manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
