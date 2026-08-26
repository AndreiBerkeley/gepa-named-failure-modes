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

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BenchmarkPipeline:
    build_script: str
    run_script: str
    base_dir_name: str


BENCHMARKS = {
    "hotpotqa": BenchmarkPipeline("build_hotpotqa_base_val.py", "run_hotpotqa_seed.py", "base_val"),
    "ifbench": BenchmarkPipeline("build_ifbench_base_val.py", "run_ifbench_seed.py", "ifbench_base_val"),
    "hover": BenchmarkPipeline("build_hover_base_val.py", "run_hover_seed.py", "hover_base_val"),
    "livebench-math": BenchmarkPipeline(
        "build_livebench_math_base_val.py", "run_livebench_math_seed.py", "livebench_math_base_val"
    ),
    "appworld": BenchmarkPipeline("build_appworld_base_val.py", "run_appworld_seed.py", "appworld_base_val"),
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
        "--taxonomy-version",
        default="auto-v1",
        help="artifact directory name for a newly generated taxonomy",
    )
    parser.add_argument("--results", type=Path, default=REPO / "results")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--reflection-model", default="us.anthropic.claude-sonnet-4-6")
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
                    str(REPO / "scripts" / pipeline.build_script),
                    "--out",
                    str(base_dir),
                    "--workers",
                    str(args.workers),
                    "--solver-model",
                    args.solver_model,
                    "--max-retries",
                    str(args.max_retries),
                ],
                dry_run=args.dry_run,
                pythonpath=pythonpath,
            )
        else:
            print(f"\nreuse traces: {traces}")

        if not taxonomy_path.exists():
            command = [
                python,
                str(REPO / "scripts" / "generate_taxonomy.py"),
                "--traces",
                str(traces),
                "--out",
                str(taxonomy_dir),
                "--model",
                args.reflection_model,
            ]
            if args.adamast_python:
                command.extend(["--adamast-python", str(args.adamast_python.resolve())])
            _run(command, dry_run=args.dry_run, pythonpath=pythonpath)
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
            str(REPO / "scripts" / pipeline.run_script),
            "--seed",
            str(seed),
            "--budget",
            str(args.budget),
            "--taxonomy",
            str(taxonomy_path),
            "--base-val-cache",
            str(base_cache),
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
