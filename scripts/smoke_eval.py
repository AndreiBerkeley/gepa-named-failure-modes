#!/usr/bin/env python
"""Smoke-test the SWE-Bench evaluation harness. FREE -- no LM calls, no API spend.

Uses the **gold patch as the prediction**. A correct harness must mark those
instances RESOLVED, so this validates the whole evaluation path -- image build,
container run, test selection, report parsing -- end to end without spending a
cent on inference.

    uv run python scripts/smoke_eval.py --instances 2

Host-agnostic: every knob (workers, cache level, run id, dataset) is a flag or
env var, so the same script runs against local Docker on this arm64 Mac and
against the remote x86_64 Linux box later.

    SWEBENCH_MAX_WORKERS=4 uv run python scripts/smoke_eval.py

Note on arm64: swebench 4.1.0 hardcodes ``arch="x86_64"`` in ``make_test_spec``
and never consults ``platform.machine()``, so it pulls the **official x86_64
images** and runs them under Docker Desktop's amd64 emulation. Verified working
on this Mac (1/1 gold resolved). Environments are therefore leaderboard-faithful;
the cost is emulation speed, not fidelity.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "manifests" / "swebench_verified" / "val.json"


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def preflight() -> list[str]:
    """Report blocking problems rather than failing deep inside the harness."""
    problems: list[str] = []

    if shutil.which("docker") is None:
        problems.append("docker not found on PATH -- no container runtime available")
    else:
        try:
            subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=30)
        except subprocess.CalledProcessError:
            problems.append("`docker info` failed -- is Docker Desktop running?")
        except (subprocess.TimeoutExpired, OSError) as exc:
            problems.append(f"could not query docker: {exc}")

    try:
        import swebench  # noqa: F401
    except ImportError:
        problems.append("swebench not installed. This is an optional extra:\n      uv sync --extra swebench")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", type=int, default=2, help="how many instances to smoke-test")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--run-id", default="smoke-gold")
    ap.add_argument("--max-workers", type=int, default=env_int("SWEBENCH_MAX_WORKERS", 2))
    ap.add_argument(
        "--cache-level",
        default=os.environ.get("SWEBENCH_CACHE_LEVEL", "env"),
        choices=["none", "base", "env", "instance"],
    )
    ap.add_argument("--timeout", type=int, default=env_int("SWEBENCH_TIMEOUT", 1800))
    ap.add_argument("--dry-run", action="store_true", help="write predictions and print the command, do not run")
    args = ap.parse_args()

    print(f"host: {platform.system()}/{platform.machine()}")
    if platform.machine() in ("arm64", "aarch64"):
        print("  NOTE: arm64 -- swebench hardcodes arch=x86_64, so the OFFICIAL images")
        print("        are pulled and run under amd64 emulation. Faithful, but slow.")

    problems = preflight()
    if problems:
        print("\nPREFLIGHT FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2
    print("preflight: ok\n")

    from datasets import load_dataset

    from gepa_taxonomy.splits import load_manifest

    ids = load_manifest(args.manifest)[: args.instances]
    print(f"instances ({len(ids)}): {', '.join(ids)}\n")

    ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
    gold_by_id = dict(zip(ds["instance_id"], ds["patch"]))

    # Gold patch as the prediction: a correct harness marks these RESOLVED.
    out_dir = REPO_ROOT / "results" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"{args.run_id}-predictions.json"
    preds = [
        {
            "instance_id": iid,
            "model_name_or_path": "gold",
            "model_patch": gold_by_id[iid],
        }
        for iid in ids
    ]
    preds_path.write_text(json.dumps(preds, indent=2))
    print(f"wrote gold predictions -> {preds_path.relative_to(REPO_ROOT)}")

    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        "SWE-bench/SWE-bench_Verified",
        "--split",
        "test",
        "--predictions_path",
        str(preds_path),
        "--run_id",
        args.run_id,
        "--max_workers",
        str(args.max_workers),
        "--cache_level",
        args.cache_level,
        "--timeout",
        str(args.timeout),
        "--instance_ids",
        *ids,
    ]
    print("\ncommand:\n  " + " ".join(cmd) + "\n")
    if args.dry_run:
        print("--dry-run: not executing")
        return 0

    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        print(f"\nharness exited {proc.returncode}", file=sys.stderr)
        return proc.returncode

    reports = sorted(REPO_ROOT.glob(f"*{args.run_id}*.json"))
    print("\nreports:", [str(p.name) for p in reports] or "none found")
    for p in reports:
        data = json.loads(p.read_text())
        resolved = data.get("resolved_instances")
        total = data.get("total_instances")
        print(f"  {p.name}: resolved {resolved}/{total}")
        if resolved != total:
            print("  WARNING: gold patches did not all resolve -- the harness or the")
            print("           environment is not behaving correctly.")
            return 1
    print("\nSMOKE TEST PASSED: gold patches resolve; the evaluation path works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
