#!/usr/bin/env python
"""Generate a failure taxonomy from segmented traces. **This spends money.**

Runs the **public** AdaMAST (D047) from its own sibling checkout, through its own
interpreter, as a subprocess. It is not a dependency of this project and must not
become one: it pulls its own ``openai``/``pydantic`` floors, and re-resolving this
venv mid-experiment would change the environment the baseline seeds run out of
(D032).

Benchmark-agnostic: it takes a trace file, so the same command serves HotpotQA
and AppWorld.

    PYTHONUTF8=1 uv run python scripts/generate_taxonomy.py \
        --traces results/base_val/base_val.traces.jsonl \
        --out results/taxonomy/hotpotqa_v1

The output ``taxonomy.json`` is the stage boundary: ``failure_taxonomy`` reads it
and needs nothing else, so anyone bringing their own taxonomy skips this entirely.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Sibling checkout of the PUBLIC AdaMAST, branch agent/baseline-taxonomy-generation
#: (D047). Its own venv, mirroring how the pinned gepa clone is handled.
ADAMAST_ROOT = REPO.parent / "adamast-public"
ADAMAST_PYTHON = next(
    (
        path
        for path in (
            ADAMAST_ROOT / ".venv" / "bin" / "python",
            ADAMAST_ROOT / ".venv" / "Scripts" / "python.exe",
        )
        if path.exists()
    ),
    ADAMAST_ROOT / ".venv" / "bin" / "python",
)

WORKER = """
import json, sys
from adamast import generate_taxonomy, load_trace_bundle

req = json.load(sys.stdin)

# Stock configuration on purpose: reduction happens AFTER generation, from
# measured trace support (scripts/reduce_taxonomy.py), never by asking the
# generator to compress into a count it has no evidence for.

bundle = load_trace_bundle(req["traces"])
report = bundle.report()
if report.get("empty_trajectories"):
    raise SystemExit(
        f"REFUSING: {report['empty_trajectories']} traces have an empty trajectory. "
        "AdaMAST cannot generate from them and discovering it after paying is the "
        "expensive way to find out."
    )
print(f"traces validated: {report}", file=sys.stderr, flush=True)

result = generate_taxonomy(
    req["traces"],
    req["output"],
    provider=req["provider"],
    model=req.get("model"),
    max_rounds=req.get("max_rounds", 5),
    kappa_target=req.get("kappa_target", 0.75),
    coverage_floor=req.get("coverage_floor", 0.7),
    max_output_tokens=req.get("max_output_tokens", 8192),
    aws_region=req.get("aws_region"),
)
print(json.dumps(result, default=str))
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider", default="bedrock")
    parser.add_argument("--model", default="us.anthropic.claude-sonnet-4-6")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--kappa-target", type=float, default=0.75)
    parser.add_argument("--coverage-floor", type=float, default=0.7)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=1800,
        help="kill generation after this many seconds without a line of output. "
        "A hung provider call (observed 2026-08-26: a Gemini request with no "
        "client timeout) produces exactly this signature: zero CPU, zero output.",
    )
    parser.add_argument("--adamast-python", type=Path, default=ADAMAST_PYTHON)
    args = parser.parse_args()

    if not args.traces.exists():
        raise SystemExit(f"traces not found: {args.traces}")
    if not Path(args.adamast_python).exists():
        raise SystemExit(
            f"public AdaMAST interpreter not found: {args.adamast_python}\n"
            "Clone and install it as a sibling (D047):\n"
            "  git clone --branch agent/baseline-taxonomy-generation "
            "https://github.com/multi-agent-systems-failure-taxonomy/AdaMAST.git ../adamast-public\n"
            '  cd ../adamast-public && uv venv --python 3.12 && uv pip install -e ".[bedrock,google]"\n'
            "The [bedrock] extra is REQUIRED. Without it AdaMAST imports cleanly and then\n"
            "raises ProviderConfigurationError on the first provider call -- AFTER trace\n"
            "validation has already passed, so the failure looks like a data problem."
        )
    if (args.out / "taxonomy.json").exists():
        raise SystemExit(
            f"REFUSING: {args.out / 'taxonomy.json'} already exists.\n"
            "Regenerating in place would silently replace the taxonomy every judged "
            "run was keyed against. Use a new --out directory."
        )

    # Record which AdaMAST produced this, so the writeup's reproducibility claim
    # is checkable rather than asserted.
    provenance = subprocess.run(
        ["git", "-C", str(ADAMAST_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    request = {
        "traces": str(args.traces.resolve()),
        "output": str(args.out.resolve()),
        "provider": args.provider,
        "model": args.model,
        "max_rounds": args.max_rounds,
        "kappa_target": args.kappa_target,
        "coverage_floor": args.coverage_floor,
        "max_output_tokens": args.max_output_tokens,
        "aws_region": args.aws_region,
    }

    print(f"AdaMAST : {ADAMAST_ROOT} @ {provenance[:12]}")
    print(f"traces  : {args.traces}")
    print(f"model   : {args.model}  kappa>={args.kappa_target}  rounds<={args.max_rounds}")
    print("configuration: stock AdaMAST; reduction happens after generation from measured support")
    print("generating; streaming AdaMAST progress ...", flush=True)

    # Stream the worker's output live, so every draft step and agreement round
    # is visible as it happens. Two guards replace the old blanket timeout: the
    # 4-hour ceiling (measured 2026-08-09: agreement alone can be ~1,200 serial
    # LLM calls and needs every round -- a 2-hour ceiling cut it too fine), and
    # an idle kill for the silent-hang signature a stuck provider call leaves.
    proc = subprocess.Popen(
        [str(args.adamast_python), "-u", "-c", WORKER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps(request))
    proc.stdin.close()

    state = {"last": time.monotonic()}
    start = state["last"]
    killed: list[str] = []

    def _guard() -> None:
        while proc.poll() is None:
            now = time.monotonic()
            if now - start > 14400:
                killed.append("4-hour ceiling reached")
            elif now - state["last"] > args.idle_timeout:
                killed.append(f"no output for {args.idle_timeout:.0f}s; likely a hung provider call")
            else:
                time.sleep(15)
                continue
            proc.kill()
            return

    threading.Thread(target=_guard, daemon=True).start()
    for line in proc.stdout:
        state["last"] = time.monotonic()
        print(f"  adamast | {line.rstrip()}", flush=True)
    returncode = proc.wait()
    if killed:
        raise SystemExit(f"generation killed: {killed[0]}")
    if returncode != 0:
        raise SystemExit(f"generation failed (exit {returncode})")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                "adamast_repo": str(ADAMAST_ROOT),
                "adamast_commit": provenance,
                "adamast_branch": "agent/baseline-taxonomy-generation",
                "traces": str(args.traces),
                "request": request,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    taxonomy_path = args.out / "taxonomy.json"
    if taxonomy_path.exists():
        codes = json.loads(taxonomy_path.read_text(encoding="utf-8-sig")).get("codes", [])
        print(f"\n  taxonomy: {len(codes)} codes -> {taxonomy_path}")
        by_category: dict[str, int] = {}
        for c in codes:
            by_category[str(c.get("category"))] = by_category.get(str(c.get("category")), 0) + 1
        print(f"  categories: {dict(sorted(by_category.items()))}")
        scoped = sum(1 for c in codes if c.get("applies_to_role"))
        print(f"  role-scoped: {scoped}/{len(codes)}  (watch this: 4/22 last time, F018)")
    print(f"  provenance: {args.out / 'provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
