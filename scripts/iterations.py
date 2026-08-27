#!/usr/bin/env python
"""Per-iteration view of a seed run: parent, minibatch, component, outcome.

gepa's own logging spreads one iteration across several lines and omits the
minibatch scores entirely; they land in run_log.json instead. This joins the
two so an iteration reads as one row. Read-only -- it never touches the run.

    uv run python scripts/iterations.py              # table so far
    uv run python scripts/iterations.py --follow     # live, appends as it goes
    uv run python scripts/iterations.py --instances  # show instance ids, not indices
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1] / "results/runs/baseline-seed1"
COMPONENT_RE = re.compile(r"^Iteration (\d+): (?:Proposed new text for|proposed for) (\w+)", re.MULTILINE)
ACCEPT_RE = re.compile(r"^Iteration (\d+): (?:Accepted candidate|.*BETTER, accepted)", re.MULTILINE)
VALSCORE_RE = re.compile(r"^Iteration (\d+): Individual valset scores.*?\{(.*?)\}", re.MULTILINE | re.DOTALL)
# REMOVED: VALSCORE_COMPACT_RE, which was meant to read a compact
# "11/60 resolved" form from QuietLogger. It never matched anything -- verified
# against every log in results/, including the new HotpotQA runs: 0 matches.
# gepa does not emit that form, so the fallback it fed was dead code pretending
# to be a safety net.


def _console_facts(run: Path) -> tuple[dict[int, str], set[int], dict[int, float]]:
    """Components edited, iterations accepted, and val scores -- from the log."""
    text = ""
    for name in ("console.log", "gepa.log"):  # gepa.log exists even without tee
        f = run / name
        if f.exists():
            text += f.read_text(errors="replace")
    components = {int(i): c for i, c in COMPONENT_RE.findall(text)}
    accepted = {int(i) for i in ACCEPT_RE.findall(text)}
    val: dict[int, float] = {}
    for i, body in VALSCORE_RE.findall(text):
        scores = [float(v.split(":")[1]) for v in body.split(",") if ":" in v]
        if scores:
            val[int(i)] = sum(scores) / len(scores)
    return components, accepted, val


def rows(run: Path, names: list[str] | None) -> list[str]:
    log = json.loads((run / "run_log.json").read_text())
    components, accepted, val = _console_facts(run)
    out = []
    for it in log:
        i = it["i"]
        parent = it.get("selected_program_candidate")
        before = it.get("subsample_scores") or []
        after = it.get("new_subsample_scores") or []
        ids = it.get("subsample_ids") or []
        if not before and not after:
            continue  # iteration produced no candidate to compare
        shown = [names[j] if names and j < len(names) else str(j) for j in ids]
        b, a = sum(before), sum(after)
        # gepa accepts only a STRICT improvement; ties are rejected.
        mark = "ACCEPTED" if (i + 1) in accepted else ("tie" if a == b else ("worse" if a < b else "better"))
        line = (
            f"it {i:>3} | parent cand {parent} | edited {components.get(i + 1, '?'):<20}"
            f" | minibatch {b:.0f}/{len(before)} -> {a:.0f}/{len(after)} | {mark}"
        )
        if (i + 1) in val:
            v = val[i + 1]
            line += f" | full val {v:.3f}"
        out.append(line + ("   [" + ", ".join(shown) + "]" if names else ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("run_pos", nargs="?", type=Path, help="run dir (positional alternative to --run)")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--instances", action="store_true", help="show instance ids instead of train indices")
    args = ap.parse_args()
    if args.run_pos:
        args.run = args.run_pos

    names = None
    if args.instances:
        man = args.run.parents[2] / "manifests/swebench_verified/train.json"
        names = json.loads(man.read_text())

    seen = 0
    while True:
        try:
            lines = rows(args.run, names)
        except FileNotFoundError:
            lines = []
        for line in lines[seen:]:
            print(line, flush=True)
        seen = len(lines)
        if not args.follow:
            return 0
        time.sleep(20)


if __name__ == "__main__":
    raise SystemExit(main())
