#!/usr/bin/env python
"""Watch every live run for the failures a progress monitor cannot see. FREE.

    uv run python scripts/watchdog.py            # one pass, prints status
    uv run python scripts/watchdog.py --loop     # event stream for Monitor

Why this exists
---------------
A CloudCast run once hung for 88 minutes without anyone noticing. Every monitor
armed at the time watched for faults, completions, or process death -- and a
hang produces **none of those**. The process is alive, the exit code never
arrives, no exception is raised. Silence looked exactly like work.

So the primary signal here is the one that was missing: *the log stopped
advancing*. Everything else (completion, crash, chain abort) is secondary.

Thresholds are per-benchmark because "too quiet" means different things:

* **circlepack** -- a single candidate may legitimately run to its 600s cap, so
  only silence well past that is suspicious.
* **cloudcast**  -- ~6.5 min per candidate observed on the finished pilot.
* **hover**      -- each accepted iteration runs a 300-instance val evaluation,
  which is genuinely slow; a short threshold here would cry wolf every time.

Emitting once per episode
-------------------------
Each alert fires ONCE and re-arms only after the condition clears. A watchdog
that re-emits every poll floods the channel and gets shut off automatically --
which would leave exactly the blind spot it was built to close.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs"

#: Seconds of log silence that count as "not advancing", per benchmark family.
#: Set generously above each one's observed per-candidate time -- a false alarm
#: costs attention, but a threshold set too tight gets ignored, which is worse.
QUIET_S = {"circlepack": 1500, "cloudcast": 1200, "hover": 2400}
DEFAULT_QUIET_S = 1200

#: Substrings in a chain log that mean the run cannot continue. ``ExpiredToken``
#: matters disproportionately here: every run shares ONE bearer token, so its
#: expiry does not fail one seed, it fails all of them at once.
FATAL = ("Traceback", "FAILED (exit", "botocore.exceptions", "AccessDenied",
         "ExpiredToken", "UnrecognizedClient", "ModuleNotFoundError")

#: Rate limiting. Not fatal -- litellm retries -- but retries are invisible in
#: the spend log and show up only as throughput quietly collapsing, so surface
#: them. Matched case-sensitively on the exception names: a case-insensitive
#: "429" matches millisecond timestamps like ``01:13:09,429``.
THROTTLE = ("ThrottlingException", "TooManyRequestsException",
            "ServiceQuotaExceeded", "ModelTimeoutException")


def quiet_limit(name: str) -> int:
    for fam, secs in QUIET_S.items():
        if name.startswith(fam):
            return secs
    return DEFAULT_QUIET_S


#: A directory whose log has not been touched in this long was abandoned, not
#: hung. Archived ``*.netfail_*`` runs and old smoke tests otherwise register as
#: permanent hangs and bury the real alerts under noise the moment they fire.
ABANDONED_S = 6 * 3600
#: Never watched, regardless of recency.
IGNORE_MARKERS = (".netfail_", "smoke-", ".bak", ".old")


def _finished(run: Path) -> bool:
    """summary.json is the LAST thing gepa writes, so if nothing is newer than
    it the run has exited. One rule covers both awkward cases: a freshly resumed
    run has a resume log newer than its stale summary (live), while a just
    finished run has nothing newer (finished). Keying this to "a resume log
    exists and is newer or equal" instead pinned COMPLETED runs as live forever,
    because tee's final write lands in the same second as the summary -- which
    is how a finished CloudCast seed got reported as hung."""
    summary = run / "summary.json"
    if not summary.exists():
        return False
    newest = 0.0
    for p in (run / "run_log.txt", run / "run_log_stderr.txt", run / "gepa_state.bin",
              run / "reflection_spend.jsonl", run.parent / f"{run.name}.resume.log"):
        if p.exists():
            newest = max(newest, p.stat().st_mtime)
    return newest <= summary.stat().st_mtime + 2.0


def live_runs() -> list[Path]:
    """Directories with a run log newer than any summary, still being written."""
    out = []
    if not RUNS.is_dir():
        return out
    now = time.time()
    for d in sorted(RUNS.iterdir()):
        log = d / "run_log.txt"
        if not d.is_dir() or not log.exists():
            continue
        if any(m in d.name for m in IGNORE_MARKERS):
            continue
        resume_log = d.parent / f"{d.name}.resume.log"
        resumed_recently = resume_log.exists() and now - resume_log.stat().st_mtime < ABANDONED_S
        if now - log.stat().st_mtime > ABANDONED_S and not resumed_recently:
            # Stale-log check must yield to a fresh resume: a just-restarted run
            # has not written its log since before the pause, which on circle
            # packing can be many hours ago. Without this it is filtered out as
            # abandoned at exactly the moment it most needs watching.
            continue
        if _finished(d):
            continue
        out.append(d)
    return out


def last_activity(run: Path) -> float:
    """Newest mtime that means "this run is doing something".

    Not just run_log.txt: a run resumed after a pause has a log whose mtime is
    from BEFORE the pause -- ten hours ago in the case that prompted this -- so
    keying quietness to the log alone reports a brand-new run as long dead.
    """
    times = [(run / "run_log.txt").stat().st_mtime]
    for extra in (run.parent / f"{run.name}.resume.log", run / "gepa_state.bin",
                  run / "reflection_spend.jsonl"):
        if extra.exists():
            times.append(extra.stat().st_mtime)
    return max(times)


def fatal_in_chain_logs() -> list[tuple[str, str]]:
    """(logname, offending line) for chain logs that hit a fatal condition.

    Archived logs are skipped: a pre-pause chain log records the exit code from
    however that segment ended, and re-reporting it every session would train
    the reader to ignore FATAL -- the one alert that must never be ignored.
    """
    hits = []
    for log in sorted(RUNS.glob("*-chain*.log")):
        if any(m in log.name for m in IGNORE_MARKERS):
            continue
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if any(f in line for f in FATAL):
                hits.append((log.name, line.strip()[:160]))
                break
    return hits


def throttle_count() -> int:
    """Total rate-limit exceptions across every live run and chain log."""
    n = 0
    for run in live_runs():
        for fname in ("run_log.txt", "run_log_stderr.txt"):
            f = run / fname
            if f.exists():
                t = f.read_text(encoding="utf-8", errors="replace")
                n += sum(t.count(marker) for marker in THROTTLE)
    for log in RUNS.glob("*-chain*.log"):
        try:
            t = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        n += sum(t.count(marker) for marker in THROTTLE)
    return n


def scan(state: dict) -> list[str]:
    events: list[str] = []
    now = time.time()
    seen = set()

    for run in live_runs():
        name = run.name
        seen.add(name)
        age = now - last_activity(run)
        limit = quiet_limit(name)
        was_quiet = state.get(f"quiet:{name}", False)

        if age > limit and not was_quiet:
            events.append(f"HANG? {name} -- no log write for {age / 60:.0f} min (limit {limit // 60})")
            state[f"quiet:{name}"] = True
        elif age <= limit and was_quiet:
            events.append(f"RECOVERED {name} -- log advancing again")
            state[f"quiet:{name}"] = False

    # A run that disappeared from live_runs since the last pass either finished
    # or died. Distinguish by whether a summary was written.
    for name in list(state.get("live", [])):
        if name in seen:
            continue
        run = RUNS / name
        if (run / "summary.json").exists():
            try:
                d = json.loads((run / "summary.json").read_text(encoding="utf-8-sig"))
                score = d.get("best_score", d.get("best_val_score"))
                spend = (d.get("spend") or {}).get("realised_usd")
                events.append(
                    f"DONE {name} -- best {score} | ${spend:.2f} | {d.get('candidates')} candidates"
                    if isinstance(spend, (int, float))
                    else f"DONE {name} -- best {score}"
                )
            except Exception:
                events.append(f"DONE {name} -- summary written")
        else:
            events.append(f"VANISHED {name} -- no longer advancing and no summary.json (crash?)")
        state.pop(f"quiet:{name}", None)
    state["live"] = sorted(seen)

    for logname, line in fatal_in_chain_logs():
        key = f"fatal:{logname}"
        if not state.get(key):
            events.append(f"FATAL {logname} -- {line}")
            state[key] = True

    # Throttling is reported as a running total, and only when it GROWS, so a
    # steady trickle does not re-alert but a sudden burst does.
    total = throttle_count()
    seen_before = state.get("throttle", 0)
    if total > seen_before:
        events.append(f"THROTTLING {total} hits total (+{total - seen_before}) -- concurrency may be too high")
        state["throttle"] = total

    return events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop", action="store_true", help="poll forever, printing one line per event")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()

    state: dict = {}
    if not args.loop:
        runs = live_runs()
        now = time.time()
        if not runs:
            print("no live runs")
            return 0
        for run in runs:
            age = now - last_activity(run)
            flag = "  <-- QUIET" if age > quiet_limit(run.name) else ""
            print(f"{run.name:<28} last write {age / 60:6.1f} min ago   limit {quiet_limit(run.name) // 60:>2} min{flag}")
        for logname, line in fatal_in_chain_logs():
            print(f"FATAL {logname}: {line}")
        return 0

    scan(state)  # prime: record what is live without alerting on it
    while True:
        time.sleep(args.interval)
        try:
            for line in scan(state):
                print(line, flush=True)
        except Exception as exc:  # never let a transient FS error kill the watch
            print(f"watchdog error (continuing): {exc}", flush=True)
        if state.get("live") == [] :
            print("all runs finished", flush=True)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
