#!/usr/bin/env python
"""One-line status of the HotpotQA seed chain. Free: reads files only.

Written as a script rather than inline shell because the shell version got two
fields wrong: it reported the base val as "best" (the only line matching its
pattern) and printed a spurious 0 because ``grep -c`` emits 0 *and* exits 1 when
nothing matches, so the ``|| echo 0`` fallback fired too.

    uv run python scripts/chain_status.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Fires once per iteration, naming the PARENT gepa chose to mutate. Used only to
#: count iterations -- see BEST_SO_FAR for the score.
SELECTED = re.compile(r"Selected program \d+ score: ([0-9.]+)")

#: The actual best-so-far. Taking max() over SELECTED instead reports the best
#: score among candidates that have been chosen as a PARENT, which lags badly:
#: on IFBench seed 1 it read 0.547 (the base) while the true best was 0.5834,
#: because the winning candidate had not yet been selected to mutate from. gepa
#: emits this line after every full-val evaluation, which is exactly when the
#: best can change.
BEST_SO_FAR = re.compile(r"Best valset aggregate score so far: ([0-9.]+)")
ACCEPTED = re.compile(r"New program candidate index: (\d+)")
FAULTS = re.compile(r"UnicodeEncodeError|Traceback|aborting|failed to reach the model")
#: tqdm writes "1080rollouts [12:51, 1.08rollouts/s]". The trailing " [" is
#: required: without it the RATE suffix matches too, and "1.08rollouts/s" yields
#: 8 -- which silently reported $0.00 of solver spend.
ROLLOUTS = re.compile(r"(\d+)rollouts\s*\[")

#: Measured mean cost of one 4-call rollout. Was 0.00584 from
#: scripts/calibrate_hotpotqa.py, which deliberately over-estimates tokens;
#: seed 1's realised figure ($60.49 over 10,590 billed rollouts) puts it at
#: 0.00571, so the pre-run estimate was 2.2% high -- close enough that the live
#: readings were trustworthy, and now replaced by the observed value.
USD_PER_ROLLOUT = 0.00571

#: The base candidate's val evaluation is REPLAYED from the shared cache, so its
#: rollouts appear in gepa's counter but cost nothing. Not subtracting them would
#: overstate spend by ~$1.75 on every seed.
REPLAYED_ROLLOUTS = 300


def _best_so_far(text: str) -> float | None:
    """Best full-val score reached, or the base program's if nothing is accepted yet.

    gepa logs the best-so-far only after a full-val evaluation, so before the
    first accepted candidate there is no such line and the base program's score
    (the sole SELECTED entry) is the honest answer.
    """
    best = BEST_SO_FAR.findall(text)
    if best:
        return float(best[-1])
    selected = SELECTED.findall(text)
    return float(selected[0]) if selected else None


def estimated_spend(run_dir: Path) -> tuple[float, float] | None:
    """(estimated solver USD, actual reflection USD).

    Solver spend is an ESTIMATE and labelled as one: it lives in an in-process
    meter and is not written to disk until the run ends, so mid-run it can only
    be inferred from the rollout counter. The budget stopper meters the real
    figure -- this is for watching, not for enforcement.
    """
    stderr = run_dir / "run_log_stderr.txt"
    if not stderr.exists():
        return None
    counts = ROLLOUTS.findall(stderr.read_text(encoding="utf-8", errors="replace"))
    rollouts = int(counts[-1]) if counts else 0
    solver = max(0, rollouts - REPLAYED_ROLLOUTS) * USD_PER_ROLLOUT

    reflection = 0.0
    log = run_dir / "reflection_spend.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    reflection += json.loads(line)["cost_usd"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return solver, reflection


def seed_status(arm: str, seed: int) -> str | None:
    d = REPO / "results" / "runs" / f"hotpotqa-{arm}-seed{seed}"
    if not d.exists():
        return None

    summary = d / "summary.json"
    if summary.exists():
        s = json.loads(summary.read_text(encoding="utf-8"))
        spend = sum(v.get("budgeted_usd", 0.0) for v in (s.get("spend") or {}).values() if isinstance(v, dict))
        return f"seed{seed}=DONE best={s.get('best_val_score'):.3f} cands={s.get('candidates')} ${spend:.0f}"

    log = d / "run_log.txt"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    scores = [float(x) for x in SELECTED.findall(text)]
    best = _best_so_far(text)
    faults = len(FAULTS.findall(text))
    spend = estimated_spend(d)
    cost = f" ~${spend[0] + spend[1]:.1f}" if spend else ""
    head = (
        f"seed{seed}=it{len(scores)} acc={len(ACCEPTED.findall(text))} best={best:.3f}"
        if best is not None
        else f"seed{seed}=it{len(scores)} best=?"
    )
    return head + cost + (f" FAULTS={faults}" if faults else "")


def appworld_status() -> str | None:
    """AppWorld base-val build, then seeds. Separate from HotpotQA's line."""
    out: list[str] = []

    bv = REPO / "results" / "appworld_base_val"
    summary = bv / "summary.json"
    if summary.exists():
        d = json.loads(summary.read_text(encoding="utf-8"))
        out.append(
            f"AW base={d['mean_score']:.3f} tgc={d['task_goal_completion']:.2f} "
            f"${d['spend_usd']:.1f} steps={d['adapter']['mean_steps']}"
        )
    elif (bv / "base_val_cache.json").exists():
        out.append("AW base=written")
    else:
        # No progress file exists mid-build, so liveness is the only signal.
        out.append("AW base=building")

    for seed in (1, 2, 3):
        for arm in ("baseline", "taxonomy"):
            d = REPO / "results" / "runs" / f"appworld-{arm}-seed{seed}"
            if not d.exists():
                continue
            s = d / "summary.json"
            if s.exists():
                j = json.loads(s.read_text(encoding="utf-8"))
                out.append(f"AW-{arm[:3]}{seed}=DONE best={j.get('best_val_score')}")
            else:
                text = (
                    (d / "run_log.txt").read_text(encoding="utf-8", errors="replace")
                    if (d / "run_log.txt").exists()
                    else ""
                )
                scores = [float(x) for x in SELECTED.findall(text)]
                value = _best_so_far(text)
                best = f"{value:.3f}" if value is not None else "?"
                faults = len(FAULTS.findall(text))
                out.append(f"AW-{arm[:3]}{seed}=it{len(scores)} best={best}" + (f" FAULTS={faults}" if faults else ""))
    return " | ".join(out) if out else None


def livebench_math_status() -> str | None:
    """LiveBench-Math base val, then seeds. Its own segment of the status line."""
    out: list[str] = []

    bv = REPO / "results" / "livebench_math_base_val" / "summary.json"
    if bv.exists():
        d = json.loads(bv.read_text(encoding="utf-8"))
        out.append(f"LBM base={d['mean_score']:.3f} ${d['spend_usd']:.1f} ${d.get('usd_per_rollout', 0):.4f}/roll")
    elif (REPO / "results" / "livebench_math_base_val").exists():
        out.append("LBM base=building")
    else:
        return None

    for seed in (1, 2, 3):
        for arm in ("baseline", "taxonomy"):
            d = REPO / "results" / "runs" / f"livebench-math-{arm}-seed{seed}"
            if not d.exists():
                continue
            s = d / "summary.json"
            if s.exists():
                j = json.loads(s.read_text(encoding="utf-8"))
                spend = sum(v.get("budgeted_usd", 0.0) for v in (j.get("spend") or {}).values() if isinstance(v, dict))
                out.append(
                    f"LBM-{arm[:3]}{seed}=DONE best={j.get('best_val_score'):.3f} "
                    f"cands={j.get('candidates')} ${spend:.0f}"
                )
            else:
                text = (
                    (d / "run_log.txt").read_text(encoding="utf-8", errors="replace")
                    if (d / "run_log.txt").exists()
                    else ""
                )
                scores = [float(x) for x in SELECTED.findall(text)]
                value = _best_so_far(text)
                best = f"{value:.3f}" if value is not None else "?"
                faults = len(FAULTS.findall(text))
                out.append(
                    f"LBM-{arm[:3]}{seed}=it{len(scores)} acc={len(ACCEPTED.findall(text))} best={best}"
                    + (f" FAULTS={faults}" if faults else "")
                )
    return " | ".join(out)


def ifbench_status() -> str | None:
    """IFBench base val, then seeds. Its own segment of the status line."""
    out: list[str] = []

    bv = REPO / "results" / "ifbench_base_val" / "summary.json"
    if bv.exists():
        d = json.loads(bv.read_text(encoding="utf-8"))
        out.append(
            f"IFB base={d['mean_score']:.3f} (loose {d.get('instruction_level_loose', 0):.3f}) "
            f"${d['spend_usd']:.1f} ${d.get('usd_per_rollout', 0):.4f}/roll"
        )
    elif (REPO / "results" / "ifbench_base_val").exists():
        out.append("IFB base=building")
    else:
        return None

    for seed in (1, 2, 3):
        for arm in ("baseline", "taxonomy"):
            d = REPO / "results" / "runs" / f"ifbench-{arm}-seed{seed}"
            if not d.exists():
                continue
            s = d / "summary.json"
            if s.exists():
                j = json.loads(s.read_text(encoding="utf-8"))
                spend = sum(v.get("budgeted_usd", 0.0) for v in (j.get("spend") or {}).values() if isinstance(v, dict))
                out.append(
                    f"IFB-{arm[:3]}{seed}=DONE best={j.get('best_val_score'):.3f} "
                    f"cands={j.get('candidates')} ${spend:.0f}"
                )
            else:
                text = (
                    (d / "run_log.txt").read_text(encoding="utf-8", errors="replace")
                    if (d / "run_log.txt").exists()
                    else ""
                )
                scores = [float(x) for x in SELECTED.findall(text)]
                value = _best_so_far(text)
                best = f"{value:.3f}" if value is not None else "?"
                faults = len(FAULTS.findall(text))
                out.append(
                    f"IFB-{arm[:3]}{seed}=it{len(scores)} acc={len(ACCEPTED.findall(text))} best={best}"
                    + (f" FAULTS={faults}" if faults else "")
                )
    return " | ".join(out)


def taxonomy_status() -> str | None:
    """Taxonomy generation runs, reported per agreement round.

    Generation writes ``round_N.json`` incrementally but buffers its stdout until
    the subprocess exits, so the round files are the only live progress signal --
    the same blind spot the base-val builders had.
    """
    root = REPO / "results" / "taxonomy"
    if not root.exists():
        return None

    out: list[str] = []
    for run in sorted(root.iterdir()):
        if not run.is_dir():
            continue
        final = run / "taxonomy.json"
        agreement = run / "artifacts" / "agreement"
        if final.exists():
            try:
                d = json.loads(final.read_text(encoding="utf-8"))
                n = len(d.get("codes") or [])
                k = ((d.get("generation") or {}).get("agreement") or {}).get("final_kappa")
                out.append(f"TAX/{run.name}=DONE {n} codes" + (f" k={k:.2f}" if isinstance(k, int | float) else ""))
            except (json.JSONDecodeError, OSError):
                out.append(f"TAX/{run.name}=DONE")
        elif agreement.exists():
            rounds = sorted(agreement.glob("round_*.json"))
            kappa = "?"
            if rounds:
                try:
                    kappa = f"{json.loads(rounds[-1].read_text(encoding='utf-8'))['kappa']:.3f}"
                except (json.JSONDecodeError, KeyError, OSError):
                    pass
            out.append(f"TAX/{run.name}=round{len(rounds)} k={kappa}")
        elif (run / "artifacts").exists():
            out.append(f"TAX/{run.name}=drafting")
    return " | ".join(out) if out else None


def main() -> int:
    parts: list[str] = []

    bv = REPO / "results" / "base_val" / "summary.json"
    if bv.exists():
        d = json.loads(bv.read_text(encoding="utf-8"))
        parts.append(f"base={d['mean_answer_f1']:.3f}")
    else:
        parts.append("base=building")

    for seed in (1, 2, 3):
        for arm in ("baseline", "taxonomy"):
            s = seed_status(arm, seed)
            if s:
                parts.append(s if arm == "baseline" else f"tax-{s}")

    segments = [" | ".join(parts)]
    for extra in (appworld_status(), livebench_math_status(), ifbench_status(), taxonomy_status()):
        if extra:
            segments.append(extra)
    print("    ||    ".join(segments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
