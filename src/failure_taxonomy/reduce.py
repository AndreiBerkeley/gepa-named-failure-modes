"""Evidence-based taxonomy reduction.

Generation runs with AdaMAST's stock configuration; the taxonomy is reduced
afterwards from measured trace support. The generator is never asked to guess
which codes to drop: a judging pass over the generation corpus decides, and
every drop is recorded with its reason.

Pure and offline by design: same inputs give the same taxonomy and the same
report, so a re-cap is a re-run over stored artifacts rather than another
billed generation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Reasons a generated code can be dropped.
UNGROUNDED = "ungrounded"
OVER_CAP = "over_cap"
RETAINED = "retained"


def measure_support(judgements: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, set[str]]:
    """``code id -> set of trace ids`` over per-trace failure-mode citations.

    A code cited many times in one trace still counts once: support is the
    number of distinct traces exhibiting the code, not the number of mentions.
    """
    support: dict[str, set[str]] = {}
    for trace_id, modes in judgements.items():
        for mode in modes:
            code = str(mode.get("code") or "")
            if code:
                support.setdefault(code, set()).add(str(trace_id))
    return support


@dataclass(frozen=True)
class ReductionResult:
    document: dict[str, Any]
    report: dict[str, Any]


def reduce_taxonomy(
    document: Mapping[str, Any],
    judgements: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    min_support: int = 2,
    max_codes: int = 25,
) -> ReductionResult:
    """Reduce ``document`` to the codes the corpus actually supports.

    Every generated code is accounted for exactly once in the report, as
    ``retained``, ``ungrounded`` (support below ``min_support``), or
    ``over_cap`` (grounded, but ranked below the ``max_codes`` safety net).
    Ordering is deterministic: support descending, code id ascending as the
    tie-break. ``max_codes`` is meant as a safety net; when it binds before
    ``min_support`` does, the taxonomy is being shaped by a count rather than
    by evidence, and the report makes that visible.
    """
    codes = list(document.get("codes") or [])
    if not codes:
        raise ValueError("taxonomy document has no codes to reduce")
    if min_support < 1:
        raise ValueError(f"min_support must be >= 1, got {min_support}")
    if max_codes < 1:
        raise ValueError(f"max_codes must be >= 1, got {max_codes}")

    support = measure_support(judgements)
    counts = {str(c["id"]): len(support.get(str(c["id"]), ())) for c in codes}

    grounded = [c for c in codes if counts[str(c["id"])] >= min_support]
    grounded.sort(key=lambda c: (-counts[str(c["id"])], str(c["id"])))
    kept = grounded[:max_codes]
    kept_ids = {str(c["id"]) for c in kept}

    entries = []
    for c in sorted(codes, key=lambda c: (-counts[str(c["id"])], str(c["id"]))):
        cid = str(c["id"])
        if cid in kept_ids:
            outcome = RETAINED
        elif counts[cid] < min_support:
            outcome = UNGROUNDED
        else:
            outcome = OVER_CAP
        entries.append({"id": cid, "name": c.get("name"), "support": counts[cid], "outcome": outcome})

    tallies = {
        RETAINED: sum(1 for e in entries if e["outcome"] == RETAINED),
        UNGROUNDED: sum(1 for e in entries if e["outcome"] == UNGROUNDED),
        OVER_CAP: sum(1 for e in entries if e["outcome"] == OVER_CAP),
    }
    assert sum(tallies.values()) == len(codes)

    source_trace_ids = sorted(str(t) for t in judgements)
    reduced = dict(document)
    # Restore original code order among the retained, so the reduced taxonomy
    # reads like the generated one minus the dropped entries.
    reduced["codes"] = [c for c in codes if str(c["id"]) in kept_ids]
    reduced["reduction"] = {
        "min_support": min_support,
        "max_codes": max_codes,
        "generated_codes": len(codes),
        "retained_codes": len(kept),
        "cap_bound": tallies[OVER_CAP] > 0,
        "source_trace_ids": source_trace_ids,
    }

    report = {
        "min_support": min_support,
        "max_codes": max_codes,
        "judged_traces": len(source_trace_ids),
        "tallies": tallies,
        "codes": entries,
    }
    return ReductionResult(document=reduced, report=report)


class FrozenTraceLeakError(RuntimeError):
    """A taxonomy is being used to score traces it was generated from."""


def assert_generation_disjoint(
    generation_trace_ids: frozenset[str],
    scoring_ids: Sequence[str],
    *,
    context: str = "scoring",
) -> None:
    """Freeze discipline, asserted in code rather than by convention.

    The traces a taxonomy was generated from must never be traces it is later
    used to score; an overlap silently turns held-out feedback into training
    leakage.
    """
    overlap = generation_trace_ids.intersection(str(i) for i in scoring_ids)
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise FrozenTraceLeakError(
            f"{len(overlap)} {context} instance(s) were in the taxonomy's generation corpus "
            f"(e.g. {sample}). Generation and scoring splits must be disjoint."
        )
