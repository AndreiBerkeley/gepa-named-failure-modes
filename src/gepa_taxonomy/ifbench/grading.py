"""IFBench scoring, driving AllenAI's vendored verifiers.

The verifiers themselves are copied verbatim (``_vendor/``, refreshed by
``scripts/vendor_ifbench.py``); this module only decides *which number GEPA
optimises* and *what the reflective feedback says*.

Which metric
------------
IFBench reports two levels. **Prompt-level** is all-or-nothing per instance;
**instruction-level** is the fraction of that instance's constraints satisfied.
We select on the **instruction-level fraction** and record prompt-level as a
diagnostic — the same split D041 made on HotpotQA, and for the same reason:
partial credit is what keeps a minibatch comparison from being a coin flip.

Be clear about how little that buys here, though. **256 of 300 instances carry
exactly one constraint**, so for 85% of the set the two metrics coincide
and scoring really is binary. The fraction helps on the 44 two-constraint
instances and nowhere else. What actually keeps the acceptance gate informative
is the base rate: IFBench's ~35.7% leaves only ~11% of minibatch-5 draws
all-zero, against the 18.3% that made SWE-Bench's minibatches tie ~75% of the
time.

Strict, not loose
-----------------
Upstream also defines a "loose" pass that retries the check against eight
mangled variants of the response (first line removed, last line removed,
asterisks stripped, ...). We score **strict** and record loose as a diagnostic.
Loose exists to stop a chatty preamble from masking a correct answer -- but
"stop emitting a preamble" is precisely the kind of thing a prompt optimiser
should learn, and scoring it away would hide a real, prompt-addressable failure
mode from both arms. The gap between the two is reported so we can see how much
of the score is formatting rather than compliance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from gepa_taxonomy.ifbench._vendor.ifbench import instructions_registry as _ifbench_registry
from gepa_taxonomy.ifbench._vendor.ifevalg import instructions_registry as _ifevalg_registry
from gepa_taxonomy.ifbench.tasks import IFBENCH, IFEVALG, Gold, constraint_family

#: The two disjoint constraint vocabularies. Routing is explicit rather than
#: inferred from the id: train/val and test share no ids at all, so a
#: lookup in the wrong registry is a KeyError -- loud, but only if we never guess.
_REGISTRIES = {
    IFBENCH: _ifbench_registry.INSTRUCTION_DICT,
    IFEVALG: _ifevalg_registry.INSTRUCTION_DICT,
}


#: The eight response variants upstream's loose check accepts. Kept only to
#: measure how much of the strict/loose gap is preamble.
def _loose_variants(response: str) -> list[str]:
    lines = response.split("\n")
    without_first = "\n".join(lines[1:]).strip()
    without_last = "\n".join(lines[:-1]).strip()
    without_both = "\n".join(lines[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        without_first,
        without_last,
        without_both,
        without_first.replace("*", ""),
        without_last.replace("*", ""),
        without_both.replace("*", ""),
    ]


@dataclass(frozen=True, slots=True)
class Grade:
    #: Instruction-level strict: fraction of constraints satisfied. The number
    #: GEPA selects on.
    score: float
    #: Prompt-level strict: every constraint satisfied. Diagnostic only.
    all_followed: bool
    #: Per-constraint verdicts, positionally aligned with the gold ids.
    followed: tuple[bool, ...] = ()
    #: Ids that failed, for feedback and for per-family reporting.
    failed_ids: tuple[str, ...] = ()
    #: Instruction-level LOOSE. Diagnostic: the strict/loose gap is formatting.
    loose_score: float = 0.0
    #: A verifier that raised. Never silently zero -- see ``grade``.
    errors: tuple[str, ...] = ()

    @property
    def families_failed(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for i in self.failed_ids:
            seen.setdefault(constraint_family(i), None)
        return tuple(seen)


def _check_one(
    instruction_id: str,
    kwargs: Mapping[str, object],
    prompt: str,
    response: str,
    registry: str = IFBENCH,
) -> bool:
    """Run a single verifier, mirroring upstream's strict loop exactly."""
    checker = _REGISTRIES[registry][instruction_id](instruction_id)
    checker.build_description(**dict(kwargs))
    args = checker.get_instruction_args()
    if args and "prompt" in args:
        # A few constraints are defined relative to the query text itself
        # (e.g. "repeat the request verbatim"), so they need it rebuilt in.
        checker.build_description(prompt=prompt)
    return bool(response and response.strip() and checker.check_following(response))


def grade(response: str, gold: Gold, *, prompt: str) -> Grade:
    """Score one response. Deterministic, offline, no LM call.

    A verifier that raises is recorded in ``errors`` and counted as **not
    followed**, rather than propagating. All 344 constraint instances in the
    published set were verified to run clean, so an error here means something
    changed upstream or a response triggered an unhandled path -- either way the
    run should surface it rather than die, and ``errors`` is checked before the
    base val is frozen.
    """
    response = response or ""
    followed: list[bool] = []
    failed: list[str] = []
    errors: list[str] = []

    for instruction_id, kwargs in zip(gold.instruction_ids, gold.kwargs, strict=False):
        try:
            ok = _check_one(instruction_id, kwargs, prompt, response, gold.registry)
        except Exception as exc:
            errors.append(f"{instruction_id}: {type(exc).__name__}: {exc}")
            ok = False
        followed.append(ok)
        if not ok:
            failed.append(instruction_id)

    loose = 0.0
    if gold.instruction_ids:
        best = 0
        for variant in _loose_variants(response):
            passed = 0
            for instruction_id, kwargs in zip(gold.instruction_ids, gold.kwargs, strict=False):
                try:
                    passed += int(_check_one(instruction_id, kwargs, prompt, variant, gold.registry))
                except Exception:
                    pass
            best = max(best, passed)
        loose = best / len(gold.instruction_ids)

    total = len(gold.instruction_ids)
    return Grade(
        score=(sum(followed) / total) if total else 0.0,
        all_followed=bool(total) and all(followed),
        followed=tuple(followed),
        failed_ids=tuple(failed),
        loose_score=loose,
        errors=tuple(errors),
    )


# -- feedback for the reflective dataset --------------------------------------


def constraint_feedback(graded: Grade, gold: Gold) -> str:
    """Gold-revealing feedback: names the constraints that failed. TRAIN ids only.

    Deliberately strong, IFBench was ruled out
    partly because "the diagnosis is already in the baseline feedback, so a
    taxonomy can only paraphrase it" -- weakening this to make the taxonomy look
    better would be exactly the rigged comparison that objection warns about. The
    taxonomy has to earn its gain against a baseline told precisely what failed.
    """
    if not gold.instruction_ids:
        return "No constraints were attached to this instance."
    if graded.all_followed:
        return f"All {len(gold.instruction_ids)} constraint(s) satisfied."
    lines = [f"Satisfied {sum(graded.followed)} of {len(gold.instruction_ids)} constraint(s)."]
    for instruction_id, ok in zip(gold.instruction_ids, graded.followed, strict=False):
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {instruction_id}")
    return "\n".join(lines)


def score_feedback(graded: Grade, gold: Gold) -> str:
    """Gold-free feedback, used on val and test ids.

    Reports the count and the strict/loose gap but not *which* verifier failed.
    The gap is the informative part: a response that passes loose but fails
    strict is being rejected for a preamble or stray markdown, which is a
    different fix from being non-compliant.
    """
    total = len(gold.instruction_ids)
    if not total:
        return "No constraints were attached to this instance."
    text = f"Satisfied {sum(graded.followed)} of {total} constraint(s)."
    if graded.loose_score > graded.score:
        text += (
            "\nThe response would have satisfied more constraints if surrounding text"
            " (a preamble, a trailing line, or markdown emphasis) were removed."
        )
    return text


def report(grades: Sequence[Grade], golds: Sequence[Gold]) -> dict[str, object]:
    """Aggregate view: the headline plus what a headline hides."""
    n = len(grades)
    if not n:
        return {}
    per_family: dict[str, list[int]] = {}
    for graded, gold in zip(grades, golds, strict=True):
        for instruction_id, ok in zip(gold.instruction_ids, graded.followed, strict=False):
            per_family.setdefault(constraint_family(instruction_id), []).append(int(ok))
    return {
        "instruction_level_strict": sum(g.score for g in grades) / n,
        "prompt_level_strict": sum(1 for g in grades if g.all_followed) / n,
        "instruction_level_loose": sum(g.loose_score for g in grades) / n,
        "by_family": {k: {"n": len(v), "mean": sum(v) / len(v)} for k, v in sorted(per_family.items())},
        "verifier_errors": sum(len(g.errors) for g in grades),
    }


__all__ = ["Grade", "constraint_feedback", "grade", "report", "score_feedback"]
