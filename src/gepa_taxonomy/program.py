"""The candidate program: BM25 retrieval -> solver -> static feedback -> refiner.

Exactly two LM calls per rollout, fixed. No variable-length agent loop -- cost
predictability is load-bearing here, because three baseline seeds under a fixed
dollar budget must be comparable. A variable-cost agent could give one seed 40
iterations and another 12, confounding the very comparison we are running.

GEPA optimizes exactly two text components:

* ``solver_instruction``  -- problem statement + retrieved code -> patch
* ``refiner_instruction`` -- candidate patch + static feedback  -> revised patch

Everything else (retrieval, patch application, feedback construction, grading)
is fixed scaffolding. Keeping the optimizable surface to two strings is what
makes the baseline-vs-taxonomy comparison interpretable.

Gold blindness
--------------
No component here receives a :class:`~gepa_taxonomy.tasks.Gold`. The signatures
take a :class:`~gepa_taxonomy.tasks.Task`, which structurally cannot carry gold
data. Every LM prompt is additionally run through ``assert_gold_free`` before
dispatch, so a leak fails the run rather than inflating a score.

In particular the refiner sees **only cheap static signals** -- does the patch
apply, does the result parse/import. It never sees FAIL_TO_PASS or PASS_TO_PASS
output: that is the grading signal, and feeding it back at inference time would
make the results meaningless.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from gepa_taxonomy.cost import CostMeter, Phase
from gepa_taxonomy.tasks import Task, assert_gold_free

# Component names. These are the keys GEPA mutates.
SOLVER = "solver_instruction"
REFINER = "refiner_instruction"
COMPONENTS = (SOLVER, REFINER)


class LMClient(Protocol):
    """Minimal LM interface. Implemented by the Bedrock client and by fakes."""

    def complete(self, prompt: str, *, max_tokens: int = 4096) -> tuple[str, int, int]:
        """Return ``(text, input_tokens, output_tokens)``."""
        ...


# ---------------------------------------------------------------------------
# Retrieval (fixed scaffolding)
# ---------------------------------------------------------------------------


class Retriever(Protocol):
    def retrieve(self, task: Task, *, k: int) -> list[RetrievedFile]: ...


@dataclass(frozen=True, slots=True)
class RetrievedFile:
    path: str
    content: str


def render_context(files: Sequence[RetrievedFile], *, max_chars: int) -> str:
    """Render retrieved files into a prompt block, truncating deterministically.

    Truncation is per-file and by whole lines so the same inputs always produce
    the same context string -- a varying context would make prompt caching and
    run-to-run comparison unreliable.
    """
    budget = max_chars // max(1, len(files))
    parts: list[str] = []
    for f in files:
        body = f.content
        if len(body) > budget:
            lines, kept, used = body.splitlines(keepends=True), [], 0
            for line in lines:
                if used + len(line) > budget:
                    break
                kept.append(line)
                used += len(line)
            body = "".join(kept) + "\n... [truncated]\n"
        parts.append(f"[start of {f.path}]\n{body}\n[end of {f.path}]")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Static feedback (fixed scaffolding, gold-free by construction)
# ---------------------------------------------------------------------------

_DIFF_HEADER = re.compile(r"^(--- |\+\+\+ |diff --git |@@ )", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class PatchFeedback:
    """Cheap, gold-free signals about a candidate patch.

    Everything here is derivable from the patch and the checked-out repo alone.
    Nothing here touches the grading tests.
    """

    is_well_formed: bool
    applies_cleanly: bool | None  # None when not checked (no workspace)
    syntax_ok: bool | None
    messages: tuple[str, ...]

    def render(self) -> str:
        lines = []
        lines.append(f"- Patch is a well-formed unified diff: {_yn(self.is_well_formed)}")
        if self.applies_cleanly is not None:
            lines.append(f"- Patch applies cleanly to the repository: {_yn(self.applies_cleanly)}")
        if self.syntax_ok is not None:
            lines.append(f"- Modified files parse without syntax errors: {_yn(self.syntax_ok)}")
        for m in self.messages:
            lines.append(f"- {m}")
        return "\n".join(lines)


def _yn(v: bool | None) -> str:
    return "unknown" if v is None else ("yes" if v else "NO")


def static_feedback(
    patch: str,
    *,
    applier: Callable[[str], PatchFeedback] | None = None,
    repo_dir: Path | None = None,
) -> PatchFeedback:
    """Build feedback for ``patch``.

    Without an ``applier`` (no workspace available) only the shape of the diff
    is checked. With one, apply/syntax results are filled in. The applier is
    handed the patch text and nothing else -- it has no route to gold data.
    """
    if applier is not None:
        return applier(patch)

    stripped = patch.strip()
    if not stripped:
        return PatchFeedback(
            is_well_formed=False,
            applies_cleanly=None,
            syntax_ok=None,
            messages=("The patch is empty. No edit was produced.",),
        )
    well_formed = bool(_DIFF_HEADER.search(patch))
    msgs: tuple[str, ...] = ()
    if not well_formed:
        msgs = (
            "The output does not look like a unified diff. It must contain "
            "'--- '/'+++ ' file headers and '@@' hunk headers.",
        )
    applies: bool | None = None
    if well_formed and repo_dir is not None:
        # The refiner exists to repair a patch that does not apply; without
        # this verdict it is blind to the pipeline's most common failure.
        from gepa_taxonomy.patch_gate import apply_diagnostics

        applies, reasons = apply_diagnostics(patch, repo_dir)
        if applies is False:
            msgs = msgs + (
                "The patch does not apply to the repository. git reported:",
                *reasons,
                "Rewrite it so every context line matches the file exactly.",
            )
        elif applies is True:
            msgs = msgs + ("The patch applies cleanly to the repository.",)

    return PatchFeedback(is_well_formed=well_formed, applies_cleanly=applies, syntax_ok=None, messages=msgs)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SOLVER_PROMPT = """{instruction}

## Repository
{repo}

## Issue
{problem_statement}

## Relevant code
{context}

Respond with a single unified diff patch and nothing else."""

REFINER_PROMPT = """{instruction}

## Repository
{repo}

## Issue
{problem_statement}

## Relevant code
{context}

## Candidate patch
{patch}

## Automated checks on the candidate patch
{feedback}

Respond with a single corrected unified diff patch and nothing else."""


@dataclass
class Rollout:
    """One complete pass through the program. This is also the trace record.

    Trace capture is designed in from the start rather than bolted on in Phase
    3: Phase 3's harvest becomes a filter over data already captured, and Phase
    4's taxonomy sees uniform, comparable traces.
    """

    instance_id: str
    retrieved_paths: list[str]
    solver_patch: str
    feedback: PatchFeedback
    refiner_patch: str
    solver_tokens: tuple[int, int]
    refiner_tokens: tuple[int, int]
    cost_usd: float
    #: The exact strings sent to the LMs. Retained so the harness -- which
    #: legitimately holds gold in order to grade -- can run the value-based
    #: gold-leak check against the true program boundary. Not written to the
    #: trace file verbatim (they embed the whole retrieved context); the trace
    #: carries a digest instead.
    solver_prompt: str = ""
    refiner_prompt: str = ""
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def prompts(self) -> tuple[str, str]:
        return self.solver_prompt, self.refiner_prompt

    @property
    def final_patch(self) -> str:
        """What gets graded: the refiner's output, falling back to the solver's."""
        return self.refiner_patch.strip() or self.solver_patch

    def to_trace(self) -> dict[str, Any]:
        import hashlib

        return {
            "instance_id": self.instance_id,
            "retrieved_paths": self.retrieved_paths,
            "solver_prompt_sha256": hashlib.sha256(self.solver_prompt.encode()).hexdigest(),
            "refiner_prompt_sha256": hashlib.sha256(self.refiner_prompt.encode()).hexdigest(),
            "solver_patch": self.solver_patch,
            "feedback": {
                "is_well_formed": self.feedback.is_well_formed,
                "applies_cleanly": self.feedback.applies_cleanly,
                "syntax_ok": self.feedback.syntax_ok,
                "messages": list(self.feedback.messages),
            },
            "refiner_patch": self.refiner_patch,
            "solver_tokens_in": self.solver_tokens[0],
            "solver_tokens_out": self.solver_tokens[1],
            "refiner_tokens_in": self.refiner_tokens[0],
            "refiner_tokens_out": self.refiner_tokens[1],
            "cost_usd": self.cost_usd,
            "error": self.error,
            **self.extra,
        }


def extract_patch(text: str) -> str:
    """Pull a unified diff out of an LM response, tolerating code fences."""
    fenced = re.search(r"```(?:diff|patch)?\n(.*?)```", text, re.DOTALL)
    body = fenced.group(1) if fenced else text
    match = _DIFF_HEADER.search(body)
    return body[match.start() :].strip() if match else body.strip()


@dataclass
class SolverRefinerProgram:
    """The candidate program. Two LM calls, fixed scaffolding around them."""

    retriever: Retriever
    solver_lm: LMClient
    refiner_lm: LMClient
    solver_meter: CostMeter
    refiner_meter: CostMeter
    solver_model: str
    refiner_model: str
    top_k: int = 5
    max_context_chars: int = 60_000
    max_tokens: int = 4096
    patch_applier: Callable[[str], PatchFeedback] | None = None
    #: Locates the checkout for a task so the solver patch can be apply-checked.
    #: Left None and the refiner receives no apply verdict at all.
    repo_dir_for: Callable[[Task], Path | None] | None = None

    def run(self, task: Task, candidate: dict[str, str], *, phase: Phase = "optimization") -> Rollout:
        files = self.retriever.retrieve(task, k=self.top_k)
        context = render_context(files, max_chars=self.max_context_chars)
        ctx = task.to_prompt_context()

        solver_prompt = SOLVER_PROMPT.format(
            instruction=candidate[SOLVER],
            repo=ctx["repo"],
            problem_statement=ctx["problem_statement"],
            context=context,
        )
        # Boundary check 1: nothing gold may reach the solver.
        assert_gold_free(solver_prompt, where="solver prompt")

        raw, s_in, s_out = self.solver_lm.complete(solver_prompt, max_tokens=self.max_tokens)
        cost = self.solver_meter.record(model=self.solver_model, input_tokens=s_in, output_tokens=s_out, phase=phase)
        solver_patch = extract_patch(raw)

        repo_dir = self.repo_dir_for(task) if self.repo_dir_for is not None else None
        feedback = static_feedback(solver_patch, applier=self.patch_applier, repo_dir=repo_dir)

        refiner_prompt = REFINER_PROMPT.format(
            instruction=candidate[REFINER],
            repo=ctx["repo"],
            problem_statement=ctx["problem_statement"],
            context=context,
            patch=solver_patch or "(the solver produced no patch)",
            feedback=feedback.render(),
        )
        # Boundary check 2: the feedback block is the likeliest leak path, since
        # it is the only part of the prompt derived from an execution result.
        assert_gold_free(refiner_prompt, where="refiner prompt")

        raw2, r_in, r_out = self.refiner_lm.complete(refiner_prompt, max_tokens=self.max_tokens)
        cost += self.refiner_meter.record(model=self.refiner_model, input_tokens=r_in, output_tokens=r_out, phase=phase)

        return Rollout(
            instance_id=task.instance_id,
            retrieved_paths=[f.path for f in files],
            solver_patch=solver_patch,
            feedback=feedback,
            refiner_patch=extract_patch(raw2),
            solver_tokens=(s_in, s_out),
            refiner_tokens=(r_in, r_out),
            cost_usd=cost,
            solver_prompt=solver_prompt,
            refiner_prompt=refiner_prompt,
        )


# Seed instructions. Deliberately plain -- GEPA's job is to improve them, and
# a hand-tuned seed would confound the baseline.
SEED_CANDIDATE: dict[str, str] = {
    SOLVER: (
        "You are fixing a bug in a Python repository. Read the issue and the "
        "provided source files, then write a patch that resolves the issue. "
        "Output a unified diff against the repository root."
    ),
    REFINER: (
        "You are reviewing a candidate patch for a Python repository issue. "
        "Check the patch against the issue and the automated checks, then output "
        "a corrected unified diff against the repository root."
    ),
}
