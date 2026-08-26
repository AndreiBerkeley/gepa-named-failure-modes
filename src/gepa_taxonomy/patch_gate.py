"""Skip container evaluation for patches the harness would never score above 0.

This is a wall-clock optimisation for local runs, and it is only legitimate
because it is **score-identical** to what the SWE-Bench harness already does.
Every claim below is grounded in swebench 4.1.0 source, not assumption.

Two tiers, with different strengths of guarantee
------------------------------------------------

**Tier 1 — empty patch. Provably identical, zero risk.**
``run_evaluation.py:458-470`` filters predictions before anything runs::

    empty_patch_ids = {k for k, v in predictions.items()
                       if v[KEY_PREDICTION] == "" or v[KEY_PREDICTION] is None}
    dataset = [i for i in dataset
               if i[KEY_INSTANCE_ID] in prediction_ids
               and i[KEY_INSTANCE_ID] not in empty_patch_ids]

The harness itself never starts a container for an empty patch and reports it
unresolved. Skipping is exactly what it does.

**Tier 2 — the patch cannot possibly apply. Score-identical.**
``run_evaluation.py:166-186`` tries each of ``GIT_APPLY_CMDS`` in order and, if
all fail, raises ``EvaluationError``::

    GIT_APPLY_CMDS = ["git apply --verbose",
                      "git apply --verbose --reject",
                      "patch --batch --fuzz=5 -p1 -i"]

An earlier version of this module tried to replicate that ladder with
``--check``. That was wrong, and testing caught it: the second command is
fuzz/reject tolerant and ``git apply --check --reject`` exits 0 for *every*
input, so the replicated ladder always reported "appliable". Predicting the
ladder is therefore not something a dry run can do.

Tier 2 instead fires only on certainties:

* the output contains no unified-diff headers at all, so every command fails
  for want of anything to parse; or
* **every file the patch modifies is absent from the checkout** -- no fuzz
  factor can apply a hunk to a file that does not exist. Pure-creation diffs
  (``--- /dev/null``) are excluded.

A patch that merely has bad context is **not** skipped: ``patch --fuzz=5`` may
still apply it, and wrongly skipping would understate a score.

⚠️ **One honest difference.** The harness records an apply failure as an
*error* (``error_ids``), not as *unresolved* (``unresolved_ids``). The
**resolved/score outcome is identical -- 0 either way** -- but the bookkeeping
category differs. We record ``skipped_reason`` on the trace so this is visible
downstream and cannot silently distort a failure taxonomy built from traces.

Everything uncertain falls through to a real container evaluation.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: Mirrors swebench 4.1.0 ``run_evaluation.GIT_APPLY_CMDS``, in order. Kept
#: verbatim so ``test_patch_gate.py`` can assert we have not drifted from it.
GIT_APPLY_CMDS: tuple[str, ...] = (
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
)

#: WARNING: there is NO faithful dry-run for the middle command.
#: ``git apply --check --verbose --reject`` returns exit 0 for *everything* --
#: verified locally, including a patch targeting a file that does not exist --
#: because ``--reject`` treats partial application as success. Including it in a
#: check ladder makes every patch look appliable and the tier-2 skip never
#: fires; excluding it and skipping anyway would risk discarding a patch the
#: harness *would* have partially applied, understating a score.
#:
#: So tier 2 does not try to predict the ladder. It fires only on a certainty no
#: fuzz factor can rescue: **every file the patch modifies is absent from the
#: checkout**. No hunk can apply to a file that is not there, so all three
#: commands necessarily fail. Pure-creation diffs (``--- /dev/null``) are
#: excluded, since those legitimately target absent files.
#: Dry-run mirror of the harness ladder (run_evaluation.py:65-67):
#:   git apply --verbose  /  git apply --verbose --reject  /  patch --batch --fuzz=5 -p1
#: `git apply --check` alone is STRICTER than the harness, so a patch the
#: harness would accept under fuzz was being reported to the refiner as
#: non-applying. Only a verdict that matches what the harness will actually do
#: is safe to feed back.
GIT_APPLY_CHECK_CMDS: tuple[tuple[str, ...], ...] = (
    ("git", "apply", "--check", "--verbose"),
    ("patch", "--batch", "--fuzz=5", "-p1", "--dry-run", "-i"),
)

#: Target paths of a unified diff: the "+++ b/path" side.
_TARGET_RE = re.compile(r"^\+\+\+ (?:b/)?(\S+)", re.MULTILINE)
_SOURCE_RE = re.compile(r"^--- (?:a/)?(\S+)", re.MULTILINE)

_DIFF_HEADER = re.compile(r"^(--- |\+\+\+ |diff --git |@@ )", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Whether to skip the container, and why."""

    skip: bool
    reason: str | None = None

    @property
    def score(self) -> float:
        """The score a skipped rollout takes. Always 0.0 -- never anything else."""
        return 0.0


def is_empty_patch(patch: str) -> bool:
    """Tier 1. Matches the harness's own emptiness test.

    The harness compares against ``""`` / ``None``. We also treat
    whitespace-only as empty: such a patch reaches the container, applies
    nothing, and cannot turn a FAIL_TO_PASS test green.
    """
    return not patch or not patch.strip()


def modified_paths(patch: str) -> list[str]:
    """Files the patch writes to, from its ``+++`` headers."""
    return [t for t in _TARGET_RE.findall(patch) if t != "/dev/null"]


def creates_new_files(patch: str) -> bool:
    """True if any hunk creates a file (``--- /dev/null``)."""
    return "/dev/null" in _SOURCE_RE.findall(patch)


def all_targets_missing(patch: str, repo_dir: Path) -> bool | None:
    """Tier 2 certainty: every modified file is absent from the checkout.

    Returns None when we cannot tell (no checkout, unparseable targets, or the
    patch creates files) -- the caller must not skip in that case.
    """
    if not repo_dir or not Path(repo_dir).is_dir() or not (Path(repo_dir) / ".git").exists():
        return None
    if creates_new_files(patch):
        return None
    targets = modified_paths(patch)
    if not targets:
        return None
    return all(not (Path(repo_dir) / t).exists() for t in targets)


def applies_cleanly(patch: str, repo_dir: Path) -> bool | None:
    """Strict single-command check, for REPORTING only -- never for gating.

    Only ``git apply --check --verbose`` is a meaningful dry run, and it is
    *stricter* than the harness ladder, so a False here does not license a skip.
    """
    if not repo_dir or not Path(repo_dir).is_dir() or not (Path(repo_dir) / ".git").exists():
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as fh:
        fh.write(patch if patch.endswith("\n") else patch + "\n")
        patch_path = fh.name
    try:
        for cmd in GIT_APPLY_CHECK_CMDS:
            try:
                proc = subprocess.run([*cmd, patch_path], cwd=str(repo_dir), capture_output=True, timeout=60)
            except (OSError, subprocess.TimeoutExpired):
                return None
            if proc.returncode == 0:
                return True
        return False
    finally:
        Path(patch_path).unlink(missing_ok=True)


def gate(patch: str, repo_dir: Path | None = None, *, enabled: bool = True) -> GateDecision:
    """Decide whether a container evaluation can be skipped.

    Conservative: returns ``skip=False`` whenever the outcome is not certain.
    """
    if not enabled:
        return GateDecision(skip=False)

    if is_empty_patch(patch):
        return GateDecision(skip=True, reason="empty_patch")

    if not _DIFF_HEADER.search(patch):
        # No diff headers at all: every command in the ladder needs them, so
        # all three would fail. This is a syntactic certainty, independent of
        # the repository state.
        return GateDecision(skip=True, reason="not_a_unified_diff")

    if repo_dir is not None and all_targets_missing(patch, Path(repo_dir)) is True:
        return GateDecision(skip=True, reason="targets_absent_from_repo")

    return GateDecision(skip=False)


def apply_diagnostics(patch: str, repo_dir: Path) -> tuple[bool | None, tuple[str, ...]]:
    """``applies_cleanly`` plus the reason, for feedback to the refiner.

    The refiner can only repair a non-applying patch if it is told *why* the
    patch did not apply, so the ladder's stderr is returned alongside the
    verdict. Reporting only: a False here never licenses a skip.
    """
    if not repo_dir or not Path(repo_dir).is_dir() or not (Path(repo_dir) / ".git").exists():
        return None, ()
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as fh:
        fh.write(patch if patch.endswith("\n") else patch + "\n")
        patch_path = fh.name
    try:
        last = ""
        for cmd in GIT_APPLY_CHECK_CMDS:
            try:
                proc = subprocess.run([*cmd, patch_path], cwd=str(repo_dir), capture_output=True, timeout=60, text=True)
            except (OSError, subprocess.TimeoutExpired):
                return None, ()
            if proc.returncode == 0:
                return True, ()
            last = (proc.stderr or proc.stdout or "").strip()
        # Keep it short: this text goes into the refiner prompt.
        lines = [ln for ln in last.splitlines() if ln.strip()][:4]
        return False, tuple(lines)
    finally:
        Path(patch_path).unlink(missing_ok=True)
