"""The skip must be provably score-identical to the harness's own behaviour.

These tests check the claim against swebench 4.1.0 source, not against my
description of it -- if the harness changes, they fail.
"""

from __future__ import annotations

import subprocess

import pytest

from gepa_taxonomy.patch_gate import (
    GIT_APPLY_CMDS,
    applies_cleanly,
    gate,
)

GOOD_PATCH = """--- a/mod.py
+++ b/mod.py
@@ -1,1 +1,1 @@
-x = 1
+x = 2
"""

BAD_CONTEXT_PATCH = """--- a/mod.py
+++ b/mod.py
@@ -1,1 +1,1 @@
-this line is not in the file at all
+replacement
"""

MISSING_FILE_PATCH = """--- a/does_not_exist.py
+++ b/does_not_exist.py
@@ -1,1 +1,1 @@
-a
+b
"""

CREATION_PATCH = """--- /dev/null
+++ b/brand_new.py
@@ -0,0 +1,1 @@
+x = 1
"""


@pytest.fixture
def repo(tmp_path):
    """A real git checkout, so the apply ladder is exercised for real."""
    d = tmp_path / "repo"
    d.mkdir()
    (d / "mod.py").write_text("x = 1\n")
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "init"],
    ):
        subprocess.run(cmd, cwd=d, check=True, capture_output=True)
    return d


# --------------------------------------------------------------------------
# The equivalence claim, checked against harness source
# --------------------------------------------------------------------------


def test_our_apply_ladder_matches_the_harness_verbatim():
    """If swebench changes its commands, our skip stops being equivalent."""
    import pathlib

    swebench = pytest.importorskip("swebench")

    src = (pathlib.Path(swebench.__file__).parent / "harness" / "run_evaluation.py").read_text()
    i = src.index("GIT_APPLY_CMDS = [")
    block = src[i : src.index("]", i) + 1]
    for cmd in GIT_APPLY_CMDS:
        assert f'"{cmd}"' in block, f"harness no longer uses {cmd!r} -- re-derive the gate"
    # And we must not have invented extra commands the harness does not try.
    assert block.count('"') // 2 == len(GIT_APPLY_CMDS), "harness command count changed"


def test_harness_excludes_empty_patches_before_running_anything():
    """Tier 1 is exactly what the harness does -- verified in its source."""
    import pathlib

    swebench = pytest.importorskip("swebench")

    src = (pathlib.Path(swebench.__file__).parent / "harness" / "run_evaluation.py").read_text()
    assert "empty_patch_ids" in src
    assert 'v[KEY_PREDICTION] == ""' in src
    assert "not in empty_patch_ids" in src


def test_harness_raises_on_apply_failure_so_score_is_zero():
    """Tier 2: all commands failing raises EvaluationError -> never resolved."""
    import pathlib

    swebench = pytest.importorskip("swebench")

    src = (pathlib.Path(swebench.__file__).parent / "harness" / "run_evaluation.py").read_text()
    assert "if not applied_patch:" in src
    assert "raise EvaluationError" in src


# --------------------------------------------------------------------------
# Tier 1 -- empty
# --------------------------------------------------------------------------


@pytest.mark.parametrize("patch", ["", "   ", "\n\n", "\t\n "])
def test_empty_patches_are_skipped(patch):
    d = gate(patch)
    assert d.skip and d.reason == "empty_patch"
    assert d.score == 0.0


def test_skipped_score_is_always_zero():
    assert gate("").score == 0.0
    assert gate("garbage with no diff headers").score == 0.0


# --------------------------------------------------------------------------
# Tier 2 -- non-applying
# --------------------------------------------------------------------------


def test_non_diff_text_is_skipped_without_a_repo():
    """Syntactic certainty: no headers means every command in the ladder fails."""
    d = gate("I could not find the bug, sorry.")
    assert d.skip and d.reason == "not_a_unified_diff"


def test_applying_patch_is_not_skipped(repo):
    assert applies_cleanly(GOOD_PATCH, repo) is True
    assert gate(GOOD_PATCH, repo).skip is False


def test_patch_on_missing_file_is_skipped(repo):
    """The tier-2 certainty: no hunk can apply to a file that is not there."""
    d = gate(MISSING_FILE_PATCH, repo)
    assert d.skip and d.reason == "targets_absent_from_repo"


def test_bad_context_patch_is_NOT_skipped(repo):
    """Deliberately conservative.

    `git apply --check` rejects this, but the harness ladder's third command
    (`patch --batch --fuzz=5`) may still apply it. Skipping would risk
    understating a score, so we run the container.
    """
    assert applies_cleanly(BAD_CONTEXT_PATCH, repo) is False  # strict check fails
    assert gate(BAD_CONTEXT_PATCH, repo).skip is False  # but we do not skip


def test_check_reject_is_useless_as_a_signal(repo, tmp_path):
    """Records WHY tier 2 cannot replicate the harness ladder.

    `git apply --check --reject` exits 0 for everything -- even a patch on a
    nonexistent file -- so a check-based ladder would never fire.
    """
    p = tmp_path / "m.diff"
    p.write_text(MISSING_FILE_PATCH)
    rc = subprocess.run(
        ["git", "apply", "--check", "--verbose", "--reject", str(p)],
        cwd=repo,
        capture_output=True,
    ).returncode
    assert rc == 0, "premise changed: --check --reject now discriminates; revisit the gate"


def test_creation_patch_is_not_skipped(repo):
    """A file-creating diff legitimately targets an absent path."""
    assert gate(CREATION_PATCH, repo).skip is False


def test_gate_is_conservative_without_a_checkout():
    """A well-formed patch with no repo to check against must NOT be skipped."""
    assert gate(GOOD_PATCH, None).skip is False


def test_gate_is_conservative_when_the_path_is_not_a_repo(tmp_path):
    assert applies_cleanly(GOOD_PATCH, tmp_path) is None
    assert gate(GOOD_PATCH, tmp_path).skip is False


def test_gate_can_be_disabled():
    assert gate("", enabled=False).skip is False


def test_real_git_agrees_with_our_verdict(repo):
    """Cross-check the positive case against real git."""
    p = repo / "p.diff"
    p.write_text(GOOD_PATCH)
    rc = subprocess.run(["git", "apply", "--check", "--verbose", str(p)], cwd=repo, capture_output=True).returncode
    assert rc == 0, "premise: this patch must genuinely apply"
    assert gate(GOOD_PATCH, repo).skip is False
