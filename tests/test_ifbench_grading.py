"""IFBench scoring. FREE: vendored verifiers run offline, no network.

The verifiers themselves are AllenAI's and are not re-tested here -- they ship
with their own suite. What IS tested is everything we wrapped around them:
routing, the strict/loose split, partial credit, error containment, and the
gold boundary in feedback.
"""

from __future__ import annotations

import pytest

from gepa_taxonomy.ifbench.grading import Grade, constraint_feedback, grade, report, score_feedback
from gepa_taxonomy.ifbench.tasks import Gold, constraint_family

# "between 5 and 10 words" -- a verifier with simple, checkable behaviour.
WORDS_5_10 = Gold(
    example_id="t1",
    instruction_ids=("count:word_count_range",),
    kwargs=({"min_words": 5, "max_words": 10},),
)
TWO_CONSTRAINTS = Gold(
    example_id="t2",
    instruction_ids=("count:word_count_range", "format:title_case"),
    kwargs=({"min_words": 5, "max_words": 10}, {}),
)


class TestSingleConstraint:
    def test_a_compliant_response_scores_one(self):
        g = grade("one two three four five six seven", WORDS_5_10, prompt="")
        assert g.score == 1.0
        assert g.all_followed
        assert g.failed_ids == ()

    def test_a_non_compliant_response_scores_zero(self):
        g = grade("too short", WORDS_5_10, prompt="")
        assert g.score == 0.0
        assert not g.all_followed
        assert g.failed_ids == ("count:word_count_range",)

    def test_an_empty_response_scores_zero_and_does_not_raise(self):
        """A dead rollout must score, not crash -- gepa's contract wants a
        failure score plus a trajectory, not an exception."""
        assert grade("", WORDS_5_10, prompt="").score == 0.0


class TestPartialCredit:
    def test_one_of_two_constraints_scores_a_half(self):
        """The only place partial credit exists: 44 of 300 instances."""
        g = grade("one two three four five six seven", TWO_CONSTRAINTS, prompt="")
        assert 0.0 < g.score < 1.0
        assert g.score == pytest.approx(0.5)
        assert not g.all_followed
        assert len(g.followed) == 2

    def test_prompt_level_and_instruction_level_diverge(self):
        """Instruction-level is what GEPA selects on precisely BECAUSE it can
        report something other than 0 or 1 here."""
        g = grade("one two three four five six seven", TWO_CONSTRAINTS, prompt="")
        assert g.score == 0.5
        assert g.all_followed is False


class TestStrictVersusLoose:
    def test_loose_forgives_a_preamble_that_strict_rejects(self):
        """A chatty first line breaks the word count under strict but is stripped
        by one of upstream's loose variants. The GAP is the diagnostic: it says
        the failure is formatting, which is a different fix from non-compliance."""
        response = "Sure! Here is my answer:\none two three four five six seven"
        g = grade(response, WORDS_5_10, prompt="")
        assert g.score == 0.0, "strict should count the preamble's words"
        assert g.loose_score == 1.0, "loose should strip the first line"

    def test_a_clean_response_scores_the_same_both_ways(self):
        g = grade("one two three four five six seven", WORDS_5_10, prompt="")
        assert g.score == g.loose_score == 1.0


class TestErrorContainment:
    def test_an_unknown_verifier_is_recorded_not_raised(self):
        """A verifier that blows up is scored as not-followed AND recorded. It
        must never propagate: one bad instance would kill a whole evaluation.
        It must never be silent either -- the base val refuses to freeze a cache
        with any verifier error, because a broken verifier depresses one
        constraint class in BOTH arms without failing loudly."""
        bad = Gold(example_id="t3", instruction_ids=("does:not_exist",), kwargs=({},))
        g = grade("anything at all here", bad, prompt="")
        assert g.score == 0.0
        assert len(g.errors) == 1
        assert "does:not_exist" in g.errors[0]

    def test_a_gold_with_no_constraints_scores_zero_without_dividing_by_zero(self):
        g = grade("anything", Gold(example_id="t4", instruction_ids=(), kwargs=()), prompt="")
        assert g.score == 0.0


class TestFeedback:
    def test_train_feedback_names_the_failed_constraint(self):
        """Deliberately strong, on the recorded requirement principle. PLAN.md ruled IFBench out
        partly because the diagnosis is already in the baseline feedback --
        weakening it to flatter the taxonomy would be the rigged comparison that
        objection warns about."""
        g = grade("too short", WORDS_5_10, prompt="")
        text = constraint_feedback(g, WORDS_5_10)
        assert "count:word_count_range" in text
        assert "FAIL" in text

    def test_val_feedback_withholds_which_constraint_failed(self):
        g = grade("too short", WORDS_5_10, prompt="")
        text = score_feedback(g, WORDS_5_10)
        assert "count:word_count_range" not in text
        assert "0 of 1" in text

    def test_val_feedback_flags_a_formatting_failure_without_naming_gold(self):
        g = grade("Sure! Here is my answer:\none two three four five six seven", WORDS_5_10, prompt="")
        text = score_feedback(g, WORDS_5_10)
        assert "surrounding text" in text
        assert "count:word_count_range" not in text

    def test_a_fully_compliant_response_is_told_so(self):
        g = grade("one two three four five six seven", WORDS_5_10, prompt="")
        assert "All 1 constraint(s) satisfied" in constraint_feedback(g, WORDS_5_10)


class TestReport:
    def test_report_breaks_the_headline_down_by_family(self):
        """A single mean hides that `format:` is prompt-addressable while
        `words:consonants` is character-level counting."""
        golds = [WORDS_5_10, TWO_CONSTRAINTS]
        grades = [
            grade("one two three four five six seven", WORDS_5_10, prompt=""),
            grade("too short", TWO_CONSTRAINTS, prompt=""),
        ]
        out = report(grades, golds)
        assert set(out["by_family"]) == {"count", "format"}
        assert out["by_family"]["count"]["n"] == 2
        assert out["verifier_errors"] == 0

    def test_report_is_empty_rather_than_dividing_by_zero(self):
        assert report([], []) == {}


class TestFamily:
    @pytest.mark.parametrize(
        ("instruction_id", "expected"),
        [("count:word_count_range", "count"), ("words:consonants", "words"), ("", "unknown")],
    )
    def test_family_is_the_prefix(self, instruction_id, expected):
        assert constraint_family(instruction_id) == expected


class TestGradeShape:
    def test_families_failed_deduplicates(self):
        g = Grade(score=0.0, all_followed=False, failed_ids=("count:a", "count:b", "format:c"))
        assert g.families_failed == ("count", "format")
