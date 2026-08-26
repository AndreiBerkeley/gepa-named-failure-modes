"""LiveBench-Math scorers. FREE: pure functions, no network.

These are ported from upstream (see grading.py), so the tests exist to pin the
port against the original's behaviour -- and to pin the ONE place we knowingly
diverge.
"""

from __future__ import annotations

import pytest

from gepa_taxonomy.livebench_math.grading import (
    aime_process_results,
    answer_feedback,
    extract_expression_completions,
    grade,
    mathcontest_process_results,
    proof_rearrangement_process_results,
    score_feedback,
    scorer_for,
)

AMC_STATEMENT = (
    r"What is $x$? $\textbf{(A) }5\qquad\textbf{(B) }6\qquad\textbf{(C) }7\qquad\textbf{(D) }8\qquad\textbf{(E) }9$"
)


class TestScorerRouting:
    @pytest.mark.parametrize(
        ("subtask", "expected"),
        [
            ("aime_i_2024", "aime"),
            ("aime_ii_2024", "aime"),
            ("imo", "olympiad"),
            ("usamo", "olympiad"),
            ("amc_12a_2023", "multiple_choice"),
            ("updated_amc_12b_2023", "multiple_choice"),
            ("smc", "multiple_choice"),
        ],
    )
    def test_every_subtask_in_the_pool_routes(self, subtask, expected):
        assert scorer_for(subtask) == expected


class TestMultipleChoice:
    def test_repeated_letter_scores(self):
        assert mathcontest_process_results("C", "...so the answer is CCCCC", AMC_STATEMENT) == 1.0

    def test_boxed_letter_scores(self):
        assert mathcontest_process_results("D", r"Therefore $\boxed{D}$", AMC_STATEMENT) == 1.0

    def test_solution_tags_score(self):
        assert mathcontest_process_results("B", "<solution>BBBBB</solution>", AMC_STATEMENT) == 1.0

    def test_last_line_letter_scores(self):
        assert mathcontest_process_results("A", "long reasoning\nA", AMC_STATEMENT) == 1.0

    def test_answer_value_from_the_statement_scores(self):
        """The letter is never emitted, but its answer TEXT is."""
        assert mathcontest_process_results("C", "After simplifying we get 7", AMC_STATEMENT) == 1.0

    def test_a_wrong_letter_scores_zero(self):
        assert mathcontest_process_results("C", r"$\boxed{B}$", AMC_STATEMENT) == 0.0

    def test_a_non_letter_ground_truth_is_a_routing_bug_not_a_zero(self):
        """Upstream raises here, and so do we: silently scoring 0 would make a
        mis-routed AIME question look like a wrong answer on every rollout."""
        with pytest.raises(ValueError):
            mathcontest_process_results("025", "025", AMC_STATEMENT)


class TestAIME:
    def test_answer_in_the_tail_scores(self):
        assert aime_process_results("025", "...therefore the answer is 025") == 1.0

    def test_answer_far_from_the_end_does_not_score(self):
        """Upstream only looks at the last 50 characters."""
        assert aime_process_results("025", "025" + "x" * 200) == 0.0

    def test_wrong_answer_scores_zero(self):
        assert aime_process_results("025", "the answer is 026") == 0.0


class TestOlympiadPartialCredit:
    def test_a_fully_correct_ordering_scores_one(self):
        assert proof_rearrangement_process_results("1,2,3", "Answer: 1, 2, 3") == 1.0

    def test_partial_credit_is_fractional(self):
        # positions 0 and 1 right, position 2 wrong -> 2/3
        assert proof_rearrangement_process_results("1,2,3", "Answer: 1, 2, 9") == pytest.approx(2 / 3)

    def test_nothing_parsed_scores_zero(self):
        assert proof_rearrangement_process_results("1,2,3", "I could not solve this.") == 0.0

    def test_a_short_answer_CANNOT_score_one(self):
        """The documented deviation from upstream, and the reason for it.

        Upstream divides by len(completions), so emitting only the two positions
        you are confident about scores 2/2 = 1.0 on a 7-position question. GEPA
        optimizes this number directly, so that is a prompt-level edit that buys
        a perfect score without solving anything -- and it would lift both arms,
        flattening the partial-credit signal olympiad was kept for.
        """
        score = proof_rearrangement_process_results("1,2,3,4,5,6,7", "Answer: 1, 2")
        assert score == pytest.approx(2 / 7), "short answers must not be rewarded"
        assert score < 1.0

    def test_extra_answers_beyond_the_truth_are_ignored_not_rewarded(self):
        assert proof_rearrangement_process_results("1,2", "Answer: 1, 2, 3, 4, 5") == 1.0

    @pytest.mark.parametrize(
        "generation",
        [
            "Answer: 1, 2, 3",
            r"The ordering is $\boxed{1, 2, 3}$",
            "reasoning here\n1, 2, 3",
        ],
    )
    def test_each_upstream_extraction_fallback_works(self, generation):
        assert proof_rearrangement_process_results("1,2,3", generation) == 1.0

    def test_unparseable_tokens_become_sentinels_not_silent_matches(self):
        completions = extract_expression_completions("Answer: 1, banana, 3")
        assert "NO ANSWER" in completions
        # The sentinel holds its POSITION -- it neither matches its own slot nor
        # shifts the ones after it, so 1 and 3 still score and only 2 is lost.
        assert proof_rearrangement_process_results("1,2,3", "Answer: 1, banana, 3") == pytest.approx(2 / 3)


class TestGradeRouter:
    def test_olympiad_reports_positions(self):
        g = grade("Answer: 1, 2, 9", "1,2,3", subtask="imo", question="")
        assert g.scorer == "olympiad"
        assert g.positions == (2, 3)
        assert g.score == pytest.approx(2 / 3)

    def test_multiple_choice_routes_and_scores(self):
        g = grade(r"$\boxed{C}$", "C", subtask="amc_12a_2023", question=AMC_STATEMENT)
        assert (g.scorer, g.score) == ("multiple_choice", 1.0)

    def test_aime_routes_and_scores(self):
        g = grade("the answer is 025", "025", subtask="aime_i_2024", question="")
        assert (g.scorer, g.score) == ("aime", 1.0)

    def test_an_empty_answer_never_raises(self):
        """A dead rollout must score, not crash: gepa's contract wants a failure
        score plus a trajectory, not an exception."""
        for subtask in ("imo", "aime_i_2024", "amc_12a_2023"):
            assert grade("", "1,2,3" if subtask == "imo" else "C", subtask=subtask, question=AMC_STATEMENT).score == 0.0


class TestFeedback:
    def test_gold_feedback_names_the_correct_answer(self):
        g = grade(r"$\boxed{B}$", "C", subtask="amc_12a_2023", question=AMC_STATEMENT)
        assert "C" in answer_feedback(g, "C")

    def test_gold_free_feedback_never_contains_the_answer(self):
        g = grade(r"$\boxed{B}$", "C", subtask="amc_12a_2023", question=AMC_STATEMENT)
        text = score_feedback(g)
        assert "Correct answer" not in text

    def test_unparseable_output_is_reported_as_such(self):
        """A response that reasons correctly but never emits the required format
        scores 0. Without saying so, the optimizer cannot tell that apart from
        being wrong -- and only one of those is fixable by a prompt."""
        g = grade("I think it is seven.", "C", subtask="amc_12a_2023", question="no choices here")
        assert "No answer could be parsed" in score_feedback(g)

    def test_olympiad_feedback_reports_the_fraction(self):
        g = grade("Answer: 1, 2, 9", "1,2,3", subtask="imo", question="")
        assert "2 of 3" in score_feedback(g)
