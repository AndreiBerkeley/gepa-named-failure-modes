"""Tests for the AppWorld HTTP client. Offline: no server, no model."""

from __future__ import annotations

import pytest

from gepa_taxonomy.appworld.client import NO_OP_LABEL, AppWorldClient, TaskResult


def _payload(passes, failures, success=False, num_tests=None):
    return {
        "success": success,
        "difficulty": 1,
        "num_tests": num_tests if num_tests is not None else len(passes) + len(failures),
        "passes": passes,
        "failures": failures,
    }


def test_no_op_passes_are_excluded_from_the_score():
    """A do-nothing agent passes 'assert no model changes' and would otherwise
    score 0.50 on a 2-requirement task -- inflating the floor and compressing the
    range the optimizer selects on. Measured on a real task."""
    result = TaskResult.from_payload(
        "t1",
        _payload(
            passes=[{"requirement": "assert no model changes.", "label": NO_OP_LABEL}],
            failures=[{"requirement": "assert answers match."}],
        ),
    )
    assert result.score == 0.0, "a do-nothing rollout must not earn partial credit"
    assert result.no_op_passes == ("assert no model changes.",)
    assert result.passes == ()


def test_substantive_passes_earn_partial_credit():
    result = TaskResult.from_payload(
        "t1",
        _payload(
            passes=[
                {"requirement": "assert answers match."},
                {"requirement": "assert no model changes.", "label": NO_OP_LABEL},
            ],
            failures=[{"requirement": "assert email sent."}],
        ),
    )
    # 1 substantive pass, 1 failure -> 0.5. The no-op is in neither term.
    assert result.score == 0.5
    assert result.num_tests == 3


def test_full_credit_when_every_substantive_requirement_passes():
    result = TaskResult.from_payload(
        "t1",
        _payload(
            passes=[{"requirement": "a"}, {"requirement": "b"}],
            failures=[],
            success=True,
        ),
    )
    assert result.score == 1.0
    assert result.success is True


def test_success_is_kept_separate_from_score():
    """TGC is the reported metric and must not be redefined by our selection
    metric; the two are allowed to disagree."""
    result = TaskResult.from_payload(
        "t1",
        _payload(passes=[{"requirement": "a"}], failures=[{"requirement": "b"}], success=False),
    )
    assert result.score == 0.5 and result.success is False


def test_all_no_op_falls_back_to_appworld_verdict():
    """Nothing substantive to grade -- inventing a score from an empty set would
    be worse than deferring to AppWorld's own answer."""
    passes = [{"requirement": "no change", "label": NO_OP_LABEL}]
    assert TaskResult.from_payload("t", _payload(passes, [], success=True)).score == 1.0
    assert TaskResult.from_payload("t", _payload(passes, [], success=False)).score == 0.0


def test_plain_string_requirements_are_tolerated():
    result = TaskResult.from_payload("t1", _payload(passes=["a"], failures=["b"]))
    assert result.score == 0.5
    assert result.passes == ("a",)


def test_client_reports_unhealthy_rather_than_raising_when_the_server_is_down():
    client = AppWorldClient(base_url="http://localhost:59999")
    assert client.health() is False


def test_unreachable_server_raises_a_clear_error():
    from gepa_taxonomy.appworld.client import AppWorldServerError

    client = AppWorldClient(base_url="http://localhost:59999", max_retries=1)
    with pytest.raises(AppWorldServerError, match="unreachable"):
        client.initialize("t1")
