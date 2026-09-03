"""Tests for the dollar-budget stopper and its accounting exclusions."""

from __future__ import annotations

import pytest

from gepa_taxonomy.cost import (
    PRICE_ENV,
    PRICE_OVERRIDES,
    CostMeter,
    MaxTotalCostStopper,
    UnpricedModelError,
    assert_priced,
    load_price_overrides,
    lookup_price,
    parse_price_spec,
    price_call,
    set_price,
)

HAIKU = "test-model-small"
SONNET = "test-model-large"


@pytest.fixture(autouse=True)
def _prices():
    """Every test prices two synthetic models through the override path."""
    PRICE_OVERRIDES.clear()
    set_price(HAIKU, 1.0, 5.0)
    set_price(SONNET, 3.0, 15.0)
    yield
    PRICE_OVERRIDES.clear()


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


def test_price_call_uses_registered_prices():
    assert price_call(HAIKU, 1_000_000, 0) == pytest.approx(1.0)
    assert price_call(HAIKU, 0, 1_000_000) == pytest.approx(5.0)


def test_bedrock_prefix_is_accepted():
    assert price_call(f"bedrock/{HAIKU}", 1000, 100) == price_call(HAIKU, 1000, 100)


def test_provider_prefix_and_bare_id_agree():
    set_price("anthropic/claude-example", 3.0, 15.0)
    assert price_call("anthropic/claude-example", 1000, 1000) == price_call("claude-example", 1000, 1000)


def test_litellm_table_prices_the_default_model():
    pytest.importorskip("litellm")
    assert lookup_price("gpt-5-mini") is not None
    assert price_call("openai/gpt-5-mini", 1000, 1000) == price_call("gpt-5-mini", 1000, 1000) > 0


def test_override_beats_litellm_table():
    pytest.importorskip("litellm")
    set_price("gpt-5-mini", 100.0, 100.0)
    assert price_call("gpt-5-mini", 1_000_000, 0) == pytest.approx(100.0)


def test_unknown_model_raises_rather_than_metering_zero():
    """The critical fail-safe: unpriced models must not be treated as free.

    litellm's completion_cost returns 0.0 for unknown models. If we inherited
    that, the stopper would meter $0 forever and never fire.
    """
    with pytest.raises(UnpricedModelError, match="--price"):
        price_call("nobody/some-unreleased-model", 1000, 1000)


def test_assert_priced_fails_before_any_spend():
    assert_priced(HAIKU, SONNET)
    with pytest.raises(UnpricedModelError):
        assert_priced(HAIKU, "nobody/unpriced")


def test_parse_price_spec():
    assert parse_price_spec("gemini/gemini-3.5-flash=1.50,9.00") == ("gemini/gemini-3.5-flash", 1.5, 9.0)
    for bad in ("no-equals", "m=1", "m=a,b", "=1,2", "m=-1,2"):
        with pytest.raises(ValueError):
            parse_price_spec(bad)


def test_load_price_overrides_from_flags_and_env(monkeypatch):
    monkeypatch.setenv(PRICE_ENV, "env-model=2,4;other=1,1")
    load_price_overrides(["flag-model=0.5,1.5"])
    assert price_call("env-model", 1_000_000, 1_000_000) == pytest.approx(6.0)
    assert price_call("other", 1_000_000, 0) == pytest.approx(1.0)
    assert price_call("flag-model", 0, 1_000_000) == pytest.approx(1.5)


# --------------------------------------------------------------------------
# Metering and exclusions
# --------------------------------------------------------------------------


def test_meter_accumulates_budgeted_spend():
    m = CostMeter()
    m.record(model=HAIKU, input_tokens=10_000, output_tokens=1_000)
    m.record(model=SONNET, input_tokens=10_000, output_tokens=1_000)
    assert m.calls == 2
    assert m.budgeted_usd > 0
    assert m.excluded_usd == 0.0


@pytest.mark.parametrize("phase", ["seed_val", "final_test", "generation"])
def test_excluded_phases_do_not_count_against_budget(phase):
    """Exclusion (1): seed val eval, final test eval, and generation runs are
    shared one-time pipeline costs and must not consume the per-seed budget."""
    m = CostMeter()
    m.record(model=SONNET, input_tokens=1_000_000, output_tokens=100_000, phase=phase)
    assert m.budgeted_usd == 0.0
    assert m.excluded_usd > 0.0
    assert m.total_usd == pytest.approx(m.excluded_usd)


def test_stopper_ignores_excluded_spend_entirely():
    """A large excluded charge must not move the stopper one inch."""
    m = CostMeter()
    stopper = MaxTotalCostStopper(budget_usd=1.0, meters=m)
    m.record(model=SONNET, input_tokens=5_000_000, output_tokens=500_000, phase="final_test")
    assert m.excluded_usd > 10.0  # far past the budget, if it had counted
    assert stopper(None) is False
    assert stopper.realised_usd == 0.0


def test_stopper_fires_on_budgeted_spend():
    m = CostMeter()
    stopper = MaxTotalCostStopper(budget_usd=0.05, meters=m)
    assert stopper(None) is False
    # ~$0.0303 for haiku at 20k in / 1.5k out
    m.record(model=HAIKU, input_tokens=20_000, output_tokens=1_500, phase="optimization")
    assert stopper(None) is False
    m.record(model=HAIKU, input_tokens=20_000, output_tokens=1_500, phase="optimization")
    assert stopper(None) is True
    assert stopper.fired_at_usd == pytest.approx(stopper.realised_usd)


def test_stopper_sums_multiple_meters():
    """Solver and refiner meter separately; the budget covers their sum."""
    solver, refiner = CostMeter(), CostMeter()
    stopper = MaxTotalCostStopper(budget_usd=0.05, meters=[solver, refiner])
    solver.record(model=HAIKU, input_tokens=20_000, output_tokens=1_500)
    refiner.record(model=HAIKU, input_tokens=20_000, output_tokens=1_500)
    assert stopper.realised_usd == pytest.approx(solver.budgeted_usd + refiner.budgeted_usd)
    assert stopper(None) is True


def test_stopper_requires_a_meter():
    with pytest.raises(ValueError):
        MaxTotalCostStopper(budget_usd=1.0, meters=[])


def test_stopper_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        MaxTotalCostStopper(budget_usd=0.0, meters=CostMeter())


# --------------------------------------------------------------------------
# Behaviour neutrality
# --------------------------------------------------------------------------


def test_does_not_expose_max_metric_calls():
    """The one concrete neutrality constraint.

    gepa's engine duck-types on an attribute named `max_metric_calls` in two
    places (engine.py:1002 `_get_remaining_budget`, engine.py:564 the tqdm
    total). Both are reporting-only, but exposing the name would hijack the
    progress bar and the BudgetUpdatedEvent field.
    """
    stopper = MaxTotalCostStopper(budget_usd=1.0, meters=CostMeter())
    assert not hasattr(stopper, "max_metric_calls")
    assert getattr(stopper, "max_metric_calls", None) is None


def test_satisfies_gepa_stopper_protocol():
    stopper = MaxTotalCostStopper(budget_usd=1.0, meters=CostMeter())
    from gepa.utils.stop_condition import StopperProtocol

    assert isinstance(stopper, StopperProtocol)


def test_composes_with_gepa_composite_stopper():
    """We must be able to add a wall-clock safety net alongside the budget."""
    from gepa.utils.stop_condition import CompositeStopper, TimeoutStopCondition

    m = CostMeter()
    composite = CompositeStopper(
        MaxTotalCostStopper(budget_usd=0.05, meters=m),
        TimeoutStopCondition(timeout_seconds=1e9),
        mode="any",
    )
    assert composite(None) is False
    m.record(model=SONNET, input_tokens=100_000, output_tokens=10_000)
    assert composite(None) is True


def test_stopper_does_not_read_state():
    """It must observe only its own meters, never the optimization state."""

    class Exploding:
        def __getattr__(self, name):
            raise AssertionError(f"stopper read GEPAState.{name}")

    stopper = MaxTotalCostStopper(budget_usd=1.0, meters=CostMeter())
    assert stopper(Exploding()) is False
