"""Tests for the dollar-budget stopper and its accounting exclusions."""

from __future__ import annotations

import pytest

from gepa_taxonomy.cost import (
    ALL_REFINER_MODELS,
    ALT2_REFINER_MODEL,
    ALT_REFINER_MODEL,
    BEDROCK_PRICES_USD_PER_TOKEN,
    REFINER_MODEL,
    SONNET_5_POST_INTRO_USD_PER_TOKEN,
    CostMeter,
    MaxTotalCostStopper,
    UnpricedModelError,
    price_call,
)

HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET = "us.anthropic.claude-sonnet-5"


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


def test_price_call_matches_table():
    cin, cout = BEDROCK_PRICES_USD_PER_TOKEN[HAIKU]
    assert price_call(HAIKU, 1_000_000, 0) == pytest.approx(cin * 1_000_000)
    assert price_call(HAIKU, 0, 1_000_000) == pytest.approx(cout * 1_000_000)


def test_bedrock_prefix_is_accepted():
    assert price_call(f"bedrock/{HAIKU}", 1000, 100) == price_call(HAIKU, 1000, 100)


def test_cross_region_profile_carries_premium():
    """`us.` inference profiles cost ~10% more than the base model."""
    base_in, base_out = BEDROCK_PRICES_USD_PER_TOKEN["anthropic.claude-sonnet-5"]
    prof_in, prof_out = BEDROCK_PRICES_USD_PER_TOKEN["us.anthropic.claude-sonnet-5"]
    assert prof_in == pytest.approx(base_in * 1.1)
    assert prof_out == pytest.approx(base_out * 1.1)


def test_unknown_model_raises_rather_than_metering_zero():
    """The critical fail-safe: unpriced models must not be treated as free.

    litellm's completion_cost returns 0.0 for unknown models. If we inherited
    that, the stopper would meter $0 forever and never fire.
    """
    with pytest.raises(UnpricedModelError):
        price_call("anthropic.some-unreleased-model", 1000, 1000)


def test_our_table_agrees_with_litellm():
    """Guards against a typo in the hand-written table."""
    litellm = pytest.importorskip("litellm")
    for model, (cin, cout) in BEDROCK_PRICES_USD_PER_TOKEN.items():
        info = litellm.model_cost.get(model)
        if info is None:
            continue
        assert cin == pytest.approx(info["input_cost_per_token"]), model
        assert cout == pytest.approx(info["output_cost_per_token"]), model


# --------------------------------------------------------------------------
# The costed alternative refiner (Sonnet 4.6) -- for the budget decision
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_REFINER_MODELS)
def test_refiner_rates_cross_checked_against_litellm(model):
    """Every candidate refiner rate is verified offline, not assumed."""
    litellm = pytest.importorskip("litellm")
    cin, cout = BEDROCK_PRICES_USD_PER_TOKEN[model]
    info = litellm.model_cost[model]
    assert cin == pytest.approx(info["input_cost_per_token"])
    assert cout == pytest.approx(info["output_cost_per_token"])
    assert (cin, cout) == (3.00e-6, 15.00e-6)


def test_alt_refiner_is_a_distinct_active_sonnet():
    assert ALT_REFINER_MODEL != REFINER_MODEL
    assert "sonnet" in ALT_REFINER_MODEL


def test_pinned_refiner_rate_is_verified():
    """Sonnet 4.6 at $3/$15, cross-checked against litellm."""
    assert BEDROCK_PRICES_USD_PER_TOKEN[REFINER_MODEL] == (3.00e-6, 15.00e-6)
    assert BEDROCK_PRICES_USD_PER_TOKEN[REFINER_MODEL] == SONNET_5_POST_INTRO_USD_PER_TOKEN


@pytest.mark.parametrize("model", ALL_REFINER_MODELS)
def test_pricing_guards_hold_for_every_refiner(model):
    """Every candidate id must price without raising, so the budget stopper
    stays valid whichever refiner is pinned."""
    assert price_call(model, 10_000, 1_000) > 0
    assert price_call(f"bedrock/{model}", 10_000, 1_000) > 0


def test_alt2_refiner_is_priced_and_cross_checked():
    """Sonnet 4.5 pricing verified offline against litellm, not assumed."""
    litellm = pytest.importorskip("litellm")
    assert ALT2_REFINER_MODEL in BEDROCK_PRICES_USD_PER_TOKEN
    cin, cout = BEDROCK_PRICES_USD_PER_TOKEN[ALT2_REFINER_MODEL]
    info = litellm.model_cost[ALT2_REFINER_MODEL]
    assert cin == pytest.approx(info["input_cost_per_token"])
    assert cout == pytest.approx(info["output_cost_per_token"])
    assert (cin, cout) == (3.00e-6, 15.00e-6), "Sonnet 4.5 is $3/$15 per MTok"


def test_every_available_sonnet_shares_one_rate():
    """Sonnet 4.6 and 4.5 are both $3/$15, so stepping down the remaining
    ladder buys nothing on cost."""
    assert BEDROCK_PRICES_USD_PER_TOKEN[REFINER_MODEL] == BEDROCK_PRICES_USD_PER_TOKEN[ALT_REFINER_MODEL]


def test_all_older_sonnets_share_one_rate():
    assert BEDROCK_PRICES_USD_PER_TOKEN[ALT_REFINER_MODEL] == BEDROCK_PRICES_USD_PER_TOKEN[ALT2_REFINER_MODEL]


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
