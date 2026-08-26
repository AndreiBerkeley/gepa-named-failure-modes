"""Routing lock-in: Bedrock + bearer auth, never direct-Anthropic fallback.

A silent reroute to the direct Anthropic API would hit a different account, a
different price sheet, and different credentials -- and would invalidate the
dollar accounting without any visible error. These tests make that impossible
to do accidentally.

No test here invokes a model. Invocation is billed and stays gated.
"""

from __future__ import annotations

import pytest

from gepa_taxonomy.bedrock import (
    BEARER_ENV,
    BEDROCK_PREFIX,
    BedrockLM,
    CredentialsMissingError,
    RoutingError,
    bedrock_model_id,
    require_credentials,
)
from gepa_taxonomy.cost import (
    ALL_REFINER_MODELS,
    BEDROCK_PRICES_USD_PER_TOKEN,
    REFINER_MODEL,
    SOLVER_MODEL,
)

# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", [SOLVER_MODEL, *ALL_REFINER_MODELS])
def test_configured_models_route_to_bedrock(model):
    """Guards must hold for the pinned refiner AND the costed alternative, so
    swapping refiners is a one-line change with no loss of protection."""
    assert bedrock_model_id(model).startswith(BEDROCK_PREFIX)


@pytest.mark.parametrize("model", [SOLVER_MODEL, *ALL_REFINER_MODELS])
def test_litellm_routes_both_refiners_to_bedrock(model):
    litellm = pytest.importorskip("litellm")
    _, provider, _, _ = litellm.get_llm_provider(bedrock_model_id(model))
    assert provider == "bedrock", f"{model} resolved to {provider!r}"


def test_prefix_is_idempotent():
    once = bedrock_model_id(SOLVER_MODEL)
    assert bedrock_model_id(once) == once
    assert once.count(BEDROCK_PREFIX) == 1


def test_rejects_bare_non_bedrock_ids():
    """The core guard: a bare id that is not a Bedrock Anthropic id is a hard error."""
    for bad in ("claude-sonnet-5", "gpt-4", "gemini-2.5-flash"):
        with pytest.raises(RoutingError):
            bedrock_model_id(bad)


def test_explicit_provider_prefix_passes_through():
    """An explicit prefix is a deliberate routing choice and reaches litellm verbatim."""
    for model in ("gemini/gemini-2.5-flash", "openai/gpt-4", "anthropic/claude-sonnet-5"):
        assert bedrock_model_id(model) == model


def test_rejects_prefix_nested_inside_bedrock():
    with pytest.raises(RoutingError):
        bedrock_model_id("bedrock/openai/gpt-4")


def test_litellm_resolves_prefixed_ids_to_bedrock():
    """litellm itself must agree the routed id is a Bedrock call."""
    litellm = pytest.importorskip("litellm")
    for model in (SOLVER_MODEL, REFINER_MODEL):
        _, provider, _, _ = litellm.get_llm_provider(bedrock_model_id(model))
        assert provider == "bedrock", f"{model} resolved to {provider!r}, not bedrock"


@pytest.mark.parametrize("bare_name", ["claude-sonnet-5", "claude-haiku-4-5"])
def test_bare_model_names_would_bypass_bedrock(bare_name):
    """Demonstrates *why* the routing guard exists.

    litellm resolves a bare first-party model name to provider 'anthropic',
    i.e. api.anthropic.com -- a different account, price sheet, and credential.
    Verified against litellm 1.95.0. Our guard rejects these ids outright.
    """
    litellm = pytest.importorskip("litellm")
    _, provider, _, _ = litellm.get_llm_provider(bare_name)
    assert provider == "anthropic", "premise changed: re-check the routing guard"

    with pytest.raises(RoutingError):
        bedrock_model_id(bare_name)


def test_profile_ids_resolve_to_bedrock_even_unprefixed():
    """Recorded fact, not a licence to drop the prefix.

    litellm 1.95.0 already maps `<region>.anthropic.*` to bedrock. We still add
    `bedrock/` explicitly so routing does not depend on litellm's inference
    heuristics staying the same across versions.
    """
    litellm = pytest.importorskip("litellm")
    for model in (SOLVER_MODEL, REFINER_MODEL):
        _, provider, _, _ = litellm.get_llm_provider(model)
        assert provider == "bedrock"


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------


def test_missing_credentials_fail_fast(monkeypatch):
    """Runs must abort at startup, not error mid-run."""
    monkeypatch.delenv(BEARER_ENV, raising=False)
    with pytest.raises(CredentialsMissingError):
        require_credentials()


def test_credential_error_explains_the_zshrc_wrapper(monkeypatch):
    monkeypatch.delenv(BEARER_ENV, raising=False)
    with pytest.raises(CredentialsMissingError, match="zshrc"):
        require_credentials()


def test_bedrock_lm_construction_requires_credentials(monkeypatch):
    monkeypatch.delenv(BEARER_ENV, raising=False)
    with pytest.raises(CredentialsMissingError):
        BedrockLM(model=SOLVER_MODEL)


def test_bedrock_lm_uses_bearer_auth_not_sigv4_keys(monkeypatch):
    """Bearer token alone must be sufficient -- no AWS access keys required."""
    monkeypatch.setenv(BEARER_ENV, "x" * 32)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    lm = BedrockLM(model=SOLVER_MODEL)
    assert lm.routed_model.startswith(BEDROCK_PREFIX)
    assert lm.region


def test_region_defaults_are_sane(monkeypatch):
    monkeypatch.setenv(BEARER_ENV, "x" * 32)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    assert BedrockLM(model=SOLVER_MODEL).region == "us-west-2"


def test_credentials_never_surface_in_repr(monkeypatch):
    """A token must not leak into logs via an accidental repr()."""
    secret = "SECRET-TOKEN-VALUE-DO-NOT-LOG"
    monkeypatch.setenv(BEARER_ENV, secret)
    lm = BedrockLM(model=SOLVER_MODEL)
    assert secret not in repr(lm)


# --------------------------------------------------------------------------
# Model selection is priced and profile-based
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", [SOLVER_MODEL, REFINER_MODEL])
def test_selected_models_are_priced(model):
    """An unpriced model would make the budget stopper fail open."""
    assert model in BEDROCK_PRICES_USD_PER_TOKEN


@pytest.mark.parametrize("model", [SOLVER_MODEL, REFINER_MODEL])
def test_selected_models_use_an_inference_profile(model):
    """Both are INFERENCE_PROFILE-only on this account: the bare
    `anthropic.*` id is not directly invocable, so a profile prefix is
    mandatory. Verified via ListFoundationModels on 2026-08-07."""
    assert model.startswith(("global.anthropic.", "us.anthropic."))


def test_global_profile_is_cheaper_than_us_profile():
    """Documents why we picked `global.`: it is the base rate, while `us.`
    carries a ~10% cross-region premium."""
    g = BEDROCK_PRICES_USD_PER_TOKEN["global.anthropic.claude-sonnet-4-6"]
    u = BEDROCK_PRICES_USD_PER_TOKEN["us.anthropic.claude-sonnet-4-6"]
    assert g[0] < u[0] and g[1] < u[1]


def test_sonnet_5_is_removed_from_consideration():
    """Permanently unavailable to this account (403 on invocation)."""
    from gepa_taxonomy.cost import ALL_REFINER_MODELS, UNAVAILABLE_MODELS

    assert REFINER_MODEL not in UNAVAILABLE_MODELS
    assert not any("sonnet-5" in m for m in ALL_REFINER_MODELS)


def test_profile_prefix_is_configurable():
    """A profile-scoped authorization problem must be a one-line fix."""
    from gepa_taxonomy.cost import REFINER_BASE, with_profile

    assert with_profile(REFINER_BASE, "us.").startswith("us.")
    assert with_profile(REFINER_BASE, "global.").startswith("global.")
    # idempotent: never double-prefix
    once = with_profile(REFINER_BASE, "us.")
    assert with_profile(once, "global.") == once


def test_solver_is_haiku_and_refiner_is_sonnet():
    assert "haiku" in SOLVER_MODEL
    assert "sonnet" in REFINER_MODEL
