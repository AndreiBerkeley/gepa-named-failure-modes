# Phase 1 — Bedrock model IDs, pricing, and routing (verified, not assumed)

Date: 2026-08-07 · All checks below are free: control-plane listings and offline
pricing-map inspection. **No model was invoked.**

## (a) What is actually enabled on this Bedrock account

Bearer auth (`AWS_BEARER_TOKEN_BEDROCK`) **does** cover the control plane, so I
was able to run the listing directly — no need to hand it to you.
`boto3.client("bedrock").list_foundation_models(byProvider="anthropic")`,
region `us-east-1`, botocore 1.43.67: **15 Anthropic models visible.**

Reproduce with `scripts/list_bedrock_models.py`.

### The two we need — newest of each class, both ACTIVE

| Class | Model ID | Status | Inference types |
|---|---|---|---|
| Haiku | `anthropic.claude-haiku-4-5-20251001-v1:0` | ACTIVE | **INFERENCE_PROFILE only** |
| Sonnet | `anthropic.claude-sonnet-5` | ACTIVE | **INFERENCE_PROFILE only** |

Haiku 4.5 is the newest Haiku-class model in existence (there is no Haiku 5).
Sonnet 5 is the newest Sonnet.

### ⚠️ Both are inference-profile-only — the bare ID is not invocable

Neither supports `ON_DEMAND`, so `anthropic.claude-sonnet-5` cannot be called
directly; a profile prefix is mandatory. Two prefixes are available, and **they
are priced differently**:

| Prefix | Rate | Example |
|---|---|---|
| `global.` | base | `global.anthropic.claude-sonnet-5` |
| `us.` | base **+10%** (US cross-region premium) | `us.anthropic.claude-sonnet-5` |

**Decision: use `global.`** — same models, 10% cheaper across the whole
programme. Pinned in `cost.py` as `SOLVER_MODEL` / `REFINER_MODEL`.

## (b) Does `litellm.completion_cost` price these correctly?

**Yes** — verified offline against litellm 1.95.0, both with and without the
`bedrock/` prefix, matching the pricing map to the cent:

```
us.anthropic.claude-haiku-4-5-20251001-v1:0    20k in / 1.5k out -> $0.030250  OK
us.anthropic.claude-sonnet-5                   20k in / 1.5k out -> $0.060500  OK
bedrock/... (both)                             identical                       OK
```

### But we built the explicit price table anyway, and it is load-bearing

Two reasons `completion_cost` alone is not safe enough for a budget stopper:

1. **It returns `0.0` for unknown models rather than raising.** gepa's own
   `gskill/cost_tracker.py` even wraps it in `except: cost = 0.0`. A stopper
   metering $0 never fires — the budget fails *open*, which is the worst
   possible failure for a spend control. `cost.py` raises `UnpricedModelError`
   instead, covered by
   `test_unknown_model_raises_rather_than_metering_zero`.
2. **Sonnet 5 is on introductory pricing that ends 2026-08-31** — $2/$10 per
   MTok now, $3/$15 after. That is **24 days from today**. Pinning the rate we
   budgeted against stops a mid-experiment repricing from silently changing our
   accounting. See the cost-implications note below.

`test_our_table_agrees_with_litellm` cross-checks every table entry against
litellm, so the pin cannot silently drift from reality.

### Verified prices (USD per million tokens)

| Model | in | out |
|---|---:|---:|
| `global.anthropic.claude-haiku-4-5-20251001-v1:0` | $1.00 | $5.00 |
| `us.anthropic.claude-haiku-4-5-20251001-v1:0` | $1.10 | $5.50 |
| `global.anthropic.claude-sonnet-5` | $2.00 | $10.00 |
| `us.anthropic.claude-sonnet-5` | $2.20 | $11.00 |

> **⚠️ Budget risk to decide on.** Sonnet 5's intro pricing ends **2026-08-31**.
> The refiner is Sonnet, and it is ~67% of per-rollout cost, so a post-intro
> repricing raises the total programme by roughly **50% on the refiner term**
> (~$0.044 → ~$0.066 per rollout; ~$1,374 → ~$1,900 at $200/seed). Either launch
> the baseline runs before 2026-08-31, or budget at post-intro rates. **Your call.**

## Routing: Bedrock + bearer auth, never direct-Anthropic

Locked in by `tests/test_bedrock_routing.py`. The failure mode is real and
silent — verified against litellm 1.95.0:

```
claude-sonnet-5            -> provider='anthropic'   # api.anthropic.com!
claude-haiku-4-5           -> provider='anthropic'
anthropic/claude-sonnet-5  -> provider='anthropic'
```

A bare model name routes to the **direct Anthropic API** — a different account,
a different price sheet, different credentials — with no error. Our
`bedrock_model_id()` raises `RoutingError` on all three.

Recorded honestly: litellm 1.95.0 *already* maps `global.anthropic.*` and
`us.anthropic.*` to bedrock even unprefixed. We still add `bedrock/` explicitly
so routing does not depend on litellm's inference heuristics holding across
versions.

## Credentials

`AWS_BEARER_TOKEN_BEDROCK` + `AWS_REGION` live in `~/.zshrc`. No `.env` file
exists and none was created. Values are never printed or logged
(`test_credentials_never_surface_in_repr`).

Non-interactive shells do not auto-load `~/.zshrc`, so free checks wrap as:

```bash
zsh -c 'source ~/.zshrc >/dev/null 2>&1; uv run python scripts/list_bedrock_models.py'
```

Run scripts call `require_credentials()` at startup and abort with that exact
hint if the token is absent, so a long run cannot die halfway through on auth
(`test_missing_credentials_fail_fast`).
