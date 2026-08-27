"""Dollar-budget accounting and the behaviour-neutral total-cost stopper.

Why this exists
---------------
gepa v0.1.4 ships ``MaxReflectionCostStopper``, but it meters the *reflection
LM only*. On SWE-Bench the solver dominates spend, so it cannot bound a run.
This module adds a stopper that meters solver + refiner + reflection together.

Behaviour neutrality (baseline purity)
--------------------------------------------
``_should_stop()`` has exactly one call site in gepa v0.1.4 -- the main loop
condition at ``engine.py:731``. A stopper is therefore a pure loop-exit
observer: it cannot influence candidate selection, reflection, sampling, merge
scheduling, or acceptance.

One constraint follows from reading the engine: ``_get_remaining_budget()``
(engine.py:1002) and the tqdm total (engine.py:564) both duck-type on an
attribute literally named ``max_metric_calls``. Both consumers are
reporting-only, but this stopper still must not expose that name -- doing so
would hijack the progress bar and the ``BudgetUpdatedEvent`` field. Enforced by
``test_stopper.py::test_does_not_expose_max_metric_calls``.

Budget accounting -- what is IN and what is OUT
----------------------------------------------
The per-seed dollar budget meters **the optimization loop only**:

  IN   minibatch rollouts, reflection calls, val evaluations of promoted
       candidates.

  OUT  (a) the base candidate's initial val evaluation -- run once and reused
           by every seed and both arms, so all runs start from identical state;
       (b) final test-set evaluations after runs complete;
       (c) generation-set runs for taxonomy building.

Those three are shared, one-time pipeline costs, accounted separately.

Exclusion (a) needs no special-casing here and that is by design: the base
candidate's val results are served from a replay cache (see ``seed_cache.py``),
so no LM call is issued and no spend is recorded. Exclusions (b) and (c) are
enforced by construction -- those phases run outside ``gepa.optimize()`` and so
never touch a meter that is attached to a run. ``CostMeter.excluded_usd``
records anything explicitly booked to a shared phase, so a run can report both
numbers.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

Phase = Literal["optimization", "seed_val", "final_test", "generation"]

#: Phases whose spend counts against the per-seed dollar budget.
BUDGETED_PHASES: frozenset[str] = frozenset({"optimization"})

#: Explicit Bedrock price table, USD per token, verified against litellm 1.95.0
#: Kept explicit rather than trusting ``litellm.completion_cost``
#: alone for two reasons:
#:
#: 1. ``completion_cost`` returns ``0.0`` for an unknown model rather than
#:    raising. A stopper that silently meters $0 never fires -- unbounded spend.
#: 2. Sonnet 5 is on introductory pricing that ends 2026-08-31. Pinning the
#:    rate we budgeted against keeps a mid-experiment repricing from silently
#:    changing our accounting.
#:
#: Both target models are ``INFERENCE_PROFILE``-only on this account -- the bare
#: ``anthropic.*`` id is NOT directly invocable, so a profile prefix is
#: mandatory. Two prefixes are available and they are priced differently:
#:
#:   ``global.``  base rate            <- what we use
#:   ``us.``      base rate + ~10%     (US cross-region premium)
#:
#: Bare ids are listed only so a mistaken direct call is priced rather than
#: raising; they are not what we invoke.
BEDROCK_PRICES_USD_PER_TOKEN: dict[str, tuple[float, float]] = {
    # model id: (input $/token, output $/token)
    # Gemini ids route through litellm's gemini provider (explicit prefix).
    # Pinned for the same reason as the Bedrock ids below: the pinned litellm
    # predates the model, and its fallback table must not decide budget stops.
    # $1.50 / $9.00 per million tokens (Gemini API list price, 2026-08).
    "gemini/gemini-3.5-flash": (1.50e-6, 9.00e-6),
    "anthropic.claude-haiku-4-5-20251001-v1:0": (1.00e-6, 5.00e-6),
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": (1.00e-6, 5.00e-6),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (1.10e-6, 5.50e-6),
    # Opus 4.5 / 4.6 are the strongest models this API key can actually invoke
    # Opus 5/4.8/4.7, Sonnet 5 and Fable 5 all 403). Pinned here rather than
    # left to litellm's fallback table: that table ships with the library and
    # can change under a version bump, which would silently reprice a run and
    # move the budget stopper's firing point.
    "anthropic.claude-opus-4-6-v1": (5.00e-6, 25.00e-6),
    "global.anthropic.claude-opus-4-6-v1": (5.00e-6, 25.00e-6),
    "us.anthropic.claude-opus-4-6-v1": (5.50e-6, 27.50e-6),
    "anthropic.claude-opus-4-5-20251101-v1:0": (5.00e-6, 25.00e-6),
    "global.anthropic.claude-opus-4-5-20251101-v1:0": (5.00e-6, 25.00e-6),
    "us.anthropic.claude-opus-4-5-20251101-v1:0": (5.50e-6, 27.50e-6),
    "anthropic.claude-sonnet-5": (2.00e-6, 10.00e-6),
    "global.anthropic.claude-sonnet-5": (2.00e-6, 10.00e-6),
    "us.anthropic.claude-sonnet-5": (2.20e-6, 11.00e-6),
    "anthropic.claude-sonnet-4-6": (3.00e-6, 15.00e-6),
    "global.anthropic.claude-sonnet-4-6": (3.00e-6, 15.00e-6),
    "us.anthropic.claude-sonnet-4-6": (3.30e-6, 16.50e-6),
    "anthropic.claude-sonnet-4-5-20250929-v1:0": (3.00e-6, 15.00e-6),
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.00e-6, 15.00e-6),
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.30e-6, 16.50e-6),
}

#: Inference-profile prefix. Both target models are INFERENCE_PROFILE-only on
#: this account, so a prefix is mandatory. ``global.`` is the base rate; ``us.``
#: carries a ~10% cross-region premium. Configurable so a region change or a
#: profile-scoped authorization problem is a one-line fix.
INFERENCE_PROFILE_PREFIX = "global."


def with_profile(base_model_id: str, prefix: str = INFERENCE_PROFILE_PREFIX) -> str:
    """Attach an inference-profile prefix to a bare foundation-model id."""
    if base_model_id.startswith(("global.", "us.", "eu.", "apac.")):
        return base_model_id
    return f"{prefix}{base_model_id}"


#: Base (unprefixed) foundation-model ids.
SOLVER_BASE = "anthropic.claude-haiku-4-5-20251001-v1:0"
REFINER_BASE = "anthropic.claude-sonnet-4-6"

#: Solver: Haiku 4.5, newest Haiku. **Invocation-proven** -- it completed a real
#: paid call during the failed calibration run.
SOLVER_MODEL = with_profile(SOLVER_BASE)

#: Refiner: Sonnet 4.6, the strongest Sonnet below Sonnet 5.
#:
#: WARNING -- invocability is NOT verified. Sonnet 5 is permanently unavailable
#: to this account, yet every Bedrock control-plane surface
#: (GetFoundationModelAvailability, GetInferenceProfile, ListFoundationModels)
#: reports it AUTHORIZED / AVAILABLE / ACTIVE. The restriction is enforced
#: outside Bedrock's model-access layer -- an IAM identity policy or SCP scoped
#: to model ARNs -- which no control-plane call exposes. So the same APIs that
#: wrongly cleared Sonnet 5 also "clear" Sonnet 4.6, and they cannot be trusted.
#: Run ``scripts/probe_invocation.py`` (~$0.0002) before committing a budget.
REFINER_MODEL = with_profile(REFINER_BASE)

#: Sonnet 5 -- REMOVED FROM CONSIDERATION. Permanently unavailable to this
#: account (403 on invocation). Retained only as a price row so that a stray
#: reference is priced rather than silently metered at $0.
UNAVAILABLE_MODELS: frozenset[str] = frozenset(
    {
        "anthropic.claude-sonnet-5",
        "global.anthropic.claude-sonnet-5",
        "us.anthropic.claude-sonnet-5",
    }
)

#: Sonnet candidates in strength order, for the invocation probe. Sonnet 5 is
#: deliberately absent.
SONNET_CANDIDATES: tuple[str, ...] = (
    "anthropic.claude-sonnet-4-6",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
)

#: The rung below the pin, if 4.6 also turns out to be blocked.
ALT_REFINER_MODEL = with_profile("anthropic.claude-sonnet-4-5-20250929-v1:0")
ALT2_REFINER_MODEL = ALT_REFINER_MODEL

#: Every refiner id the routing and pricing guards must hold for.
ALL_REFINER_MODELS: tuple[str, ...] = (REFINER_MODEL, ALT_REFINER_MODEL)

#: Sonnet 5's post-introductory rate, for budgeting past 2026-08-31. Not a
#: model id -- a rate pair applied to ``REFINER_MODEL``.
SONNET_5_POST_INTRO_USD_PER_TOKEN: tuple[float, float] = (3.00e-6, 15.00e-6)

#: Sonnet 5 introductory pricing expires on this date.
SONNET_5_INTRO_PRICING_ENDS = "2026-08-31"


class UnpricedModelError(RuntimeError):
    """Raised when a model has no known price.

    Deliberately fatal. The alternative -- treating unknown models as free --
    is how a dollar-budget stopper silently fails open.
    """


def _normalise(model: str) -> str:
    """Strip a leading ``bedrock/`` provider prefix, which litellm accepts."""
    return model.removeprefix("bedrock/")


def price_call(model: str, input_tokens: int, output_tokens: int) -> float:
    """Price one call in USD from the explicit table.

    Falls back to litellm only for models absent from the table, and raises if
    litellm cannot price them either.
    """
    key = _normalise(model)
    entry = BEDROCK_PRICES_USD_PER_TOKEN.get(key)
    if entry is not None:
        cost_in, cost_out = entry
        return input_tokens * cost_in + output_tokens * cost_out

    try:  # pragma: no cover - exercised only for off-table models
        import litellm

        info = litellm.model_cost.get(key)
        if info:
            return input_tokens * info["input_cost_per_token"] + output_tokens * info["output_cost_per_token"]
    except Exception:
        pass

    raise UnpricedModelError(
        f"no price for model {model!r}. Add it to BEDROCK_PRICES_USD_PER_TOKEN. "
        "Refusing to meter it as $0 -- that would make the budget stopper fail open."
    )


# ---------------------------------------------------------------------------
# Metering
# ---------------------------------------------------------------------------


@dataclass
class CostMeter:
    """Thread-safe running total of spend, split by budget phase.

    The adapter records every LM call here. Only ``budgeted_usd`` is what the
    stopper compares against the budget; ``excluded_usd`` captures the shared
    one-time phases so a run can report both.
    """

    budgeted_usd: float = 0.0
    excluded_usd: float = 0.0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    by_phase: dict[str, float] = field(default_factory=dict)
    by_model: dict[str, float] = field(default_factory=dict)

    #: Where to mirror the running total, so spend is knowable WHILE a run is in
    #: flight and survives one that never reaches its summary.
    #:
    #: Without this the meter lives only in memory and reaches disk once, in
    #: ``summary.json``, at the end. Three consequences, each of which has cost
    #: real money to discover:
    #:
    #: * A crashed or aborted run leaves NO spend record at all. Every such
    #:   segment then has to be reconstructed from rollout counts times an
    #:   assumed rate -- and an interrupted run accumulates one unrecorded
    #:   segment per interruption.
    #: * A resumed run's summary covers only its final segment, so the recorded
    #:   cost understates the true one silently.
    #: * Live spend cannot be read at all. The only thing on disk mid-run was
    #:   the reflection log, which on a finished HoVer seed was $1.26 of $49.60
    #:   -- 2.5% of the truth, and easily mistaken for the whole.
    #:
    #: A SNAPSHOT is rewritten rather than a line appended per call: the solver
    #: makes tens of thousands of calls per run, and the useful question is
    #: "what is the total now", not "what did call 14,203 cost".
    spend_log: Path | str | None = None
    #: Records between snapshot writes. The file is small and rewritten whole,
    #: but flushing on literally every call would add an fsync to each rollout.
    flush_every: int = 25

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _since_flush: int = field(default=0, repr=False)

    def record(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        phase: Phase = "optimization",
    ) -> float:
        """Record one LM call and return its cost in USD."""
        cost = price_call(model, input_tokens, output_tokens)
        with self._lock:
            self.calls += 1
            self.tokens_in += input_tokens
            self.tokens_out += output_tokens
            self.by_phase[phase] = self.by_phase.get(phase, 0.0) + cost
            self.by_model[model] = self.by_model.get(model, 0.0) + cost
            if phase in BUDGETED_PHASES:
                self.budgeted_usd += cost
            else:
                self.excluded_usd += cost
            self._since_flush += 1
            due = self.spend_log is not None and self._since_flush >= self.flush_every
            if due:
                self._since_flush = 0
                snapshot = self._snapshot_locked()
        if due:
            self._write(snapshot)
        return cost

    def flush(self) -> None:
        """Write the snapshot now, regardless of ``flush_every``."""
        if self.spend_log is None:
            return
        with self._lock:
            self._since_flush = 0
            snapshot = self._snapshot_locked()
        self._write(snapshot)

    def _write(self, snapshot: dict[str, Any]) -> None:
        """Atomic replace, so a reader never sees a half-written file and a
        crash mid-write cannot destroy the previous good snapshot."""
        path = Path(self.spend_log)  # type: ignore[arg-type]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
            tmp.replace(path)
        except Exception:
            # Cost tracking must never take down a paid run.
            pass

    @property
    def total_usd(self) -> float:
        """Everything spent, budgeted or not. Reported, never compared."""
        with self._lock:
            return self.budgeted_usd + self.excluded_usd

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        """Caller must hold ``_lock``."""
        if True:
            return {
                "budgeted_usd": round(self.budgeted_usd, 6),
                "excluded_usd": round(self.excluded_usd, 6),
                "total_usd": round(self.budgeted_usd + self.excluded_usd, 6),
                "calls": self.calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "by_phase": {k: round(v, 6) for k, v in sorted(self.by_phase.items())},
                "by_model": {k: round(v, 6) for k, v in sorted(self.by_model.items())},
            }


class MaxTotalCostStopper:
    """Stop once budgeted spend reaches ``budget_usd``.

    Satisfies ``gepa.utils.stop_condition.StopperProtocol`` structurally: it is
    a callable taking ``GEPAState`` and returning ``bool``. The state argument
    is accepted and **not read** -- spend is metered by the adapter, so this
    stopper observes only its own meters.

    Deliberately exposes no attribute named ``max_metric_calls`` (see module
    docstring).

    Overshoot
    ---------
    The engine consults this between iterations, so realised spend overshoots
    the budget by up to one iteration's cost. On SWE-Bench that is material.
    We do not pretend to a hard ceiling: ``realised_usd`` is reported per seed
    and seeds are compared on realised cost. The caveat belongs in the blog
    post's methods section.
    """

    def __init__(
        self,
        budget_usd: float,
        meters: Iterable[CostMeter] | CostMeter,
    ) -> None:
        if budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        self.budget_usd = float(budget_usd)
        self._meters: tuple[CostMeter, ...] = (meters,) if isinstance(meters, CostMeter) else tuple(meters)
        if not self._meters:
            raise ValueError(
                "at least one CostMeter is required; a stopper with nothing to meter would never fire (unbounded spend)"
            )
        self.fired_at_usd: float | None = None

    @property
    def realised_usd(self) -> float:
        """Budgeted spend across all attached meters."""
        return sum(m.budgeted_usd for m in self._meters)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.budget_usd - self.realised_usd)

    def __call__(self, gepa_state: Any = None) -> bool:
        spent = self.realised_usd
        if spent >= self.budget_usd:
            if self.fired_at_usd is None:
                self.fired_at_usd = spent
            return True
        return False

    def __repr__(self) -> str:  # pragma: no cover
        return f"MaxTotalCostStopper(budget_usd={self.budget_usd}, realised_usd={self.realised_usd:.4f})"
