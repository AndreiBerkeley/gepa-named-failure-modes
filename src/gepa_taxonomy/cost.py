"""Dollar-budget accounting and the behaviour-neutral total-cost stopper.

Why this exists
---------------
gepa v0.1.4 ships ``MaxReflectionCostStopper``, but it meters the *reflection
LM only*. In a multi-step program the task model usually dominates spend, so
it cannot bound a run. This module adds a stopper that meters every model call
together.

Behaviour neutrality (baseline purity)
--------------------------------------------
``_should_stop()`` has exactly one call site in gepa v0.1.4 -- the main loop
condition at ``engine.py:731``. A stopper is therefore a pure loop-exit
observer: it cannot influence candidate selection, reflection, sampling, merge
scheduling, or acceptance.

One constraint follows from reading the engine: ``_get_remaining_budget()``
(engine.py:1002) and the tqdm total in gepa's engine both duck-type on an
attribute literally named ``max_metric_calls``. Both consumers are
reporting-only, but this stopper still must not expose that name -- doing so
would hijack the progress bar and the ``BudgetUpdatedEvent`` field. Enforced by
``tests/test_cost.py``.

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
import os
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

#: Environment variable holding extra prices, ``MODEL=IN,OUT`` specs separated
#: by ``;`` with IN and OUT in USD per million tokens.
PRICE_ENV = "GEPA_TAXONOMY_PRICES"

#: Prices supplied by the user for models litellm's table does not know:
#: model id -> (USD per input token, USD per output token). Filled from
#: ``--price`` flags and :data:`PRICE_ENV`; nothing is hard-coded here.
PRICE_OVERRIDES: dict[str, tuple[float, float]] = {}


class UnpricedModelError(RuntimeError):
    """Raised when a model has no known price.

    Deliberately fatal. The alternative -- treating unknown models as free --
    is how a dollar-budget stopper silently fails open.
    """


def _normalise(model: str) -> str:
    """Strip a leading ``bedrock/`` provider prefix, which litellm accepts."""
    return model.removeprefix("bedrock/")


def _price_keys(model: str) -> list[str]:
    """Lookup keys for a model id, most specific first.

    A litellm id may carry a provider prefix (``anthropic/claude-sonnet-4-6``,
    ``openai/gpt-5-mini``); litellm's table keys some entries with the prefix
    and some without, so both forms are tried.
    """
    key = _normalise(model)
    keys = [key]
    if "/" in key:
        keys.append(key.split("/", 1)[1])
    return keys


def parse_price_spec(spec: str) -> tuple[str, float, float]:
    """Parse ``MODEL=IN,OUT`` with IN and OUT in USD per million tokens."""
    try:
        model, rates = spec.split("=", 1)
        cost_in, cost_out = (float(x) for x in rates.split(",", 1))
    except ValueError as exc:
        raise ValueError(f"bad price spec {spec!r}; expected MODEL=IN,OUT in USD per million tokens") from exc
    if not model.strip() or cost_in < 0 or cost_out < 0:
        raise ValueError(f"bad price spec {spec!r}; expected MODEL=IN,OUT in USD per million tokens")
    return model.strip(), cost_in, cost_out


def set_price(model: str, input_usd_per_million: float, output_usd_per_million: float) -> None:
    """Register a price for ``model`` (USD per million tokens)."""
    prices = (input_usd_per_million / 1e6, output_usd_per_million / 1e6)
    for key in _price_keys(model):
        PRICE_OVERRIDES[key] = prices


def load_price_overrides(specs: Iterable[str] | None = None) -> None:
    """Register prices from ``specs`` and from :data:`PRICE_ENV`."""
    env = os.environ.get(PRICE_ENV, "")
    for spec in [s for s in env.split(";") if s.strip()] + list(specs or []):
        set_price(*parse_price_spec(spec))


def lookup_price(model: str) -> tuple[float, float] | None:
    """Per-token prices for ``model``: user overrides first, then litellm's table."""
    keys = _price_keys(model)
    for key in keys:
        if key in PRICE_OVERRIDES:
            return PRICE_OVERRIDES[key]
    try:
        import litellm

        for key in keys:
            info = litellm.model_cost.get(key)
            if info and "input_cost_per_token" in info and "output_cost_per_token" in info:
                return float(info["input_cost_per_token"]), float(info["output_cost_per_token"])
    except Exception:
        pass
    return None


def price_call(model: str, input_tokens: int, output_tokens: int) -> float:
    """Price one call in USD, raising for a model nobody has priced."""
    prices = lookup_price(model)
    if prices is None:
        raise UnpricedModelError(
            f"no price for model {model!r}. litellm's table does not list it; pass "
            f"--price {model}=IN,OUT (USD per million tokens) or set {PRICE_ENV}. "
            "Refusing to meter it as $0 -- that would make the budget stopper fail open."
        )
    cost_in, cost_out = prices
    return input_tokens * cost_in + output_tokens * cost_out


def assert_priced(*models: str) -> None:
    """Fail before any spend if one of ``models`` cannot be priced."""
    for model in models:
        price_call(model, 1, 1)


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
    #: * Live spend cannot be read at all. The only thing on disk mid-run is
    #:   the reflection log, a small fraction of the total and easily mistaken
    #:   for the whole.
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
    the budget by up to one iteration's cost. We do not pretend to a hard
    ceiling: ``realised_usd`` is reported per seed and seeds are compared on
    realised cost.
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
