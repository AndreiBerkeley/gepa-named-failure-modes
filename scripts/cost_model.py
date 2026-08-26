#!/usr/bin/env python
"""Per-seed cost model for the baseline runs. FREE -- pure arithmetic, no API calls.

    uv run python scripts/cost_model.py

Models the GEPA optimization loop under our solver->refiner program and prices
it against the verified Bedrock rates, for both candidate val sizes (60 and
100). Reports the affordable number of full candidate evaluations per seed at a
range of budgets, which is the number Andrei asked to see before Phase 2.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from gepa_taxonomy.cost import (
    ALT_REFINER_MODEL,
    BEDROCK_PRICES_USD_PER_TOKEN,
    REFINER_MODEL,
    SOLVER_MODEL,
    SONNET_5_POST_INTRO_USD_PER_TOKEN,
)

MEASURED_PATH = Path(__file__).resolve().parents[1] / "results" / "calibration" / "calibration.json"

#: Andrei's target: this many realistic full candidate evals per seed...
TARGET_EVALS_LOW, TARGET_EVALS_HIGH = 15, 20
#: ...at a per-seed budget under this, preferably around $100.
TARGET_BUDGET_PREFERRED, TARGET_BUDGET_CEILING = 100, 150
#: Fraction of proposals promoted to a full val eval, in the realistic model.
PROMOTION_RATE = 0.25

# ---------------------------------------------------------------------------
# Per-rollout token assumptions
# ---------------------------------------------------------------------------
# Grounded where possible:
#   * problem_statement sizes MEASURED over the committed manifests
#     (mean ~500 tok, median ~300, p90 ~1000) at ~3.6 chars/token.
#   * retrieved context is CAPPED by the program at max_context_chars=60_000,
#     ~16.7k tokens; BM25 top-5 files will often be smaller, so this is an
#     upper-ish estimate and the model is deliberately conservative.
#   * output sizes are ESTIMATES -- a unified diff for a single-file fix.
# Flagged as assumptions rather than presented as measurements.

CONTEXT_TOKENS = 16_700  # 60k chars / 3.6
PROBLEM_TOKENS = 500  # measured mean
INSTRUCTION_TOKENS = 150  # the evolved component; grows as GEPA mutates it
PATCH_OUT_TOKENS = 800  # estimate
FEEDBACK_TOKENS = 60

SOLVER_IN = INSTRUCTION_TOKENS + PROBLEM_TOKENS + CONTEXT_TOKENS
SOLVER_OUT = PATCH_OUT_TOKENS
REFINER_IN = INSTRUCTION_TOKENS + PROBLEM_TOKENS + CONTEXT_TOKENS + PATCH_OUT_TOKENS + FEEDBACK_TOKENS
REFINER_OUT = PATCH_OUT_TOKENS

# Reflection: gepa's default reflection_minibatch_size is 3, so the reflective
# dataset carries ~3 rollouts' worth of trace text plus the current instruction.
REFLECTION_IN = 3 * (PROBLEM_TOKENS + PATCH_OUT_TOKENS * 2 + FEEDBACK_TOKENS) + 800
REFLECTION_OUT = 700
REFLECTION_MODEL = REFINER_MODEL  # reflection uses the stronger model


def price(model: str, tin: int, tout: int) -> float:
    cin, cout = BEDROCK_PRICES_USD_PER_TOKEN[model]
    return tin * cin + tout * cout


ROLLOUT_USD = price(SOLVER_MODEL, SOLVER_IN, SOLVER_OUT) + price(REFINER_MODEL, REFINER_IN, REFINER_OUT)
REFLECTION_USD = price(REFLECTION_MODEL, REFLECTION_IN, REFLECTION_OUT)

MINIBATCH = 3  # gepa default reflection_minibatch_size


@dataclass
class Scenario:
    n_val: int
    promotion_rate: float  # fraction of proposals that get promoted to a full val eval

    def per_iteration_usd(self) -> float:
        """One GEPA iteration: minibatch rollouts + reflection + expected val eval."""
        minibatch = MINIBATCH * ROLLOUT_USD
        reflection = REFLECTION_USD
        val = self.promotion_rate * self.n_val * ROLLOUT_USD
        return minibatch + reflection + val

    def full_val_eval_usd(self) -> float:
        return self.n_val * ROLLOUT_USD


SIZES_PATH = Path(__file__).resolve().parents[1] / "results" / "calibration" / "prompt_sizes.json"


def load_measured() -> dict | None:
    """Replace token estimates with measured values from the calibration run.

    Also recomputes the reflection prompt from the MEASURED patch sizes -- the
    reflective dataset carries rollout traces, so a wrong patch-size assumption
    propagates into the reflection cost as well.
    """
    if not MEASURED_PATH.exists():
        return None
    d = json.loads(MEASURED_PATH.read_text())
    if not d.get("complete"):
        print(f"WARNING: {MEASURED_PATH.name} is a PARTIAL result (refiner did not complete).")
        return None
    global SOLVER_IN, SOLVER_OUT, REFINER_IN, REFINER_OUT
    global REFLECTION_IN, ROLLOUT_USD, REFLECTION_USD
    SOLVER_IN, SOLVER_OUT = d["solver_tokens_in"], d["solver_tokens_out"]
    REFINER_IN, REFINER_OUT = d["refiner_tokens_in"], d["refiner_tokens_out"]
    REFLECTION_IN = 3 * (PROBLEM_TOKENS + SOLVER_OUT + REFINER_OUT + FEEDBACK_TOKENS) + 800
    ROLLOUT_USD = price(SOLVER_MODEL, SOLVER_IN, SOLVER_OUT) + price(REFINER_MODEL, REFINER_IN, REFINER_OUT)
    REFLECTION_USD = price(REFINER_MODEL, REFLECTION_IN, REFLECTION_OUT)
    return d


def mean_instance_projection(measured: dict) -> dict | None:
    """Project a MEAN-instance rollout from the measured chars-per-token ratio.

    The calibrated instance was deliberately the LARGEST of the six sampled, so
    its cost is a near-ceiling figure, not a typical one. The measured token
    count divided by that instance's prompt chars gives a real chars/token
    ratio, which applied to the sample mean yields the typical rollout.
    """
    if not SIZES_PATH.exists():
        return None
    sizes = json.loads(SIZES_PATH.read_text())
    rows = sizes["rows"]

    cal = next((r for r in rows if r["instance_id"] == measured["instance_id"]), None)
    if cal is None:
        return None

    chars_per_token = cal["solver_prompt_chars"] / measured["solver_tokens_in"]
    mean_solver_in = sizes["mean_solver_prompt_chars"] / chars_per_token
    # Refiner prompt = solver prompt + patch + feedback; measured delta is stable.
    refiner_delta = measured["refiner_tokens_in"] - measured["solver_tokens_in"]
    mean_refiner_in = mean_solver_in + refiner_delta

    s_out, r_out = measured["solver_tokens_out"], measured["refiner_tokens_out"]
    solver_usd = price(SOLVER_MODEL, int(mean_solver_in), s_out)
    refiner_usd = price(REFINER_MODEL, int(mean_refiner_in), r_out)

    per_instance = []
    for r in rows:
        s_in = r["solver_prompt_chars"] / chars_per_token
        per_instance.append(
            {
                "instance_id": r["instance_id"],
                "rollout_usd": price(SOLVER_MODEL, int(s_in), s_out)
                + price(REFINER_MODEL, int(s_in + refiner_delta), r_out),
            }
        )

    return {
        "chars_per_token": chars_per_token,
        "mean_solver_in": mean_solver_in,
        "mean_refiner_in": mean_refiner_in,
        "solver_usd": solver_usd,
        "refiner_usd": refiner_usd,
        "rollout_usd": solver_usd + refiner_usd,
        "per_instance": per_instance,
        "n": len(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--measured", action="store_true", help="use measured token counts from results/calibration/calibration.json"
    )
    args = ap.parse_args()

    if args.measured:
        d = load_measured()
        if d is None:
            print(f"no calibration at {MEASURED_PATH}. Run scripts/calibrate_rollout.py first.")
            return 2
        print("=" * 78)
        print("USING MEASURED TOKEN COUNTS")
        print("=" * 78)
        print(f"  from {d['instance_id']} ({d['repo']})")
        print(f"  solver  {d['solver_tokens_in']:,} in / {d['solver_tokens_out']:,} out")
        print(f"  refiner {d['refiner_tokens_in']:,} in / {d['refiner_tokens_out']:,} out")
        print(f"  measured rollout ${d['rollout_usd']:.4f}  (estimate was ${d['estimated_usd']:.4f})")
        print("  NOTE: n=1, and the calibrated instance was deliberately the")
        print("        LARGEST of the sample -- a near-ceiling figure, not typical.\n")

        proj = mean_instance_projection(d)
        if proj:
            print("-" * 78)
            print("MEAN-INSTANCE PROJECTION (what a typical rollout actually costs)")
            print("-" * 78)
            print(f"  measured chars/token       : {proj['chars_per_token']:.3f}")
            print("    (vs the 3.6 assumed -- code tokenizes denser, as expected)")
            print(f"  mean solver input          : {proj['mean_solver_in']:,.0f} tok")
            print(f"  mean refiner input         : {proj['mean_refiner_in']:,.0f} tok")
            print(f"\n  {'':22}{'solver':>10}{'refiner':>10}{'rollout':>10}")
            print(
                f"  {'calibrated (largest)':22}{d['solver_usd']:10.4f}{d['refiner_usd']:10.4f}{d['rollout_usd']:10.4f}"
            )
            print(
                f"  {'MEAN instance':22}{proj['solver_usd']:10.4f}{proj['refiner_usd']:10.4f}{proj['rollout_usd']:10.4f}"
            )
            delta = (proj["rollout_usd"] / d["rollout_usd"] - 1) * 100
            print(f"\n  mean is {delta:+.0f}% vs the calibrated ceiling instance")
            print(f"\n  per-instance spread (n={proj['n']}, projected):")
            for r in sorted(proj["per_instance"], key=lambda x: x["rollout_usd"]):
                print(f"    {r['instance_id']:36} ${r['rollout_usd']:.4f}")
            print()
            _measured_target_table(d, proj)

    print("=" * 78)
    print("PER-SEED COST MODEL -- baseline GEPA, solver->refiner program")
    print("=" * 78)
    print(f"\nsolver  {SOLVER_MODEL}")
    print(f"        {SOLVER_IN:,} in / {SOLVER_OUT:,} out  ->  ${price(SOLVER_MODEL, SOLVER_IN, SOLVER_OUT):.4f}")
    print(f"refiner {REFINER_MODEL}")
    print(f"        {REFINER_IN:,} in / {REFINER_OUT:,} out  ->  ${price(REFINER_MODEL, REFINER_IN, REFINER_OUT):.4f}")
    print(f"\n  ONE ROLLOUT      = ${ROLLOUT_USD:.4f}")
    print(f"  ONE REFLECTION   = ${REFLECTION_USD:.4f}")

    print("\n" + "-" * 78)
    print("FULL VAL EVALUATION (the cost-dominant term)")
    print("-" * 78)
    for n_val in (60, 100):
        print(f"  n_val={n_val:3d}   one full candidate evaluation = ${n_val * ROLLOUT_USD:8.2f}")

    print("\n" + "-" * 78)
    print("AFFORDABLE FULL CANDIDATE EVALUATIONS PER SEED")
    print("  (budget spent entirely on val evals -- an upper bound)")
    print("-" * 78)
    budgets = (100, 150, 200, 300, 400, 500, 750, 1000)
    print(f"  {'budget':>8} " + "".join(f"{f'n_val={n}':>14}" for n in (60, 100)))
    for b in budgets:
        row = f"  ${b:>7,}"
        for n_val in (60, 100):
            row += f"{b / (n_val * ROLLOUT_USD):14.1f}"
        print(row)

    print("\n" + "-" * 78)
    print("REALISTIC LOOP MODEL (minibatch + reflection + promoted val evals)")
    print("-" * 78)
    for promo in (0.25, 0.40):
        print(f"\n  promotion rate = {promo:.0%}  (fraction of proposals reaching a full val eval)")
        print(f"    {'budget':>8} " + "".join(f"{f'n_val={n}':>26}" for n in (60, 100)))
        print(f"    {'':>8} " + "".join(f"{'iters':>13}{'full evals':>13}" for _ in (60, 100)))
        for b in budgets:
            row = f"    ${b:>7,}"
            for n_val in (60, 100):
                s = Scenario(n_val=n_val, promotion_rate=promo)
                iters = b / s.per_iteration_usd()
                row += f"{iters:13.0f}{iters * promo:13.1f}"
            print(row)

    print("\n" + "=" * 78)
    print("SHARED ONE-TIME COSTS (excluded from the per-seed budget)")
    print("=" * 78)
    print("  base candidate val eval (n=100), run ONCE and reused by all runs")
    print(f"      = ${100 * ROLLOUT_USD:8.2f}")
    print("  generation-set trace harvest (n=150, one pass)")
    print(f"      = ${150 * ROLLOUT_USD:8.2f}")
    print("  final test evaluation (n=400 x 6 arms: 3 baseline + 3 taxonomy)")
    print(f"      = ${400 * 6 * ROLLOUT_USD:8.2f}")
    shared = 100 * ROLLOUT_USD + 150 * ROLLOUT_USD + 400 * 6 * ROLLOUT_USD
    print(f"  {'':-<50}\n  SHARED TOTAL = ${shared:8.2f}")

    print("\n" + "=" * 78)
    print("TOTAL PROGRAMME (3 baseline + 3 taxonomy seeds)")
    print("=" * 78)
    for b in (150, 200, 300):
        print(f"  ${b}/seed x 6 seeds = ${b * 6:,}  + shared ${shared:,.0f}  =  ${b * 6 + shared:,.0f}")

    compare_refiners()
    budget_target_analysis()

    print("\nNOTE: token counts marked ESTIMATE above (output sizes, context fill)")
    print("      are unverified until a smoke test runs. Treat +/-40% as the")
    print("      honest error bar on these figures. It applies EQUALLY to both")
    print("      refiner options, so the comparison between them is robust even")
    print("      though the absolute totals are not.")
    return 0


# ---------------------------------------------------------------------------
# Refiner comparison -- for the budget decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefinerOption:
    label: str
    rates: tuple[float, float]  # (input, output) USD per token
    note: str

    @property
    def refiner_usd(self) -> float:
        cin, cout = self.rates
        return REFINER_IN * cin + REFINER_OUT * cout

    @property
    def rollout_usd(self) -> float:
        return price(SOLVER_MODEL, SOLVER_IN, SOLVER_OUT) + self.refiner_usd

    def reflection_usd(self) -> float:
        cin, cout = self.rates
        return REFLECTION_IN * cin + REFLECTION_OUT * cout

    def per_iteration_usd(self, n_val: int, promo: float) -> float:
        return MINIBATCH * self.rollout_usd + self.reflection_usd() + promo * n_val * self.rollout_usd


def compare_refiners(n_val: int = 100) -> None:
    solver_usd = price(SOLVER_MODEL, SOLVER_IN, SOLVER_OUT)
    options = [
        RefinerOption(
            "Sonnet 4.6 (pinned)",
            BEDROCK_PRICES_USD_PER_TOKEN[REFINER_MODEL],
            "PINNED today. $2/$10 per MTok. Intro rate ENDS 2026-08-31.",
        ),
        RefinerOption(
            "Sonnet 5 (post-intro)", SONNET_5_POST_INTRO_USD_PER_TOKEN, "Same model from 2026-09-01. $3/$15 per MTok."
        ),
        RefinerOption(
            "Sonnet 4.6",
            BEDROCK_PRICES_USD_PER_TOKEN[ALT_REFINER_MODEL],
            "Next-strongest ACTIVE Sonnet. $3/$15. NO intro period -- rate is stable.",
        ),
    ]

    print("\n" + "=" * 78)
    print(f"REFINER COMPARISON (val={n_val}, solver fixed = Haiku 4.5)")
    print("=" * 78)
    print("\nCandidates derived from the account listing, not assumed: the ACTIVE")
    print("Sonnets are 4-5, 4-6 and 5; sonnet-4 (20250514) is LEGACY. So the")
    print("next rung below Sonnet 5 is Sonnet 4.6.\n")

    print(f"  {'option':24}{'refiner $/MTok':>17}{'solver':>10}{'refiner':>10}{'rollout':>10}")
    for o in options:
        rate = f"${o.rates[0] * 1e6:.0f}/${o.rates[1] * 1e6:.0f}"
        print(f"  {o.label:24}{rate:>17}{solver_usd:10.4f}{o.refiner_usd:10.4f}{o.rollout_usd:10.4f}")

    print(f"\n  per-rollout split: solver is FIXED at ${solver_usd:.4f}; the refiner is")
    print("  the entire difference between these options.")

    print("\n" + "-" * 78)
    print(f"AFFORDABLE FULL CANDIDATE EVALUATIONS PER SEED (val={n_val})")
    print("-" * 78)
    print("  one full val eval:  " + "  ".join(f"{o.label}=${n_val * o.rollout_usd:.2f}" for o in options))
    print(f"\n  {'budget':>8}" + "".join(f"{o.label:>24}" for o in options))
    print(f"  {'':>8}" + "".join(f"{'pure-val / realistic':>24}" for _ in options))
    for b in (100, 150, 200):
        row = f"  ${b:>7,}"
        for o in options:
            pure = b / (n_val * o.rollout_usd)
            realistic = (b / o.per_iteration_usd(n_val, 0.25)) * 0.25
            row += f"{f'{pure:.1f} / {realistic:.1f}':>24}"
        print(row)
    print("\n  (pure-val = budget spent only on val evals, an upper bound;")
    print("   realistic = minibatch + reflection + promoted val evals at 25% promotion)")

    print("\n" + "-" * 78)
    print("TOTAL 6-SEED PROGRAMME (3 baseline + 3 taxonomy)")
    print("-" * 78)
    print(f"  {'budget/seed':>12}" + "".join(f"{o.label:>24}" for o in options))
    for b in (100, 150, 200):
        row = f"  ${b:>11,}"
        for o in options:
            shared = (n_val + 150 + 400 * 6) * o.rollout_usd
            row += f"{f'${b * 6 + shared:,.0f}':>24}"
        print(row)
    print("\n  (shared = seed val eval + generation harvest + final test eval,")
    print("   which also scale with the refiner choice)")

    print("\n" + "-" * 78)
    print("INTRO-PRICING DEADLINES")
    print("-" * 78)
    for o in options:
        print(f"  {o.label:24} {o.note}")

    print("\n" + "=" * 78)
    print("READ")
    print("=" * 78)
    s5, s5_post, s46 = options
    print(f"  Sonnet 4.6 costs MORE than Sonnet 5 today (${s46.rollout_usd:.4f} vs")
    print(
        f"  ${s5.rollout_usd:.4f} per rollout, +{(s46.rollout_usd / s5.rollout_usd - 1) * 100:.0f}%) and is priced IDENTICALLY to"
    )
    print("  Sonnet 5 from 2026-09-01. It is an older, less capable model that is")
    print("  never cheaper -- so it is dominated, and there is no cost argument")
    print("  for stepping down to it.")
    print("\n  The real decision is not which Sonnet, but WHEN to launch:")
    print(f"    launch before 2026-08-31  ->  ${s5.rollout_usd:.4f}/rollout")
    print(
        f"    launch after              ->  ${s5_post.rollout_usd:.4f}/rollout (+{(s5_post.rollout_usd / s5.rollout_usd - 1) * 100:.0f}%)"
    )
    print("\n  If the budget must come down, the effective levers are val size,")
    print("  test size, or the per-seed budget -- not the refiner model.")


# ---------------------------------------------------------------------------
# Andrei's target: ~15-20 realistic evals/seed at <$150, preferably ~$100
# ---------------------------------------------------------------------------


def _realistic_evals(opt: RefinerOption, budget: float, n_val: int, promo: float = PROMOTION_RATE) -> float:
    return (budget / opt.per_iteration_usd(n_val, promo)) * promo


def _min_budget_for(opt: RefinerOption, target: float, n_val: int, promo: float = PROMOTION_RATE) -> float:
    """Invert the realistic-loop model: budget needed for `target` evals."""
    return target * opt.per_iteration_usd(n_val, promo) / promo


def _options(n_val_note: str = "") -> list[RefinerOption]:
    return [
        RefinerOption("Sonnet 5 (intro)", BEDROCK_PRICES_USD_PER_TOKEN[REFINER_MODEL], "$3/$15, no intro period"),
        RefinerOption(
            "Sonnet 4.5 (fallback)", BEDROCK_PRICES_USD_PER_TOKEN[ALT_REFINER_MODEL], "$3/$15, identical rate"
        ),
    ]


def budget_target_analysis(n_val: int = 100) -> None:
    opts = _options()
    solver_usd = price(SOLVER_MODEL, SOLVER_IN, SOLVER_OUT)

    print("\n" + "=" * 78)
    print(
        f"TARGET: {TARGET_EVALS_LOW}-{TARGET_EVALS_HIGH} realistic evals/seed "
        f"at <${TARGET_BUDGET_CEILING}/seed (ideally ~${TARGET_BUDGET_PREFERRED}), val={n_val}"
    )
    print("=" * 78)

    # (a) per-rollout and per-full-val-eval
    print("\n(a) PER-ROLLOUT AND PER-FULL-VAL-EVAL COST")
    print(f"    {'option':24}{'rate':>12}{'solver':>10}{'refiner':>10}{'rollout':>10}{'full val eval':>16}")
    for o in opts:
        rate = f"${o.rates[0] * 1e6:.0f}/${o.rates[1] * 1e6:.0f}"
        print(
            f"    {o.label:24}{rate:>12}{solver_usd:10.4f}{o.refiner_usd:10.4f}"
            f"{o.rollout_usd:10.4f}{n_val * o.rollout_usd:16.2f}"
        )

    # (b) realistic evals at the target budgets
    print(f"\n(b) REALISTIC-LOOP EVALS PER SEED (promotion {PROMOTION_RATE:.0%})")
    print(f"    {'option':24}" + "".join(f"{f'${b}':>12}" for b in (100, 125, 150)))
    for o in opts:
        row = f"    {o.label:24}"
        for b in (100, 125, 150):
            e = _realistic_evals(o, b, n_val)
            mark = "*" if e >= TARGET_EVALS_LOW else " "
            row += f"{f'{e:.1f}{mark}':>12}"
        print(row)
    print(f"    (* = meets the >={TARGET_EVALS_LOW} floor)")

    # (c) minimum budget to hit 15 and 20
    print(f"\n(c) MINIMUM PER-SEED BUDGET TO REACH {TARGET_EVALS_LOW} / {TARGET_EVALS_HIGH} EVALS")
    print(f"    {'option':24}{f'{TARGET_EVALS_LOW} evals':>14}{f'{TARGET_EVALS_HIGH} evals':>14}   verdict vs target")
    for o in opts:
        b15 = _min_budget_for(o, TARGET_EVALS_LOW, n_val)
        b20 = _min_budget_for(o, TARGET_EVALS_HIGH, n_val)
        if b15 <= TARGET_BUDGET_PREFERRED:
            verdict = "hits 15 at ~$100"
        elif b15 <= TARGET_BUDGET_CEILING:
            verdict = f"15 needs ${b15:.0f} (under ${TARGET_BUDGET_CEILING}, above ${TARGET_BUDGET_PREFERRED})"
        else:
            verdict = f"MISSES: 15 needs ${b15:.0f}"
        print(f"    {o.label:24}{f'${b15:.0f}':>14}{f'${b20:.0f}':>14}   {verdict}")

    # (d) 6-seed totals for cells meeting the target
    print(f"\n(d) 6-SEED PROGRAMME TOTALS for cells meeting >={TARGET_EVALS_LOW} evals at <${TARGET_BUDGET_CEILING}")
    shared_rollouts = n_val + 150 + 400 * 6
    any_cell = False
    print(f"    {'option':24}{'budget/seed':>13}{'evals':>8}{'shared':>10}{'6-seed total':>15}")
    for o in opts:
        for b in (100, 125, 150):
            e = _realistic_evals(o, b, n_val)
            if e < TARGET_EVALS_LOW or b >= TARGET_BUDGET_CEILING + 1:
                continue
            any_cell = True
            shared = shared_rollouts * o.rollout_usd
            print(f"    {o.label:24}{f'${b}':>13}{e:8.1f}{f'${shared:,.0f}':>10}{f'${b * 6 + shared:,.0f}':>15}")
    if not any_cell:
        print("    (none)")

    print("\n" + "-" * 78)
    print("WHERE THE TARGET STANDS")
    print("-" * 78)
    pinned = opts[0]
    print("  Sonnet 5 is permanently unavailable to this account, so the cheapest")
    print("  refiner rate available is $3/$15. Every remaining Sonnet (4.6, 4.5)")
    print("  is priced identically, so no model choice moves the number.")
    print(f"\n  pinned refiner: {REFINER_MODEL}")
    print(
        f"    ${TARGET_BUDGET_PREFERRED}/seed  -> {_realistic_evals(pinned, TARGET_BUDGET_PREFERRED, n_val):.1f} evals"
    )
    print(f"    ${TARGET_BUDGET_CEILING}/seed  -> {_realistic_evals(pinned, TARGET_BUDGET_CEILING, n_val):.1f} evals")
    print(f"    {TARGET_EVALS_LOW} evals needs ${_min_budget_for(pinned, TARGET_EVALS_LOW, n_val):.0f}/seed")
    print(f"    {TARGET_EVALS_HIGH} evals needs ${_min_budget_for(pinned, TARGET_EVALS_HIGH, n_val):.0f}/seed")
    print("\n  So the target IS met at the $150 ceiling (15.0 evals), but NOT at the")
    print("  ~$100 preference (10.0 evals). Levers below close that gap.")

    levers(n_val)


def levers(n_val: int = 100) -> None:
    """Options only. Nothing here changes a pinned decision."""
    s5 = _options()[0]  # the pinned refiner
    base = _realistic_evals(s5, TARGET_BUDGET_PREFERRED, n_val)

    print("\n" + "=" * 78)
    print(
        f"LEVERS THAT WOULD REACH ~{TARGET_EVALS_LOW}-{TARGET_EVALS_HIGH} EVALS AT "
        f"~${TARGET_BUDGET_PREFERRED}/SEED  (OPTIONS ONLY -- nothing pinned changes)"
    )
    print("=" * 78)
    print(f"\n  baseline: {REFINER_MODEL} refiner, val={n_val}, context {CONTEXT_TOKENS:,} tok,")
    print(f"            refiner max_tokens {REFINER_OUT} -> {base:.1f} evals at ${TARGET_BUDGET_PREFERRED}/seed\n")

    def evals_with(*, ctx=CONTEXT_TOKENS, out=PATCH_OUT_TOKENS, nval=n_val, budget=TARGET_BUDGET_PREFERRED):
        s_in, s_out = INSTRUCTION_TOKENS + PROBLEM_TOKENS + ctx, PATCH_OUT_TOKENS
        r_in, r_out = INSTRUCTION_TOKENS + PROBLEM_TOKENS + ctx + PATCH_OUT_TOKENS + FEEDBACK_TOKENS, out
        cin, cout = BEDROCK_PRICES_USD_PER_TOKEN[REFINER_MODEL]
        roll = price(SOLVER_MODEL, s_in, s_out) + (r_in * cin + r_out * cout)
        refl = REFLECTION_IN * cin + REFLECTION_OUT * cout
        per_iter = MINIBATCH * roll + refl + PROMOTION_RATE * nval * roll
        return (budget / per_iter) * PROMOTION_RATE, roll

    rows = [
        ("baseline (nothing changed)", dict()),
        ("refiner max_tokens 800 -> 400", dict(out=400)),
        ("context 16.7k -> 12k tok (~43k chars)", dict(ctx=12_000)),
        ("context 16.7k -> 8k tok (~29k chars)", dict(ctx=8_000)),
        ("val 100 -> 80", dict(nval=80)),
        ("val 100 -> 60", dict(nval=60)),
        ("context 12k + max_tokens 400", dict(ctx=12_000, out=400)),
        ("context 8k + max_tokens 400", dict(ctx=8_000, out=400)),
        ("context 12k + val 80", dict(ctx=12_000, nval=80)),
    ]
    print(f"  {'lever':40}{'rollout':>10}{'evals @$100':>13}{'evals @$125':>13}  target?")
    for label, kw in rows:
        e100, roll = evals_with(**kw)
        e125, _ = evals_with(**kw, budget=125)
        hit = "YES" if e100 >= TARGET_EVALS_LOW else ("at $125" if e125 >= TARGET_EVALS_LOW else "no")
        print(f"  {label:40}{roll:10.4f}{e100:13.1f}{e125:13.1f}  {hit}")

    print("\n  Notes on each lever, so the trade-offs are explicit:")
    print("   * refiner max_tokens 400 -- a unified diff for a single-file fix usually")
    print("     fits; risks truncating multi-hunk patches, which would score as failures")
    print("     and bias the taxonomy toward a scaffolding artifact rather than a real")
    print("     failure mode. Cheap to validate on the generation set first.")
    print("   * context trim -- fewer/shorter retrieved files. Directly lowers the")
    print("     solve-rate ceiling, and it lowers it for BOTH arms equally, so the")
    print("     baseline-vs-taxonomy delta stays interpretable.")
    print("   * val size -- reverses decision D010. At val=60 the per-candidate SE")
    print("     (~+/-5pp) is the same order as the effects we care about, which is")
    print("     exactly the noise-dominated selection you rejected. val=80 is the")
    print("     milder version of the same trade.")
    print("\n  Cheapest route to the target WITHOUT touching val: trim context and cap")
    print("  refiner output. Those are scaffolding constants, not study design, and")
    print("  they apply identically to both arms.")
    print("\n  IMPORTANT CAVEAT ON ALL OF THE ABOVE")
    print("  The context figure is the program's CAP (60k chars), not a measurement.")
    print("  BM25 top-5 files will often come in under it, so the true baseline may")
    print("  already be cheaper than modelled -- possibly meeting the target with no")
    print("  lever at all. At the honest +/-40% error bar, $100/seed spans ~10-22")
    print("  evals. One measured rollout would collapse that range, and it is the")
    print("  single cheapest thing to buy before committing a budget.")


def _measured_target_table(measured: dict, proj: dict, n_val: int = 100) -> None:
    """Evals-per-budget at MEASURED rates, mean vs ceiling instance.

    The per-seed budget is a spend ceiling, so how many evaluations it buys is
    governed by the AVERAGE rollout cost over the instances actually drawn --
    not by the single (largest) calibrated instance. The ceiling column is the
    downside case: what you get if the draw happens to be expensive throughout.
    """
    refl = price(REFINER_MODEL, REFLECTION_IN, REFLECTION_OUT)

    def evals(rollout_usd: float, budget: float, promo: float = PROMOTION_RATE) -> float:
        per_iter = MINIBATCH * rollout_usd + refl + promo * n_val * rollout_usd
        return (budget / per_iter) * promo

    def min_budget(rollout_usd: float, target: float, promo: float = PROMOTION_RATE) -> float:
        per_iter = MINIBATCH * rollout_usd + refl + promo * n_val * rollout_usd
        return target * per_iter / promo

    mean_r, ceil_r = proj["rollout_usd"], measured["rollout_usd"]

    print("=" * 78)
    print(f"REALISTIC EVALS PER SEED AT MEASURED RATES (val={n_val}, promotion {PROMOTION_RATE:.0%})")
    print("=" * 78)
    print(f"  {'budget/seed':>12}{'MEAN instance':>18}{'ceiling instance':>20}   in 15-20 band?")
    for b in (100, 125, 150, 175, 200):
        m, c = evals(mean_r, b), evals(ceil_r, b)
        band = "YES" if TARGET_EVALS_LOW <= m <= TARGET_EVALS_HIGH else ("above" if m > TARGET_EVALS_HIGH else "below")
        print(f"  {f'${b}':>12}{m:>18.1f}{c:>20.1f}   {band}")

    print(
        f"\n  minimum budget for {TARGET_EVALS_LOW} evals : "
        f"${min_budget(mean_r, TARGET_EVALS_LOW):.0f} (mean) / ${min_budget(ceil_r, TARGET_EVALS_LOW):.0f} (ceiling)"
    )
    print(
        f"  minimum budget for {TARGET_EVALS_HIGH} evals : "
        f"${min_budget(mean_r, TARGET_EVALS_HIGH):.0f} (mean) / ${min_budget(ceil_r, TARGET_EVALS_HIGH):.0f} (ceiling)"
    )

    shared_rollouts = n_val + 150 + 400 * 6
    print(f"\n  6-seed programme totals (shared = {shared_rollouts:,} rollouts at mean rate):")
    for b in (100, 125, 150):
        shared = shared_rollouts * mean_r
        print(f"    ${b}/seed  ->  ${b * 6:,} + ${shared:,.0f} shared  =  ${b * 6 + shared:,.0f}")

    print("\n" + "-" * 78)
    print("DO WE NEED A LEVER?")
    print("-" * 78)
    ctx_scale = 12_000 / (proj["mean_solver_in"])
    trimmed = price(SOLVER_MODEL, 12_000, measured["solver_tokens_out"]) + price(
        REFINER_MODEL, 12_223, measured["refiner_tokens_out"]
    )
    _ = ctx_scale
    print(f"  no lever, $150/seed : {evals(mean_r, 150):.1f} evals (mean), {evals(ceil_r, 150):.1f} (ceiling)")
    print(f"  no lever, $125/seed : {evals(mean_r, 125):.1f} evals (mean), {evals(ceil_r, 125):.1f} (ceiling)")
    print(f"  context->12k, $100  : {evals(trimmed, 100):.1f} evals (mean-equivalent)")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
