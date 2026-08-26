#!/usr/bin/env python
"""Minimal invocation probe. **BILLED, but ~$0.0002 total.**

Why a billed probe is unavoidable
---------------------------------
Every Bedrock control-plane authorization surface reports Sonnet 5 as
``AUTHORIZED`` / ``AVAILABLE`` / ``ACTIVE`` on this account -- and Sonnet 5
403s on invocation. Checked and all uninformative:

  GetFoundationModelAvailability   authorizationStatus=AUTHORIZED,
                                   entitlementAvailability=AVAILABLE
  GetInferenceProfile              status=ACTIVE
  ListFoundationModels             modelLifecycle=ACTIVE
  ListFoundationModelAgreementOffers  returns an offer

The restriction is therefore enforced *outside* Bedrock's model-access layer --
an IAM identity policy or SCP scoped to model ARNs on the API key -- and no
Bedrock control-plane call exposes it. The only signal that matches reality is
an actual invocation.

So this probe sends the smallest possible request per model: a 3-token prompt
with ``max_tokens=1``. At Sonnet rates that is well under a hundredth of a cent
each; the whole run is ~$0.0002.

    zsh -c 'source ~/.zshrc >/dev/null 2>&1; \
      uv run python scripts/probe_invocation.py'

Writes results/access/invocation_probe.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "access"

PROMPT = "Say OK."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefixes", nargs="*", default=["global.", "us."])
    ap.add_argument("--dry-run", action="store_true", help="list what would be probed; free")
    args = ap.parse_args()

    from gepa_taxonomy.cost import SOLVER_BASE, SONNET_CANDIDATES, price_call, with_profile

    targets: list[str] = []
    for base in (*SONNET_CANDIDATES, SOLVER_BASE):
        for pref in args.prefixes:
            targets.append(with_profile(base, pref))

    est = sum(price_call(t, 10, 1) for t in targets)
    print(f"probing {len(targets)} model ids, max_tokens=1 each")
    print(f"estimated total cost: ${est:.6f}\n")
    for t in targets:
        print(f"  {t}")

    if args.dry_run:
        print("\n--dry-run: nothing invoked, nothing spent.")
        return 0

    from gepa_taxonomy.bedrock import BedrockLM, require_credentials

    require_credentials()

    print("\n" + "=" * 78)
    results: list[dict] = []
    for model in targets:
        try:
            lm = BedrockLM(model=model)
            _text, tin, tout = lm.complete(PROMPT, max_tokens=1)
            cost = price_call(model, tin, tout)
            results.append(
                {"model": model, "invocable": True, "tokens_in": tin, "tokens_out": tout, "usd": cost, "error": None}
            )
            print(f"  OK    {model:52} {tin} in / {tout} out  ${cost:.6f}")
        except Exception as exc:
            name = type(exc).__name__
            msg = str(exc)[:160]
            denied = any(k in msg for k in ("AccessDenied", "403", "not authorized", "don't have access"))
            results.append({"model": model, "invocable": False, "error": f"{name}: {msg}", "looks_like_authz": denied})
            print(f"  FAIL  {model:52} {name}{' (authz)' if denied else ''}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "invocation_probe.json"
    out.write_text(json.dumps({"prompt": PROMPT, "results": results}, indent=2) + "\n")

    ok = [r["model"] for r in results if r["invocable"]]
    spent = sum(r.get("usd", 0.0) for r in results)
    print("\n" + "=" * 78)
    print(f"  invocable ({len(ok)}/{len(results)}):")
    for m in ok:
        print(f"    {m}")
    print(f"\n  actually spent: ${spent:.6f}")
    print(f"  wrote {out.relative_to(REPO_ROOT)}")

    sonnets = [m for m in ok if "sonnet" in m]
    if not sonnets:
        print("\n  NO SONNET IS INVOCABLE. The refiner cannot be a Sonnet on this")
        print("  account; the next question is which non-Sonnet model to use.")
        return 1
    print(f"\n  strongest invocable Sonnet: {sonnets[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
