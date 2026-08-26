#!/usr/bin/env python
"""Per-account INVOCATION authorization preflight. FREE -- control plane only.

Why this exists
---------------
``ListFoundationModels`` reports a model's **lifecycle** (ACTIVE / LEGACY).
That is a property of the model, not of this account, and it says nothing about
whether we are allowed to invoke it. Conflating the two is what let a refiner
get pinned to a model this account cannot call; the error only surfaced as a
403 partway through a paid rollout.

``GetFoundationModelAvailability`` is the call that answers the real question.
It reports, per account and region:

  authorizationStatus     AUTHORIZED / NOT_AUTHORIZED  <- the decisive field
  entitlementAvailability AVAILABLE / NOT_AVAILABLE
  agreementAvailability   status of the EULA/marketplace agreement
  regionAvailability      whether the model exists in this region at all

A model is invocable only if authorization AND entitlement are green.

Usage (free, no invocation):

    zsh -c 'source ~/.zshrc >/dev/null 2>&1; \
      uv run python scripts/check_model_access.py'

Writes results/access/model_access.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "access"

#: Inference-profile prefixes to report alongside each foundation model.
PROFILE_PREFIXES = ("global.", "us.")


def _availability(client, model_id: str) -> dict:
    try:
        r = client.get_foundation_model_availability(modelId=model_id)
    except Exception as exc:
        return {"error": type(exc).__name__, "detail": str(exc)[:200]}
    return {
        "authorizationStatus": r.get("authorizationStatus"),
        "entitlementAvailability": r.get("entitlementAvailability"),
        "agreementAvailability": (r.get("agreementAvailability") or {}).get("status"),
        "regionAvailability": r.get("regionAvailability"),
    }


def invocable(av: dict) -> bool:
    return av.get("authorizationStatus") == "AUTHORIZED" and av.get("entitlementAvailability") == "AVAILABLE"


def main() -> int:
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print(
            "AWS_BEARER_TOKEN_BEDROCK is not set.\n"
            "  zsh -c 'source ~/.zshrc >/dev/null 2>&1; uv run python scripts/check_model_access.py'",
            file=sys.stderr,
        )
        return 2

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="claude", help="substring filter on modelId")
    args = ap.parse_args()

    import boto3

    region = os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client("bedrock", region_name=region)
    print(f"region {region}\n")

    models = client.list_foundation_models(byProvider="anthropic")["modelSummaries"]
    targets = sorted({m["modelId"] for m in models if args.family in m["modelId"]})
    lifecycle = {m["modelId"]: m.get("modelLifecycle", {}).get("status", "") for m in models}
    profiles = {p["inferenceProfileId"] for p in client.list_inference_profiles().get("inferenceProfileSummaries", [])}

    rows: list[dict] = []
    print(f"{'modelId':46}{'lifecycle':>10}{'authorization':>16}{'entitlement':>14}  invocable")
    print("-" * 100)
    for mid in targets:
        av = _availability(client, mid)
        ok = invocable(av)
        rows.append({"modelId": mid, "lifecycle": lifecycle.get(mid), **av, "invocable": ok})
        auth = av.get("authorizationStatus") or av.get("error", "?")
        ent = av.get("entitlementAvailability") or "-"
        print(f"{mid:46}{lifecycle.get(mid, ''):>10}{auth:>16}{ent:>14}  {'YES' if ok else 'no'}")

    # Which profile ids exist for the invocable models.
    print("\nINFERENCE PROFILES for invocable models")
    print("-" * 100)
    usable: list[str] = []
    for row in rows:
        if not row["invocable"]:
            continue
        for pref in PROFILE_PREFIXES:
            pid = pref + row["modelId"]
            if pid in profiles:
                usable.append(pid)
                print(f"  {pid}")
    if not usable:
        print("  (none)")

    sonnets = [r for r in rows if "sonnet" in r["modelId"] and r["invocable"]]
    haikus = [r for r in rows if "haiku" in r["modelId"] and r["invocable"]]

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  invocable Sonnets : {[r['modelId'] for r in sonnets] or 'NONE'}")
    print(f"  invocable Haikus  : {[r['modelId'] for r in haikus] or 'NONE'}")
    blocked = [r["modelId"] for r in rows if not r["invocable"] and r.get("lifecycle") == "ACTIVE"]
    if blocked:
        print(f"\n  ACTIVE but NOT invocable by this account ({len(blocked)}):")
        for b in blocked:
            print(f"    {b}")
        print("  ^ exactly the trap that a lifecycle-only check misses.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "model_access.json"
    out.write_text(
        json.dumps(
            {"region": region, "models": rows, "usable_profiles": sorted(usable)},
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0 if sonnets else 1


if __name__ == "__main__":
    raise SystemExit(main())
