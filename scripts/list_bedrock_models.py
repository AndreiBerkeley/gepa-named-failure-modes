#!/usr/bin/env python
"""List Anthropic models available on this Bedrock account. FREE -- control plane only.

`bedrock:ListFoundationModels` and `ListInferenceProfiles` are control-plane
calls: they enumerate metadata and do NOT invoke a model, so they cost nothing.
Model *invocation* remains billed and gated.

    zsh -c 'source ~/.zshrc >/dev/null 2>&1; uv run python scripts/list_bedrock_models.py'

Never prints credential values.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print(
            "AWS_BEARER_TOKEN_BEDROCK is not set.\n"
            "Non-interactive shells do not auto-load ~/.zshrc; run this as:\n"
            "  zsh -c 'source ~/.zshrc >/dev/null 2>&1; uv run python scripts/list_bedrock_models.py'",
            file=sys.stderr,
        )
        return 2

    import boto3
    import botocore

    region = os.environ.get("AWS_REGION", "us-east-1")
    print(f"botocore {botocore.__version__} | region {region}\n")

    client = boto3.client("bedrock", region_name=region)

    try:
        summaries = client.list_foundation_models(byProvider="anthropic")["modelSummaries"]
    except Exception as exc:
        print(f"ListFoundationModels FAILED: {type(exc).__name__}", file=sys.stderr)
        print(str(exc)[:600], file=sys.stderr)
        print(
            "\nIf this is an auth error, the bearer token may not cover the control "
            "plane. Run the listing from a shell with working credentials.",
            file=sys.stderr,
        )
        return 1

    print(f"{len(summaries)} Anthropic foundation models visible\n")
    header = f"  {'modelId':62} {'inference types':30} status"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in sorted(summaries, key=lambda x: x["modelId"]):
        types = ",".join(m.get("inferenceTypesSupported", []))
        status = m.get("modelLifecycle", {}).get("status", "")
        print(f"  {m['modelId']:62} {types:30} {status}")

    # Cross-region inference profiles are what you actually call for models that
    # are INFERENCE_PROFILE-only (no ON_DEMAND support).
    try:
        profiles = client.list_inference_profiles()["inferenceProfileSummaries"]
    except Exception as exc:
        print(f"\nListInferenceProfiles unavailable: {type(exc).__name__}", file=sys.stderr)
        return 0

    anth = [p for p in profiles if "anthropic" in p["inferenceProfileId"].lower()]
    print(f"\n{len(anth)} Anthropic inference profiles\n")
    print(f"  {'inferenceProfileId':62} status")
    print("  " + "-" * 72)
    for p in sorted(anth, key=lambda x: x["inferenceProfileId"]):
        print(f"  {p['inferenceProfileId']:62} {p.get('status', '')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
