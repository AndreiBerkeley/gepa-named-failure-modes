#!/usr/bin/env python
"""Vendor the two constraint-verifier sets the IFBench arm needs. FREE: downloads.

The arm follows GEPA's published setup, which uses **two different
datasets with two disjoint constraint vocabularies**:

* **train / val** come from ``allenai/IF_multi_constraints_upto5`` (IF-RLVR
  Train), whose constraints are IFEval-style and are verified by AllenAI's
  ``open-instruct`` IFEvalG registry.
* **test** is ``allenai/IFBench_test``, whose 58 constraints are new and
  deliberately out-of-distribution, and are verified by the IFBench registry.

Measured overlap between the two vocabularies: **zero**. That separation is the
benchmark -- the paper splits this way "to ensure that the optimizers do not
access the new, unseen constraints being tested in IFBench" -- so both registries
have to be present and grading has to route by split.

Both are Apache-2.0 and are vendored **verbatim** apart from one mechanical edit:
upstream uses imports that only resolve from their own repo roots, so those are
rewritten to package-relative form.

Vendoring rather than reimplementing is deliberate. There are 58 + ~170
constraint types and each is a place to introduce a silent scoring bug -- a wrong
verifier does not crash, it mis-scores an entire constraint class in both arms.

    uv run python demo/vendor_ifbench.py
    uv run python demo/vendor_ifbench.py --check   # detect drift, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "src" / "gepa_taxonomy" / "ifbench" / "_vendor"

FILES = ("instructions.py", "instructions_util.py", "instructions_registry.py")

#: package dir -> (raw base URL, provenance line, import rewrites)
SOURCES: dict[str, tuple[str, str, tuple[tuple[str, str], ...]]] = {
    "ifbench": (
        "https://raw.githubusercontent.com/allenai/IFBench/main",
        "github.com/allenai/IFBench @ main",
        (
            ("import instructions_util\n", "from gepa_taxonomy.ifbench._vendor.ifbench import instructions_util\n"),
            ("import instructions\n", "from gepa_taxonomy.ifbench._vendor.ifbench import instructions\n"),
        ),
    ),
    "ifevalg": (
        "https://raw.githubusercontent.com/allenai/open-instruct/main/open_instruct/IFEvalG",
        "github.com/allenai/open-instruct @ main (open_instruct/IFEvalG)",
        (
            (
                "from open_instruct.IFEvalG import instructions_util",
                "from gepa_taxonomy.ifbench._vendor.ifevalg import instructions_util",
            ),
            (
                "from open_instruct.IFEvalG import instructions",
                "from gepa_taxonomy.ifbench._vendor.ifevalg import instructions",
            ),
        ),
    ),
}

HEADER = '''"""VENDORED from {origin} -- do not edit by hand.

Apache-2.0. Refresh with demo/vendor_ifbench.py.
The only change from upstream is that its imports are made package-relative.
"""

'''


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read().decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare against upstream, write nothing")
    args = parser.parse_args()

    VENDOR.mkdir(parents=True, exist_ok=True)
    (VENDOR / "__init__.py").write_text(
        '"""Vendored constraint verifiers. See demo/vendor_ifbench.py."""\n', encoding="utf-8"
    )

    drift = False
    for package, (base, origin, rewrites) in SOURCES.items():
        target_dir = VENDOR / package
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "__init__.py").write_text(f'"""Vendored from {origin}."""\n', encoding="utf-8")
        print(f"{package}  <- {origin}")

        for name in FILES:
            source = fetch(f"{base}/{name}")
            for old, new in rewrites:
                source = source.replace(old, new)
            payload = HEADER.format(origin=origin) + source
            digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
            target = target_dir / name

            if args.check:
                if not target.exists():
                    print(f"  {name:<26} MISSING")
                    drift = True
                elif target.read_text(encoding="utf-8") != payload:
                    print(f"  {name:<26} DRIFTED from upstream")
                    drift = True
                else:
                    print(f"  {name:<26} ok ({digest})")
                continue

            target.write_text(payload, encoding="utf-8")
            print(f"  {name:<26} {len(payload.splitlines()):>5} lines  sha256:{digest}")

    if args.check:
        if drift:
            print("\nvendored copy differs from upstream. Re-run without --check to refresh.")
            return 1
        return 0

    print(f"\nvendored to {VENDOR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
