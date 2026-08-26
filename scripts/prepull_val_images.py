#!/usr/bin/env python
"""Pre-pull the val-split Docker images. FREE (bandwidth + disk only, no LM calls).

Why this exists
---------------
Only ``sweb.eval.*`` (per-instance) images are published; there are no public
``sweb.env.*`` images. So on the pull path each evaluation fetches a ~1.19 GB
image that expands to ~4 GB on disk.

``should_remove()`` in ``swebench/harness/docker_utils.py`` deletes an instance
image after use only when ``clean or not existed_before``. So an image that was
already present **before** the run is kept:

    if cache_level in {"none","base","env"} and (clean or not existed_before):
        return True          # <- skipped when existed_before is True

That is the lever. Pre-pulling the 100 val images makes them permanent under
``--cache_level env --clean False``, while transient train-minibatch images are
still cleaned up after each use. Without this, every val evaluation re-downloads
1.19 GB -- and val accounts for ~1,236 of the ~1,384 graded rollouts per seed.

    uv run python scripts/prepull_val_images.py            # pull
    uv run python scripts/prepull_val_images.py --dry-run  # just report size

Budget ~400 GB of Docker disk for these, plus transient headroom.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = REPO_ROOT / "manifests" / "swebench_verified"

#: The harness maps instance ids to image names by lowercasing and replacing
#: the '__' separator with '_1776_' (see swebench test_spec.instance_image_key).
NAMESPACE = "swebench"
ARCH = "x86_64"  # swebench 4.1.0 hardcodes this; arm64 runs them under emulation


def image_name(instance_id: str) -> str:
    return f"{NAMESPACE}/sweb.eval.{ARCH}.{instance_id.lower().replace('__', '_1776_')}:latest"


def local_images() -> set[str]:
    out = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True)
    return set(out.stdout.split())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFESTS / "val.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="pull only the first N (for a partial warm-up)")
    args = ap.parse_args()

    from gepa_taxonomy.splits import load_manifest

    ids = load_manifest(args.manifest)
    if args.limit:
        ids = ids[: args.limit]
    images = [image_name(i) for i in ids]

    have = local_images()
    missing = [im for im in images if im not in have]

    print(f"val instances      : {len(ids)}")
    print(f"already present    : {len(images) - len(missing)}")
    print(f"to pull            : {len(missing)}")
    print(f"download estimate  : {len(missing) * 1.19:.0f} GB")
    print(f"on-disk estimate   : {len(missing) * 4.0:.0f} GB")
    print("\nThese stay resident: swebench's should_remove() keeps images that")
    print("existed before the run, so --cache_level env will not delete them.")

    if args.dry_run:
        print("\n--dry-run: nothing pulled.")
        return 0

    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        print("docker is not running", file=sys.stderr)
        return 2

    failures: list[str] = []
    for n, im in enumerate(missing, 1):
        print(f"[{n}/{len(missing)}] {im}")
        if subprocess.run(["docker", "pull", "--quiet", im], capture_output=True).returncode != 0:
            failures.append(im)
            print("    FAILED", file=sys.stderr)

    print(f"\npulled {len(missing) - len(failures)}/{len(missing)}")
    if failures:
        print("failed:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
