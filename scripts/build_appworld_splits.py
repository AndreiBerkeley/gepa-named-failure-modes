#!/usr/bin/env python
"""Write the committed AppWorld split manifests. Free: reads task ids only.

Unlike HotpotQA, **nothing is sampled here**. AppWorld ships canonical splits and
the published results (ACE, the leaderboard) are reported on them, so inventing
our own would make our numbers incomparable with the only GEPA baseline that
exists for this benchmark. The manifests exist to *record* which ids we used, as
committed stage artifacts, not to choose them.

The mapping to GEPA's three roles:

======  ===================  ====  =========================================
role    AppWorld split          n  why
======  ===================  ====  =========================================
train   train                  90  minibatch sampling only
val     dev                    57  candidate selection
test    test_normal           168  held-out; the headline comparison
======  ===================  ====  =========================================

``test_challenge`` (417) is deliberately left out of the main comparison. It is
the harder distribution, and ACE reports it separately; mixing it into test
would make our test score incomparable with the published numbers.

**val = 57 is small and is the known weakness of this arm**. It is close
to the val=60 that proved noise-dominated on SWE-Bench. What makes it tolerable
is partial credit: AppWorld scores a fraction of substantive requirements per
task rather than 0/1, so a 57-task val carries far more information than 60
binary outcomes did. It is still the number to watch if selection looks unstable.

    uv run python scripts/build_appworld_splits.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "manifests" / "appworld"

#: GEPA role -> AppWorld split name.
ROLE_TO_SPLIT = {"train": "train", "val": "dev", "test": "test_normal"}

#: AppWorld lives in its own WSL venv: it pins pydantic <2 while gepa and litellm
#: need pydantic v2, so its task ids are read out through a subprocess rather
#: than imported.
WSL_DISTRO = "Ubuntu-24.04"
WSL_PYTHON = "~/appworld/.venv/bin/python"


def load_task_ids(split: str) -> list[str]:
    code = f"from appworld.task import load_task_ids; print('\\n'.join(load_task_ids('{split}')))"
    proc = subprocess.run(
        ["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc", f'cd ~/appworld && {WSL_PYTHON} -c "{code}"'],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise SystemExit(f"could not read AppWorld split {split!r}: {proc.stderr.strip()[:400]}")
    ids = [line.strip() for line in proc.stdout.replace("\r", "").splitlines() if line.strip()]
    # The WSL shell prints profile noise on some setups; task ids look like
    # "82e2fac_1", so anything without an underscore is not one.
    return [i for i in ids if "_" in i and " " not in i]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()

    for role, split in ROLE_TO_SPLIT.items():
        ids = load_task_ids(split)
        if not ids:
            raise SystemExit(f"AppWorld split {split!r} returned no task ids")

        overlap = seen & set(ids)
        if overlap:
            raise SystemExit(f"FATAL: {role} overlaps an earlier split: {sorted(overlap)[:5]}")
        seen |= set(ids)

        manifest = {
            "name": role,
            "benchmark": "appworld",
            "appworld_split": split,
            "n": len(ids),
            # Sorted: gepa keys val subscores and the Pareto frontier by POSITION,
            # not by id, so a stable order is load-bearing.
            "task_ids": sorted(ids),
        }
        path = args.out / f"{role}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"  {role:6} <- appworld/{split:12} n={len(ids):4}  -> {path.relative_to(REPO)}")

    print(f"\ntotal {len(seen)} tasks, all disjoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
