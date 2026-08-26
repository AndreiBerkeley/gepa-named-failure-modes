# Phase 0 — Canonical GEPA repo verification and baseline pin

Date: 2026-08-07

## Repo verification

`github.com/gepa-ai/gepa` is confirmed canonical:

| Field | Value |
|---|---|
| `full_name` | `gepa-ai/gepa` |
| Description | "Optimize prompts, code, and more with AI-powered Reflective Optimization" |
| Stars | 6,025 |
| `fork` | `false` |
| Created | 2025-08-05 |
| Last push | 2026-08-06 |
| License | MIT |
| Homepage | https://gepa-ai.github.io/gepa/ |
| Default branch | `main` |

The PyPI package `gepa` lists `Homepage = https://github.com/gepa-ai/gepa`, and its
latest version matches the repo's latest tag — the repo and the published package
are the same project. No competing/renamed upstream was found.

## Baseline pin

All three baseline seed runs use this exact release, unmodified:

| | |
|---|---|
| Release tag | **`v0.1.4`** |
| Commit SHA | **`8b0ce6cd99a234f6b74daf37558a2ac0ce18f975`** |
| Published | 2026-07-15 |
| PyPI version | `0.1.4` (sdist + wheel uploaded 2026-07-15) |
| `requires_python` | `>=3.10,<3.15` |

`v0.1.4` is the latest release as of 2026-08-07. Note that `main` has moved past it
(HEAD at clone time was `8a2bed96385202f69caaeb5327a843ed2f5ea225`, 2026-08-05) —
**baselines pin the release, not `main`**, so the baseline is reproducible from PyPI
by anyone.

The local clone at `GEPA/gepa/` is checked out at the `v0.1.4` tag (detached HEAD).
PR branches for upstream-bound work will be cut from `main`, not from this tag.

## Reproducing the pin

```bash
git clone https://github.com/gepa-ai/gepa.git && cd gepa && git checkout v0.1.4
```

Environment note: system Python here is 3.9.6, which is below gepa's floor of 3.10.
The `experiments/` workspace is uv-managed on Python 3.12 for this reason.
