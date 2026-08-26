# Phase 1 — SWE-Bench harness smoke test on local Docker (arm64)

Date: 2026-08-07 · Cost: **$0** — no LM calls. Uses the gold patch as the
prediction, so it exercises the whole evaluation path without inference spend.

## Result: PASSED

```
instances (1): astropy__astropy-13477
Instances resolved: 1
Instances unresolved: 0
Instances with errors: 0
Unstopped containers: 0
SMOKE TEST PASSED: gold patches resolve; the evaluation path works.
```

Reproduce: `uv run python scripts/smoke_eval.py --instances 1 --max-workers 1`

## The correction this produced

My earlier feasibility writeup predicted that arm64 would have to **build every
image from source**, with a warning that locally built environments would not be
the ones the leaderboard used. **That prediction was wrong**, and the smoke test
is what caught it.

What actually happens, verified in the installed package:

* `swebench/harness/test_spec/test_spec.py:180` — `make_test_spec(..., arch: str = "x86_64")`.
* A grep of the **entire** `swebench` package for `platform.machine`, `uname`, or
  any `arch =` assignment finds **no host detection at all**. The only `arch`
  assignment is `arch=arch` passing the default straight through.
* The `arm64` branches in `test_spec.py:149` and `dockerfiles/__init__.py:66`
  are therefore dead code on the default path.

So on arm64 the harness resolves x86_64 image names, **pulls the official
prebuilt images**, and runs them under Docker Desktop's amd64 emulation.
Observed directly during the run:

```
IMAGE      swebench/sweb.eval.x86_64.astropy_1776_astropy-13477:latest   4GB
CONTAINER  sweb.eval.astropy__astropy-13477.smoke-gold                   Up
```

**Consequence: environments are leaderboard-faithful even locally.** The earlier
fidelity caveat does not apply to this path. What remains is the *speed* cost of
emulation — which is now the primary local-execution concern, not correctness of
the environment.

## What this does and does not tell us

**Does:** the full path works on this machine — image pull, container run, patch
application, test selection, report parsing. The plumbing is sound, and
`scripts/smoke_eval.py` is a free regression check we can re-run any time.

**Does not:** one instance is not a timing-flakiness study. The concern that
emulated execution can perturb timing-sensitive tests — and so corrupt the
reward signal GEPA optimizes against — is *unaddressed* by n=1. Before trusting
local numbers for anything but plumbing, the honest check is to run the same
handful of instances several times and confirm the verdicts are stable. Still
free; worth doing before Phase 2 if you want local eval on the critical path.

## Resource notes

* One instance image is **~4 GB**. Our val split alone is 100 instances; a full
  local eval of val+test would be well past the 120 GB the SWE-Bench README
  recommends. `--cache_level env` (the default here) keeps env images and
  discards instance images to bound this.
* Docker Desktop is allocated **7.75 GiB** of RAM on this Mac — below the 16 GB
  the README recommends. It was sufficient for one worker; raising
  `--max-workers` will need more, and is the first thing to raise if you plan
  meaningful local parallelism.

## Bearing on the venue decision

This makes **local-arm64 a genuinely viable option for smoke tests and small
batches**, which it was not under my earlier (mistaken) reading. It does not
change the recommendation for the *real* runs: those go to the remote x86_64
Linux box, where there is no emulation penalty and no 7.75 GiB ceiling. Modal
stays a fallback we are not building against, per your instruction.
