# Phase 1 — SWE-Bench acquisition + evaluation-harness feasibility on this Mac

Date: 2026-08-07 · Updated 2026-08-07 after an end-to-end smoke test.

> **CORRECTION (supersedes finding B below).** I predicted that arm64 would force
> building every image from source, with a fidelity caveat. **That was wrong.**
> swebench 4.1.0 hardcodes `arch="x86_64"` in `make_test_spec` and never calls
> `platform.machine()` anywhere in the package — so on arm64 it pulls the
> **official prebuilt x86_64 images** and runs them under Docker Desktop's amd64
> emulation. A gold-patch smoke test resolved **1/1** on this Mac.
> Local evaluation works, with leaderboard-faithful environments. The real cost
> is emulation *speed*, not environment fidelity. Details in
> [`phase1-harness-smoke-test.md`](phase1-harness-smoke-test.md).
>
> Finding A (only the 2,294-instance `test` split is gradeable) **stands** and is
> unaffected.

## 1. Dataset acquired

`SWE-bench/SWE-bench` (the full dataset) downloads cleanly — public, ungated, ~120 MB
of parquet.

| Split | Rows | Distinct repos |
|---|---|---|
| `train` | 19,008 | 35 |
| `test` | 2,294 | 12 |
| `dev` | 225 | — |
| **Total** | **21,527** | |

Columns: `repo`, `instance_id`, `base_commit`, `patch`, `test_patch`,
`problem_statement`, `hints_text`, `created_at`, `version`, `FAIL_TO_PASS`,
`PASS_TO_PASS`, `environment_setup_commit`.

### ⚠️ Blocking finding A — only the 2,294-instance `test` split is actually evaluable

CLAUDE.md picks the full split "for task count, so the train/eval/test/generation
subsets are large enough", which implies a ~21.5k pool. **That pool is not real.**

Execution-based grading needs a per-instance Docker image with the repo at
`base_commit` and a validated test environment. Evidence that these exist only for
`test`:

- The `swebench` Docker Hub namespace holds 4,503 repositories. I paged 1,000 of them
  and mapped each image name back to an `instance_id`: **1,000 / 1,000 matched the
  `test` split; 0 matched `train`; 0 matched `dev`.**
- The 12 distinct repos in `test` vs 35 in `train` is consistent with this: the
  extra 23 train-only repos have no environment images at all.

`train` instances were scraped but never environment-validated — they are intended as
fine-tuning/retrieval data, not as a gradeable benchmark. Building images for them
ourselves would mean solving per-repo environment setup for 23 unsupported projects,
which is a research project in itself.

**Therefore the usable pool is 2,294 instances (+225 `dev`), not 21,527.** This is
still ample for four disjoint subsets, but the split proposal must partition 2,294 —
and the blog post's framing of "full split" needs this caveat. It also weakens "full
over Verified" as a *task-count* argument: full-test is 2,294 vs Verified's 500, a
4.6× edge, not the 43× the 21,527 figure suggests.

## 2. Evaluation harness on this machine

### ⚠️ Blocking finding B — Docker is not installed, and this is arm64

Machine: **Apple Silicon (arm64)**, macOS 26.5.1, 16 GB RAM, 10 cores, 836 GB free.

```
$ which docker    → not found
$ ls /Applications → no Docker Desktop / OrbStack / Podman / Rancher
$ which colima podman lima nerdctl → none found
```

There is **no container runtime of any kind** on this Mac. Nothing about the SWE-Bench
harness can run until one is installed.

### Against the harness's own stated requirements

From the SWE-bench README (lines 96–102):

> SWE-bench evaluation can be resource intensive. We recommend running on an
> `x86_64` machine with at least 120GB of free storage, 16GB of RAM, and 8 CPU cores.
> […] Support for `arm64` machines is experimental.

| Requirement | This Mac | |
|---|---|---|
| `x86_64` | arm64 | ❌ "experimental" |
| 120 GB free storage | 836 GB | ✅ |
| 16 GB RAM | 16 GB | ⚠️ exactly at floor, shared with Docker VM |
| 8 CPU cores | 10 | ✅ |
| Container runtime | none | ❌ blocking |

### What arm64 actually costs us

The harness *does* have a native arm64 code path — it is not simply broken:

- `TestSpec.platform` maps `arch == "arm64"` → `linux/arm64/v8`
  (`harness/test_spec/test_spec.py:146–152`).
- The Dockerfile generator branches on it, selecting the `aarch64` Miniconda installer
  (`harness/dockerfiles/__init__.py:65–79`).

But: **there are no prebuilt arm64 images.** Direct check —

```
swebench/sweb.eval.arm64.astropy_1776_astropy-12057  → HTTP 404
swebench/sweb.eval.x86_64.astropy_1776_astropy-12057 → HTTP 200
```

— plus 0 arm64 hits across the 1,000 repos scanned. So on arm64 every image must be
**built locally from source**, per instance. Three consequences:

1. **Time.** Each build resolves and compiles a full scientific-Python environment.
   Multiply by the number of distinct instances in our eval/generation subsets, and
   again by every GEPA iteration that re-evaluates them.
2. **Fidelity.** Locally built arm64 environments are not the images the published
   leaderboard used. Some instances are known to behave differently or fail to build
   under aarch64 (native wheels, pinned manylinux deps). Our numbers would not be
   strictly comparable to published SWE-Bench results — acceptable for a controlled
   baseline-vs-taxonomy comparison, but it must be stated in the blog post.
3. **Emulation is not a fix.** Running the x86_64 images under QEMU on Apple Silicon
   is functional but drastically slower, and test suites with timing-sensitive tests
   become flaky — which corrupts the reward signal GEPA optimizes against. Not
   recommended.

### The realistic options

| Option | What it is | Trade-off |
|---|---|---|
| **1. Modal (cloud)** | `run_evaluation.py` has a first-class `--modal` path (`--modal` arg; `run_instances_modal`, `validate_modal_credentials`). Runs official x86_64 images in the cloud. | Leaderboard-faithful, no local Docker, parallel. Costs money (compute, not API tokens) — **your call**. Strongest option. |
| **2. x86_64 cloud VM** | One rented Linux box meeting the stated specs; pull official images. | Faithful, full control, predictable cost. Needs provisioning + we run GEPA there too, or split solver/eval across machines. |
| **3. Local arm64 + build** | Install OrbStack/Docker Desktop, build images natively. | Free. Slow, 16 GB RAM is tight, fidelity caveat, some instances may not build. Viable for *smoke-testing* the pipeline on a handful of instances. |
| **4. Local x86_64 emulation** | QEMU under Docker Desktop. | Not recommended — flaky timing corrupts the signal. |

**Recommendation: Option 1 (Modal) for real runs, Option 3 for local smoke tests.**
Modal keeps us leaderboard-faithful and sidesteps both blocking findings; a local
OrbStack install lets us validate plumbing end-to-end on ~3 instances without
burning cloud spend. These are complementary, not exclusive.

### Immediately actionable, free

Installing a runtime is free and unblocks the smoke-test path:

```bash
brew install --cask orbstack
```

(OrbStack over Docker Desktop: substantially lighter on RAM, which matters at our
16 GB floor.) I have not run this — install it when you want the local path opened.

## Open questions for you

1. Evaluation venue — Modal, x86 VM, or local-only? Blocks any real evaluation.
2. Given finding A, confirm we partition the **2,294-instance `test` split**. If you
   want a larger pool, the alternative is a different benchmark, not a different split.
3. Is the arm64/local fidelity caveat acceptable for smoke tests only, or do you want
   everything on faithful x86_64 images?
