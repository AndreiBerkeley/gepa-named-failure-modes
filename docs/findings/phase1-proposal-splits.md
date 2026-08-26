# Phase 1 — PROPOSAL: split sizes (needs approval before implementation)

Date: 2026-08-07 · Status: **awaiting Andrei's decision**. Nothing built yet.

## Pool

Per `phase1-swebench-feasibility.md` finding A, the evaluable pool is the
**2,294-instance `test` split**, not 21,527. All numbers below partition 2,294.

Repo distribution is heavily skewed — this drives the stratification requirement:

| repo | n | % |
|---|---:|---:|
| django/django | 850 | 37.1% |
| sympy/sympy | 386 | 16.8% |
| scikit-learn/scikit-learn | 229 | 10.0% |
| sphinx-doc/sphinx | 187 | 8.2% |
| matplotlib/matplotlib | 184 | 8.0% |
| pytest-dev/pytest | 119 | 5.2% |
| pydata/xarray | 110 | 4.8% |
| astropy/astropy | 95 | 4.1% |
| pylint-dev/pylint | 57 | 2.5% |
| psf/requests | 44 | 1.9% |
| mwaskom/seaborn | 22 | 1.0% |
| pallets/flask | 11 | 0.5% |

## The cost asymmetry that determines sizing

The four subsets have wildly different cost profiles. Sizing them uniformly would be
a mistake:

| Subset | How often evaluated | Cost scaling |
|---|---|---|
| **val** (GEPA's valset) | **fully re-evaluated per accepted candidate** | `n_val × n_candidates` — **dominant cost** |
| **train** | only sampled minibatches are run | ~**independent of size** — larger is free |
| **generation** | one pass, after baselines finish | `n_gen × n_seeds` (or ×1) |
| **test** | once per arm at the very end | `n_test × n_arms` (6 arms: 3 baseline + 3 taxonomy) |

So: **val must be small, train can be huge for free, test is a one-time but
multiplied cost.**

## Test sizing is a statistical-power question

The headline claim is "taxonomy-conditioned GEPA beats baseline GEPA". Both arms
score the *same* test instances, so this is a **paired** comparison — use McNemar,
not a two-sample proportion test. Paired is markedly more efficient here.

Power at α=0.05 two-sided, assuming 20% of instances are discordant between arms:

| n_test | +3pp | +5pp | +8pp | +10pp | +15pp |
|---:|---:|---:|---:|---:|---:|
| 150 | 0.12 | 0.27 | 0.60 | 0.82 | 1.00 |
| 200 | 0.15 | 0.35 | 0.73 | 0.92 | 1.00 |
| 300 | 0.21 | 0.49 | 0.89 | 0.99 | 1.00 |
| **400** | **0.27** | **0.61** | **0.96** | **1.00** | 1.00 |
| 600 | 0.37 | 0.79 | 1.00 | 1.00 | 1.00 |

**Honest read: a +3pp effect is not detectable at any size we can afford, and +5pp is
marginal even at 600.** We are powered for a moderate-to-large effect (≥8pp). That is
worth knowing *now*, not after the runs — and it should be stated in the blog post
rather than discovered as a null result and spun.

## Recommended split — Option B

| Subset | n | Rationale |
|---|---:|---|
| `train` | **1,684** | Free to enlarge; maximum minibatch diversity for reflection. |
| `val` | **60** | Cost-dominant. 60 (not 50) because proportional stratification at n=50 drops 2 of 12 repos; at 60 with a ≥1-per-repo floor all 12 survive. |
| `generation` | **150** | At a plausible ~35% solve rate this yields ~95 failure traces — enough material for AdaMAST taxonomy generation with inter-annotator rounds. |
| `test` | **400** | 0.96 power at +8pp; 6 arms × 400 = 2,400 graded rollouts. |
| **Total** | **2,294** | Fully partitioned, no reserve. |

### Alternatives

| | train | val | gen | test | Trade-off |
|---|---:|---:|---:|---:|---|
| **A — cheap** | 1,834 | 60 | 100 | 300 | ~25% less final-eval cost; power drops to 0.89 @ +8pp; thinner taxonomy material. |
| **B — recommended** | 1,684 | 60 | 150 | 400 | Balanced. |
| **C — powered** | 1,434 | 80 | 180 | 600 | 0.79 power @ +5pp. Costs ~50% more on final eval *and* raises the dominant val cost by a third. |

I recommend **B**. C's extra power buys detection of a +5pp effect, but it inflates
the cost-dominant `val` term as well as final eval; if budget allows more spend, it
is better spent on more GEPA iterations per seed than on test precision.

## Construction rules (identical under any option)

1. **Disjoint.** No instance appears in two subsets. Enforced by a test.
2. **Stratified by repo**, proportional, with a **floor of ≥1 per repo** in `val`,
   `generation`, and `test` so no repo is invisible in any evaluated subset.
3. **Deterministic.** One fixed seed (proposal: `20260807`), stratified shuffle. The
   generator script is committed; re-running it reproduces the manifests byte-for-byte,
   verified by a test.
4. **Committed as manifests**, not code-generated at runtime — one JSON file per
   subset under `manifests/swebench_full/`, each containing the seed, the git SHA of
   the generator, the dataset revision, and the sorted `instance_id` list. This is the
   stage boundary that lets someone skip Phases 1–3.
5. **Splits are fixed once and never re-drawn.** Re-drawing after seeing results is
   p-hacking.

### One assignment subtlety worth flagging

`train` and `generation` should ideally be disjoint (rule 1) — a taxonomy built from
failures on instances GEPA already trained against would describe *memorised*
failures, not generalisable ones. Rule 1 gives this for free. But note CLAUDE.md's
leading idea is that traces come from "base candidate only on the generation set",
which is compatible and preserves the cleaner interpretation.

## Blocked on you

1. **Which option — A, B, or C?** (I recommend B.)
2. **Confirm the 2,294 pool** given feasibility finding A.
3. **Seed value** — `20260807` unless you prefer another.

Once you pick, the generator + manifests + disjointness/reproducibility tests are
free work and I will build them immediately.
