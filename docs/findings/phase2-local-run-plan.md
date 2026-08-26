# Phase 2 — Local-only run plan (server unavailable)

Date: 2026-08-07 · All analysis free. Every number below is derived from
measurements already taken, not from vendor guidance.

## Headline, without optimism

**Phase 2 is feasible locally, but it is a multi-day machine commitment, and the
final test evaluation is a bigger job than the seeds themselves.**

| stage | wall clock (4 workers) |
|---|---|
| val image pre-pull (one-time) | ~3 h (119 GB download) |
| base-candidate val evaluation | ~0.8 h |
| **per seed at $100** | **~15.8 h** |
| 3 seeds | **~47 h** |
| final test eval (400 × 6 arms) | **~45 h** (see caveat) |

Three seeds is roughly **two days of continuous running**; the whole programme
including final evaluation is closer to **four to five days**. If that is
unacceptable, the levers are at the end.

## 1. Evaluation backend: local Docker

### Rollout accounting per seed at $100

From measured rates (mean rollout $0.0714, reflection $0.0237):

```
per-iteration $2.0229  ->  49 iterations
  val rollouts      1,236   (all on the SAME 100 val instances)
  minibatch rollouts  148   (mostly distinct train instances)
  TOTAL             1,384 graded rollouts
```

**89% of grading work lands on the 100 val instances.** That single fact drives
the whole disk and time strategy.

### Measured evaluation time

From the stability batches, 1 worker, images already local:

| batch | instances | time/round | per instance |
|---|---|---|---|
| A (astropy ×3) | 3 | 6.1 min | 2.03 min |
| B (astropy, django, matplotlib) | 3 | 5.1–5.3 min | ~1.75 min |

I use **1.9 min per evaluation** as the steady-state figure, plus **~1.5 min**
when the image must be pulled first (1.19 GB).

### Worker count and Docker resources

| setting | recommendation | reasoning |
|---|---|---|
| `--max-workers` | **4** | 10 CPU cores; emulated test suites are largely single-threaded, so 4 concurrent containers leave headroom for the harness and macOS. |
| Docker VM memory | **12 GB** (from 7.75 GB) | ~1.5–2 GB per running container × 4, plus daemon overhead. Leaves ~4 GB for macOS — workable only if the Mac is dedicated during runs. |
| Docker VM disk | **500 GB** | see §3 |
| `--cache_level` | **env** | with pre-pulled val images (§3) |

I model 4 workers as a **3.0× effective speedup**, not 4×. Emulation is
CPU-bound and contended; assuming linear scaling here would be exactly the
optimism you asked me to avoid. If measured throughput beats that, the pilot
will show it.

```
workers=1 (×1.0):  47.5 h/seed
workers=2 (×1.8):  26.4 h/seed
workers=4 (×3.0):  15.8 h/seed   <- recommended
workers=6 (×3.8):  12.5 h/seed   (likely RAM-bound at 16 GB; not recommended)
```

## 2. Interruption tolerance

Resume is verified state-faithful for everything gepa persists plus our spend
meter (`docs/findings/phase1-resume-fidelity.md`). Two gaps must be closed
before "laptop sleep/reboot mid-run" is genuinely safe:

### ⚠️ "Zero paid-work loss" is not free — it needs one more component

gepa saves state **once per iteration** (`engine.py:742`). A crash mid-iteration
therefore loses that iteration's spend: **up to ~$2.02**, about 28 rollouts.
That is small in dollars but ~50 minutes of local wall clock.

To get true rollout-granularity durability, the adapter needs a **write-through
rollout cache**: append each graded `(candidate_hash, instance_id) -> score` to
a JSONL as soon as it completes, and consult it on restart before re-running.
This is the seed-cache mechanism generalised, and it also removes the cost of
re-grading val instances after any restart. **Not yet built** — I flag it rather
than claim the requirement is met.

Until it exists, the honest statement is: **loss is bounded by one iteration
(~$2 and ~50 min), not zero.**

### Launch wrapper

```bash
caffeinate -dimsu <command> 2>&1 | tee -a results/runs/<run>/console.log
```

`-d` display, `-i` idle, `-m` disk, `-s` system sleep on AC, `-u` user-active
assertion. Note `caffeinate` cannot prevent **lid-close** sleep on battery —
runs must be on AC with the lid open, or with an external display attached.

Logging is `tee -a` (append) so a resumed run adds to the same log rather than
truncating it. `PYTHONUNBUFFERED=1` keeps the log current if the process dies.

## 3. Disk

Only `sweb.eval.*` (per-instance) images are published — there are **no public
`sweb.env.*` images**, so the "keep env images" reading of `--cache_level` does
not apply on the pull path. Each evaluation fetches ~1.19 GB, expanding to
~4.0 GB on disk (measured: 4.00 GB disk / 1.19 GB content).

The lever is in `swebench/harness/docker_utils.py::should_remove`:

```python
if cache_level in {"none","base","env"} and (clean or not existed_before):
    return True     # skipped when existed_before is True
```

**Images that existed before the run are never removed.** So pre-pulling the 100
val images makes them permanent, while transient train-minibatch images are
still cleaned after use.

| item | size |
|---|---|
| val images, pre-pulled and resident | 100 × 4.0 GB = **400 GB** |
| transient instance images (4 workers) | 4 × 4.0 GB = 16 GB |
| base layers + slack | ~60 GB |
| **Docker disk to allocate** | **~500 GB** |
| one-time download | 100 × 1.19 GB = **119 GB** |

836 GB is free, so this fits with room to spare. Distinct env footprints per
split, for reference: val 47, generation 56, test 88, train 120 — the repeats
are why the *pull* strategy, not the *env* strategy, is what matters here.

### The test-evaluation caveat

The final test evaluation touches **400 distinct instances × 6 arms**. Caching
all 400 would be 1.6 TB, which does not fit. So test evaluation re-pulls, and
its wall clock (~45 h) is dominated by download, not compute. Options when we
get there: evaluate arms in sequence reusing a 400-image cache (1.6 TB — no), or
accept the pull cost, or revisit test size. **Not a Phase 2 blocker, but it
should not be a surprise later.**

## 4. Sequencing (all gated on Andrei)

1. **Pre-pull val images** — free, ~3 h, 119 GB down / 400 GB disk.
2. **Base-candidate val evaluation** — 100 rollouts, **~$7.14**, ~0.8 h. Run
   once; replayed by every seed and both arms (D009).
3. **SEED 1 pilot** — $100, ~15.8 h. Validates wall clock and spend telemetry
   against these estimates *before* committing the other two seeds.
4. **Seeds 2–3** — after the pilot's numbers are checked.

## 5. Migration back to the server

**Nothing in this configuration blocks a later migration**, and I checked the
specific mechanisms rather than assuming:

- `run_dir` state is a pickle of plain containers (lists, dicts, floats) plus
  `adapter_state`, which we keep JSON-serialisable. Portable.
- Split manifests and the seed cache are committed JSON.
- Every host-specific knob is a flag or env var: workers, cache level, paths,
  run id, profile prefix.
- Docker images are re-pulled on the target; nothing local is baked in.

Three conditions for the state to load cleanly on another machine:

1. **Same gepa version** (v0.1.4) — pickle contains gepa-defined classes.
2. **Same Python minor version** (3.12) — pickle protocol/class layout.
3. **`gepa_taxonomy` importable** at the same module paths.

All three are already pinned. One caveat worth stating: migrating mid-run
inherits the same resume behaviour as any restart — prior work is preserved, but
the minibatch stream restarts, so the trajectory differs from an uninterrupted
run. That is acceptable under the uniform-tranche policy (D014), provided a
migration is applied at the same point for all seeds or its timing is recorded.

## 6. Levers, if ~47 h for three seeds is too much

Options only; none changes a pinned decision.

| lever | effect | cost |
|---|---|---|
| **Skip Docker for non-applying patches** | 20% → 12.7 h/seed, 30% → 11.1, 40% → 9.5 | none — an empty/malformed patch scores 0 either way, which is exactly how the harness already treats it. **Recommended.** |
| Reduce val 100 → 60 | ~40% fewer graded rollouts | reverses D010, reintroduces noise-dominated selection |
| Lower promotion rate | fewer full val evals | not directly controllable; a consequence of search behaviour |
| Run seeds concurrently | no total speedup | 16 GB RAM cannot host two 4-worker runs |

The first is nearly free and I would take it: it removes container startup for
rollouts whose patch cannot apply, which with a plain seed instruction is likely
a substantial fraction early in each run.
