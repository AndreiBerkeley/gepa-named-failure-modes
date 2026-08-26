# Phase 1 — Emulation stability + cost calibration

Date: 2026-08-07 · Stability checks and prompt sizing cost **$0**. The single
calibration rollout is billed and gated on Andrei.

## 1. Emulation stability — PASSED (batch A)

The open question from the smoke test was whether amd64 emulation perturbs
timing-sensitive tests. A flaky verdict is not merely noise in the headline
number — it corrupts the reward signal GEPA optimizes against, so a candidate
can be promoted or rejected for reasons unrelated to its patch.

Method: run the same gold-patch evaluations three times and require identical
verdicts. Gold patches must always resolve, so any disagreement is environmental.

```
3 instances x 3 repeats, gold patches, $0

  round 1/3: 3/3 resolved  (6.1 min)
  round 2/3: 3/3 resolved  (6.1 min)
  round 3/3: 3/3 resolved  (6.1 min)

  astropy__astropy-13477   ['PASS', 'PASS', 'PASS']
  astropy__astropy-13803   ['PASS', 'PASS', 'PASS']
  astropy__astropy-14528   ['PASS', 'PASS', 'PASS']

STABLE: no emulation-induced flakiness detected at this sample size.
```

Round wall-time was identical to 0.1 min across all three rounds, which is
itself evidence that emulation is behaving deterministically here.

### Honest limitation

All three instances are **astropy** — they are simply the first three in the
sorted val manifest, so this is a single-repo, single-test-framework sample. It
is weak evidence for the benchmark as a whole. A cross-repo batch
(`--spread`, distinct repos) was therefore run as batch B; see below.

Even with both batches, "no flakiness observed in 3 repeats" is not "no
flakiness". It is enough to trust local Docker for **plumbing and smoke tests**,
which is all we intend to use it for. Real runs still go to the x86_64 box.

## 2. Prompt sizing — measured, free, and it moved the numbers

`scripts/calibrate_rollout.py --sample 6` runs real BM25 retrieval over real
checkouts for six train instances spanning six repos. No LM calls.

| instance | context chars | solver prompt chars |
|---|---:|---:|
| astropy__astropy-11693 | 60,352 | 74,272 |
| django__django-10087 | 35,122 | 35,656 |
| matplotlib__matplotlib-13913 | 55,297 | 59,259 |
| mwaskom__seaborn-2389 | 60,274 | 60,946 |
| pallets__flask-4045 | 54,448 | 54,982 |
| psf__requests-1142 | 42,502 | 43,186 |

```
context chars : mean 51,332  median 54,872  min 35,122  max 60,352
solver prompt : mean 54,717  median 57,120   (~15,199 tok at 3.6 chars/tok)
2/6 instances SATURATE the 60k-char context cap.
```

**This partly corrects a caveat I gave earlier.** I said the true baseline "may
already be cheaper than modelled, possibly meeting the target with no lever at
all". Retrieval fills far more of the cap than that implied, and saturates it
outright on a third of the sample. The mean still lands ~12% *below* the
modelled 17,350 solver input tokens — but chars-per-token for source code is
typically below 3.6, which pushes the true token count back up. The two effects
work against each other, and only a billed call settles which wins.

## 3. The calibration rollout — one billed call

What it establishes that free sizing cannot:

1. **Real chars→tokens ratio** for this prompt mix (the ±40% error bar is mostly
   this).
2. **Real output sizes** — `PATCH_OUT_TOKENS = 800` is a pure guess today, and
   output is billed at 5× the input rate on both models.
3. That the end-to-end path works against live Bedrock with bearer auth.

Design choices worth noting:
- Runs on a **train** instance, so val / generation / test stay uncontaminated.
- Hard `--max-spend` ceiling (default $0.50) independent of the estimate.
- Writes `results/calibration/calibration.json`, which
  `scripts/cost_model.py --measured` consumes to regenerate every table.

### Cost estimate

The default instance (`astropy__astropy-11693`) is the **largest** of the six
sampled — a deliberately conservative choice:

| | tokens (est.) | rate | cost |
|---|---:|---|---:|
| solver (Haiku 4.5) | 20,631 in / 800 out | $1/$5 | $0.0247 |
| refiner (Sonnet 5) | 21,491 in / 800 out | $2/$10 | $0.0510 |
| **total** | | | **≈ $0.076** |

Under eight cents, ceiling-bounded at $0.50.
