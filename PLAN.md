# Benchmark Plan — where to run taxonomy-conditioned GEPA next

Written 2026-08-10, after the SWE-Bench Verified round completed end to end.
This document records what we learned, the benchmark options considered, the
recommendation, and how to implement it.

---

## 1. What the SWE-Bench round established

The pipeline works. The experiment did not discriminate.

| | baseline arm | taxonomy arm |
|---|---|---|
| budget / realised | $100 / $100.88 | $100 / $101.59 |
| iterations | 40 | 40 |
| candidates accepted | 13 | 10 |
| base program on val | 11/60 (18.3%) | 11/60 (18.3%) |
| best candidate on val | 13/60 (21.7%) | 13/60 (21.7%) |
| **best candidate on test (150)** | **21/150 (14.0%)** | **23/150 (15.3%)** |

Paired McNemar over the 150 test instances: 14 both-resolved, 7 baseline-only,
9 taxonomy-only, 120 neither. **p = 0.80** on the 16 discordant pairs. The arms
are statistically indistinguishable.

Two findings matter more than the null result:

**A. The acceptance gate selects noise.** With an ~18% base resolve rate and a
binary metric, a 6-instance minibatch is 0-0 roughly 75% of the time. Accepted
candidates scattered around the base score with no upward trend
(`10, 11, 12, 9, 11, 13, 12, 10, 11` on val, base 11). One candidate that went
0/6 -> 3/6 on its minibatch scored *below* the base program on the full val set.
GEPA was hill-climbing on variance.

**B. Both candidates got worse from val to test** (21.7% -> 14-15%). The
selection overfit the 60-instance val set rather than the task distribution.

The taxonomy machinery itself was sound: 34 codes certified at kappa 0.908,
pruned to 22 by observed support, per-subagent judging on the parent minibatch,
91% of judge evidence quotes verifiable verbatim in the traces, judging only 14%
of budget. It produced diagnoses the harness cannot give (retrieval gaps,
instruction non-compliance, output truncation). It had nothing to select on.

**Conclusion:** the bottleneck is benchmark economics, not the taxonomy.

---

## 2. What a benchmark needs, for this specific claim

The claim under test is *taxonomy-conditioned reflection beats plain gold
feedback*. That requires two properties that pull in opposite directions:

1. **A wide gold-to-diagnosis gap.** The taxonomy only earns its keep where gold
   tells you *that* you failed but not *why*. Long multi-step trajectories with
   terminal outcomes maximise this.
2. **Enough signal to select on.** High base rate and/or partial credit, so a
   minibatch comparison is not a coin flip.

SWE-Bench has (1) in abundance and fails (2). Benchmarks that are cheap and
dense usually fail (1).

**Explicitly ruled out:**

- **IFBench** — gold *is* a per-constraint checklist ("did not use exactly 5
  bullets"). The diagnosis is already in the baseline feedback, so a taxonomy
  can only paraphrase it. Most likely of any option to produce a null result by
  construction.
- **AIME / MATH / LiveBench-Math** — single answer, no intermediate structure to
  attribute failure to. Base rate 16.7% on AIME, i.e. the SWE-Bench noise
  problem without the trace richness.
- **Terminal-Bench** — two independent objections. Only **89 tasks** total, so
  train/val/test splits leave too few val instances for stable selection. And
  GEPA's shipped adapter optimises a **single** `instruction_prompt` for one
  monolithic Terminus agent, so there are no sub-agents to attribute failures
  between and per-subagent judging degenerates to whole-trajectory judging.
- **tau-bench** — no public GEPA result and no adapter, in the repo or the
  paper. Interesting for FailureRank's cross-seed axis; zero comparability here.

---

## 3. Public GEPA results and their initial programs

Verified against the pinned `gepa` v0.1.4 clone and the paper's Appendix L
(which prints the base prompt of every module, per benchmark).

| Benchmark | GEPA result | Initial program published | Modules |
|---|---|---|---|
| **HotpotQA** | paper + repo | yes, 4 prompts | `summarize1` -> `create_query_hop2` -> `summarize2` -> `final_answer` |
| **HoVer** | paper | yes, 4 prompts | `summarize1` -> `create_query_hop2` -> `summarize2` -> `create_query_hop3` |
| **IFBench** | paper | yes, 2 prompts | `generate_response_module` -> `ensure_correct_response_module` |
| **PUPA** | paper | yes, 1 prompt | `craft_redacted_request` |
| **Terminal-Bench** | repo adapter + trainer | yes, 1 prompt | `instruction_prompt` (Terminus agent) |
| **AppWorld** | third-party only (RLM-GEPA) | no | n/a |
| **tau-bench** | none | no | n/a |

Baselines reported across the GEPA paper and the Cisco/FAPO post, for signal
density: PUPA 73.6%, LiveBench-Math 51.0%, HotpotQA 50.9%, HoVer 35.9%,
IFBench 35.7%, AIME 16.7%. SWE-Bench Verified, ours: 18.3%.

---

## 4. HoVer and HotpotQA are near-siblings

Three of four module names are identical. Both are multi-hop Wikipedia
retrieval pipelines with the same alternating summarize/reformulate skeleton;
HoVer goes one hop deeper and ends in claim verification instead of answer
extraction. Same corpus, same failure surface (missed first-hop retrieval,
broken hop chaining, bad final extraction), same cost profile.

Running both is a **replication, not a generalization**. For a claim about
helping across different kinds of failure, that is a weak second data point.

---

## 5. Recommendation

**Primary: HoVer.** Deepest hop chain of the published programs (four modules,
three of them query/summary construction), a documented 4-prompt initial
program to seed from, a published baseline to compare against, rollouts roughly
a fifth the cost of SWE-Bench, no Docker, and we already own HoVer data
(12 candidates x 300 tasks with repeats 0-4).

**Contrast arm: SWE-Bench Verified — keep it.** It is the genuinely different
second benchmark and it is already built.

| | HoVer | SWE-Bench Verified |
|---|---|---|
| failure surface | retrieval misses, hop chaining | patch application, wrong file, truncation, logic |
| program | 4-module RAG chain | 2-module solver -> refiner |
| metric | recall / partial credit | binary resolve |
| base rate | ~36% | ~18% |

**Do not run HotpotQA as the second benchmark.** If a third is wanted later,
AppWorld is the honest candidate (tool-use trajectories, state-based partial
credit, a failure surface unlike either) at the cost of building the program
ourselves.

---

## 6. Implementation

Reuse everything; only the benchmark adapter changes. The following already
exist and are benchmark-agnostic or nearly so:

| Component | File | Change needed |
|---|---|---|
| dollar-budget stopper | `src/gepa_taxonomy/cost.py` | none |
| rollout cache | `src/gepa_taxonomy/rollout_cache.py` | none |
| seed-eval cache | `src/gepa_taxonomy/seed_cache.py` | none |
| taxonomy judge | `src/gepa_taxonomy/taxonomy_judge.py` | role list per benchmark |
| AdaMAST trace emitter | `src/gepa_taxonomy/adamast_trace.py` | none |
| watchdog | `scripts/seed_watchdog.sh` | none |
| iteration viewer | `scripts/iterations.py` | none |
| test evaluation | `scripts/eval_test.py` | grader swap |
| gold-blindness audit | `src/gepa_taxonomy/tasks.py` | gold fields per benchmark |

### Steps

1. **Splits.** Build seeded train/val/test manifests from HoVer, mirroring
   `scripts/build_splits.py`. Val should be large enough that selection is not
   noise-dominated: at ~36% base rate, val 100-150 is affordable here because
   rollouts are cheap.
2. **Program.** Port the paper's 4-module pipeline as the initial candidate:
   `summarize1`, `create_query_hop2`, `summarize2`, `create_query_hop3`, with
   the Appendix L base prompts as the seed. Keep the same `Rollout` /
   `to_trace()` shape so the judge and trace emitter work unchanged.
3. **Grader.** Replace `LocalDockerGrader` with HoVer's metric (supporting-fact
   recall / label accuracy). No containers, so `eval_test.py`'s instance-major
   image logic becomes a no-op path.
4. **Minibatch.** Set `--minibatch-size 15-20`. This is the single change most
   likely to fix the selection noise, and it is affordable only because HoVer
   rollouts are cheap. Measure first with `scripts/calibrate_rollout.py`.
5. **Base-val + taxonomy.** Run the base candidate on val once (shared starting
   state, and the taxonomy-generation trace source), then `adamast generate`
   over those traces, then prune to codes with observed support.
6. **Arms.** Baseline and taxonomy at equal dollars, same seed, same splits,
   same starting candidate. The treatment arm differs by exactly one key in the
   reflective dataset (`failure_modes`).
7. **Test.** `eval_test.py` with both frozen best candidates, paired McNemar on
   the discordant pairs.

### Open decisions for Andrei

- Per-seed budget on HoVer (cheaper rollouts mean $100 buys far more).
- Val size (100-150 suggested; drives selection stability).
- Whether to re-run a SWE-Bench baseline seed under current code, so the
  contrast arm is internally comparable.
- Judge model for the treatment arm (Sonnet 4.6 currently; Haiku is ~1/5 cost).

### Known issues to fix before the next round

- `scripts/iterations.py` has a `VALSCORE_COMPACT_RE` that has never matched
  anything (dead code).
- A rollout served from the rollout cache cannot be judged: the cached trace
  keeps prompt digests, not prompts. Matters only when a candidate first
  evaluated as a child is later selected as a parent.
- ~19% of judge firings restate a signal reflection already has
  (patch-does-not-apply codes when `applies_cleanly` is already false). A
  one-line filter removes them with no information loss.

---

## 7. Sources

- GEPA paper (ICLR 2026 oral): https://arxiv.org/abs/2507.19457 — Appendix L
  prints every module's base prompt per benchmark.
- GEPA repo (pinned v0.1.4): https://github.com/gepa-ai/gepa
- FAPO / Cisco Foundation AI:
  https://cisco-foundation-ai.github.io/blogs/fully-automated-prompt-optimization/
- Arize, GEPA vs Prompt Learning:
  https://arize.com/blog/gepa-vs-prompt-learning-benchmarking-different-prompt-optimization-approaches/
- ACE (AppWorld, beats GEPA offline): https://arxiv.org/abs/2510.04618
- MAS-PromptBench: https://arxiv.org/abs/2606.23664
- tau-bench: https://arxiv.org/abs/2406.12045
- AppWorld: https://github.com/StonyBrookNLP/appworld
