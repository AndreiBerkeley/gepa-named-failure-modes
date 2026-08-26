# GEPA × Failure Taxonomy — Clean Experiment Workspace

> **Location note (2026-08-10).** This file now lives *inside* the experiments
> repo (`AndreiBerkeley/GEPA-Post`), which is the git root. Paths written as
> `experiments/...` below refer to this directory itself. The pinned `gepa`
> v0.1.4 clone is a sibling checkout and is NOT in this repo -- see PLAN.md.

## Mission

Build, from zero, a clean and fully reproducible pipeline for
**taxonomy-conditioned GEPA**:

1. Run plain baseline GEPA (3 independent seeds) on a benchmark.
2. Collect execution traces on a held-out generation set.
3. Generate a failure-mode taxonomy from those traces with the AdaMAST
   public pipeline (inter-annotator rounds included).
4. Run GEPA with the taxonomy integrated into its refinement prompt
   (integration design TBD — do not design it unilaterally).

The results feed a blog post. The reusable components become a PR to the
**official `gepa` repo** (verified: `github.com/gepa-ai/gepa`, pinned at
v0.1.4). A second benchmark will be chosen later.

**Benchmark — SUPERSEDED 2026-08-07.** Originally SWE-Bench full split. Now
**SWE-Bench Verified** (`SWE-bench/SWE-bench_Verified`, 500 instances, all
covered by prebuilt x86_64 eval images, and a strict subset of the full test
split). Two changes follow:

* **Three subsets, not four.** val 60 / test 300 / train 140. There is no
  separate generation set — the base candidate's val evaluation doubles as the
  taxonomy-generation trace source, so test absorbs what generation would have
  taken.
* **val is deliberately hard-weighted**, using Verified's own `difficulty`
  labels: 15 from the hard pool (>1 hour) + 45 at random from the rest. val is
  25% hard against a 9% base rate; the share is capped at 15 so test retains
  21 hard instances (7%) -- test representativeness wins, since that is what
  the headline comparison is measured on (D027).

Phase 3 ("trace harvest on the generation set") is therefore folded into the
base-candidate val evaluation; its traces must be generation-grade.

Prior pilots on HoVer and IfBench live in `../GEPA_Experiments`. That folder
is now a **sandbox for reference only**: you may read it to understand the
solver→refiner program shape used on IfBench, but do not import or depend on
code from it. Everything here is reimplemented cleanly.

## Hard rules

1. **Never launch billed runs.** Anything that spends API tokens (GEPA
   optimization, LLM calls, reflection, evaluation with LLM components) is
   prepared, never executed: write the code, print the exact command in its
   own ```bash block (one command per block), state estimated cost, and let
   Andrei start it. Free/offline checks are fine to run directly: unit
   tests, `--help`, dry runs, schema validation, dataset downloads, Docker
   builds, file audits.
2. **Baseline purity.** The 3 baseline seed runs use the latest released
   GEPA, unmodified — no behavioral additions of any kind. The only thing we
   add is a **dollar-budget stop criterion**, and it must be strictly
   behavior-neutral: it observes spend and decides when to stop, and touches
   nothing else (no influence on candidate selection, reflection, sampling,
   scheduling). First verify whether the latest gepa release already
   supports cost-based budgets before building anything.
3. **Swappable-stage architecture.** Every stage boundary is a plain,
   documented artifact: program definition, split manifests, trace bundles,
   taxonomy file. Each stage must be runnable standalone. A user who brings
   their own taxonomy must be able to skip stages 1–3 entirely and run
   taxonomy-conditioned GEPA directly. Design for this from the first line.
4. **PR-grade code.** Reusable components (budget stop, taxonomy-conditioned
   refinement, benchmark adapter) follow the gepa repo's conventions and
   adapter patterns, with tests. Experiment orchestration (configs, split
   manifests, run scripts, results) lives separately from PR-bound code.
5. **Deferred decisions — surface, don't decide.** The following are
   Andrei's calls; propose options with trade-offs and wait: budget amount
   per seed, solver/refiner model choice, split sizes, trace-source rule for
   taxonomy generation (leading idea: base candidate only on the generation
   set), taxonomy-integration mechanism, second benchmark.

## Suggested layout

```
GEPA/
  CLAUDE.md          # this file
  gepa/              # clone of official gepa (PR branches live here)
  experiments/       # clean experiment workspace, its own git repo:
                     #   configs, split manifests, run scripts, results, logs
```

## Phase plan

- **Phase 0 — Scaffold (now).** Verify + clone the official gepa repo; pin
  and record the exact release/commit used for baselines. Set up
  `experiments/` as a uv-managed git repo with the layout above.
- **Phase 1 — SWE-Bench foundation (now).**
  - Acquire SWE-Bench full; build deterministic, seeded
    train/eval/test/generation splits written as committed manifest files.
    Propose split sizes for approval first.
  - Feasibility check, early: SWE-Bench evaluation harness (Docker) on this
    arm64 Mac — can it run locally, or do we need x86 emulation/cloud?
    Report findings before building around it.
  - Initial candidate program: two-stage solver→refiner pipeline (IfBench
    pilot shape, adapted to SWE-Bench patch generation). Propose the design
    before implementing.
  - Verify latest gepa's budget controls; then build the dollar-budget stop
    component (isolated, tested, behavior-neutral).
- **Phase 2 — Baseline runs.** 3 seeds, same initial candidate, same dollar
  budget. Andrei launches; you prepare commands and monitoring.
- **Phase 3 — Trace harvest** on the generation set (rule TBD).
- **Phase 4 — Taxonomy generation** via the AdaMAST public pipeline (see
  its GitHub repo for the full 8-step generation + inter-annotator
  procedure). One taxonomy shared across all seeds.
- **Phase 5 — Taxonomy-conditioned GEPA** (design discussed with Andrei
  first) + comparison runs against Phase 2.
- **Phase 6 — Blog post + PR.**

## Working style

- Keep `PROGRESS.md` in this folder current: what's done, what's blocked,
  what needs Andrei (decisions or run launches). It is the coordination
  surface other sessions read.
- Log every decision made and every deferred decision resolved in
  `DECISIONS.md` with a one-line rationale.
- Small, clean commits from the start; the history is part of the artifact.
