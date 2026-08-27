# Taxonomy-conditioned GEPA

Companion repository for the study
[Named Failure Modes: Diagnosing the Program, Not Just the Output](https://andreiberkeley.github.io/gepa-named-failure-modes/blog/2026/08/18/named-failure-modes/).
It contains the method, its pipeline, and one runnable benchmark demo. The
optimizer-side hook itself lives in [gepa](https://github.com/gepa-ai/gepa) as
the `reflective_dataset_enricher` argument on `gepa.optimize`.

## Layout

```
src/failure_taxonomy/   the method: taxonomy schema, outcome-blind judge,
                        reflective-dataset enricher, evidence-based reduction
src/gepa_taxonomy/      pipeline, model routing, cost metering, AdaMAST
                        transport, observability, and the IFBench demo program
scripts/                bootstrap plus one entry point per pipeline stage
manifests/ifbench/      committed train/val/test split definitions for the demo
patches/                the gepa hook and the AdaMAST parallel-annotator patch,
                        applied by bootstrap until each lands upstream
tests/                  free, offline suite
results/                created at run time; never committed
```

## Setup

One idempotent command prepares everything, including the sibling checkouts,
and finishes with the free offline checks:

```bash
./scripts/bootstrap.sh
```

It installs the project (`uv sync`), clones gepa v0.1.4 and applies the hook
patch, and clones the public
[AdaMAST](https://github.com/multi-agent-systems-failure-taxonomy/AdaMAST)
pipeline into its own sibling environment (`../adamast-public`, with the
`[bedrock,google]` extras) for taxonomy generation.

## Run the demo

```bash
uv run gepa-taxonomy ifbench --seed 1 --budget 5 --gepa-root ../gepa-taxonomy-hook
```

One command runs the whole method:

1. **Harvest** — evaluate the base program over the validation split and keep
   its component-level traces.
2. **Generate** — run stock AdaMAST over the traces to draft a failure
   taxonomy through iterative agreement rounds.
3. **Judge the corpus** — apply the drafted taxonomy back over the same
   traces to measure each code's support (distinct traces citing it).
4. **Reduce** — keep the codes the evidence supports: drop below
   `--min-support` (default 2), apply `--max-codes` (default 25) only as a
   safety net, and account for every code in `reduction_report.json`. The
   full draft survives as `taxonomy.full.json`, so re-capping is offline.
5. **Optimize** — launch GEPA with the frozen taxonomy: an outcome-blind
   judge diagnoses each rollout and the enricher adds the named failure
   modes to reflection, alongside the adapter's ordinary feedback.

Artifacts are reused on re-runs. Bring your own taxonomy to skip steps 1-4:

```bash
uv run gepa-taxonomy ifbench --seed 1 --budget 5 --taxonomy path/to/taxonomy.json --gepa-root ../gepa-taxonomy-hook
```

Model ids are litellm ids: a bare id routes to Bedrock, an explicit provider
prefix routes there instead, and taxonomy generation maps the prefix to the
matching AdaMAST provider automatically (`gemini/` becomes `google`). Example
with a cheap solver and a stronger model for generation, reflection, and the
judge:

```bash
uv run gepa-taxonomy ifbench --seed 1 --budget 5 --gepa-root ../gepa-taxonomy-hook --solver-model gemini/gemini-2.5-flash-lite --reflection-model gemini/gemini-3.5-flash
```

Use `--dry-run` to print every phase without spending, and `--prepare-only`
to stop once the taxonomy is frozen.

## Using the method on your own task

The demo is IFBench end to end, but the method does not depend on it. With
your own GEPA setup (any adapter, any task), three steps apply it:

1. **Harvest traces.** Run your program over a held-out split and write one
   JSON line per rollout with `problem_id`, `task`, `raw_trajectory`, and
   `metadata`. `failure_taxonomy.harvest_traces` and
   `write_generation_traces` produce exactly this from GEPA trajectories.
2. **Generate and reduce.** `scripts/generate_taxonomy.py`,
   `scripts/judge_corpus.py`, and `scripts/reduce_taxonomy.py` consume a
   trace file, not a benchmark; point them at your bundle and they produce a
   frozen, evidence-reduced `taxonomy.json`.
3. **Optimize.** In your own `gepa.optimize` call, pass
   `reflective_dataset_enricher=TaxonomyFeedbackEnricher(judge=LLMFailureJudge(taxonomy=load_taxonomy("taxonomy.json"), lm=reflection_lm))`.
   Your adapter is unchanged.

The IFBench-named files (`src/gepa_taxonomy/ifbench/`, the build and run
scripts, `manifests/ifbench/`) are a complete worked example of those three
steps wired into the one-command pipeline. Copy them as a template only if
you want the same orchestration for your benchmark: implement your program
and grader in a new `src/gepa_taxonomy/<yours>/` package, write a split
manifest, adapt the two scripts, and add one entry to `BENCHMARKS` in
`pipeline.py`.

## Observability

Every run writes its own audit trail into the run directory:

* `reflection_datasets.jsonl` — one record per reflection round, exactly the
  dataset reflection consumed, including each example's injected
  `failure_modes` (disable with `--no-log-reflection-datasets`);
* `judge_cache.jsonl` — every judge diagnosis: code, name, evidence span,
  component;
* `spend.solver.json`, `spend.reflection.json`, `spend.judge.json` — live
  per-stream spend, flushed at exit, plus a once-a-minute heartbeat;
* `reduction_report.json` — every generated code as retained, ungrounded, or
  over cap.

The taxonomy also records the trace ids it was generated from, and the
enricher refuses at run time to diagnose an instance from that corpus.

## Tests

```bash
uv run pytest
```

The suite is free and offline. Tests that exercise the optimizer hook need
the patched checkout on the path:
`PYTHONPATH=../gepa-taxonomy-hook/src uv run pytest`.

## Until the hook is in a gepa release

The pinned `gepa==0.1.4` predates the hook, so bootstrap applies
`patches/gepa-reflective-dataset-enricher.patch` to the sibling checkout and
runs point at it with `--gepa-root ../gepa-taxonomy-hook`. This section and
the patch disappear once a released gepa carries the hook.
