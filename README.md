# Taxonomy-conditioned GEPA

Companion repository for the study
[Learned Error Diagnosis for GEPA's Reflection](https://andreiberkeley.github.io/gepa-named-failure-modes/blog/2026/08/18/named-failure-modes/).
It contains the method, its pipeline, and one runnable benchmark demo. The
optimizer-side hook itself lives in [gepa](https://github.com/gepa-ai/gepa) as
the `reflective_dataset_enricher` argument on `gepa.optimize`.

## Layout

```
src/failure_taxonomy/   the method: taxonomy schema, outcome-blind judge,
                        reflective-dataset enricher, evidence-based reduction
src/gepa_taxonomy/      pipeline, model routing, cost metering, AdaMAST
                        transport, observability, and the demo's program code
scripts/                the generic pieces: bootstrap plus the three
                        benchmark-agnostic taxonomy stages
demo/                   the IFBench worked example: split builder, harvest,
                        run, test eval, grader vendoring, and its committed
                        train/val/test manifests
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
pipeline into its own sibling environment (`../adamast-public`, with all
provider extras) for taxonomy generation.

## Run the demo

`demo/` ships everything needed: a 10/10/10 IFBench split (small enough that
the whole pipeline costs cents) and `demo/taxonomy.json`, an evidence-reduced
taxonomy generated with `gemini/gemini-3.5-flash` from 300 validation traces of
the full-size IFBench split and reduced from 20 to 13 failure modes; the demo
split's validation traces are a subset of that corpus. Two ways to run it:

**1. From zero** -- generate the taxonomy yourself, then optimize with it:

```bash
uv run gepa-taxonomy ifbench --seed 1 --budget 2 --gepa-root ../gepa-taxonomy-hook
```

One command runs the whole method: harvest the base program's traces over the
demo validation split, generate a taxonomy with stock AdaMAST, judge it back
over the same traces with AdaMAST's own judge to measure each code's support, reduce it to the codes
the evidence supports (`--min-support`, default 2; `--max-codes`, default 25,
as a safety net; every code accounted for in `reduction_report.json`), and
launch GEPA with the frozen taxonomy: an outcome-blind judge diagnoses each
rollout and the enricher adds the named failure modes to reflection.

**2. With the provided taxonomy** -- skip generation entirely:

```bash
uv run gepa-taxonomy ifbench --seed 1 --budget 1 --taxonomy demo/taxonomy.json --gepa-root ../gepa-taxonomy-hook
```

Harvested traces and a generated taxonomy are reused on re-runs; to continue
an existing run directory, pass `--run-arg=--resume`. Model ids are plain litellm ids and default
to OpenAI (`gpt-5-mini`); pass any litellm id to use another provider --
`gemini/...`, `anthropic/...`, `bedrock/...` -- and taxonomy generation maps
the prefix to the matching AdaMAST provider automatically (`gemini/` becomes
`google`); for example
`--solver-model gemini/gemini-2.5-flash-lite --reflection-model gemini/gemini-3.5-flash`
puts a cheap model on rollouts and a stronger one on generation, reflection,
and the judge. The dollar budget prices every call from litellm's own price
table; for a model that table does not know yet, pass
`--price MODEL=IN,OUT` in USD per million tokens (for example
`--price gemini/gemini-3.5-flash=1.50,9.00`) or set `GEPA_TAXONOMY_PRICES`,
and the run refuses to start rather than metering an unpriced model as free.
Use `--dry-run` to print every phase without spending, and `--prepare-only`
to stop once the taxonomy is frozen.

Provider credentials are read by litellm from the usual environment variables
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, whichever applies).
Taxonomy generation and corpus judging run in the sibling AdaMAST environment
that bootstrap creates; set `ADAMAST_PYTHON` to use a different interpreter.

## Using the method on your own task

If you already run gepa, nothing about your run changes: the method is your
existing `gepa.optimize` call plus one argument,
`reflective_dataset_enricher`. The steps below exist only to produce that
argument's input, a frozen `taxonomy.json`, once. Your program, adapter, and
data stay in your own codebase:

**1. Harvest traces from your own program.** Evaluate it once over a held-out
split with `capture_traces=True` and write the bundle:

```python
from failure_taxonomy import harvest_traces, write_generation_traces

batch = my_adapter.evaluate(heldout, seed_candidate, capture_traces=True)
traces = harvest_traces(batch, instance_ids=[x.id for x in heldout])
write_generation_traces(traces, "traces.jsonl")
```

**2. Prepare the taxonomy with the generic stages.** These consume a trace
file; no benchmark concept is involved:

```bash
uv run python scripts/generate_taxonomy.py --traces traces.jsonl --out taxonomy_dir
uv run python scripts/judge_corpus.py --taxonomy taxonomy_dir/taxonomy.json --traces traces.jsonl --out taxonomy_dir/judgements.jsonl
uv run python scripts/reduce_taxonomy.py --taxonomy taxonomy_dir/taxonomy.json --judgements taxonomy_dir/judgements.jsonl
```

**3. Optimize with the taxonomy in your own `gepa.optimize` call.** Your
adapter is unchanged:

```python
from failure_taxonomy import LLMFailureJudge, TaxonomyFeedbackEnricher, load_taxonomy

taxonomy = load_taxonomy("taxonomy_dir/taxonomy.json")
enricher = TaxonomyFeedbackEnricher(judge=LLMFailureJudge(taxonomy=taxonomy, lm=reflection_lm))

result = gepa.optimize(..., adapter=my_adapter, reflective_dataset_enricher=enricher)
```

The `demo/` directory is those same three steps pre-wired for IFBench so the
whole flow is runnable before you write a line of your own.

## Observability

Every run writes its own audit trail into the run directory:

* `reflection_datasets.jsonl` — one record per reflection round, exactly the
  dataset reflection consumed, including each example's injected
  `failure_modes` (disable with `--run-arg=--no-log-reflection-datasets`);
* `judge_cache.jsonl` — every judge diagnosis: code, name, evidence span,
  component;
* `spend.solver.json`, `spend.reflection.json`, `spend.judge.json` — live
  per-stream spend, flushed at exit, plus a once-a-minute heartbeat;
* `reduction_report.json`, written next to the taxonomy rather than in the run
  directory — every generated code as retained, ungrounded, or over cap.

The taxonomy also records the trace ids it was judged over, and at run time
the enricher leaves a reflection batch undiagnosed if it contains one of them.

## Tests

```bash
uv run pytest
```

The suite makes no model calls. The four tests that exercise the optimizer
hook fail without the patched checkout on the path, so run
`PYTHONPATH=../gepa-taxonomy-hook/src uv run pytest`; one split test reads
`allenai/IF_multi_constraints_upto5` from the Hugging Face cache and needs it
downloaded once.

## Until the hook is in a gepa release

The pinned `gepa==0.1.4` predates the hook, so bootstrap applies
`patches/gepa-reflective-dataset-enricher.patch` to the sibling checkout and
runs point at it with `--gepa-root ../gepa-taxonomy-hook`. This section and
the patch disappear once a released gepa carries the hook.
