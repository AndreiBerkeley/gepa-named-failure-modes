# Learned error diagnosis for GEPA

Companion repository for the post
[Learned Error Diagnosis for GEPA's Reflection](https://andreiberkeley.github.io/gepa-named-failure-modes/blog/2026/08/18/named-failure-modes/).
It contains the method, the pipeline that prepares its input, and one runnable
demo. The hook that plugs the method into the optimizer lives in
[gepa](https://github.com/gepa-ai/gepa) as the `reflective_dataset_enricher`
argument on `gepa.optimize`.

## What the method does

Before optimization, the base program is run once over a held-out split and
its traces are handed to [AdaMAST](https://github.com/multi-agent-systems-failure-taxonomy/AdaMAST),
which learns a **failure taxonomy** for that program: a fixed, named list of
the ways it tends to fail. During optimization, an LLM **judge** reads every
trace GEPA is about to reflect on, reports the failure modes it finds with the
step responsible and the supporting evidence, and never sees the score, the
gold answer, or the evaluator's feedback. The **enricher** attaches those
findings to the reflection records the adapter already produced. Everything
else in GEPA is unchanged.

## Layout

```
src/failure_taxonomy/   the method: taxonomy schema, outcome-blind judge,
                        enricher, and evidence-based reduction
src/gepa_taxonomy/      the pipeline around it: model routing, cost metering,
                        the AdaMAST bridge, observability, and the demo program
scripts/                bootstrap plus the three taxonomy stages, which work on
                        any trace file
demo/                   the IFBench example: split builder, harvest, run, test
                        evaluation, grader vendoring, and the committed
                        train/val/test manifests
patches/                the gepa hook and the AdaMAST parallel-annotator patch,
                        applied by bootstrap until each lands upstream
tests/                  offline suite, no model calls
results/                created at run time, never committed
```

## Setup

One command prepares everything and ends with the offline checks. It is safe
to rerun:

```bash
./scripts/bootstrap.sh
```

It installs the project (`uv sync`), clones gepa v0.1.4 into a sibling
directory and applies the hook patch, and clones AdaMAST into its own sibling
environment (`../adamast-public`, with all provider extras).

## Run the demo

`demo/` ships a 10/10/10 IFBench split, small enough that the whole pipeline
costs cents, and `demo/taxonomy.json`, a taxonomy you can use without
generating one. That file was generated with `gemini/gemini-3.5-flash` from
the 300 validation traces of the full-size IFBench split and reduced from 20
to 13 failure modes; the demo split's validation traces are part of that
corpus. There are two ways to run the demo.

**From zero**, generating the taxonomy first:

```bash
uv run gepa-taxonomy ifbench --seed 1 --budget 2 --gepa-root ../gepa-taxonomy-hook
```

This runs five phases in order:

1. harvest the base program's traces over the demo validation split;
2. generate a taxonomy from them with stock AdaMAST;
3. judge that taxonomy back over the same traces with AdaMAST's own judge, to
   measure how many traces support each failure mode;
4. reduce it to the failure modes with enough support (`--min-support`,
   default 2; `--max-codes`, default 25, as a cap), writing every decision to
   `reduction_report.json`;
5. launch GEPA with the frozen taxonomy, the outcome-blind judge, and the
   enricher.

**With the provided taxonomy**, skipping generation:

```bash
uv run gepa-taxonomy ifbench --seed 1 --budget 1 --taxonomy demo/taxonomy.json --gepa-root ../gepa-taxonomy-hook
```

Harvested traces and a generated taxonomy are reused on re-runs. To continue
an existing run directory, pass `--run-arg=--resume`. `--dry-run` prints every
phase without spending anything, and `--prepare-only` stops once the taxonomy
is frozen.

**Models.** Model ids are litellm ids and default to `gpt-5-mini` everywhere.
Any litellm id works (`gemini/...`, `anthropic/...`, `bedrock/...`), and
taxonomy generation maps the prefix to the matching AdaMAST provider
(`gemini/` becomes `google`). For example,
`--solver-model gemini/gemini-2.5-flash-lite --reflection-model gemini/gemini-3.5-flash`
puts a cheap model on rollouts and a stronger one on generation, reflection,
and the judge. Credentials are read by litellm from the usual variables
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`). Generation and
corpus judging run in the sibling AdaMAST environment; set `ADAMAST_PYTHON`
to use a different interpreter.

**Prices.** The dollar budget prices every call from litellm's own price
table. For a model that table does not know yet, pass `--price MODEL=IN,OUT`
in USD per million tokens (for example `--price gemini/gemini-3.5-flash=1.50,9.00`)
or set `GEPA_TAXONOMY_PRICES`. A run with an unpriced model refuses to start
rather than metering it as free.

## Using the method on your own task

If you already run gepa, your run changes by one argument:
`reflective_dataset_enricher`. The steps below produce that argument's input,
a frozen `taxonomy.json`, once. Your program, adapter, and data stay where
they are.

**1. Harvest traces from your program.** Evaluate it once over a held-out
split with `capture_traces=True` and write the trace file:

```python
from failure_taxonomy import harvest_traces, write_generation_traces

batch = my_adapter.evaluate(heldout, seed_candidate, capture_traces=True)
traces = harvest_traces(batch, instance_ids=[x.id for x in heldout])
write_generation_traces(traces, "traces.jsonl")
```

**2. Generate, judge, and reduce the taxonomy.** These stages take a trace
file and know nothing about benchmarks:

```bash
uv run python scripts/generate_taxonomy.py --traces traces.jsonl --out taxonomy_dir
uv run python scripts/judge_corpus.py --taxonomy taxonomy_dir/taxonomy.json --traces traces.jsonl --out taxonomy_dir/judgements.jsonl
uv run python scripts/reduce_taxonomy.py --taxonomy taxonomy_dir/taxonomy.json --judgements taxonomy_dir/judgements.jsonl
```

**3. Optimize with the taxonomy.** Your adapter is unchanged:

```python
from failure_taxonomy import LLMFailureJudge, TaxonomyFeedbackEnricher, load_taxonomy

taxonomy = load_taxonomy("taxonomy_dir/taxonomy.json")
enricher = TaxonomyFeedbackEnricher(judge=LLMFailureJudge(taxonomy=taxonomy, lm=reflection_lm))

result = gepa.optimize(..., adapter=my_adapter, reflective_dataset_enricher=enricher)
```

The `demo/` directory is these three steps wired up for IFBench, so the whole
flow can be run before you write anything of your own.

## Observability

Every run writes its own audit trail into the run directory:

* `reflection_datasets.jsonl`: one record per reflection round, exactly the
  dataset reflection consumed, including each example's `failure_modes`
  (disable with `--run-arg=--no-log-reflection-datasets`);
* `judge_cache.jsonl`: every finding, with its failure mode, step, and
  evidence;
* `spend.solver.json`, `spend.reflection.json`, `spend.judge.json`: live spend
  per model stream, flushed at exit, plus a once-a-minute heartbeat on stdout.

Taxonomy preparation writes `reduction_report.json` next to the taxonomy, with
every generated failure mode marked as retained, unsupported, or over the cap.
The taxonomy also records the ids of the traces it was judged over, and the
enricher leaves a reflection batch undiagnosed if it contains one of them, so
a taxonomy is never applied to the traces it was built from.

## Tests

```bash
uv run pytest
```

The suite makes no model calls. The four tests that exercise the optimizer
hook fail without the patched checkout on the path, so run
`PYTHONPATH=../gepa-taxonomy-hook/src uv run pytest`. One split test reads
`allenai/IF_multi_constraints_upto5` from the Hugging Face cache and needs it
downloaded once.

## Until the hook is in a gepa release

The pinned `gepa==0.1.4` predates the hook, so bootstrap applies
`patches/gepa-reflective-dataset-enricher.patch` to the sibling checkout, and
runs point at it with `--gepa-root ../gepa-taxonomy-hook`. This section and
the patch disappear once a released gepa carries the hook.
