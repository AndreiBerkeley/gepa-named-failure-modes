# Taxonomy-conditioned reflection for GEPA

GEPA improves a prompt by showing a proposer LM what went wrong. The feedback it
gets is whatever the metric emits — often a score, sometimes a sentence. This
package adds a second, structured signal: **which known failure modes this
rollout exhibits, with verbatim evidence, attributed to the component that
caused them.**

The task adapter is never wrapped. With no enricher configured, GEPA follows its
ordinary baseline path and never calls the taxonomy judge.

## Usage

```python
from failure_taxonomy import LLMFailureJudge, TaxonomyFeedbackEnricher, load_taxonomy

taxonomy = load_taxonomy("taxonomy.json")
judge = LLMFailureJudge(taxonomy=taxonomy, lm=my_reflection_lm)
taxonomy_feedback = TaxonomyFeedbackEnricher(judge=judge)

result = gepa.optimize(
    seed_candidate=seed,
    trainset=train,
    valset=val,
    adapter=my_adapter,
    reflection_lm=my_reflection_lm,
    reflective_dataset_enricher=taxonomy_feedback,
    max_metric_calls=200,
)
```

The same model is used for both calls: first it reviews the traces against the
frozen taxonomy, then it reflects over the task feedback plus that report.

## The taxonomy

Any JSON file whose codes carry an `id` and a `name`:

```json
{"codes": [
  {"id": "A.1", "name": "Output_Truncation",
   "description": "The output ends mid-structure at the token limit."},
  {"id": "B.2", "name": "Ignored_Provided_Context",
   "description": "The component did not use context it was given.",
   "when_to_use": "Only when the context demonstrably contains the answer."}
]}
```

`description` is optional but strongly recommended — it is most of what the
judge selects on. `when_to_use` / `when_not_to_use` are used when present.
Everything else (`category`, `severity`, `applies_to_role`, …) is preserved for
your logs and **never affects routing**.

This is the stage boundary. Bring your own taxonomy — hand-written, MAST,
AdaMAST, anything — and the generation stage is skippable entirely.

## Getting per-component precision

Attribution needs to know which component produced which text. Expose it on
your trajectory:

```python
EvaluationBatch(
    outputs=outputs,
    scores=scores,
    trajectories=[
        {
            "instance_id": "q17",
            "module_calls": [
                {"component": "summarize1", "prompt": ..., "output": ...},
                {"component": "create_query_hop2", "prompt": ..., "output": ...},
            ],
        }
    ],
)
```

**Adapters that don't do this still work.** The trajectory is judged whole, no
component vocabulary is offered, and every occurrence comes back general — which
routes it to every component. Precision degrades; nothing breaks.

This matters more than it looks. Downstream prompts routinely quote upstream
outputs verbatim — a refiner's prompt embeds the solver's candidate. Given flat
text, a judge cannot tell which component *authored* a span from which merely
*received* it. The structure is what makes attribution answerable.

## What reflection receives

One extra key on examples that have a diagnosis:

```python
{
  "instance_id": "q17",
  ...,                       # whatever your adapter already produced
  "failure_modes": [
    {"name": "Ignored_Provided_Context",
     "evidence": "the passage states 1971, the summary says 1968"},
  ],
}
```

Name and evidence only. The code id (`B.2`) is meaningless to a language model
and is withheld; it stays in the cache and logs, where cross-run analysis joins
on it.

## Design notes

**Every instance is judged, not only failures.** The judge is deliberately not
shown scores. If the only traces it ever received were failures, the outcome
would leak back in through selection, and "no failure mode applies" would be an
answer it could never give.

**Occurrences, not codes.** The same code firing three times with three
different quotes is three occurrences, and all three are shown. Collapsing them
would discard exactly the multiplicity that tells reflection how bad a problem
is.

**Nothing in the taxonomy restricts which codes may apply where.** Placement is
decided by observation — the judge attributes each occurrence — not by
declaration. A taxonomy with no role information is fully supported, which is
the common case.

**Fail-soft throughout.** Any judge error yields no occurrences and logs once.
A lost diagnosis must never cost a paid run.

**Positional alignment is checked, not assumed.** Reflective examples are
matched to trajectories by index, which is the contract in practice — GEPA
guarantees trajectories align with outputs and scores, and adapters build
examples by walking that list. If the counts disagree, the batch is left
undiagnosed rather than mis-attributed.

## Generating a taxonomy

Optional; skip it if you have one. `harvest_traces` and
`write_generation_traces` export segmented traces for a generator:

```python
from failure_taxonomy import harvest_traces, trace_report, write_generation_traces

traces = harvest_traces(eval_batch)
print(trace_report(traces))  # check `segmented` and `components` first
write_generation_traces(traces, "traces.jsonl")
```

`trace_report` is worth reading before you spend anything. A generator that has
to recover component structure from trace prose will recover whatever the prose
contains — on one real run, a generator scraped Python identifiers out of source
code embedded in the prompts and reported them as the system's agents, finding
one component where the program had two. Writing the real names into the export
removes the guess.

Outcomes go in `metadata`, never into the trajectory: a generator shown the
outcome writes codes about *being wrong* rather than about observable behaviour,
and those are unjudgeable later, when no outcome is available.

The experiment package also provides `gepa-taxonomy`, a one-command launcher
that performs trace harvest and taxonomy generation before it enters
`gepa.optimize`. These remain separate persisted phases internally, and an
existing `taxonomy.json` skips both.

## API

| | |
|---|---|
| `TaxonomyFeedbackEnricher(judge)` | adds diagnoses immediately before reflective proposal |
| `LLMFailureJudge(taxonomy, lm, *, cache=None)` | default judge, one call per rollout |
| `FailureJudge` | protocol — implement to plug in your own |
| `load_taxonomy(path)` → `Taxonomy` | reads bare lists, `{"codes": …}`, and layered `category_a/b/c` |
| `JudgeCache.open(path)` | write-through JSONL cache, keyed on taxonomy × candidate × trace |
| `harvest_traces` / `write_generation_traces` / `trace_report` | generation-stage export |
