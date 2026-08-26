# Taxonomy-conditioned GEPA

Companion repository for the study
[Named Failure Modes: Diagnosing the Program, Not Just the Output](https://andreiberkeley.github.io/gepa-named-failure-modes/blog/2026/08/18/named-failure-modes/):
the failure-taxonomy components, the experiment pipeline, split manifests, run
scripts, frozen taxonomies, and offline tests. The optimizer-side hook itself
lives in [gepa](https://github.com/gepa-ai/gepa) as the
`reflective_dataset_enricher` argument on `gepa.optimize`.

## Layout

```
manifests/          committed split manifests, a stage boundary artifact
scripts/            standalone stage entrypoints
src/failure_taxonomy/  taxonomy schema, judge, and enricher
src/gepa_taxonomy/  benchmark programs, adapters, and run machinery
tests/              free, offline tests
results/            per-run outputs (raw/ is gitignored) and frozen taxonomies
docs/findings/      investigation writeups
patches/            the enricher hook as a patch, for pre-release gepa
```

## Setup

One command prepares everything, including the sibling checkouts described
below, and finishes with the free offline checks. It is idempotent, so rerun
it after any failure:

```bash
./scripts/bootstrap.sh
```

The pieces it sets up, for reference or manual installation:

```bash
uv sync
```

Taxonomy-conditioned runs use GEPA's optimizer-side
`reflective_dataset_enricher` hook. The hook runs after an adapter creates its
normal reflection records and before GEPA requests a revision. It does not
wrap, replace, or proxy the adapter, so DSPy, LangChain, OpenAI, and custom
adapters keep their normal behavior.

The SWE-Bench harness is optional (it pulls docker + modal) and only needed where
evaluation actually runs:

```bash
uv sync --extra swebench
```

## Taxonomy generation requires AdaMAST

Stage 4 (generating a taxonomy from traces) shells out to the public
[AdaMAST](https://github.com/multi-agent-systems-failure-taxonomy/AdaMAST)
pipeline, running in its own sibling checkout with its own interpreter. It is
deliberately not a dependency of this project: it pins its own `openai` and
`pydantic` floors, and installing it here would re-resolve the environment the
baseline seeds run from. One-time setup:

```bash
git clone --branch agent/baseline-taxonomy-generation https://github.com/multi-agent-systems-failure-taxonomy/AdaMAST.git ../adamast-public
cd ../adamast-public && uv venv --python 3.12 && uv pip install -e ".[bedrock]"
```

The `[bedrock]` extra is required; without it AdaMAST imports cleanly and then
fails on its first provider call. Runs that bring an existing taxonomy via
`--taxonomy` never touch AdaMAST.

## Stage boundaries

Every stage is standalone and communicates through a documented artifact, so a user
who brings their own taxonomy can skip stages 1–3:

| Stage | Consumes | Produces |
|---|---|---|
| 1. Splits | benchmark dataset | `manifests/<benchmark>/*.json` |
| 2. Baseline GEPA | manifests, program def | optimized candidate + run state |
| 3. Trace harvest | candidate, generation manifest | trace bundle (JSONL) |
| 4. Taxonomy | trace bundle | `taxonomy.json` |
| 5. Conditioned GEPA | manifests, program def, **`taxonomy.json`** | optimized candidate |

Stage 5 depends on the taxonomy *file*, not on stages 1–4 having run.

## One-command taxonomy run

The stage boundaries remain available, but the normal treatment workflow can be
started with one command:

```bash
uv run gepa-taxonomy hotpotqa --seed 1 --budget 60
```

That command reuses existing artifacts when present. Otherwise it evaluates the
base candidate on the study's held-out validation set, saves the captured
traces, generates and freezes the taxonomy, and then starts GEPA. Taxonomy
generation finishes before optimization begins. The runtime judge and GEPA's
reflector use the same `--reflection-model` value.

Repeat `--seed` to prepare once and launch several runs:

```bash
uv run gepa-taxonomy hotpotqa --seed 1 --seed 2 --seed 3 --budget 60
```

Bring an existing taxonomy to skip preparation entirely:

```bash
uv run gepa-taxonomy hotpotqa --seed 1 --budget 60 --taxonomy path/to/taxonomy.json
```

Model ids are litellm ids. A bare id is pinned to Bedrock (the study
configuration); an explicit provider prefix routes there deliberately, and
taxonomy generation maps the prefix to the matching AdaMAST provider
automatically (`gemini/` becomes `google`). A complete from-zero preparation
on Gemini, harvest plus taxonomy generation, is therefore one command with two
model settings:

```bash
uv run gepa-taxonomy ifbench --seed 1 --budget 2 --solver-model gemini/gemini-3-flash-preview --reflection-model gemini/gemini-3-flash-preview --prepare-only
```

Use `--dry-run` to print all phases without making calls or writing artifacts.
The supported benchmarks are `hotpotqa`, `ifbench`, `hover`, `livebench-math`,
and `appworld`.

## Tests

The suite is free and offline:

```bash
uv run pytest
```

Three `test_patch_gate.py` tests additionally need the optional SWE-Bench
harness (`uv sync --extra swebench`).

## Until the hook is in a gepa release

The pinned `gepa==0.1.4` from PyPI predates the hook. Until a release includes
it, apply the committed patch to a v0.1.4 checkout and point runs and tests at
it:

```bash
git clone --branch v0.1.4 https://github.com/gepa-ai/gepa.git ../gepa-taxonomy-hook
git -C ../gepa-taxonomy-hook apply "$PWD/patches/gepa-reflective-dataset-enricher.patch"
```

Add `--gepa-root ../gepa-taxonomy-hook` to `gepa-taxonomy` commands, and run
the tests with `PYTHONPATH=../gepa-taxonomy-hook/src uv run pytest`. This
section disappears once the released gepa carries the hook.
