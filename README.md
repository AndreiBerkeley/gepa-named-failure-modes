# experiments — taxonomy-conditioned GEPA

Experiment orchestration for the GEPA × failure-taxonomy study: configs, split
manifests, run scripts, results, logs. Kept separate from PR-bound code per hard
rule 4.

Baseline GEPA is pinned to **v0.1.4** (`8b0ce6c`). See
[`docs/findings/phase0-gepa-pin.md`](docs/findings/phase0-gepa-pin.md).

## Layout

```
configs/            run configs (one per seed / arm)
manifests/          committed split manifests — a stage boundary artifact
scripts/            standalone stage entrypoints
src/gepa_taxonomy/  reusable components (PR-bound code lives upstream in ../gepa)
tests/              free, offline tests
results/            per-run outputs (raw/ is gitignored)
logs/
docs/findings/      investigation writeups
```

## Setup

```bash
uv sync
```

Taxonomy-conditioned runs use a small optimizer-side GEPA hook. Keep that
checkout separate from the unmodified v0.1.4 environment used for baseline
runs:

```bash
git clone --branch v0.1.4 https://github.com/gepa-ai/gepa.git ../gepa-taxonomy-hook
git -C ../gepa-taxonomy-hook apply "$PWD/patches/gepa-reflective-dataset-enricher.patch"
```

The hook runs after an adapter creates its normal reflection records and before
GEPA requests a revision. It does not wrap, replace, or proxy the adapter, so
DSPy, LangChain, OpenAI, and custom adapter hooks keep their normal behavior.

The SWE-Bench harness is optional (it pulls docker + modal) and only needed where
evaluation actually runs:

```bash
uv sync --extra swebench
```

## Stage boundaries

Every stage is standalone and communicates through a documented artifact, so a user
who brings their own taxonomy can skip stages 1–3:

| Stage | Consumes | Produces |
|---|---|---|
| 1. Splits | SWE-Bench dataset | `manifests/swebench_full/*.json` |
| 2. Baseline GEPA | manifests, program def | optimized candidate + run state |
| 3. Trace harvest | candidate, generation manifest | trace bundle (JSONL) |
| 4. Taxonomy | trace bundle | `taxonomy.json` |
| 5. Conditioned GEPA | manifests, program def, **`taxonomy.json`** | optimized candidate |

Stage 5 depends on the taxonomy *file*, not on stages 1–4 having run.

## One-command taxonomy run

The stage boundaries remain available, but the normal treatment workflow can be
started with one command:

```bash
uv run gepa-taxonomy hotpotqa --seed 1 --budget 60 --gepa-root ../gepa-taxonomy-hook
```

That command reuses existing artifacts when present. Otherwise it evaluates the
base candidate on the study's held-out validation set, saves the captured
traces, generates and freezes the taxonomy, and then starts GEPA. Taxonomy
generation finishes before optimization begins. The runtime judge and GEPA's
reflector use the same `--reflection-model` value.

Repeat `--seed` to prepare once and launch several runs:

```bash
uv run gepa-taxonomy hotpotqa --seed 1 --seed 2 --seed 3 --budget 60 --gepa-root ../gepa-taxonomy-hook
```

Bring an existing taxonomy to skip preparation entirely:

```bash
uv run gepa-taxonomy hotpotqa --seed 1 --budget 60 --taxonomy path/to/taxonomy.json --gepa-root ../gepa-taxonomy-hook
```

Use `--dry-run` to print all phases without making calls or writing artifacts.

## Tests

The suite is free and offline. The engine tests exercise the optimizer-side
hook, so they need the patched checkout on the path:

```bash
PYTHONPATH=../gepa-taxonomy-hook/src uv run pytest
```

Three `test_patch_gate.py` tests additionally need the optional SWE-Bench
harness (`uv sync --extra swebench`).

## Writeup

The study and its results are described in the blog post
[Named Failure Modes: Diagnosing the Program, Not Just the Output](https://andreiberkeley.github.io/gepa-named-failure-modes/blog/2026/08/18/named-failure-modes/).
