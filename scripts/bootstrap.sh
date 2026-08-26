#!/usr/bin/env bash
# One-command environment setup for a fresh checkout. Idempotent: each piece
# is skipped when already present, so rerunning after a failure is safe.
#
#   ./scripts/bootstrap.sh
#
# Sets up, in order:
#   1. project dependencies (uv sync)
#   2. ../gepa-taxonomy-hook   pinned gepa v0.1.4 with the enricher hook
#      (needed until a gepa release carries reflective_dataset_enricher)
#   3. ../adamast-public       public AdaMAST, its own venv and interpreter
#      (needed only to GENERATE a taxonomy; --taxonomy runs never touch it)
# then runs the free offline checks: the test suite and a pipeline dry run.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [1/4] project dependencies"
uv sync

HOOK=../gepa-taxonomy-hook
echo "==> [2/4] gepa v0.1.4 + enricher hook: $HOOK"
if [ ! -d "$HOOK" ]; then
  git clone --branch v0.1.4 https://github.com/gepa-ai/gepa.git "$HOOK"
fi
if ! grep -q reflective_dataset_enricher "$HOOK/src/gepa/api.py"; then
  git -C "$HOOK" apply "$PWD/patches/gepa-reflective-dataset-enricher.patch"
  echo "    hook patch applied"
else
  echo "    hook already present"
fi

ADAMAST=../adamast-public
echo "==> [3/4] public AdaMAST (taxonomy generation): $ADAMAST"
if [ ! -d "$ADAMAST" ]; then
  git clone --branch agent/baseline-taxonomy-generation \
    https://github.com/multi-agent-systems-failure-taxonomy/AdaMAST.git "$ADAMAST"
fi
if [ ! -x "$ADAMAST/.venv/bin/python" ]; then
  (cd "$ADAMAST" && uv venv --python 3.12 && uv pip install -e ".[bedrock]")
  echo "    AdaMAST environment built"
else
  echo "    AdaMAST environment already present"
fi
if ! grep -q "_discover_one" "$ADAMAST/adamast/pipeline/agreement.py"; then
  git -C "$ADAMAST" apply "$PWD/patches/adamast-parallel-annotators.patch"
  echo "    parallel-annotator patch applied (4x faster agreement rounds)"
else
  echo "    parallel-annotator patch already present"
fi

echo "==> [4/4] offline checks (free, no model calls)"
PYTHONPATH="$HOOK/src" uv run pytest -q
uv run gepa-taxonomy hotpotqa --seed 1 --budget 1 --gepa-root "$HOOK" --dry-run >/dev/null
echo
echo "bootstrap complete. Billed runs additionally need AWS_BEARER_TOKEN_BEDROCK in the environment."
