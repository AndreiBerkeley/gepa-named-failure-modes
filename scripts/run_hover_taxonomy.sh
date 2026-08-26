#!/usr/bin/env bash
# HoVer taxonomy arm, 3 seeds in parallel, live output. **SPENDS MONEY.**
#
#   ./scripts/run_hover_taxonomy.sh            # $75/seed, 3 seeds, ~$225
#   BUDGET=60 SEEDS="1" ./scripts/run_hover_taxonomy.sh
#   SEQUENTIAL=1 ./scripts/run_hover_taxonomy.sh   # one at a time, cleaner output
#
# Runs in the FOREGROUND and stays attached. Ctrl-C stops every seed it started;
# gepa checkpoints per iteration, so at most the current iteration is lost.
#
# What you see
# ------------
# gepa writes its log lines to stdout and its carriage-return progress bar to
# stderr. stdout is prefixed per seed and streamed to the terminal; stderr goes
# to <run>.err.log. Both are captured in full to <run>.log.
#
# Budget: $75/seed, not the baseline's $60
# ---------------------------------------
# The judge is paid out of the same budget as rollouts and reflection (D032), so
# at equal dollars the treatment arm buys less search depth. HotpotQA and IFBench
# handled this by giving their taxonomy arms MORE budget -- hotpotqa-taxonomy-seed1
# spent $67 solver + $33 judge and reached 38 candidates against its baseline's 32.
# Measured judge spend on those arms was $7-33/seed; HoVer traces are ~4x larger
# than CloudCast's, so budget nearer the top of that range.
#
# Depth, not dollars, is the stopping rule
# ----------------------------------------
# The baseline arm is depth-matched at 22 candidates (D065). Arm the same watcher
# after launch so the treatment seeds stop at the same depth rather than wherever
# their budget happens to run out:
#
#   uv run python scripts/stop_at_depth.py --prefix hover-taxonomy --candidates 22 --loop

set -u

BUDGET="${BUDGET:-75}"
SEEDS="${SEEDS:-1 2 3}"
SEQUENTIAL="${SEQUENTIAL:-0}"
TAXONOMY="${TAXONOMY:-results/taxonomy/hover_v1/taxonomy.json}"

cd "$(dirname "$0")/.." || exit 1

if [[ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" ]]; then
  echo "AWS_BEARER_TOKEN_BEDROCK is not set. Run:  source ~/.bashrc" >&2
  exit 2
fi
if [[ ! -f "$TAXONOMY" ]]; then
  echo "taxonomy not found: $TAXONOMY" >&2
  exit 2
fi
for s in $SEEDS; do
  d="results/runs/hover-taxonomy-seed${s}"
  if [[ -e "$d/gepa_state.bin" ]]; then
    echo "REFUSING: $d already has state. Move it, or resume deliberately." >&2
    exit 2
  fi
done

echo "=============================================================="
echo " HoVer TAXONOMY arm | seeds: ${SEEDS} | \$${BUDGET}/seed"
echo " taxonomy: ${TAXONOMY}"
PYTHONUTF8=1 uv run python -c "
import sys; sys.path.insert(0,'src')
from failure_taxonomy import load_taxonomy
t = load_taxonomy('${TAXONOMY}')
print(f'           {len(t)} codes, fingerprint {t.fingerprint}')" 2>/dev/null
echo " baseline to match: 22 candidates/seed, best val 0.6033 / 0.6233 / 0.6400"
echo " live output below. Ctrl-C stops everything."
echo "=============================================================="

PIDS=()
cleanup() {
  echo ""
  echo "!!! stopping -- gepa checkpoints per iteration, so at most the current one is lost."
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  pkill -f run_hover_seed.py 2>/dev/null
  exit 130
}
trap cleanup INT TERM

for s in $SEEDS; do
  name="hover-taxonomy-seed${s}"
  echo ">>> ${name}"
  cmd="PYTHONUTF8=1 uv run python scripts/run_hover_seed.py --seed ${s} \
--budget ${BUDGET} --workers 8 --minibatch-size 6 --taxonomy ${TAXONOMY}"
  if [[ "$SEQUENTIAL" == "1" ]]; then
    bash -c "$cmd" 2> "results/runs/${name}.err.log" | tee "results/runs/${name}.log"
  else
    # sed -u so lines appear as they happen rather than sitting in a pipe buffer.
    ( bash -c "$cmd" 2> "results/runs/${name}.err.log" \
        | tee "results/runs/${name}.log" \
        | sed -u "s|^|[tax-s${s}] |" ) &
    PIDS+=($!)
  fi
done

[[ "$SEQUENTIAL" != "1" ]] && wait

echo ""
echo "=============================================================="
PYTHONUTF8=1 uv run python scripts/track.py 2>/dev/null | head -12
echo "=============================================================="
