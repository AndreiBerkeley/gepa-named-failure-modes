#!/usr/bin/env bash
# Circle packing TAXONOMY arm, 3 seeds in parallel, live output. **SPENDS MONEY.**
#
#   ./scripts/run_circlepack_taxonomy.sh              # $12/seed, ~$36
#   SEQUENTIAL=1 ./scripts/run_circlepack_taxonomy.sh # one at a time
#
# Runs in the FOREGROUND and stays attached, so the run is visible and Ctrl-C
# stops it. This exists because the `nohup ... &` loop it replaces was launched
# twice by accident: twelve processes, two per seed, **both writing the same
# --out directory** -- racing on one run_log.txt and one gepa_state.bin. Nothing
# detected it, because a detached loop prints nothing and each launch looked
# fine on its own. A foreground launcher that refuses a dirty directory makes
# that failure impossible rather than merely unlikely.
#
# The refusal below is the load-bearing part: a second invocation while the
# first is running finds a live directory and exits, instead of silently
# corrupting it.

set -u

BUDGET="${BUDGET:-12}"
SEEDS="${SEEDS:-1 2 3}"
SEQUENTIAL="${SEQUENTIAL:-0}"
TAXONOMY="${TAXONOMY:-results/taxonomy/circlepack_v1/taxonomy.json}"
#: Matches the stock arm exactly, so both arms hit the same rollout cap.
METRIC_CALLS="${METRIC_CALLS:-150}"

cd "$(dirname "$0")/.." || exit 1
GEPA_PY="../gepa-v0.1.4/.venv/Scripts/python.exe"

if [[ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" ]]; then
  echo "AWS_BEARER_TOKEN_BEDROCK is not set. Run:  source ~/.bashrc" >&2
  exit 2
fi
if [[ ! -f "$TAXONOMY" ]]; then
  echo "taxonomy not found: $TAXONOMY" >&2
  exit 2
fi
for s in $SEEDS; do
  d="results/runs/circlepack-taxonomy-seed${s}"
  if [[ -e "$d" ]]; then
    echo "REFUSING: $d already exists." >&2
    echo "  Another launch may still be running -- check first:" >&2
    echo "    powershell -c \"Get-CimInstance Win32_Process | ? { \\\$_.CommandLine -like '*run_circle_packing*' }\"" >&2
    echo "  If it is genuinely dead, move it aside, then re-run." >&2
    exit 2
  fi
done

echo "=============================================================="
echo " circle packing TAXONOMY arm | seeds: ${SEEDS} | \$${BUDGET}/seed"
PYTHONUTF8=1 uv run python -c "
import sys; sys.path.insert(0,'src')
from failure_taxonomy import load_taxonomy
t = load_taxonomy('${TAXONOMY}')
print(f' taxonomy: {len(t)} codes, fingerprint {t.fingerprint}')" 2>/dev/null
echo " stock arm to beat: 2.6255 / 2.6282 / 2.6310 / 2.6182  (AlphaEvolve 2.6358)"
echo " max_metric_calls=${METRIC_CALLS}, same cap the stock seeds ran under"
echo " live output below. Ctrl-C stops everything."
echo "=============================================================="

PIDS=()
cleanup() {
  echo ""
  echo "!!! stopping -- gepa checkpoints per iteration, so at most the current one is lost."
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  pkill -f run_circle_packing.py 2>/dev/null
  exit 130
}
trap cleanup INT TERM

for s in $SEEDS; do
  name="circlepack-taxonomy-seed${s}"
  echo ">>> ${name}"
  cmd="PYTHONUTF8=1 \"$GEPA_PY\" scripts/run_circle_packing.py --arm taxonomy \
--budget ${BUDGET} --seed ${s} --max-metric-calls ${METRIC_CALLS} \
--taxonomy ${TAXONOMY} --out results/runs/${name}"
  if [[ "$SEQUENTIAL" == "1" ]]; then
    bash -c "$cmd" 2> "results/runs/${name}.err.log" | tee "results/runs/${name}.log"
  else
    # stdout prefixed to the terminal; the \r progress bar goes to its own file.
    # sed -u so lines appear as they happen rather than buffering until exit.
    ( bash -c "$cmd" 2> "results/runs/${name}.err.log" \
        | tee "results/runs/${name}.log" \
        | sed -u "s|^|[cp-tax-s${s}] |" ) &
    PIDS+=($!)
  fi
done

[[ "$SEQUENTIAL" != "1" ]] && wait

echo ""
echo "=============================================================="
PYTHONUTF8=1 uv run python scripts/track.py 2>/dev/null | head -12
echo "=============================================================="
