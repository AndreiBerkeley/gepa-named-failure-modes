#!/usr/bin/env bash
# Run one optimize_anything arm across 3 seeds, back to back. **SPENDS MONEY.**
#
#   ./scripts/run_oa_all_seeds.sh circlepack stock 20
#   ./scripts/run_oa_all_seeds.sh cloudcast  stock 15
#   ./scripts/run_oa_all_seeds.sh circlepack taxonomy 20 results/taxonomy/circlepack_v1/taxonomy.json
#
# Why 3 seeds and not 1
# ---------------------
# GEPA's search is stochastic. Baseline seeds of an IDENTICAL configuration span
# 1.9pp on HotpotQA and 3.9pp on IFBench, and re-evaluating a single FROZEN
# candidate moved 2.42pp. A one-seed arm therefore cannot separate a feedback
# effect from a draw -- which is the entire question these runs exist to answer.
#
# Sequential WITHIN one invocation, but seeds can safely be run in PARALLEL by
# launching one invocation per seed with SEEDS="n". The earlier claim that circle
# packing pins a core per candidate was wrong: measured live, the pilot used
# 0.16 of 24 cores with a flat thread count. Both benchmarks are dominated by the
# reflection LM call, not local compute, so concurrent seeds wait on the network
# side by side instead of contending. Believing otherwise costed this arm an
# estimated 24h of wall clock that it does not actually need.
#
# Already-finished seeds (summary.json newer than run_log.txt) are skipped, so
# an interrupted chain resumes by re-running the same command.

set -u

BENCH="${1:-}"
ARM="${2:-}"
BUDGET="${3:-}"
TAXONOMY="${4:-}"
SEEDS="${SEEDS:-1 2 3}"
#: Match the arms on DEPTH, not dollars: the judge competes for the same budget,
#: so at equal spend the taxonomy arm explores fewer candidates. Set this to the
#: candidate count the STOCK arm reached.
MAX_CANDIDATES="${MAX_CANDIDATES:-}"

if [[ -z "$BENCH" || -z "$ARM" || -z "$BUDGET" ]]; then
  echo "usage: $0 <circlepack|cloudcast> <stock|taxonomy|score> <budget-per-seed> [taxonomy.json]" >&2
  exit 2
fi
if [[ "$ARM" == "taxonomy" && -z "$TAXONOMY" ]]; then
  echo "the taxonomy arm needs a taxonomy.json, or it silently degrades to score-only" >&2
  exit 2
fi

cd "$(dirname "$0")/.." || exit 1
GEPA_PY="../gepa-v0.1.4/.venv/Scripts/python.exe"

# max_metric_calls MUST be passed explicitly. Both run scripts default to their
# upstream CLI default (100 for cloudcast), which is far below what these budgets
# need: seed 1 exhausted 100 rollouts at iteration 15 and stopped with 2
# candidates having spent $1.59 of $15. The dollar budget never got a chance to
# bind. These values are the ones the PILOTS ran, so seeds are comparable to them.
case "$BENCH" in
  circlepack) SCRIPT=scripts/run_circle_packing.py; METRIC_CALLS="${METRIC_CALLS:-150}" ;;
  cloudcast)  SCRIPT=scripts/run_cloudcast.py;      METRIC_CALLS="${METRIC_CALLS:-1000}" ;;
  *) echo "unknown benchmark: $BENCH" >&2; exit 2 ;;
esac

EXTRA=(--max-metric-calls "$METRIC_CALLS")
[[ -n "$TAXONOMY" ]] && EXTRA+=(--taxonomy "$TAXONOMY")
[[ -n "$MAX_CANDIDATES" ]] && EXTRA+=(--max-candidate-proposals "$MAX_CANDIDATES")

n=$(echo "$SEEDS" | wc -w)
echo "=============================================================="
echo " ${BENCH} ${ARM} arm | seeds: ${SEEDS} | \$${BUDGET} per seed"
echo " estimated total: \$$(echo "$BUDGET * $n" | bc 2>/dev/null || echo "${BUDGET}x${n}")"
echo " depth cap: ${MAX_CANDIDATES:-none} | max_metric_calls: ${METRIC_CALLS}"
echo "=============================================================="

for seed in $SEEDS; do
  out="results/runs/${BENCH}-${ARM}-seed${seed}"

  if [[ -f "$out/summary.json" && "$out/summary.json" -nt "$out/run_log.txt" ]]; then
    echo ""
    echo ">>> seed ${seed}: already complete -- skipping"
    continue
  fi

  echo ""
  echo ">>> ${BENCH} ${ARM} seed ${seed} starting at $(date +%H:%M:%S)"

  # PYTHONUTF8=1 is mandatory: gepa writes its run log with the platform default
  # encoding, and a proposed candidate containing a non-cp1252 character kills
  # the run on Windows.
  PYTHONUTF8=1 "$GEPA_PY" "$SCRIPT" \
    --arm "$ARM" --budget "$BUDGET" --seed "$seed" \
    --out "$out" "${EXTRA[@]}"
  status=$?

  if [[ $status -ne 0 ]]; then
    echo ""
    echo "!!! seed ${seed} FAILED (exit ${status}) -- aborting the chain."
    echo "!!! later seeds would hit the same fault. inspect: $out"
    exit "$status"
  fi
  echo ">>> seed ${seed} finished at $(date +%H:%M:%S)"
done

echo ""
echo "=============================================================="
for seed in $SEEDS; do
  s="results/runs/${BENCH}-${ARM}-seed${seed}/summary.json"
  [[ -f "$s" ]] && PYTHONUTF8=1 uv run python -c "
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8-sig'))
print(f\"  seed {d['seed']}: best {d['best_score']:.6f} | {d['candidates']} candidates | \${d['spend']['realised_usd']:.2f} | {d['elapsed_hours']}h\")
" "$s"
done
echo "=============================================================="
