#!/usr/bin/env bash
# Run the HotpotQA seeds back to back, one at a time. **THIS SPENDS MONEY.**
#
# Sequential, not parallel: --workers 8 already saturates the Bedrock quota, so
# concurrent seeds would contend and drive up transport errors -- the failure
# mode that scores 0.0 and is indistinguishable to the optimizer from a bad
# candidate.
#
# The chain ABORTS on the first failure rather than continuing. If seed 1 dies
# of a configuration fault, seeds 2 and 3 would die of the same one, and three
# corrupted runs are worse than one.
#
#   ./scripts/run_hotpotqa_all_seeds.sh 60                  # baseline arm
#   ./scripts/run_hotpotqa_all_seeds.sh 60 path/to/tax.json # treatment arm
#
# Budget is per seed and required -- no default, so this cannot be run by
# accident. Already-finished seeds (summary.json present) are skipped, so an
# interrupted chain can be resumed by re-running the same command.

set -u

BUDGET="${1:-}"
TAXONOMY="${2:-}"
SEEDS="${SEEDS:-1 2 3}"
# Lower when another chain shares the Bedrock quota. Three HotpotQA baseline
# seeds ran at 8 with zero transport errors across 33,000 rollouts, so 8 is the
# proven ceiling for ONE chain; two concurrent chains should use 4 each.
WORKERS="${WORKERS:-8}"

if [[ -z "$BUDGET" ]]; then
  echo "usage: $0 <budget-per-seed-usd> [taxonomy.json]" >&2
  exit 2
fi

cd "$(dirname "$0")/.." || exit 1

ARM="baseline"
TAX_ARGS=()
if [[ -n "$TAXONOMY" ]]; then
  ARM="taxonomy"
  TAX_ARGS=(--taxonomy "$TAXONOMY")
fi

n_seeds=$(echo "$SEEDS" | wc -w)
echo "=============================================================="
echo " HotpotQA ${ARM} arm | seeds: ${SEEDS} | \$${BUDGET} per seed"
echo " estimated total: \$$(echo "$BUDGET * $n_seeds" | bc 2>/dev/null || echo "${BUDGET}x${n_seeds}")"
echo " sequential, ~2h per seed, aborts on first failure"
echo "=============================================================="

# Step 0: the shared base-candidate val evaluation. Built once and replayed by
# every seed and both arms, so all runs start from byte-identical state.
# It also produces the traces the taxonomy is generated from, so this is
# spend we owe regardless -- paid once here instead of six times.
# Idempotent: it exits immediately if the cache already exists.
echo ""
echo ">>> base val (shared starting state) at $(date +%H:%M:%S)"
PYTHONUTF8=1 uv run python scripts/build_hotpotqa_base_val.py
if [[ $? -ne 0 ]]; then
  echo "!!! base val failed -- not starting any seed. Every run depends on it."
  exit 1
fi

for seed in $SEEDS; do
  out="results/runs/hotpotqa-${ARM}-seed${seed}"

  if [[ -f "$out/summary.json" ]]; then
    echo ""
    echo ">>> seed ${seed}: already complete ($out/summary.json exists) -- skipping"
    continue
  fi

  echo ""
  echo ">>> seed ${seed} starting at $(date +%H:%M:%S)"

  # PYTHONUTF8=1 is mandatory: gepa writes its run log with the platform
  # default encoding, and a proposed prompt containing an emoji kills the run
  # on Windows (F031). The run script refuses to start without it anyway.
  PYTHONUTF8=1 uv run python scripts/run_hotpotqa_seed.py \
    --seed "$seed" --budget "$BUDGET" --workers "$WORKERS" "${TAX_ARGS[@]}"
  status=$?

  if [[ $status -ne 0 ]]; then
    echo ""
    echo "!!! seed ${seed} FAILED (exit ${status}) at $(date +%H:%M:%S)"
    echo "!!! aborting the chain -- later seeds would hit the same fault."
    echo "!!! inspect: $out"
    exit "$status"
  fi

  echo ">>> seed ${seed} finished at $(date +%H:%M:%S)"
done

echo ""
echo "=============================================================="
echo " all seeds complete"
for seed in $SEEDS; do
  s="results/runs/hotpotqa-${ARM}-seed${seed}/summary.json"
  if [[ -f "$s" ]]; then
    python -c "
import json,sys
d=json.load(open(sys.argv[1]))
sp=d.get('spend',{})
total=sum(v.get('budgeted_usd',0) for v in sp.values() if isinstance(v,dict))
print(f\"  seed {d['seed']}: best val {d.get('best_val_score')}  \"
      f\"candidates {d.get('candidates')}  \\\${total:.2f}  {d.get('elapsed_hours')}h\")
" "$s"
  fi
done
echo "=============================================================="
