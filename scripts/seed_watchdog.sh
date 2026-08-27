#!/bin/zsh
# Watchdog for baseline seed runs. Waits for a run_seed.py process, then
# monitors. Kills the run (resume-safe) on any hard failure condition and
# records the verdict. Conditions:
#   1. no-candidate loop: >=3 "did not propose a new candidate" lines
#   2. hard spend ceiling: rollout-cache spend > $100.50 (target: stop at $100)
#   3. stall: no rollouts.jsonl append AND no console.log growth for 35 min
#   4. error storm: >=6 tracebacks in console.log
#   5. OOM signature: >=5 errored evaluations AND >8% of rollouts
cnt() { local n; n=$(grep -c "$1" "$2" 2>/dev/null | head -1 | tr -dc '0-9'); echo ${n:-0}; }
RUN=${1:-/Users/andreicojocaru/Desktop/GEPA/experiments/results/runs/baseline-seed1}
#: Ceiling as $2 so a test-eval run can be guarded at its own budget.
CEIL=${2:-100.5}
LOG=$RUN/console.log
CACHE=$RUN/rollouts.jsonl
RUNLOG=$RUN/run_log.json
VERDICT=$RUN/watchdog_verdict.txt

echo "watchdog armed $(date '+%F %T'); waiting for run_seed.py" > $VERDICT
while ! pgrep -f 'run_seed.py|eval_test.py' >/dev/null; do sleep 20; done
PID=$(pgrep -f "python.*(run_seed|eval_test).py" | head -1); : ${PID:=$(pgrep -f "(run_seed|eval_test).py" | head -1)}
echo "watchdog: run detected pid=$PID $(date '+%F %T')" >> $VERDICT
# Baselines: console.log accumulates across attempts (tee -a), so every
# count-based rule must act on the DELTA since this run started.
NOCAND0=0; [ -f $LOG ] && NOCAND0=$(cnt 'did not propose a new candidate' $LOG)
SKIPS0=0; [ -f $LOG ] && SKIPS0=$(cnt ', skipping' $LOG)
ACCEPTS0=0; [ -f $LOG ] && ACCEPTS0=$(cnt 'Accepted candidate' $LOG)
TB0=0; [ -f $LOG ] && TB0=$(cnt 'Traceback (most recent call last)' $LOG)
SPEND0=$(jq -s '[.[]|(.cost_usd//.cost//0)]|add // 0' $CACHE 2>/dev/null || echo 0)

while true; do
  if ! kill -0 $PID 2>/dev/null; then
    echo "ENDED: process exited on its own $(date '+%F %T')" >> $VERDICT
    exit 0
  fi
  [ -f $LOG ] && NOCAND=$(( $(cnt 'did not propose a new candidate' $LOG) - NOCAND0 )) || NOCAND=0
  if [ "$NOCAND" -ge 3 ]; then
    kill $PID
    echo "KILLED: no-candidate loop ($NOCAND occurrences this run) $(date '+%F %T')" >> $VERDICT
    exit 1
  fi
  # Unproductive-optimization guard: proposals keep coming but none is ever
  # accepted -- burns money producing nothing. Read from
  # run_log.json, not the console: a run launched without `tee` has no log.
  [ -f $RUNLOG ] || { SCORED=0; ACCEPTS=1; }
  [ -f $RUNLOG ] && SCORED=$(jq '[.[] | select((.subsample_scores // []) != [] or (.new_subsample_scores // []) != [])] | length' $RUNLOG 2>/dev/null || echo 0)
  [ -f $RUNLOG ] && ACCEPTS=$(jq '[.[] | select(.new_program_idx != null)] | length' $RUNLOG 2>/dev/null || echo 0)
  if [ "${SCORED:-0}" -ge 12 ] && [ "${ACCEPTS:-0}" -eq 0 ]; then
    kill $PID
    echo "KILLED: $SCORED scored iterations, zero accepted candidates $(date '+%F %T')" >> $VERDICT
    exit 1
  fi
  # Equal-dollar experiment: the ceiling must count every metered dollar, not
  # just rollouts. Reflection and judge spend live in the run summary/meters,
  # so add the judge cache when the treatment arm is running.
  SPEND_R=$(jq -s '[.[]|(.cost_usd//.cost//0)]|add // 0' $CACHE 2>/dev/null || echo 0)
  SPEND_J=$(jq -s '[.[]|(.cost_usd//0)]|add // 0' $RUN/judgements.jsonl 2>/dev/null || echo 0)
  SPEND_F=$(jq -s '[.[]|(.cost_usd//0)]|add // 0' $RUN/reflection.jsonl 2>/dev/null || echo 0)
  # Matches gepa's in-process MaxTotalCostStopper, which sums all four meters.
  # Two enforcement layers with different definitions of "spend" would measure
  # the two arms differently -- and the arm comparison is at equal dollars.
  SPEND=$(echo "$SPEND_R + $SPEND_J + $SPEND_F" | bc 2>/dev/null || echo $SPEND_R)
  if [ "$(echo "$SPEND > $CEIL" | bc 2>/dev/null || echo 0)" = "1" ]; then
    kill $PID
    echo "KILLED: spend runaway (\$$SPEND cumulative > \$$CEIL) $(date '+%F %T')" >> $VERDICT
    exit 1
  fi
  [ -f $LOG ] && TB=$(( $(cnt 'Traceback (most recent call last)' $LOG) - TB0 )) || TB=0
  if [ "$TB" -ge 6 ]; then
    kill $PID
    echo "KILLED: error storm ($TB tracebacks this run) $(date '+%F %T')" >> $VERDICT
    exit 1
  fi
  if [ -f "$CACHE" ] && [ -f "$LOG" ]; then
    NOW=$(date +%s)
    C_AGE=$(( NOW - $(stat -f %m $CACHE) ))
    L_AGE=$(( NOW - $(stat -f %m $LOG) ))
    if [ $C_AGE -gt 2100 ] && [ $L_AGE -gt 2100 ]; then
      kill $PID
      echo "KILLED: stall (no cache/log activity for 35 min) $(date '+%F %T')" >> $VERDICT
      exit 1
    fi
  fi
  sleep 60
done
