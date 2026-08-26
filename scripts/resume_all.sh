#!/usr/bin/env bash
# Resume every paused seed of ONE benchmark, streaming their logs live.
# **SPENDS MONEY.**
#
#   ./scripts/resume_all.sh cloudcast          # parallel, interleaved live output
#   SEQUENTIAL=1 ./scripts/resume_all.sh hover # one seed at a time, clean output
#
# Runs in the FOREGROUND and stays attached. Ctrl-C stops every seed it started.
#
# What you see
# ------------
# gepa writes its real log lines ("Iteration 7: Accepted candidate ...") to
# stdout and its progress bar to stderr, and the bar is carriage-return animated
# -- thousands of \r updates that would bury everything else. So stdout is
# prefixed with the seed name and streamed to your terminal, while stderr goes
# to <run>.resume.err.log. Both are also written in full to <run>.resume.log.
#
# Budgets are not written here. Each seed's command comes from `resume.py
# --emit`, which reads that seed's spend off disk and subtracts it from the
# original budget -- resume is NOT cost-continuous (F064): the CostMeter is
# process-local and restarts at zero, so passing the original budget would
# authorise the whole amount a second time. Computing it at launch keeps these
# commands correct across any number of pause/resume cycles.
#
# `resume.py --emit` also deletes any leftover `gepa.stop` for the seeds it
# emits -- that file is how a pause is performed, and left in place a resumed run
# reads it on iteration 1 and exits immediately, looking exactly like a crash.
#
# Parallel by default: both benchmarks wait on the reflection LM rather than on
# local compute -- measured at 0.16 of 24 cores -- so concurrent seeds share
# network latency instead of contending (D063). Use SEQUENTIAL=1 when you would
# rather read one seed's output without interleaving.

set -u

BENCH="${1:-}"
SEQUENTIAL="${SEQUENTIAL:-0}"
case "$BENCH" in
  hover|cloudcast|circlepack) ;;
  *) echo "usage: $0 <hover|cloudcast|circlepack>    (SEQUENTIAL=1 for one at a time)" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.." || exit 1

if [[ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" ]]; then
  echo "AWS_BEARER_TOKEN_BEDROCK is not set. Run:  source ~/.bashrc" >&2
  exit 2
fi

mapfile -t CMDS < <(PYTHONUTF8=1 uv run python scripts/resume.py --emit "$BENCH")
if [[ ${#CMDS[@]} -eq 0 ]]; then
  echo "nothing to resume for ${BENCH}: no seed has both saved state and budget left."
  exit 0
fi

PIDS=()
cleanup() {
  echo ""
  echo "!!! stopping ${#PIDS[@]} seed(s) -- gepa checkpoints per iteration, so at most"
  echo "!!! the current iteration is lost. Re-run this command to pick up again."
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  pkill -f run_cloudcast.py 2>/dev/null
  pkill -f run_circle_packing.py 2>/dev/null
  pkill -f run_hover_seed.py 2>/dev/null
  exit 130
}
trap cleanup INT TERM

name_of() {
  if [[ "$1" =~ --out\ ([^\ ]+) ]]; then basename "${BASH_REMATCH[1]}"
  else echo "hover-baseline-seed$(sed -n 's/.*--seed \([0-9]*\).*/\1/p' <<<"$1")"; fi
}

echo "=============================================================="
echo " resuming ${BENCH}: ${#CMDS[@]} seed(s)$([[ "$SEQUENTIAL" == "1" ]] && echo ', one at a time' || echo ', in parallel')"
echo " live output below. Ctrl-C stops everything."
echo "=============================================================="

for cmd in "${CMDS[@]}"; do
  name=$(name_of "$cmd")
  budget=$(sed -n 's/.*--budget \([0-9.]*\).*/\1/p' <<<"$cmd")
  short=${name##*-}
  echo ">>> ${name}  (\$${budget} remaining)"

  if [[ "$SEQUENTIAL" == "1" ]]; then
    bash -c "$cmd" 2> "results/runs/${name}.resume.err.log" | tee "results/runs/${name}.resume.log"
  else
    # stdout -> prefixed to terminal AND to the log; stderr (the \r progress bar)
    # -> its own file only. sed -u so lines appear as they happen rather than
    # sitting in a 4KB pipe buffer until the run ends.
    ( bash -c "$cmd" 2> "results/runs/${name}.resume.err.log" \
        | tee "results/runs/${name}.resume.log" \
        | sed -u "s|^|[${BENCH}-${short}] |" ) &
    PIDS+=($!)
  fi
done

if [[ "$SEQUENTIAL" != "1" ]]; then
  wait
fi

echo ""
echo "=============================================================="
echo " all ${BENCH} seeds finished. summary:"
PYTHONUTF8=1 uv run python scripts/track.py 2>/dev/null | head -20
echo "=============================================================="
