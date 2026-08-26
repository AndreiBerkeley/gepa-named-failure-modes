# Phase 1 — Pause/resume state fidelity in gepa v0.1.4

Date: 2026-08-07 · Free: no LM calls. Proven by
`tests/test_resume_fidelity.py` (8 tests) running a real `gepa.optimize()` loop
against a deterministic fake adapter.

## Verdict

**Partially faithful — and the distinction matters for the tranche policy.**

| Andrei's checklist | Restored? | Mechanism |
|---|---|---|
| Candidate pool | ✅ exact | full pickle of `GEPAState.__dict__` |
| Per-instance results | ✅ exact | `prog_candidate_val_subscores` in that pickle |
| Pareto frontier | ✅ exact | `pareto_front_valset` / `program_at_pareto_front_valset` |
| Evaluation cache | ✅ exact | `evaluation_cache` is a state field |
| Spend meter | ✅ exact | **only via `adapter_state`** — see below |
| **RNG / sampler position** | ❌ **not restored** | built outside state; rebuilt on resume |

So: **everything already discovered survives a pause exactly. The future
trajectory does not match what an uninterrupted longer run would have taken.**

## Mechanism

`GEPAState.save()` pickles `self.__dict__` wholesale, excluding only
`_budget_hooks`. `load()` restores it and asserts structural invariants. That is
why candidates, scores and frontiers come back bit-exact.

The gap: `rng = random.Random(seed)` is constructed in **`api.py:304`**, outside
`GEPAState`, and `EpochShuffledBatchSampler` keeps `shuffled_ids` / `epoch` /
`id_freqs` / `_calls_in_iteration` on the *strategy object*, which is likewise
not part of the pickle. On resume both are rebuilt from scratch, so the
minibatch stream restarts at the beginning of a fresh epoch instead of
continuing.

Our spend meter is not gepa state at all. gepa provides `adapter_state` — a dict
synced each iteration via `get_adapter_state()` / `set_adapter_state()`
(`engine.py:159-174`) and restored on load. Routing `CostMeter` through that hook
makes the meter exactly restored. **Without it the meter silently resets to $0
on resume and the run spends its whole budget again**; that is pinned by
`test_spend_meter_survives_resume_via_adapter_state`.

## Measured divergence

Uninterrupted 120-call run vs. pause-at-60-and-resume, same seed:

```
uninterrupted : base, base+1, base+10, base+12, base+13, base+14   (6)
resumed       : base, base+1, base+1,  base+3                      (4)
common prefix : 2      best_idx: 5 vs 0
```

What **is** guaranteed, and tested: everything discovered before the pause is
preserved, in order, as a prefix of the resumed pool
(`test_prepause_candidates_are_a_prefix_of_the_resumed_pool`). Nothing is lost
or reordered — the resumed run simply explores differently from there.

## ⚠️ A serious bug this investigation surfaced

The first version of the equality test **passed**, and it was wrong. Both pools
contained only the seed candidate, so it was comparing two single-element lists.

Root cause: `ReflectiveMutationProposer._propose_texts_batch` reads
`self.adapter.propose_new_texts` **unconditionally**
(`reflective_mutation.py:176`). An adapter that omits the attribute raises
`AttributeError`, which gepa catches and logs as *"Reflective mutation did not
propose a new candidate"* — then continues.

**Our real `SweBenchAdapter` also omitted it.** Every baseline run would have
burned its entire dollar budget, logged a warning per iteration, and finished
with nothing but the seed candidate. Fixed by declaring
`propose_new_texts = None` ("I don't own proposal; use the reflection LM"), with
`test_adapter_declares_propose_new_texts` guarding it.

Worth noting how it was caught: not by the test, but by the test looking *too
clean*. A green suite is not evidence on its own.

## Consequence for the tranche policy

Extending $100 → $150 per seed is **safe** — no discovered work is lost, and the
spend meter continues rather than resetting.

But a tranche-extended run is **not identical** to a single-shot $150 run. Two
things follow:

1. **Extend all seeds uniformly, by the same increment, in one decision** —
   which is already the policy. Extending arms or seeds unequally would confound
   the comparison outright.
2. **The blog post's methods section must state that runs were extended in
   tranches**, and that resume restarts the minibatch shuffle. This is a
   reproducibility detail a reader needs.

If bit-identical replay ever becomes a requirement, the fix is upstream:
persist the RNG and sampler state inside `GEPAState`. That is a clean, small PR
against `gepa` — and a natural companion to the `evaluation_cache` passthrough
already on our list.
