#!/usr/bin/env python
"""Evaluate FROZEN candidates on the SWE-Bench Verified TEST split. **THIS SPENDS API TOKENS.**

    zsh -c 'source ~/.zshrc >/dev/null 2>&1; \
      caffeinate -dimsu uv run python scripts/eval_test.py \
        --candidates base=results/runs/baseline-seed1/candidates.json#0 \
        --candidates baseline=results/runs/baseline-seed1 \
        --candidates taxonomy=results/runs/taxonomy-seed1 \
        --n 150 --budget 60'

A MEASUREMENT script, not an optimizer. No GEPA, no reflection, no judging: it
takes candidates that are already frozen, runs each one on the same held-out
instances, and reports resolve rates so a val gain can be checked for
generalisation.

Why the loop is INSTANCE-major, not candidate-major
---------------------------------------------------
Only the 60 val images are pre-pulled (61 ``sweb.eval.*`` locally). Test images
are not, and they do not all fit: measured locally they run ~3.7 GB each and up
(matplotlib's is 11.4 GB), so 300 instances is well over a terabyte against
~400 GB free in the Docker VM. Every test image must therefore be pulled, used,
and released -- and at a measured ~46 GB/h the downloads, not the containers,
are the wall clock.

Two candidate-major passes would therefore download every image TWICE. This
script instead walks instances on the outside and candidates on the inside: an
image is pulled once, every candidate's patch is graded against it, and only
then is it released. Doubling the candidate count costs container time, not
bandwidth. Pinned by ``tests/test_eval_test.py::test_grader_sees_every_candidate_before_next_instance``.

What the harness can and cannot do (verified against swebench 4.1.0 source)
---------------------------------------------------------------------------
One ``run_evaluation`` call CANNOT grade several candidates on one instance.
``run_evaluation.py:515`` collapses the predictions list to a dict::

    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

The key is the instance id ALONE, so a second prediction for the same instance
silently overwrites the first no matter what ``model_name_or_path`` says, and
only the survivor is ever graded. ``make_run_report`` (reporting.py:51-57) then
reads back one prediction per instance, so the report cannot express more
either. Multi-prediction-per-instance is therefore not available, and this
script issues one harness call per (instance group, candidate).

Keeping the image alive across those calls is the part that matters, and it
turns on ``should_remove`` (docker_utils.py:295-311):

    elif image_name.startswith("sweb.eval"):
        if cache_level in {"none", "base", "env"} and (clean or not existed_before):
            return True

``existed_before`` is residency at the START of that invocation. Under the
repo's usual ``--cache_level env``, the first call for an instance pulls the
image and then DELETES it, so candidate 2 re-pulls -- exactly the download this
ordering exists to avoid. Hence ``--cache-level instance`` here: at that level
the harness removes no eval image, and this script frees them itself, only ever
deleting an image it introduced. That last condition is what keeps a run from
evicting the 60 pre-pulled val images.

Resumability
------------
Every paid rollout is fsync'd to ``<out>/rollouts.jsonl`` the instant it
completes, keyed by (candidate label, candidate hash, instance id). An
interruption six hours in re-pays nothing: patches come back from the cache and
only ungraded instances go back to Docker. Same file, same conventions, as
``src/gepa_taxonomy/rollout_cache.py``, and the ``cost_usd`` ledger field is
what ``scripts/seed_watchdog.sh`` sums.

Gold blindness
--------------
Unchanged and absolute (D008). The program receives a ``Task``, which cannot
carry gold, and every rollout goes through the adapter's own ``_audit`` -- the
same code path, not a restatement of it. Gold reaches only the grader.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from gepa_taxonomy.seed_cache import candidate_hash

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = REPO_ROOT / "manifests" / "swebench_verified"
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "test_eval"
CACHE_DIR = REPO_ROOT / ".cache" / "repos"

#: Seed for the stratified draw. Shares the splits' seed so every deterministic
#: choice in this project traces to one number; the algorithms are unrelated, so
#: sharing it creates no correlation between which instances are in test and
#: which prefix of test we evaluate.
DEFAULT_SEED = 20260807

#: Stratum for an instance the dataset gives no difficulty label. Named rather
#: than dropped: a silently discarded instance would shrink `n` without saying so.
UNKNOWN_DIFFICULTY = "unknown"

#: The harness's own default namespace. ``LocalDockerGrader`` passes no
#: ``--namespace``, so image keys must be computed at this namespace or the
#: residency check inspects a tag the harness never creates.
HARNESS_NAMESPACE = "swebench"


def log(message: str = "") -> None:
    """Single print helper so every line is flushed. A 6-hour run watched
    through `tail -f` must not sit in a 4 KB stdio buffer."""
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Candidate resolution
# ---------------------------------------------------------------------------


class CandidateSelectionError(RuntimeError):
    """Raised when the best candidate cannot be established from a run's logs.

    Deliberately fatal. Guessing an index here would silently measure a
    different program than the one whose val gain we are trying to reproduce,
    and the results would look perfectly plausible.
    """


@dataclass(frozen=True)
class ResolvedCandidate:
    """One frozen program, plus the provenance of why THIS index was chosen."""

    label: str
    text: dict[str, str]
    index: int
    source: Path
    #: Full-val score parsed from the run's logs; None when an explicit
    #: ``#INDEX`` was given and no log evidence was consulted.
    val_score: float | None = None
    #: How the index was arrived at, printed verbatim so the choice is auditable.
    selection: str = ""

    @property
    def hash(self) -> str:
        return candidate_hash(self.text)


def parse_candidate_spec(spec: str) -> tuple[str, Path, int | None]:
    """Split ``LABEL=PATH`` or ``LABEL=PATH#INDEX``.

    ``#`` is split from the RIGHT: run directories are user-chosen paths and may
    legitimately contain a '#'.
    """
    if "=" not in spec:
        raise CandidateSelectionError(
            f"--candidates {spec!r} is not LABEL=PATH. Example: baseline=results/runs/baseline-seed1"
        )
    label, _, rest = spec.partition("=")
    label = label.strip()
    if not label:
        raise CandidateSelectionError(f"--candidates {spec!r} has an empty label")
    index: int | None = None
    if "#" in rest:
        rest, _, raw = rest.rpartition("#")
        try:
            index = int(raw)
        except ValueError as exc:
            raise CandidateSelectionError(f"--candidates {spec!r}: {raw!r} is not an integer index") from exc
        if index < 0:
            raise CandidateSelectionError(f"--candidates {spec!r}: index must be >= 0")
    return label, Path(rest).expanduser(), index


#: gepa's own log lines. Both files are read because they carry different
#: subsets: gepa.log exists even when the launch was not wrapped in `tee`, and
#: console.log additionally holds QuietLogger's compact rewrites.
_BASE_VAL_RE = re.compile(r"^Iteration \d+: Base program full valset score: ([0-9.eE+-]+)", re.MULTILINE)
_BASE_VAL_N_RE = re.compile(r"^Iteration \d+: Base program full valset score: [0-9.eE+-]+ over \d+ / (\d+)", re.MULTILINE)
_NEW_SCORE_RE = re.compile(r"^Iteration (\d+): Valset score for new program: ([0-9.eE+-]+)", re.MULTILINE)
_NEW_INDEX_RE = re.compile(r"^Iteration (\d+): New program candidate index: (\d+)", re.MULTILINE)
#: iterations.py reads this one; it is the per-instance dict, averaged.
_INDIVIDUAL_RE = re.compile(r"^Iteration (\d+): Individual valset scores.*?\{(.*?)\}", re.MULTILINE | re.DOTALL)
#: QuietLogger's rewrite of gepa's "Selected program N score: X". Independent of
#: the three above -- it survives even when the iteration that produced a
#: candidate scrolled out of a truncated log.
_SELECTED_RE = re.compile(r"^Iteration \d+: candidate (\d+) selected \(val (\d+)/(\d+)\)", re.MULTILINE)


def _read_run_logs(run_dir: Path) -> str:
    """Concatenate the run's logs, exactly as scripts/iterations.py does."""
    text = ""
    for name in ("console.log", "gepa.log"):
        f = run_dir / name
        if f.exists():
            text += f.read_text(errors="replace")
    return text


def val_scores_by_candidate(run_dir: Path) -> tuple[dict[int, float], int | None]:
    """Map candidate index -> full-val score, from the run's console/gepa logs.

    Returns ``(scores, valset_size)``. Three independent sources are merged, and
    they agree where they overlap:

    * the base line, which is candidate 0 by construction (engine.py:686);
    * ``Valset score for new program`` joined to ``New program candidate index``
      on the iteration number (both emitted by logging/utils.py for the same
      iteration, so the join is exact);
    * QuietLogger's ``candidate N selected (val k/m)``.

    Merging is not belt-and-braces: a run killed mid-iteration can hold the
    index line without the score line, or vice versa, and either alone would
    then omit a candidate that a later line does describe.
    """
    text = _read_run_logs(run_dir)
    if not text:
        raise CandidateSelectionError(
            f"no console.log or gepa.log under {run_dir}. The best candidate is read from the run's own "
            "logs; without them the choice would be a guess. Pass an explicit LABEL=PATH#INDEX instead."
        )

    scores: dict[int, float] = {}

    m = _BASE_VAL_N_RE.search(text)
    valset_size = int(m.group(1)) if m else None

    m = _BASE_VAL_RE.search(text)
    if m:
        scores[0] = float(m.group(1))

    by_iter_score: dict[int, float] = {int(i): float(v) for i, v in _NEW_SCORE_RE.findall(text)}
    for i, body in _INDIVIDUAL_RE.findall(text):
        values = [float(v.split(":")[1]) for v in body.split(",") if ":" in v]
        if values and int(i) not in by_iter_score:
            by_iter_score[int(i)] = sum(values) / len(values)
            valset_size = valset_size or len(values)
    for i, idx in _NEW_INDEX_RE.findall(text):
        if int(i) in by_iter_score:
            scores[int(idx)] = by_iter_score[int(i)]

    for idx, num, den in _SELECTED_RE.findall(text):
        scores.setdefault(int(idx), int(num) / max(int(den), 1))
        valset_size = valset_size or int(den)

    return scores, valset_size


def best_candidate_index(run_dir: Path, n_candidates: int) -> tuple[int, float, str]:
    """Pick the candidate with the best full-val score. Returns (index, score, why).

    Ties go to the LOWEST index, which is what gepa's own ``idxmax`` does when
    it reports "Best program as per aggregate score on valset". The tie is named
    in the returned explanation rather than hidden, because two candidates at
    the same val score is exactly the situation where the reader needs to know
    the choice was arbitrary.
    """
    scores, valset_size = val_scores_by_candidate(run_dir)
    if not scores:
        raise CandidateSelectionError(
            f"{run_dir} has logs but no full-val score lines in them, so the best candidate is "
            "undeterminable. Pass an explicit LABEL=PATH#INDEX."
        )
    known = {i: s for i, s in scores.items() if i < n_candidates}
    if not known:
        raise CandidateSelectionError(
            f"{run_dir}: every scored candidate index {sorted(scores)} is out of range for the "
            f"{n_candidates} candidates in candidates.json. The logs and the candidate file disagree."
        )
    best = max(known.values())
    winners = sorted(i for i, s in known.items() if s == best)
    index = winners[0]
    shown = f"{best:.4f}"
    if valset_size:
        shown += f" ({round(best * valset_size)}/{valset_size} resolved)"
    why = f"best full-val score {shown} among {len(known)} scored candidates"
    if len(winners) > 1:
        why += f"; TIED with {winners[1:]}, lowest index taken"
    return index, best, why


def resolve_candidate(label: str, path: Path, index: int | None) -> ResolvedCandidate:
    """Load one frozen candidate, choosing the index from logs when not given."""
    if path.is_dir():
        run_dir, cand_file = path, path / "candidates.json"
    else:
        run_dir, cand_file = path.parent, path
    if not cand_file.exists():
        raise CandidateSelectionError(f"{label}: no candidates.json at {cand_file}")

    candidates = json.loads(cand_file.read_text())
    if not isinstance(candidates, list) or not candidates:
        raise CandidateSelectionError(f"{label}: {cand_file} is not a non-empty list of candidates")

    if index is None:
        index, score, why = best_candidate_index(run_dir, len(candidates))
        selection = f"selected from logs: {why}"
    else:
        score, selection = None, "index given explicitly on the command line"
    if index >= len(candidates):
        raise CandidateSelectionError(f"{label}: index {index} but {cand_file} holds only {len(candidates)}")

    text = candidates[index]
    if not isinstance(text, dict) or not all(isinstance(v, str) for v in text.values()):
        raise CandidateSelectionError(f"{label}: candidate {index} in {cand_file} is not a dict of strings")
    return ResolvedCandidate(
        label=label, text=text, index=index, source=cand_file, val_score=score, selection=selection
    )


# ---------------------------------------------------------------------------
# Instance selection
# ---------------------------------------------------------------------------


def stratified_order(
    instance_ids: Iterable[str],
    difficulty: dict[str, str],
    *,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    """Order the manifest so that EVERY prefix preserves the difficulty mix.

    Not a stratified sample -- a stratified *ordering*, which is strictly
    stronger and is what makes an extension cheap. Each stratum is shuffled with
    the seeded RNG, then an item at rank ``r`` of a stratum of size ``n`` is
    given the position key ``(2r+1)/2n`` and everything is merged on that key.
    Interleaving on relative rank means any prefix of length N holds each
    stratum in proportion to its share, to within one instance.

    Two consequences we rely on:

    * ``order[:N]`` is difficulty-stratified for every N, so ``--n`` needs no
      re-derivation; and
    * ``order[:N]`` is a strict subset of ``order[:M]`` for N < M, so raising
      ``--n`` later extends the measurement instead of replacing it -- the
      already-paid instances stay in, and the cache still covers them.

    Taking a stratified *sample* per N would satisfy neither: largest-remainder
    allocation is not prefix-stable, so N=150 and N=200 would disagree about the
    first 150.
    """
    strata: dict[str, list[str]] = defaultdict(list)
    for iid in sorted(instance_ids):  # sorted first: dict order must not leak in
        strata[difficulty.get(iid) or UNKNOWN_DIFFICULTY].append(iid)

    rng = random.Random(seed)
    keyed: list[tuple[float, str, str]] = []
    for stratum in sorted(strata):  # fixed label order, so the RNG stream is fixed
        members = strata[stratum]
        rng.shuffle(members)
        n = len(members)
        for rank, iid in enumerate(members):
            keyed.append(((2 * rank + 1) / (2 * n), stratum, iid))
    keyed.sort()
    return [iid for _key, _stratum, iid in keyed]


def difficulty_mix(instance_ids: Iterable[str], difficulty: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for iid in instance_ids:
        counts[difficulty.get(iid) or UNKNOWN_DIFFICULTY] += 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Durable cache -- also the watchdog's spend ledger
# ---------------------------------------------------------------------------


@dataclass
class EvalCache:
    """Write-through cache of (candidate label, candidate hash, instance id).

    Conventions are ``rollout_cache.RolloutCache``'s: append-only JSONL, fsync
    on every record, replayed into memory at open, and a truncated final line
    dropped rather than fatal -- an abrupt kill must cost at most the rollout
    that was in flight.

    Two differences, both deliberate:

    **The key carries the label.** The hash alone would be sufficient for
    correctness, but a label is a claim about provenance ("this is the taxonomy
    arm's winner"). Keying on both means repointing a label at a different
    candidate invalidates that label's rows instead of silently reusing patches
    generated by the program it no longer names.

    **A rollout is written twice**: once when the patch is generated and paid
    for, once when the harness has scored it. The first write is what makes
    Phase A resumable -- deferring to Phase B would put every dollar of the run
    at risk of a single interruption. ``cost_usd`` is a per-LINE ledger delta
    and is 0.0 on the second write, so summing the file (which is what
    seed_watchdog.sh does) gives realised spend exactly once.
    ``rollout_cost_usd`` repeats the rollout's own price on both lines and is
    what per-candidate reporting reads.
    """

    path: Path
    _entries: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _fh: Any = field(default=None, repr=False)
    #: Rollouts served from cache this session -- work not paid for twice.
    hits: int = 0
    truncated_records: int = 0

    @classmethod
    def open(cls, path: str | Path) -> EvalCache:
        cache = cls(path=Path(path))
        cache.load()
        cache._fh = cache.path.open("a", buffering=1)
        return cache

    def load(self) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            return 0
        loaded = 0
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (rec["label"], rec["candidate_hash"], rec["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    self.truncated_records += 1
                    continue
                self._entries[key] = rec  # later line wins: the graded record supersedes
                loaded += 1
        return loaded

    def _write(self, rec: dict[str, Any]) -> None:
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    @staticmethod
    def key(label: str, candidate: dict[str, str], instance_id: str) -> tuple[str, str, str]:
        return (label, candidate_hash(candidate), instance_id)

    def get(self, label: str, candidate: dict[str, str], instance_id: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self._entries.get(self.key(label, candidate, instance_id))
            if rec is not None:
                self.hits += 1
            return rec

    def has_patch(self, label: str, candidate: dict[str, str], instance_id: str) -> bool:
        with self._lock:
            return self.key(label, candidate, instance_id) in self._entries

    def is_graded(self, label: str, candidate: dict[str, str], instance_id: str) -> bool:
        with self._lock:
            rec = self._entries.get(self.key(label, candidate, instance_id))
            return bool(rec and rec.get("graded"))

    def put_patch(
        self,
        label: str,
        candidate: dict[str, str],
        instance_id: str,
        *,
        patch: str,
        cost_usd: float,
        trace: dict[str, Any] | None = None,
    ) -> None:
        key = self.key(label, candidate, instance_id)
        rec = {
            "label": label,
            "candidate_hash": key[1],
            "instance_id": instance_id,
            "patch": patch,
            "score": None,
            "graded": False,
            # Ledger delta for this line; the watchdog sums this column.
            "cost_usd": round(cost_usd, 8),
            # The rollout's own price, repeated on every line about it.
            "rollout_cost_usd": round(cost_usd, 8),
            "output": {"patch": patch, "instance_id": instance_id},
            "trace": trace or {},
        }
        with self._lock:
            self._entries[key] = rec
            self._write(rec)

    def put_score(
        self,
        label: str,
        candidate: dict[str, str],
        instance_id: str,
        *,
        score: float,
        detail: dict[str, Any],
    ) -> None:
        key = self.key(label, candidate, instance_id)
        with self._lock:
            base = self._entries.get(key)
            if base is None:
                raise KeyError(f"no patch cached for {key}; a score cannot be attached to nothing")
            rec = dict(base)
            rec["score"] = float(score)
            rec["graded"] = True
            rec["grading"] = detail
            # Already counted on the patch line. Anything but 0.0 here would
            # make seed_watchdog.sh report double the real spend.
            rec["cost_usd"] = 0.0
            self._entries[key] = rec
            self._write(rec)

    def spend_usd(self, label: str | None = None) -> float:
        with self._lock:
            return sum(
                r.get("rollout_cost_usd", 0.0)
                for k, r in self._entries.items()
                if label is None or k[0] == label
            )

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class TotalSpendMeter:
    """Presents a ``CostMeter``'s TOTAL spend under the name the stopper reads.

    ``MaxTotalCostStopper`` compares ``meter.budgeted_usd``, and ``budgeted_usd``
    counts only ``BUDGETED_PHASES`` -- which is ``{"optimization"}``. This script
    books to ``final_test``, because cost.py excludes final test evaluation from
    the per-seed budget by design (exclusion (b)). Handing the raw meters to the
    stopper would therefore show it $0.00 forever and it would never fire: a
    ceiling that silently fails open, on the one script whose whole job is a
    bounded measurement.

    So the stopper class, its comparison and its overshoot caveat are reused
    unchanged; only the number it reads is redirected to what this script
    actually spends.
    """

    def __init__(self, meter: Any) -> None:
        self._meter = meter

    @property
    def budgeted_usd(self) -> float:
        return self._meter.total_usd


# ---------------------------------------------------------------------------
# Phase A -- patch generation (API-bound)
# ---------------------------------------------------------------------------


@dataclass
class PhaseAResult:
    executed: int = 0
    cached: int = 0
    #: Instances left untouched because the budget stopper fired first.
    skipped_for_budget: list[str] = field(default_factory=list)
    budget_hit: bool = False


def generate_patches(
    *,
    instance_ids: Sequence[str],
    candidates: Sequence[ResolvedCandidate],
    instances: dict[str, Any],
    program: Any,
    cache: EvalCache,
    auditor: Any,
    max_workers: int = 4,
    stopper: Callable[[], bool] | None = None,
    on_progress: Callable[[str, PhaseAResult], None] | None = None,
) -> PhaseAResult:
    """Run every (candidate, instance) rollout that is not already cached.

    The unit of work is one INSTANCE with all its candidates, for two reasons.

    *Pairing.* When the budget stopper fires it fires between instances, so
    every instance that made it into the cache has a patch from every candidate.
    A per-rollout cut-off would leave half-populated instances, which cannot
    enter a paired comparison at all.

    *The checkout.* ``BM25Retriever`` force-checks-out one shared directory per
    repo (``retrieval.ensure_checkout``), so two threads on the same repo would
    move the working tree under each other -- retrieval would index the wrong
    commit and the refiner's apply verdict would be against the wrong tree. This
    hazard does not exist in run_seed.py, where gepa drives rollouts serially;
    it is created by running Phase A concurrently, so it is fenced here with a
    per-repo lock. Holding that lock across an instance's candidates also means
    the checkout is moved once, not once per candidate.

    Overshoot: the stopper is consulted at instance boundaries, so realised
    spend can exceed the budget by up to one instance x every candidate. Same
    shape of caveat as MaxTotalCostStopper's, one iteration wide instead.
    """
    result = PhaseAResult()
    repo_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
    guard = threading.Lock()

    def work(instance_id: str) -> None:
        inst = instances[instance_id]
        todo = [c for c in candidates if not cache.has_patch(c.label, c.text, instance_id)]
        with guard:
            result.cached += len(candidates) - len(todo)
        if not todo:
            if on_progress:
                on_progress(instance_id, result)
            return
        # Checked before any paid work for this instance, never inside it.
        if stopper is not None and stopper():
            with guard:
                result.budget_hit = True
                result.skipped_for_budget.append(instance_id)
            return

        with repo_locks[inst.task.repo]:
            for cand in todo:
                rollout = program.run(inst.task, cand.text, phase="final_test")
                # The adapter's own audit, called rather than restated: a second
                # implementation of the gold check is a second thing to drift.
                auditor._audit(rollout, inst.gold)
                cache.put_patch(
                    cand.label,
                    cand.text,
                    instance_id,
                    patch=rollout.final_patch,
                    cost_usd=rollout.cost_usd,
                    trace=rollout.to_trace(),
                )
                with guard:
                    result.executed += 1
        if on_progress:
            on_progress(instance_id, result)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(work, instance_ids))
    return result


# ---------------------------------------------------------------------------
# Phase B -- grading (instance-major)
# ---------------------------------------------------------------------------


class ImagePool(Protocol):
    """Owns the residency of the per-instance evaluation images (~4 GB each)."""

    def acquire(self, instance_ids: Sequence[str]) -> None: ...

    def release(self, instance_ids: Sequence[str]) -> int: ...


@dataclass
class DockerImagePool:
    """Notes which eval images we introduce, and deletes only those.

    The pull itself is left to the harness: it already resolves the right
    platform for an emulated arm64 host, and a second implementation here could
    fetch a manifest the harness would not have chosen. What the harness cannot
    do is keep the image alive across calls at ``--cache_level env`` -- see the
    module docstring -- so this script runs it at ``instance`` and frees images
    here instead.

    ``acquire`` therefore records residency; it does not download. ``release``
    removes an image only if it was absent when we started, which is what stops
    a test run from evicting the 60 pre-pulled val images.
    """

    rows: dict[str, dict[str, Any]]
    namespace: str = HARNESS_NAMESPACE
    dry_run: bool = False
    _keys: dict[str, str] = field(default_factory=dict, repr=False)
    _preexisting: set[str] = field(default_factory=set, repr=False)
    _introduced: set[str] = field(default_factory=set, repr=False)
    released: int = 0

    def image_key(self, instance_id: str) -> str:
        if instance_id not in self._keys:
            # Imported lazily: swebench is an optional dependency, installed
            # only on a machine that actually grades.
            from swebench.harness.test_spec.test_spec import make_test_spec

            spec = make_test_spec(self.rows[instance_id], namespace=self.namespace)
            self._keys[instance_id] = spec.instance_image_key
        return self._keys[instance_id]

    def _is_local(self, key: str) -> bool:
        proc = subprocess.run(
            ["docker", "image", "inspect", key], capture_output=True, text=True, check=False
        )
        return proc.returncode == 0

    def acquire(self, instance_ids: Sequence[str]) -> None:
        for iid in instance_ids:
            key = self.image_key(iid)
            if self._is_local(key):
                self._preexisting.add(key)
            else:
                self._introduced.add(key)

    def release(self, instance_ids: Sequence[str]) -> int:
        freed = 0
        for iid in instance_ids:
            key = self._keys.get(iid)
            if key is None or key in self._preexisting or key not in self._introduced:
                continue
            if self.dry_run:
                continue
            proc = subprocess.run(["docker", "rmi", "-f", key], capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                # Never fatal, always visible: a silent leak here fills the
                # Docker VM and the run dies hundreds of instances later with an
                # unrelated-looking error.
                log(f"    WARNING: could not remove {key}: {(proc.stderr or '').strip()[:200]}")
                continue
            self._introduced.discard(key)
            freed += 1
        self.released += freed
        return freed


def _chunks(items: Sequence[str], size: int) -> list[list[str]]:
    size = max(1, size)
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def grade_instances(
    *,
    instance_ids: Sequence[str],
    candidates: Sequence[ResolvedCandidate],
    instances: dict[str, Any],
    grader: Any,
    cache: EvalCache,
    images: ImagePool,
    group_size: int = 1,
    on_instance: Callable[[str], None] | None = None,
) -> None:
    """Grade every candidate's patch, walking INSTANCES on the outside.

    A group is the set of instances whose images are resident at once; inside a
    group the candidate loop runs to completion before the group is released, so
    an image is downloaded once and serves every candidate. ``group_size`` above
    1 exists only to restore harness parallelism: ``--max-workers`` parallelises
    *within* one invocation, so a one-instance call is serial no matter what the
    flag says. It does not weaken the ordering guarantee -- an instance's
    candidates are still all graded before any instance outside its group.

    Empty patches never reach Docker. That is not an optimisation with a risk
    attached: run_evaluation.py:458-470 filters empty predictions out of the
    dataset before a container starts and reports them unresolved, so this is
    the harness's own behaviour, moved earlier to avoid pulling gigabytes for a
    patch it would refuse to run. The tier-2 apply check from ``patch_gate`` is
    deliberately NOT used here -- it needs the checkout parked at this task's
    base_commit, and by Phase B the shared per-repo checkout has moved on, so
    its verdict would be against the wrong tree.
    """
    for group in _chunks(instance_ids, group_size):
        pending = [i for i in group if any(not cache.is_graded(c.label, c.text, i) for c in candidates)]
        needs_docker = False
        for iid in pending:
            for cand in candidates:
                rec = cache.get(cand.label, cand.text, iid)
                if rec and not rec.get("graded") and (rec.get("patch") or "").strip():
                    needs_docker = True
        if needs_docker:
            images.acquire(pending)
        try:
            for cand in candidates:  # inner loop, one harness call per candidate
                items = []
                for iid in pending:
                    rec = cache.get(cand.label, cand.text, iid)
                    if rec is None or rec.get("graded"):
                        continue
                    patch = rec.get("patch") or ""
                    if not patch.strip():
                        cache.put_score(
                            cand.label,
                            cand.text,
                            iid,
                            score=0.0,
                            detail={"resolved": False, "empty_patch": True, "skipped": True},
                        )
                        continue
                    inst = instances[iid]
                    items.append((inst.task, inst.gold, patch))
                if not items:
                    continue
                graded = grader.grade_batch(items)
                for task, _gold, _patch in items:
                    score, detail = graded[task.instance_id]
                    cache.put_score(cand.label, cand.text, task.instance_id, score=score, detail=detail)
        finally:
            if needs_docker:
                images.release(pending)
        for iid in group:
            if on_instance:
                on_instance(iid)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_results(
    *,
    instance_ids: Sequence[str],
    candidates: Sequence[ResolvedCandidate],
    cache: EvalCache,
    difficulty: dict[str, str],
) -> dict[str, Any]:
    """Per-candidate totals plus the per-instance matrix a paired test needs."""
    rows: list[dict[str, Any]] = []
    resolved: dict[str, int] = {c.label: 0 for c in candidates}
    graded: dict[str, int] = {c.label: 0 for c in candidates}

    for iid in instance_ids:
        row: dict[str, Any] = {
            "instance_id": iid,
            "difficulty": difficulty.get(iid) or UNKNOWN_DIFFICULTY,
            "scores": {},
        }
        for cand in candidates:
            rec = cache.get(cand.label, cand.text, iid)
            score = rec.get("score") if rec and rec.get("graded") else None
            row["scores"][cand.label] = score
            if score is not None:
                graded[cand.label] += 1
                if score > 0:
                    resolved[cand.label] += 1
        rows.append(row)

    per_candidate = []
    for cand in candidates:
        n = graded[cand.label]
        per_candidate.append(
            {
                "label": cand.label,
                "source": str(cand.source),
                "index": cand.index,
                "candidate_hash": cand.hash,
                "val_score": cand.val_score,
                "selection": cand.selection,
                "resolved": resolved[cand.label],
                "n": n,
                "resolve_rate": (resolved[cand.label] / n) if n else None,
                "spend_usd": round(cache.spend_usd(cand.label), 4),
            }
        )
    return {"candidates": per_candidate, "instances": rows}


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def load_difficulty(instance_ids: Sequence[str]) -> dict[str, str]:
    """SWE-bench Verified's own ``difficulty`` column, for the stratification.

    A module-level function so tests can substitute it: the strata are a
    property of the dataset, not something this script should re-derive.
    """
    from datasets import load_dataset

    from gepa_taxonomy.splits import DATASET_NAME, DATASET_SPLIT

    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    wanted = set(instance_ids)
    return {i: d for i, d in zip(ds["instance_id"], ds["difficulty"], strict=True) if i in wanted}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--candidates",
        action="append",
        default=None,
        metavar="LABEL=PATH[#INDEX]",
        help="repeatable; PATH is a run dir or a candidates.json. Without #INDEX the "
        "best full-val candidate is read from that run's logs.",
    )
    ap.add_argument("--n", type=int, default=150, help="test instances to evaluate")
    ap.add_argument("--test-manifest", type=Path, default=MANIFESTS / "test.json")
    ap.add_argument("--max-workers", type=int, default=4, help="harness workers inside one invocation")
    ap.add_argument("--lm-workers", type=int, default=None, help="Phase A concurrency (default: max-workers)")
    ap.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="instances resident at once in Phase B (default: max-workers). Larger trades disk for parallelism.",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--budget", type=float, default=None, help="dollar ceiling on Phase A (MaxTotalCostStopper)")
    ap.add_argument("--profile-prefix", default=None)
    ap.add_argument(
        "--cache-level",
        default="instance",
        help="harness cache level. 'env' DELETES the eval image after each invocation, "
        "which re-pulls it for every candidate; see the module docstring.",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the resolved config and exit; free")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    args = build_parser().parse_args(argv)

    if not args.candidates:
        log("Refusing to run: at least one --candidates LABEL=PATH is required.")
        return 2

    resolved: list[ResolvedCandidate] = []
    seen: set[str] = set()
    for spec in args.candidates:
        label, path, index = parse_candidate_spec(spec)
        if label in seen:
            raise CandidateSelectionError(f"duplicate candidate label {label!r}; labels key the cache and the report")
        seen.add(label)
        resolved.append(resolve_candidate(label, path, index))

    from gepa_taxonomy.splits import load_manifest

    manifest_ids = load_manifest(args.test_manifest)
    difficulty = load_difficulty(manifest_ids)
    order = stratified_order(manifest_ids, difficulty, seed=args.seed)
    if args.n > len(order):
        log(f"Refusing to run: --n {args.n} exceeds the {len(order)} instances in {args.test_manifest}.")
        return 2
    chosen = order[: args.n]

    out = args.out or (DEFAULT_OUT_ROOT / time.strftime("%Y-%m-%d_%H%M%S"))
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lm_workers = args.lm_workers or args.max_workers
    group_size = args.group_size or args.max_workers

    log("=" * 72)
    log("TEST-SPLIT EVALUATION OF FROZEN CANDIDATES  (local Docker)")
    log("=" * 72)
    for cand in resolved:
        log(f"  candidate '{cand.label}'  <- {cand.source}[{cand.index}]")
        log(f"      {cand.selection}")
        log(f"      candidate hash {cand.hash[:16]}")
    log(f"  instances     {len(chosen)} of {len(manifest_ids)} in {args.test_manifest.name}  (seed {args.seed})")
    log(f"  difficulty    chosen {difficulty_mix(chosen, difficulty)}")
    log(f"                pool   {difficulty_mix(manifest_ids, difficulty)}")
    log(f"  rollouts      {len(chosen) * len(resolved)}  ({len(resolved)} candidates x {len(chosen)} instances)")
    log(f"  budget        {f'${args.budget:.2f}' if args.budget else 'NONE (unbounded)'}")
    log(f"  workers       phase A {lm_workers} threads / phase B {args.max_workers} harness workers")
    log(f"  group size    {group_size} instances resident at once   cache_level {args.cache_level}")
    log(f"  out           {out}")

    (out / "instances.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n": len(chosen),
                "manifest": str(args.test_manifest),
                "manifest_n": len(manifest_ids),
                "stratified_by": "difficulty",
                "difficulty_mix": difficulty_mix(chosen, difficulty),
                # The full order, so a later --n extends this measurement rather
                # than replacing it: order[:n] is always a prefix of order[:m].
                "order": order,
                "instance_ids": chosen,
            },
            indent=2,
        )
        + "\n"
    )
    log(f"  wrote         {out / 'instances.json'}")

    cache = EvalCache.open(out / "rollouts.jsonl")
    if len(cache):
        log(f"  resuming      {len(cache)} cached rollouts (${cache.spend_usd():.2f} will not be re-paid)")

    if args.dry_run:
        # Everything above this line is free: manifests, logs, and the dataset's
        # difficulty column. Credentials are required BELOW, so a dry run works
        # on a machine that could not spend a cent even if it tried.
        pending = sum(
            1 for i in chosen for c in resolved if not cache.has_patch(c.label, c.text, i)
        )
        log(f"  would run     {pending} rollouts, {len(chosen) * len(resolved) - pending} already cached")
        cache.close()
        log("\n--dry-run: nothing generated, nothing graded, nothing spent.")
        return 0

    from gepa_taxonomy.bedrock import BedrockLM, require_credentials

    require_credentials()

    from datasets import load_dataset

    from gepa_taxonomy.adapter import SweBenchAdapter
    from gepa_taxonomy.cost import (
        REFINER_BASE,
        SOLVER_BASE,
        CostMeter,
        MaxTotalCostStopper,
        with_profile,
    )
    from gepa_taxonomy.cost import REFINER_MODEL as _DEF_REF
    from gepa_taxonomy.cost import SOLVER_MODEL as _DEF_SOL
    from gepa_taxonomy.grading import LocalDockerGrader
    from gepa_taxonomy.program import SolverRefinerProgram
    from gepa_taxonomy.retrieval import BM25Retriever
    from gepa_taxonomy.splits import DATASET_NAME, DATASET_SPLIT
    from gepa_taxonomy.tasks import split_row

    solver_model = with_profile(SOLVER_BASE, args.profile_prefix) if args.profile_prefix else _DEF_SOL
    refiner_model = with_profile(REFINER_BASE, args.profile_prefix) if args.profile_prefix else _DEF_REF
    log(f"  solver        {solver_model}")
    log(f"  refiner       {refiner_model}")

    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    wanted = set(chosen)
    rows = {r["instance_id"]: dict(r) for r in ds if r["instance_id"] in wanted}
    instances = {i: split_row(r) for i, r in rows.items()}
    missing = wanted - set(instances)
    if missing:
        log(f"Refusing to run: {len(missing)} chosen instances are not in the dataset, e.g. {sorted(missing)[:3]}")
        cache.close()
        return 2

    solver_meter, refiner_meter = CostMeter(), CostMeter()
    program = SolverRefinerProgram(
        retriever=BM25Retriever(cache_dir=CACHE_DIR),
        # Gives the refiner a real apply verdict; the checkout is already at
        # this task's base_commit because retrieval just placed it there.
        repo_dir_for=lambda t: CACHE_DIR / t.repo.replace("/", "__"),
        solver_lm=BedrockLM(model=solver_model),
        refiner_lm=BedrockLM(model=refiner_model),
        solver_meter=solver_meter,
        refiner_meter=refiner_meter,
        solver_model=solver_model,
        refiner_model=refiner_model,
    )
    grader = LocalDockerGrader(
        work_dir=out / "harness",
        max_workers=args.max_workers,
        cache_level=args.cache_level,
        run_id_prefix="testeval",
    )
    # Constructed for ONE method: ``_audit``. Nothing here evaluates or grades
    # through the adapter -- reusing its audit verbatim is the point, so that
    # the gold check on this path cannot drift from the optimization path's.
    auditor = SweBenchAdapter(program=program, grader=grader, instances=instances, strict_gold_check=True)

    stopper = None
    if args.budget:
        stopper = MaxTotalCostStopper(
            args.budget, meters=[TotalSpendMeter(solver_meter), TotalSpendMeter(refiner_meter)]
        )

    t0 = time.time()
    log("\n--- PHASE A: generating patches (API-bound) ---")
    phase_a_done = 0

    def phase_a_progress(instance_id: str, res: PhaseAResult) -> None:
        nonlocal phase_a_done
        phase_a_done += 1
        spent = solver_meter.total_usd + refiner_meter.total_usd
        # Cumulative counters, not this instance's: the threads share them.
        log(
            f"  [A {phase_a_done:>4}/{len(chosen)}] {instance_id:<40} "
            f"rollouts so far: ran {res.executed} cached {res.cached} | spent ${spent:.2f}"
        )

    try:
        phase_a = generate_patches(
            instance_ids=chosen,
            candidates=resolved,
            instances=instances,
            program=program,
            cache=cache,
            auditor=auditor,
            max_workers=lm_workers,
            stopper=(lambda: bool(stopper())) if stopper else None,
            on_progress=phase_a_progress,
        )
    except KeyboardInterrupt:
        log("\ninterrupted -- every completed rollout is in the durable cache; re-run to resume.")
        cache.close()
        return 130

    spent = solver_meter.total_usd + refiner_meter.total_usd
    log(f"\n  phase A: {phase_a.executed} rollouts run, {phase_a.cached} from cache, ${spent:.2f} spent")
    if phase_a.budget_hit:
        log(
            f"  BUDGET REACHED: {len(phase_a.skipped_for_budget)} instances have no patches and are excluded.\n"
            "  Phase B grades only fully-populated instances, so the comparison stays paired."
        )

    # Only instances every candidate produced a patch for. A half-populated
    # instance cannot enter a paired comparison, and silently scoring it 0 for
    # the candidate that never ran would invent a failure.
    gradeable = [i for i in chosen if all(cache.has_patch(c.label, c.text, i) for c in resolved)]
    if len(gradeable) != len(chosen):
        log(f"  grading {len(gradeable)} of {len(chosen)} instances ({len(chosen) - len(gradeable)} incomplete)")

    log("\n--- PHASE B: grading, instance-major (one image, every candidate) ---")
    images = DockerImagePool(rows=rows)
    graded_count = 0

    def on_instance(instance_id: str) -> None:
        nonlocal graded_count
        graded_count += 1
        running = []
        for cand in resolved:
            hits = sum(
                1
                for i in gradeable
                for rec in [cache.get(cand.label, cand.text, i)]
                if rec and rec.get("graded") and (rec.get("score") or 0) > 0
            )
            running.append(f"{cand.label} {hits}")
        log(
            f"  [B {graded_count:>4}/{len(gradeable)}] {instance_id:<40} "
            f"resolved: {' | '.join(running)}  ({(time.time() - t0) / 3600:.2f} h)"
        )

    try:
        grade_instances(
            instance_ids=gradeable,
            candidates=resolved,
            instances=instances,
            grader=grader,
            cache=cache,
            images=images,
            group_size=group_size,
            on_instance=on_instance,
        )
    except KeyboardInterrupt:
        log("\ninterrupted -- scores so far are in the durable cache; re-run to resume.")
        cache.close()
        return 130

    report = build_results(
        instance_ids=gradeable, candidates=resolved, cache=cache, difficulty=difficulty
    )
    report.update(
        {
            "n_requested": args.n,
            "n_graded": len(gradeable),
            "seed": args.seed,
            "manifest": str(args.test_manifest),
            "difficulty_mix": difficulty_mix(gradeable, difficulty),
            "budget_usd": args.budget,
            "budget_hit": phase_a.budget_hit,
            "realised_usd": round(spent, 4),
            "elapsed_hours": round((time.time() - t0) / 3600, 2),
            "solver_model": solver_model,
            "refiner_model": refiner_model,
            "harness": grader.summary(),
            "images_released": images.released,
        }
    )
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    cache.close()

    log("\n" + "=" * 72)
    for row in report["candidates"]:
        rate = "n/a" if row["resolve_rate"] is None else f"{row['resolve_rate']:.1%}"
        log(
            f"  {row['label']:<16} {row['resolved']:>3}/{row['n']:<4} = {rate:<7} "
            f"(val {row['val_score'] if row['val_score'] is not None else 'n/a'})  ${row['spend_usd']:.2f}"
        )
    log(f"  elapsed {report['elapsed_hours']:.1f} h, images released {images.released}")
    log(f"  wrote {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
