"""Held-out test evaluation of frozen candidates (scripts/eval_test.py).

Everything here is free: the LMs are stubs, the grader is a recorder, and no
container is ever started. What the tests pin are the four properties that a
6-hour, ~$50 measurement cannot afford to get wrong.

**Instance-major ordering.** The images are not pre-pulled and do not all fit;
two candidate-major passes would download every one of them twice at a measured
~46 GB/h. The ordering is not an optimisation, it is the difference between a
6-hour run and a 12-hour one, so it is asserted on the grader's actual call
sequence rather than trusted.

**Prefix-stable stratification.** ``--n 200`` next week must extend ``--n 150``
today, not resample it, or the already-paid instances are thrown away.

**The cache.** An interruption five hours in must re-pay nothing.

**Candidate selection.** Picking the wrong index measures a program nobody
asked about, and the numbers look entirely plausible either way.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

from gepa_taxonomy.adapter import SweBenchAdapter
from gepa_taxonomy.cost import CostMeter
from gepa_taxonomy.program import REFINER, SOLVER, RetrievedFile, SolverRefinerProgram
from gepa_taxonomy.seed_cache import candidate_hash
from gepa_taxonomy.tasks import GoldLeakError, split_row
from tests.test_gold_blindness import RAW_ROW

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "eval_test.py"

HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET = "global.anthropic.claude-sonnet-4-6"


def load_script():
    spec = importlib.util.spec_from_file_location("eval_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def et():
    return load_script()


# ---------------------------------------------------------------------------
# Fixtures: a real program with stub LMs, real Instances, a recording grader
# ---------------------------------------------------------------------------


def make_instances(n: int, *, repo: str = "django/django") -> dict[str, object]:
    """``n`` distinct instances built from the real dataset-row splitter.

    Real ``Instance`` objects, so ``Task``/``Gold`` separation -- and therefore
    the gold audit -- is the genuine one, not a stand-in.
    """
    out = {}
    for i in range(n):
        row = dict(RAW_ROW)
        row["instance_id"] = f"inst-{i:02d}"
        row["repo"] = repo
        out[row["instance_id"]] = split_row(row)
    return out


class StubLM:
    """Returns a patch that encodes the instruction it was given.

    The encoding matters: it is how the grading tests tell which candidate
    produced a patch, so the assertions are on content that actually flowed
    through the program rather than on call counts alone.
    """

    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def complete(self, prompt: str, *, max_tokens: int = 4096):
        with self.lock:
            self.calls += 1
        marker = "A" if "INSTRUCTION-A" in prompt else ("B" if "INSTRUCTION-B" in prompt else "?")
        return f"--- a/{marker}.py\n+++ b/{marker}.py\n@@\n-x\n+y\n", 1000, 100


class StubRetriever:
    def __init__(self, content: str = "def f():\n    return 1\n"):
        self.content = content

    def retrieve(self, task, *, k: int):
        return [RetrievedFile(path="pkg/mod.py", content=self.content)]


def build_program(lm=None, retriever=None):
    lm = lm or StubLM()
    return SolverRefinerProgram(
        retriever=retriever or StubRetriever(),
        solver_lm=lm,
        refiner_lm=lm,
        solver_meter=CostMeter(),
        refiner_meter=CostMeter(),
        solver_model=HAIKU,
        refiner_model=SONNET,
    )


class RecordingGrader:
    """Records the exact (instance, patch-marker) sequence the harness sees."""

    def __init__(self, resolve: set[str] | None = None):
        self.batches: list[list[str]] = []
        self.events: list[tuple[str, str]] = []
        self.resolve = resolve or set()

    def grade_batch(self, items):
        marker = _marker(items[0][2]) if items else "?"
        self.batches.append([t.instance_id for t, _g, _p in items])
        out = {}
        for task, _gold, patch in items:
            self.events.append((task.instance_id, _marker(patch)))
            key = f"{task.instance_id}:{_marker(patch)}"
            resolved = key in self.resolve
            out[task.instance_id] = (1.0 if resolved else 0.0, {"resolved": resolved, "marker": marker})
        return out


def _marker(patch: str) -> str:
    """'A' or 'B' -- which candidate's instruction produced this patch."""
    for m in ("A", "B"):
        if f"--- a/{m}.py" in patch:
            return m
    return "?"


class RecordingImagePool:
    """Records image residency events interleaved with grading."""

    def __init__(self, events: list):
        self.events = events
        self.released = 0

    def acquire(self, instance_ids):
        self.events.append(("acquire", tuple(instance_ids)))

    def release(self, instance_ids):
        self.events.append(("release", tuple(instance_ids)))
        self.released += len(instance_ids)
        return len(instance_ids)


def two_candidates(et):
    return [
        et.ResolvedCandidate(
            label="alpha",
            text={SOLVER: "INSTRUCTION-A solve", REFINER: "INSTRUCTION-A refine"},
            index=1,
            source=Path("cands.json"),
        ),
        et.ResolvedCandidate(
            label="beta",
            text={SOLVER: "INSTRUCTION-B solve", REFINER: "INSTRUCTION-B refine"},
            index=2,
            source=Path("cands.json"),
        ),
    ]


def make_auditor(program, instances):
    return SweBenchAdapter(program=program, grader=RecordingGrader(), instances=instances, strict_gold_check=True)


# ---------------------------------------------------------------------------
# Stratified selection
# ---------------------------------------------------------------------------

#: Deliberately skewed, like Verified's own difficulty column (116/163/20/1 in
#: the committed test manifest). A uniform pool would let a plain shuffle pass.
POOL = {
    **{f"e{i:03d}": "<15 min fix" for i in range(60)},
    **{f"m{i:03d}": "15 min - 1 hour" for i in range(30)},
    **{f"h{i:03d}": "1-4 hours" for i in range(10)},
}


def test_stratified_order_is_deterministic(et):
    a = et.stratified_order(POOL, POOL, seed=7)
    b = et.stratified_order(POOL, POOL, seed=7)
    assert a == b
    assert set(a) == set(POOL), "the ordering must be a permutation, not a sample"


def test_stratified_order_ignores_input_ordering(et):
    """Determinism must come from the seed, not from dict insertion order."""
    forward = et.stratified_order(list(POOL), POOL, seed=7)
    backward = et.stratified_order(list(reversed(list(POOL))), POOL, seed=7)
    assert forward == backward


def test_seed_actually_changes_the_draw(et):
    assert et.stratified_order(POOL, POOL, seed=7) != et.stratified_order(POOL, POOL, seed=8)


def test_first_n_is_a_strict_subset_of_a_larger_n(et):
    """Raising --n must EXTEND the measurement, not resample it.

    Every instance already paid for at n=25 has to still be there at n=90, or
    the cache misses and the money is spent again.
    """
    order = et.stratified_order(POOL, POOL, seed=7)
    for small, large in ((10, 25), (25, 50), (50, 90)):
        assert set(order[:small]) < set(order[:large])
        assert order[:small] == order[:large][:small]


def test_every_prefix_preserves_the_difficulty_mix(et):
    """The property a stratified *sample* per-n would not give us.

    Checked at every n, not a few: a plain seeded shuffle happens to land on
    the right mix at some sizes, and only the every-n form rules it out.
    """
    order = et.stratified_order(POOL, POOL, seed=7)
    share = {k: v / len(POOL) for k, v in Counter(POOL.values()).items()}
    for n in range(1, len(order) + 1):
        counts = Counter(POOL[i] for i in order[:n])
        for stratum, frac in share.items():
            assert abs(counts.get(stratum, 0) - frac * n) <= 1.0, (
                f"n={n} stratum={stratum}: got {counts.get(stratum, 0)}, expected ~{frac * n:.1f}"
            )


def test_committed_test_manifest_is_orderable(et):
    """The real manifest, with a difficulty map, produces a full permutation."""
    ids = json.loads((REPO_ROOT / "manifests/swebench_verified/test.json").read_text())["instance_ids"]
    # Difficulty is unknown to this test; the point is that an absent label is
    # bucketed rather than dropped, so `--n` never silently shrinks.
    order = et.stratified_order(ids, {}, seed=1)
    assert len(order) == len(ids) == 300
    assert set(order) == set(ids)


# ---------------------------------------------------------------------------
# Best-candidate selection
# ---------------------------------------------------------------------------

#: Shaped exactly like the committed runs: gepa's own lines, in gepa's order.
#: The winner (index 3) is neither the base candidate, nor the last candidate
#: discovered, nor the highest index -- so an implementation that takes any of
#: those shortcuts fails.
FIXTURE_LOG = """\
Iteration 0: Base program full valset score: 0.18333333333333332 over 60 / 60 examples
Iteration 1: Valset score for new program: 0.15
Iteration 1: Individual valset scores for new program: {0: 1.0, 1: 0.0}
Iteration 1: New program candidate index: 1
Iteration 4: Valset score for new program: 0.2
Iteration 4: New program candidate index: 2
Iteration 9: Valset score for new program: 0.31666666666666665
Iteration 9: New program candidate index: 3
Iteration 12: Valset score for new program: 0.21666666666666667
Iteration 12: New program candidate index: 4
"""


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "candidates.json").write_text(
        json.dumps([{SOLVER: f"s{i}", REFINER: f"r{i}"} for i in range(5)])
    )
    (d / "gepa.log").write_text(FIXTURE_LOG)
    return d


def test_best_candidate_is_read_from_the_log(et, run_dir):
    index, score, why = et.best_candidate_index(run_dir, 5)
    assert index == 3
    assert score == pytest.approx(0.31666666666666665)
    # The printed reason must carry the number it was decided on, so the choice
    # can be checked against the log without re-running anything.
    assert "0.3167" in why and "19/60 resolved" in why


def test_resolved_candidate_carries_the_right_text(et, run_dir):
    cand = et.resolve_candidate("winner", run_dir, None)
    assert cand.index == 3
    assert cand.text == {SOLVER: "s3", REFINER: "r3"}
    assert cand.hash == candidate_hash({SOLVER: "s3", REFINER: "r3"})
    assert cand.val_score == pytest.approx(0.31666666666666665)


def test_base_candidate_can_win(et, tmp_path):
    """Index 0 is only present in the log as the 'Base program' line."""
    d = tmp_path / "r"
    d.mkdir()
    (d / "candidates.json").write_text(json.dumps([{SOLVER: "s0"}, {SOLVER: "s1"}]))
    (d / "gepa.log").write_text(
        "Iteration 0: Base program full valset score: 0.5 over 60 / 60 examples\n"
        "Iteration 1: Valset score for new program: 0.1\n"
        "Iteration 1: New program candidate index: 1\n"
    )
    assert et.best_candidate_index(d, 2)[0] == 0


def test_quietlogger_compact_line_is_a_source(et, tmp_path):
    """console.log's rewritten form must count; a run may show only that."""
    d = tmp_path / "r"
    d.mkdir()
    (d / "candidates.json").write_text(json.dumps([{SOLVER: "a"}, {SOLVER: "b"}, {SOLVER: "c"}]))
    (d / "console.log").write_text(
        "Iteration 1: candidate 0 selected (val 11/60)\n"
        "Iteration 6: candidate 2 selected (val 21/60)\n"
    )
    index, score, _ = et.best_candidate_index(d, 3)
    assert index == 2
    assert score == pytest.approx(21 / 60)


def test_ties_go_to_the_lowest_index_and_say_so(et, tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "candidates.json").write_text(json.dumps([{SOLVER: "a"}, {SOLVER: "b"}, {SOLVER: "c"}]))
    (d / "gepa.log").write_text(
        "Iteration 0: Base program full valset score: 0.1 over 60 / 60 examples\n"
        "Iteration 1: Valset score for new program: 0.4\n"
        "Iteration 1: New program candidate index: 1\n"
        "Iteration 2: Valset score for new program: 0.4\n"
        "Iteration 2: New program candidate index: 2\n"
    )
    index, _score, why = et.best_candidate_index(d, 3)
    assert index == 1
    assert "TIED with [2]" in why, "an arbitrary choice must be announced, not hidden"


def test_missing_logs_fail_loudly(et, tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "candidates.json").write_text(json.dumps([{SOLVER: "a"}]))
    with pytest.raises(et.CandidateSelectionError) as exc:
        et.resolve_candidate("x", d, None)
    assert "#INDEX" in str(exc.value), "the error must say how to proceed"


def test_logs_without_val_scores_fail_loudly(et, tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "candidates.json").write_text(json.dumps([{SOLVER: "a"}]))
    (d / "gepa.log").write_text("Iteration 3: proposed for solver_instruction\n")
    with pytest.raises(et.CandidateSelectionError):
        et.resolve_candidate("x", d, None)


def test_out_of_range_index_fails_loudly(et, run_dir):
    with pytest.raises(et.CandidateSelectionError):
        et.resolve_candidate("x", run_dir, 99)


def test_explicit_index_needs_no_logs(et, tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "candidates.json").write_text(json.dumps([{SOLVER: "a"}, {SOLVER: "b"}]))
    cand = et.resolve_candidate("x", d / "candidates.json", 1)
    assert cand.text == {SOLVER: "b"}
    assert cand.val_score is None


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("base=results/runs/x", ("base", "results/runs/x", None)),
        ("base=results/runs/x#4", ("base", "results/runs/x", 4)),
        ("b=a/candidates.json#0", ("b", "a/candidates.json", 0)),
    ],
)
def test_candidate_spec_parsing(et, spec, expected):
    label, path, index = et.parse_candidate_spec(spec)
    # as_posix(), not str(): the parser returns a Path, whose str() uses the
    # platform separator, so str() would fail on Windows for a correct parse.
    assert (label, path.as_posix(), index) == expected


def test_candidate_spec_without_label_is_rejected(et):
    with pytest.raises(et.CandidateSelectionError):
        et.parse_candidate_spec("results/runs/x")


REAL_RUNS = REPO_ROOT / "results" / "runs"


@pytest.mark.skipif(not (REAL_RUNS / "baseline-seed1" / "gepa.log").exists(), reason="run artifacts are gitignored")
@pytest.mark.parametrize("run", ["baseline-seed1", "taxonomy-seed1"])
def test_agrees_with_gepas_own_verdict_on_real_runs(et, run):
    """Cross-check against the line gepa itself writes.

    ``Best program as per aggregate score on valset`` is gepa's own answer,
    computed from state rather than parsed from text. If our reconstruction
    disagrees with the last one of those, our parse is wrong.
    """
    d = REAL_RUNS / run
    text = et._read_run_logs(d)
    gepa_says = int(
        [ln for ln in text.splitlines() if "Best program as per aggregate score on valset:" in ln][-1]
        .split(":")[-1]
        .strip()
    )
    n = len(json.loads((d / "candidates.json").read_text()))
    assert et.best_candidate_index(d, n)[0] == gepa_says


# ---------------------------------------------------------------------------
# Durable cache
# ---------------------------------------------------------------------------

CAND = {SOLVER: "s", REFINER: "r"}


def test_cache_roundtrips_patch_then_score(et, tmp_path):
    c = et.EvalCache.open(tmp_path / "rollouts.jsonl")
    c.put_patch("alpha", CAND, "i1", patch="PATCH-TEXT", cost_usd=0.07)
    assert c.get("alpha", CAND, "i1")["patch"] == "PATCH-TEXT"
    assert c.is_graded("alpha", CAND, "i1") is False
    c.put_score("alpha", CAND, "i1", score=1.0, detail={"resolved": True})
    rec = c.get("alpha", CAND, "i1")
    assert rec["score"] == 1.0 and rec["graded"] is True
    assert rec["patch"] == "PATCH-TEXT", "the score record must not lose the patch"
    c.close()


def test_cache_is_keyed_on_label_and_hash_and_instance(et, tmp_path):
    c = et.EvalCache.open(tmp_path / "r.jsonl")
    c.put_patch("alpha", CAND, "i1", patch="P", cost_usd=0.01)
    assert c.get("beta", CAND, "i1") is None, "a different label must not reuse a patch"
    assert c.get("alpha", {SOLVER: "s", REFINER: "CHANGED"}, "i1") is None, (
        "repointing a label at a different candidate must invalidate its rows"
    )
    assert c.get("alpha", CAND, "i2") is None
    c.close()


def test_cache_survives_reopen(et, tmp_path):
    p = tmp_path / "r.jsonl"
    c1 = et.EvalCache.open(p)
    c1.put_patch("alpha", CAND, "i1", patch="P1", cost_usd=0.07)
    c1.put_score("alpha", CAND, "i1", score=1.0, detail={})
    c1.put_patch("alpha", CAND, "i2", patch="P2", cost_usd=0.07)
    c1.close()

    c2 = et.EvalCache.open(p)
    assert len(c2) == 2
    assert c2.get("alpha", CAND, "i1")["score"] == 1.0
    assert c2.get("alpha", CAND, "i2")["patch"] == "P2"
    assert c2.is_graded("alpha", CAND, "i2") is False, "an ungraded rollout must come back ungraded"
    assert c2.spend_usd() == pytest.approx(0.14)
    c2.close()


def test_watchdog_sees_each_dollar_exactly_once(et, tmp_path):
    """seed_watchdog.sh sums `.cost_usd` over the file.

    The graded line repeats the whole record, so if it carried the price again
    the watchdog would see double the real spend and kill a run at half its
    budget.
    """
    p = tmp_path / "r.jsonl"
    c = et.EvalCache.open(p)
    c.put_patch("alpha", CAND, "i1", patch="P", cost_usd=0.07)
    c.put_score("alpha", CAND, "i1", score=0.0, detail={})
    c.put_patch("beta", CAND, "i1", patch="Q", cost_usd=0.03)
    c.close()

    lines = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    assert len(lines) == 3
    assert sum(r["cost_usd"] for r in lines) == pytest.approx(0.10)
    assert c.spend_usd() == pytest.approx(0.10)
    assert c.spend_usd("alpha") == pytest.approx(0.07)


def test_truncated_final_line_is_survivable(et, tmp_path):
    p = tmp_path / "r.jsonl"
    c1 = et.EvalCache.open(p)
    c1.put_patch("alpha", CAND, "i1", patch="P", cost_usd=0.07)
    c1.close()
    with p.open("a") as fh:
        fh.write('{"label": "alpha", "candidate_ha')

    c2 = et.EvalCache.open(p)
    assert len(c2) == 1 and c2.truncated_records == 1
    assert c2.get("alpha", CAND, "i1")["patch"] == "P"
    c2.close()


def test_score_without_a_patch_is_refused(et, tmp_path):
    c = et.EvalCache.open(tmp_path / "r.jsonl")
    with pytest.raises(KeyError):
        c.put_score("alpha", CAND, "i1", score=1.0, detail={})
    c.close()


# ---------------------------------------------------------------------------
# Phase A -- generation, caching, budget, gold
# ---------------------------------------------------------------------------


def test_cache_prevents_recomputation(et, tmp_path):
    """The resume property: a second pass must issue zero LM calls."""
    instances = make_instances(3)
    lm = StubLM()
    program = build_program(lm)
    cands = two_candidates(et)
    cache = et.EvalCache.open(tmp_path / "r.jsonl")

    first = et.generate_patches(
        instance_ids=list(instances),
        candidates=cands,
        instances=instances,
        program=program,
        cache=cache,
        auditor=make_auditor(program, instances),
        max_workers=2,
    )
    assert first.executed == 6 and first.cached == 0
    assert lm.calls == 12  # 2 LM calls per rollout, fixed by design
    patches = {(c.label, i): cache.get(c.label, c.text, i)["patch"] for c in cands for i in instances}
    cache.close()

    # A fresh process, the same directory -- the shape of a real resume.
    lm2 = StubLM()
    program2 = build_program(lm2)
    cache2 = et.EvalCache.open(tmp_path / "r.jsonl")
    second = et.generate_patches(
        instance_ids=list(instances),
        candidates=cands,
        instances=instances,
        program=program2,
        cache=cache2,
        auditor=make_auditor(program2, instances),
        max_workers=2,
    )
    assert second.executed == 0 and second.cached == 6
    assert lm2.calls == 0, "a resumed run must not re-pay for a completed rollout"
    for key, patch in patches.items():
        assert cache2.get(key[0], dict(next(c for c in cands if c.label == key[0]).text), key[1])["patch"] == patch
    cache2.close()


def test_each_candidate_gets_its_own_patch(et, tmp_path):
    """Two labels must not collapse onto one rollout."""
    instances = make_instances(2)
    program = build_program()
    cands = two_candidates(et)
    cache = et.EvalCache.open(tmp_path / "r.jsonl")
    et.generate_patches(
        instance_ids=list(instances),
        candidates=cands,
        instances=instances,
        program=program,
        cache=cache,
        auditor=make_auditor(program, instances),
        max_workers=1,
    )
    for iid in instances:
        assert _marker(cache.get("alpha", cands[0].text, iid)["patch"]) == "A"
        assert _marker(cache.get("beta", cands[1].text, iid)["patch"]) == "B"
    cache.close()


def test_budget_stops_between_instances_leaving_pairs_intact(et, tmp_path):
    """A partially-generated instance cannot enter a paired comparison.

    So the stopper is consulted at instance boundaries: whatever is in the
    cache when it fires has a patch from EVERY candidate.
    """
    instances = make_instances(6)
    program = build_program()
    cands = two_candidates(et)
    cache = et.EvalCache.open(tmp_path / "r.jsonl")

    calls = {"n": 0}

    def stopper():
        calls["n"] += 1
        return calls["n"] > 2  # allow two instances through, then stop

    result = et.generate_patches(
        instance_ids=list(instances),
        candidates=cands,
        instances=instances,
        program=program,
        cache=cache,
        auditor=make_auditor(program, instances),
        max_workers=1,
        stopper=stopper,
    )
    assert result.budget_hit is True
    assert len(result.skipped_for_budget) == 4
    for iid in instances:
        have = [c.label for c in cands if cache.has_patch(c.label, c.text, iid)]
        assert have in ([], ["alpha", "beta"]), f"{iid} is half-populated: {have}"
    assert result.executed == 4
    cache.close()


def test_budget_stopper_reads_total_not_budgeted_spend(et):
    """The trap this script has to avoid.

    ``MaxTotalCostStopper`` compares ``budgeted_usd``, which counts only the
    'optimization' phase. This script books to 'final_test', so handing it the
    raw meters would show it $0.00 forever and the ceiling would never fire.
    """
    from gepa_taxonomy.cost import MaxTotalCostStopper

    meter = CostMeter()
    meter.record(model=HAIKU, input_tokens=1_000_000, output_tokens=1_000_000, phase="final_test")
    assert meter.budgeted_usd == 0.0, "premise: final_test spend is not 'budgeted'"
    assert meter.total_usd > 0

    naive = MaxTotalCostStopper(1.0, meters=[meter])
    assert naive() is False, "premise: the raw meter never fires the stopper"

    guarded = MaxTotalCostStopper(1.0, meters=[et.TotalSpendMeter(meter)])
    assert guarded.realised_usd == pytest.approx(meter.total_usd)
    assert guarded() is True


def test_gold_never_reaches_the_program(et, tmp_path):
    """The adapter's own ``_audit`` still guards this path.

    Retrieval is made to leak the reference patch, which is the realistic leak:
    it is the one component that reads the repository. The run must die rather
    than produce a score.
    """
    instances = make_instances(1)
    program = build_program(retriever=StubRetriever(content=RAW_ROW["patch"]))
    cache = et.EvalCache.open(tmp_path / "r.jsonl")
    with pytest.raises(GoldLeakError):
        et.generate_patches(
            instance_ids=list(instances),
            candidates=two_candidates(et),
            instances=instances,
            program=program,
            cache=cache,
            auditor=make_auditor(program, instances),
            max_workers=1,
        )
    cache.close()


def test_same_repo_rollouts_never_run_concurrently(et, tmp_path):
    """A hazard this script creates and run_seed.py does not.

    ``ensure_checkout`` force-checks-out ONE shared directory per repo, so two
    threads on the same repo move the working tree under each other: retrieval
    indexes the wrong commit and the apply verdict is against the wrong tree.
    gepa drives rollouts serially, so the per-repo lock exists only here.
    """
    instances = make_instances(6, repo="django/django")
    overlaps = []
    active = {"n": 0}
    lock = threading.Lock()

    class SlowLM(StubLM):
        def complete(self, prompt, *, max_tokens=4096):
            with lock:
                active["n"] += 1
                if active["n"] > 1:
                    overlaps.append(active["n"])
            time.sleep(0.02)
            with lock:
                active["n"] -= 1
            return super().complete(prompt, max_tokens=max_tokens)

    program = build_program(SlowLM())
    cache = et.EvalCache.open(tmp_path / "r.jsonl")
    et.generate_patches(
        instance_ids=list(instances),
        candidates=two_candidates(et),
        instances=instances,
        program=program,
        cache=cache,
        auditor=make_auditor(program, instances),
        max_workers=6,
    )
    assert overlaps == [], f"same-repo rollouts overlapped {len(overlaps)} times"
    cache.close()


# ---------------------------------------------------------------------------
# Phase B -- instance-major ordering
# ---------------------------------------------------------------------------


def graded_fixture(et, tmp_path, n=3, resolve=None):
    instances = make_instances(n)
    program = build_program()
    cands = two_candidates(et)
    cache = et.EvalCache.open(tmp_path / "r.jsonl")
    et.generate_patches(
        instance_ids=list(instances),
        candidates=cands,
        instances=instances,
        program=program,
        cache=cache,
        auditor=make_auditor(program, instances),
        max_workers=1,
    )
    return instances, cands, cache, RecordingGrader(resolve=resolve)


def test_grader_sees_every_candidate_before_next_instance(et, tmp_path):
    """THE property. Two candidate-major passes would re-download every image.

    Asserted on the harness's actual call sequence, keyed by which candidate's
    instruction produced the patch, so an implementation that merely reorders
    its bookkeeping cannot pass.
    """
    instances, cands, cache, grader = graded_fixture(et, tmp_path, n=3)
    events = []
    et.grade_instances(
        instance_ids=list(instances),
        candidates=cands,
        instances=instances,
        grader=grader,
        cache=cache,
        images=RecordingImagePool(events),
        group_size=1,
    )
    assert grader.events == [
        ("inst-00", "A"),
        ("inst-00", "B"),
        ("inst-01", "A"),
        ("inst-01", "B"),
        ("inst-02", "A"),
        ("inst-02", "B"),
    ]
    cache.close()


def test_image_is_held_across_all_candidates_then_released(et, tmp_path):
    """The download economy, stated as an ordering on residency events."""
    instances, cands, cache, grader = graded_fixture(et, tmp_path, n=2)
    events = []
    pool = RecordingImagePool(events)

    real_grade = grader.grade_batch

    def spy(items):
        events.append(("grade", tuple(t.instance_id for t, _g, _p in items)))
        return real_grade(items)

    grader.grade_batch = spy
    et.grade_instances(
        instance_ids=list(instances),
        candidates=cands,
        instances=instances,
        grader=grader,
        cache=cache,
        images=pool,
        group_size=1,
    )
    assert events == [
        ("acquire", ("inst-00",)),
        ("grade", ("inst-00",)),
        ("grade", ("inst-00",)),
        ("release", ("inst-00",)),
        ("acquire", ("inst-01",)),
        ("grade", ("inst-01",)),
        ("grade", ("inst-01",)),
        ("release", ("inst-01",)),
    ]
    cache.close()


def test_grouping_never_leaks_an_instance_past_its_release(et, tmp_path):
    """group_size>1 restores harness parallelism without weakening the order.

    Every candidate of every instance in a group is graded before that group's
    images go, and no later instance appears before then.
    """
    instances, cands, cache, grader = graded_fixture(et, tmp_path, n=5)
    events = []
    et.grade_instances(
        instance_ids=list(instances),
        candidates=cands,
        instances=instances,
        grader=grader,
        cache=cache,
        images=RecordingImagePool(events),
        group_size=2,
    )
    assert grader.batches == [
        ["inst-00", "inst-01"],
        ["inst-00", "inst-01"],
        ["inst-02", "inst-03"],
        ["inst-02", "inst-03"],
        ["inst-04"],
        ["inst-04"],
    ]
    assert [e for e in events if e[0] == "release"] == [
        ("release", ("inst-00", "inst-01")),
        ("release", ("inst-02", "inst-03")),
        ("release", ("inst-04",)),
    ]
    # Each instance's two gradings are adjacent in the per-instance event log.
    seen = [i for i, _m in grader.events]
    assert seen == ["inst-00", "inst-01", "inst-00", "inst-01", "inst-02", "inst-03", "inst-02", "inst-03",
                    "inst-04", "inst-04"]
    cache.close()


def test_already_graded_instances_are_not_regraded(et, tmp_path):
    instances, cands, cache, grader = graded_fixture(et, tmp_path, n=2)
    kwargs = {
        "instance_ids": list(instances),
        "candidates": cands,
        "instances": instances,
        "cache": cache,
        "group_size": 1,
    }
    et.grade_instances(grader=grader, images=RecordingImagePool([]), **kwargs)
    assert len(grader.events) == 4

    again = RecordingGrader()
    events = []
    et.grade_instances(grader=again, images=RecordingImagePool(events), **kwargs)
    assert again.events == [], "a resumed run must not re-enter Docker"
    assert events == [], "and must not pull an image it does not need"
    cache.close()


def test_empty_patches_never_reach_docker(et, tmp_path):
    """run_evaluation.py:458-470 refuses them anyway; pulling 4 GB first is waste."""
    instances = make_instances(1)
    cands = two_candidates(et)
    cache = et.EvalCache.open(tmp_path / "r.jsonl")
    for cand in cands:
        cache.put_patch(cand.label, cand.text, "inst-00", patch="   ", cost_usd=0.05)

    grader = RecordingGrader()
    events = []
    et.grade_instances(
        instance_ids=["inst-00"],
        candidates=cands,
        instances=instances,
        grader=grader,
        cache=cache,
        images=RecordingImagePool(events),
        group_size=1,
    )
    assert grader.events == [] and events == []
    rec = cache.get("alpha", cands[0].text, "inst-00")
    assert rec["graded"] is True and rec["score"] == 0.0 and rec["grading"]["empty_patch"] is True
    cache.close()


def test_results_carry_a_paired_per_instance_matrix(et, tmp_path):
    instances, cands, cache, grader = graded_fixture(
        et, tmp_path, n=3, resolve={"inst-00:A", "inst-01:A", "inst-01:B"}
    )
    et.grade_instances(
        instance_ids=list(instances),
        candidates=cands,
        instances=instances,
        grader=grader,
        cache=cache,
        images=RecordingImagePool([]),
        group_size=1,
    )
    report = et.build_results(
        instance_ids=list(instances),
        candidates=cands,
        cache=cache,
        difficulty={"inst-00": "1-4 hours"},
    )
    alpha, beta = report["candidates"]
    assert (alpha["label"], alpha["resolved"], alpha["n"]) == ("alpha", 2, 3)
    assert (beta["label"], beta["resolved"], beta["n"]) == ("beta", 1, 3)
    assert alpha["resolve_rate"] == pytest.approx(2 / 3)
    assert alpha["spend_usd"] > 0, "realised spend must be reported per candidate"
    # The matrix a paired test needs: same instances, one column per candidate.
    assert [r["scores"] for r in report["instances"]] == [
        {"alpha": 1.0, "beta": 0.0},
        {"alpha": 1.0, "beta": 1.0},
        {"alpha": 0.0, "beta": 0.0},
    ]
    assert report["instances"][0]["difficulty"] == "1-4 hours"
    assert report["instances"][2]["difficulty"] == et.UNKNOWN_DIFFICULTY
    cache.close()


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


def test_dry_run_prints_the_config_and_spends_nothing(et, tmp_path, monkeypatch, capsys, run_dir):
    """Free, and provably so: the credentials it would need are not there.

    ``require_credentials()`` is the first thing past the dry-run branch, so a
    dry run that reached any billable code would raise instead of returning 0.
    """
    from gepa_taxonomy.bedrock import BEARER_ENV

    monkeypatch.delenv(BEARER_ENV, raising=False)
    manifest = tmp_path / "test.json"
    ids = [f"i{n:03d}" for n in range(20)]
    manifest.write_text(json.dumps({"name": "test", "n": len(ids), "instance_ids": ids}))
    monkeypatch.setattr(
        et, "load_difficulty", lambda _ids: {i: ("1-4 hours" if n % 4 == 0 else "<15 min fix") for n, i in enumerate(ids)}
    )
    out = tmp_path / "out"

    rc = et.main(
        [
            "--candidates", f"winner={run_dir}",
            "--n", "8",
            "--test-manifest", str(manifest),
            "--out", str(out),
            "--budget", "12.5",
            "--dry-run",
        ]
    )
    assert rc == 0
    text = capsys.readouterr().out

    # The resolved config, not a banner: the audit trail for which program ran.
    assert "candidates.json[3]" in text
    assert "0.3167" in text and "19/60 resolved" in text
    assert "instances     8 of 20" in text
    assert "$12.50" in text
    assert "would run     8 rollouts" in text
    assert "nothing spent" in text

    # Nothing was evaluated and nothing was billed.
    assert not (out / "results.json").exists()
    assert (out / "rollouts.jsonl").read_text() == ""

    # The plan it printed is on disk and is the stratified prefix.
    plan = json.loads((out / "instances.json").read_text())
    assert plan["n"] == 8 and plan["seed"] == et.DEFAULT_SEED
    assert plan["instance_ids"] == plan["order"][:8]
    assert plan["difficulty_mix"] == {"1-4 hours": 2, "<15 min fix": 6}
    # The WHOLE manifest order is recorded, not just the slice taken. That is
    # what makes "raise --n later" a documented extension of this run rather
    # than a re-derivation someone has to trust.
    assert len(plan["order"]) == plan["manifest_n"] == 20
    assert set(plan["order"]) == set(ids)
    # And a larger n really does extend this one.
    assert set(plan["instance_ids"]) < set(plan["order"][:12])


def test_dry_run_reports_what_the_cache_already_covers(et, tmp_path, monkeypatch, capsys, run_dir):
    """A resume must be visible before it is launched, not discovered at $0.00."""
    from gepa_taxonomy.bedrock import BEARER_ENV

    monkeypatch.delenv(BEARER_ENV, raising=False)
    ids = [f"i{n:03d}" for n in range(4)]
    manifest = tmp_path / "test.json"
    manifest.write_text(json.dumps({"instance_ids": ids}))
    monkeypatch.setattr(et, "load_difficulty", lambda _ids: dict.fromkeys(ids, "<15 min fix"))
    out = tmp_path / "out"
    out.mkdir()

    cand = et.resolve_candidate("winner", run_dir, None)
    c = et.EvalCache.open(out / "rollouts.jsonl")
    order = et.stratified_order(ids, dict.fromkeys(ids, "<15 min fix"))
    c.put_patch("winner", cand.text, order[0], patch="P", cost_usd=0.09)
    c.close()

    argv = ["--candidates", f"winner={run_dir}", "--n", "4",
            "--test-manifest", str(manifest), "--out", str(out), "--dry-run"]
    assert et.main(argv) == 0
    text = capsys.readouterr().out
    assert "resuming      1 cached rollouts ($0.09 will not be re-paid)" in text
    assert "would run     3 rollouts, 1 already cached" in text


def test_refuses_when_n_exceeds_the_manifest(et, tmp_path, monkeypatch, capsys, run_dir):
    ids = [f"i{n:03d}" for n in range(5)]
    manifest = tmp_path / "test.json"
    manifest.write_text(json.dumps({"instance_ids": ids}))
    monkeypatch.setattr(et, "load_difficulty", lambda _ids: dict.fromkeys(ids, "<15 min fix"))
    rc = et.main(
        ["--candidates", f"w={run_dir}", "--n", "50", "--test-manifest", str(manifest),
         "--out", str(tmp_path / "o"), "--dry-run"]
    )
    assert rc == 2
    assert "exceeds the 5 instances" in capsys.readouterr().out


def test_duplicate_labels_are_refused(et, run_dir):
    """Labels key the cache and the report; two of them would silently merge."""
    with pytest.raises(et.CandidateSelectionError):
        et.main(["--candidates", f"x={run_dir}", "--candidates", f"x={run_dir}", "--dry-run"])


def test_predictions_path_is_absolute(tmp_path):
    """The harness runs with cwd=work_dir. A relative predictions path resolves
    against work_dir twice and is not found -- it scored 150 instances 0 with
    harness_error on the first test-eval run."""
    from gepa_taxonomy.grading import LocalDockerGrader

    rel = Path("results/test_eval/x/harness")
    grader = LocalDockerGrader(work_dir=rel, dry_run=True)
    grader.work_dir.mkdir(parents=True, exist_ok=True)
    written = (grader.work_dir / "probe-predictions.json").resolve()
    assert written.is_absolute()
    # the grader must build the same absolute shape it writes
    assert (grader.work_dir / "any-predictions.json").resolve().is_absolute()
