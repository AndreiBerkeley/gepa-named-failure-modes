"""Tests for the HotpotQA arm. All offline: no model, no index required."""

from __future__ import annotations

import pytest
from gepa.core.adapter import EvaluationBatch

from failure_taxonomy import FAILURE_MODES_KEY, Occurrence, TaxonomyFeedbackEnricher, extract_calls
from gepa_taxonomy.hotpotqa.adapter import HotpotQAAdapter, instances_by_id
from gepa_taxonomy.hotpotqa.grading import (
    answer_f1,
    grade,
    normalize_answer,
    retrieval_feedback,
)
from gepa_taxonomy.hotpotqa.program import (
    COMPONENTS,
    CREATE_QUERY_HOP2,
    FINAL_ANSWER,
    SEED_CANDIDATE,
    SUMMARIZE1,
    SUMMARIZE2,
    MultiHopProgram,
)
from gepa_taxonomy.hotpotqa.retrieval import Passage
from gepa_taxonomy.hotpotqa.splits import build_splits, stratum_of
from gepa_taxonomy.hotpotqa.tasks import Gold, GoldLeakError, Task, assert_gold_free, instance_from_record

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def _pool(n=2000):
    levels, types = ["easy", "medium", "hard"], ["bridge", "comparison"]
    return [{"id": f"q{i}", "level": levels[i % 3], "type": types[i % 2]} for i in range(n)]


def test_splits_are_disjoint_exact_and_reproducible():
    a = build_splits(_pool(), seed=7)
    b = build_splits(_pool(), seed=7)
    assert {k: v.n for k, v in a.items()} == {"train": 150, "val": 300, "test": 300}
    ids = [set(m.example_ids) for m in a.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert {k: v.to_json() for k, v in a.items()} == {k: v.to_json() for k, v in b.items()}


def test_a_different_seed_gives_different_splits():
    a, b = build_splits(_pool(), seed=1), build_splits(_pool(), seed=2)
    assert set(a["val"].example_ids) != set(b["val"].example_ids)


def test_splits_are_stratified_by_level_and_type():
    pool = _pool()
    manifests = build_splits(pool, seed=3)
    by_id = {r["id"]: r for r in pool}
    for manifest in manifests.values():
        share = sum(1 for i in manifest.example_ids if stratum_of(by_id[i])[1] == "comparison") / manifest.n
        assert 0.4 <= share <= 0.6, f"{manifest.name} comparison share {share:.2%} off 50%"


def test_manifest_ids_are_sorted():
    """gepa keys val subscores and the Pareto frontier POSITIONALLY (F014), so a
    manifest's order is load-bearing and must be stable."""
    manifest = build_splits(_pool(), seed=5)["val"]
    assert manifest.to_json()["example_ids"] == sorted(manifest.example_ids)


def test_too_small_a_pool_is_an_error_not_a_short_split():
    with pytest.raises(ValueError, match="need 750"):
        build_splits(_pool(100), seed=1)


# ---------------------------------------------------------------------------
# Gold blindness
# ---------------------------------------------------------------------------


def test_task_cannot_carry_gold():
    task = Task(example_id="q1", question="who?")
    with pytest.raises((AttributeError, TypeError)):
        task.answer = "leaked"  # frozen + slots


def test_record_is_split_into_gold_free_and_gold_halves():
    instance = instance_from_record(
        {
            "id": "q1",
            "question": "Were X and Y the same nationality?",
            "answer": "yes",
            "level": "hard",
            "type": "comparison",
            "supporting_facts": {"title": ["Alpha", "Alpha", "Beta"], "sent_id": [0, 1, 0]},
        }
    )
    assert instance.task.question.startswith("Were X")
    assert instance.gold.answer == "yes"
    # Titles deduplicated: HotpotQA lists one row per supporting SENTENCE.
    assert instance.gold.titles == ("Alpha", "Beta")
    assert not hasattr(instance.task, "answer")


def test_gold_title_in_a_prompt_is_a_hard_failure():
    gold = Gold(example_id="q1", answer="yes", titles=("Scott Derrickson",))
    with pytest.raises(GoldLeakError, match="Scott Derrickson"):
        assert_gold_free("summarise these: Scott Derrickson directed ...", gold, where="test prompt")


def test_short_gold_answers_do_not_trip_the_leak_check():
    """'yes' appears in innocent prose constantly; flagging it would make the
    check unusable, so only distinctive answers are matched."""
    gold = Gold(example_id="q1", answer="yes", titles=())
    assert_gold_free("Answer yes or no based on the passages.", gold, where="test prompt")


def test_long_gold_answers_are_caught():
    gold = Gold(example_id="q1", answer="The Chronicles of Narnia", titles=())
    with pytest.raises(GoldLeakError):
        assert_gold_free("hint: The Chronicles of Narnia", gold, where="test prompt")


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pred", "gold", "expected"),
    [
        ("The Beatles", "the beatles", 1.0),
        ("yes", "yes", 1.0),
        ("yes", "no", 0.0),
        ("", "something", 0.0),
    ],
)
def test_answer_f1_basics(pred, gold, expected):
    assert answer_f1(pred, gold) == pytest.approx(expected)


def test_f1_gives_partial_credit_which_is_the_whole_point():
    """A binary metric at a low base rate is what made the SWE-Bench minibatch a
    coin flip; partial credit is why HotpotQA can select at all."""
    score = answer_f1("Robert Downey Junior", "Robert Downey Jr.")
    assert 0.0 < score < 1.0


def test_yes_no_answers_are_compared_exactly_not_by_overlap():
    assert answer_f1("yes it is", "yes") == 0.0


def test_normalisation_strips_articles_and_punctuation():
    assert normalize_answer("The  Beatles!") == "beatles"


def test_grade_reports_retrieval_recall_and_missing_titles():
    gold = Gold(example_id="q1", answer="Paris", titles=("Alpha", "Beta"))
    g = grade("Paris", ["alpha", "Gamma"], gold)
    assert g.score == 1.0
    assert g.retrieval_recall == 0.5
    assert g.missing_titles == ("Beta",)


def test_retrieval_recall_is_diagnostic_not_the_score():
    """Optimising recall directly would reward retrieving gold rather than answering."""
    gold = Gold(example_id="q1", answer="Paris", titles=("Alpha", "Beta"))
    perfect_retrieval_wrong_answer = grade("London", ["Alpha", "Beta"], gold)
    assert perfect_retrieval_wrong_answer.retrieval_recall == 1.0
    assert perfect_retrieval_wrong_answer.score == 0.0


def test_feedback_names_the_missing_documents():
    """This is the BASELINE arm's feedback and it is deliberately strong: the
    taxonomy arm has to beat it, not a weakened version of it."""
    gold = Gold(example_id="q1", answer="x", titles=("Alpha", "Beta"))
    text = retrieval_feedback(["Alpha"], gold)
    assert "Gold documents retrieved so far (1/2): Alpha" in text
    assert "still to be retrieved: Beta" in text


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------


class FakeLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt, *, max_tokens=1024):
        self.prompts.append(prompt)
        return (self.replies.pop(0), 100, 20)


class FakeRetriever:
    def __init__(self, passages):
        self.passages = passages
        self.queries = []

    def retrieve(self, query, *, k=None):
        self.queries.append(query)
        return list(self.passages)


class FakeMeter:
    def record(self, *, model, input_tokens, output_tokens, phase):
        return 0.001


def _program(replies, passages=None):
    return MultiHopProgram(
        retriever=FakeRetriever(passages if passages is not None else [Passage("Alpha", "a body")]),
        lm=FakeLM(replies),
        meter=FakeMeter(),
        model="fake",
    )


def test_seed_candidate_matches_the_papers_base_prompts_verbatim():
    """Transcribed from GEPA appendix L. Seeding from the paper's *optimized*
    prompts instead would start the baseline from an already-searched point."""
    assert SEED_CANDIDATE[SUMMARIZE1] == "Given the fields `question`, `passages`, produce the fields `summary`."
    assert SEED_CANDIDATE[CREATE_QUERY_HOP2] == "Given the fields `question`, `summary_1`, produce the fields `query`."
    assert (
        SEED_CANDIDATE[SUMMARIZE2]
        == "Given the fields `question`, `context`, `passages`, produce the fields `summary`."
    )
    assert (
        SEED_CANDIDATE[FINAL_ANSWER]
        == "Given the fields `question`, `summary_1`, `summary_2`, produce the fields `answer`."
    )
    assert set(SEED_CANDIDATE) == set(COMPONENTS)


def test_program_makes_exactly_four_calls_in_order():
    """Fixed cost per rollout: a variable-cost program would give one seed more
    iterations than another under the same dollar budget."""
    program = _program(["summary one", "the hop 2 query", "summary two", "Paris"])
    rollout = program.run(Task("q1", "who?"), SEED_CANDIDATE)
    assert [c.component for c in rollout.calls] == [SUMMARIZE1, CREATE_QUERY_HOP2, SUMMARIZE2, FINAL_ANSWER]
    assert rollout.answer == "Paris"
    assert rollout.query_hop2 == "the hop 2 query"


def test_second_hop_retrieves_with_the_generated_query():
    program = _program(["s1", "MY QUERY", "s2", "ans"])
    program.run(Task("q1", "who?"), SEED_CANDIDATE)
    assert program.retriever.queries == ["who?", "MY QUERY"]


def test_trace_carries_full_prompts_not_digests():
    """A digest-only trace cannot be judged and cannot seed taxonomy generation
    -- a mistake this project already paid for once (F012)."""
    program = _program(["s1", "q", "s2", "ans"])
    trace = program.run(Task("q1", "who?"), SEED_CANDIDATE).to_trace()
    calls = trace["module_calls"]
    assert len(calls) == 4
    assert all(c["prompt"] and c["output"] for c in calls)
    assert "Given the fields" in calls[0]["prompt"]


def test_trace_satisfies_the_failure_taxonomy_contract():
    program = _program(["s1", "q", "s2", "ans"])
    trace = program.run(Task("q1", "who?"), SEED_CANDIDATE).to_trace()
    assert [c.component for c in extract_calls(trace)] == list(COMPONENTS)


def test_retrieved_titles_are_deduplicated_across_hops():
    program = _program(["s1", "q", "s2", "ans"], passages=[Passage("Alpha", "x"), Passage("Alpha", "x")])
    rollout = program.run(Task("q1", "who?"), SEED_CANDIDATE)
    assert rollout.retrieved_titles == ("Alpha",)


def test_gold_titles_flowing_through_the_pipeline_do_not_break_a_rollout():
    """The bug that killed seed 1 at a 1.5% val score.

    A summarizer naming the entity it just read about is correct behaviour, and
    on HotpotQA that entity IS a gold supporting-fact title. Under the old
    value-based audit this raised at ``create_query_hop2``, the adapter scored it
    0.0, and the run produced an all-zero result indistinguishable from a real
    negative. Gold blindness is structural here instead (F027).
    """
    program = _program(
        ["Scott Derrickson is an American director.", "q", "s2", "Paris"],
        passages=[Passage("Scott Derrickson", "is an American director.")],
    )
    rollout = program.run(Task("q1", "same nationality?"), SEED_CANDIDATE)
    assert rollout.answer == "Paris"
    # The title really did propagate into a later prompt -- and that is fine.
    assert "Scott Derrickson" in rollout.calls[1].prompt


def test_the_program_cannot_receive_gold_at_all():
    """The structural guarantee that replaces the audit: `run` takes a Task, and
    a Task has no field gold could travel on."""
    import inspect

    params = inspect.signature(MultiHopProgram.run).parameters
    assert set(params) == {"self", "task", "candidate", "phase"}
    assert not hasattr(MultiHopProgram, "gold_for"), "gold must not be reachable from the program"
    assert set(Task.__slots__) == {"example_id", "question", "level", "type"}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def _instances():
    return instances_by_id(
        [
            instance_from_record(
                {
                    "id": "q1",
                    "question": "who?",
                    "answer": "Paris",
                    "level": "hard",
                    "type": "bridge",
                    "supporting_facts": {"title": ["Alpha", "Beta"], "sent_id": [0, 0]},
                }
            )
        ]
    )


def _adapter(replies, **kw):
    instances = _instances()
    program = _program(replies, passages=[Passage("Alpha", "a body")])
    return HotpotQAAdapter(program=program, instances=instances, **kw), instances


def test_adapter_returns_a_real_evaluation_batch():
    """Not a look-alike: a 3-field stand-in for gepa's 5-field dataclass killed
    a launch once, after paid calls (F013)."""
    adapter, instances = _adapter(["s1", "q", "s2", "Paris"])
    batch = adapter.evaluate(list(instances.values()), SEED_CANDIDATE, capture_traces=True)
    assert isinstance(batch, EvaluationBatch)
    assert batch.scores == [1.0]
    assert batch.num_metric_calls == 1
    assert len(batch.trajectories) == 1


def test_adapter_declares_propose_new_texts():
    """gepa reads this attribute unconditionally; omitting it makes every
    reflection silently fail (D016)."""
    adapter, _ = _adapter(["s1", "q", "s2", "a"])
    assert adapter.propose_new_texts is None


def test_a_failing_rollout_scores_zero_rather_than_raising():
    class Boom:
        def retrieve(self, query, *, k=None):
            raise RuntimeError("index down")

    instances = _instances()
    program = _program(["s1", "q", "s2", "a"])
    program.retriever = Boom()
    adapter = HotpotQAAdapter(program=program, instances=instances)
    batch = adapter.evaluate(list(instances.values()), SEED_CANDIDATE, capture_traces=True)
    assert batch.scores == [0.0]
    assert "index down" in batch.trajectories[0]["error"]


class _Throttled(Exception):
    """Stands in for litellm's RateLimitError without importing its internals."""

    def __init__(self):
        super().__init__("RateLimitError: Too many requests, please retry")


def _adapter_whose_lm_raises(exc, **kw):
    instances = _instances()
    program = _program(["s1", "q", "s2", "a"])

    class Boom:
        def retrieve(self, query, *, k=None):
            raise exc

    program.retriever = Boom()
    return HotpotQAAdapter(program=program, instances=instances, **kw), instances


def test_transport_errors_are_counted_apart_from_program_errors():
    """A 0.0 that means 'Bedrock throttled us' is indistinguishable to the
    optimizer from 'this candidate is bad'. It has to be visible."""
    adapter, instances = _adapter_whose_lm_raises(_Throttled())
    adapter.evaluate(list(instances.values()), SEED_CANDIDATE, capture_traces=True)
    assert adapter.transport_errors == 1
    assert adapter.program_errors == 0

    other, instances2 = _adapter_whose_lm_raises(ValueError("bad candidate output"))
    other.evaluate(list(instances2.values()), SEED_CANDIDATE, capture_traces=True)
    assert other.transport_errors == 0
    assert other.program_errors == 1


def test_a_storm_of_transport_errors_aborts_rather_than_scoring_zeros():
    """Silently degrading a paid run is worse than stopping it."""
    adapter, instances = _adapter_whose_lm_raises(_Throttled(), max_transport_errors=3)
    batch = list(instances.values()) * 10
    with pytest.raises(RuntimeError, match="failed to reach the model"):
        adapter.evaluate(batch, SEED_CANDIDATE, capture_traces=True)


def test_a_single_transport_error_still_scores_zero_and_continues():
    """gepa's contract: never raise for ONE example failure."""
    adapter, instances = _adapter_whose_lm_raises(_Throttled(), max_transport_errors=25)
    batch = adapter.evaluate(list(instances.values()), SEED_CANDIDATE, capture_traces=True)
    assert batch.scores == [0.0]
    assert "RateLimitError" in batch.trajectories[0]["error"]


def test_error_counts_reach_the_run_summary():
    adapter, instances = _adapter_whose_lm_raises(_Throttled())
    adapter.evaluate(list(instances.values()), SEED_CANDIDATE, capture_traces=True)
    assert adapter.summary()["transport_errors"] == 1


def test_parallel_evaluation_preserves_batch_order():
    """Load-bearing: gepa keys val subscores and the Pareto frontier POSITIONALLY
    (F014). Assembling results in completion order would attach every score to
    the wrong instance -- silently, and only on the concurrent path."""
    import random
    import time

    n = 24
    records = [
        {
            "id": f"q{i}",
            "question": f"question {i}?",
            "answer": f"answer{i}",
            "level": "hard",
            "type": "bridge",
            "supporting_facts": {"title": ["Alpha"], "sent_id": [0]},
        }
        for i in range(n)
    ]
    instances = [instance_from_record(r) for r in records]

    class JitterLM:
        """Finishes out of order, so completion order != submission order."""

        def complete(self, prompt, *, max_tokens=1024):
            time.sleep(random.uniform(0, 0.02))
            # Echo the question number so the answer identifies its instance.
            num = prompt.split("question ")[-1].split("?")[0]
            return (f"answer{num}", 10, 5)

    program = MultiHopProgram(
        retriever=FakeRetriever([Passage("Alpha", "body")]),
        lm=JitterLM(),
        meter=FakeMeter(),
        model="fake",
    )
    adapter = HotpotQAAdapter(program=program, instances=instances_by_id(instances), max_workers=8)
    batch = adapter.evaluate(instances, SEED_CANDIDATE, capture_traces=True)

    assert [t["example_id"] for t in batch.trajectories] == [f"q{i}" for i in range(n)]
    assert batch.scores == [1.0] * n, "a score landed on the wrong instance"
    assert adapter.rollouts == n


def test_parallel_and_serial_agree():
    instances = [
        instance_from_record(
            {
                "id": f"q{i}",
                "question": f"q{i}?",
                "answer": "Paris",
                "level": "hard",
                "type": "bridge",
                "supporting_facts": {"title": ["Alpha"], "sent_id": [0]},
            }
        )
        for i in range(10)
    ]

    def run(workers):
        program = MultiHopProgram(
            retriever=FakeRetriever([Passage("Alpha", "body")]),
            lm=FakeLM(["s1", "q", "s2", "Paris"] * 10),
            meter=FakeMeter(),
            model="fake",
        )
        adapter = HotpotQAAdapter(program=program, instances=instances_by_id(instances), max_workers=workers)
        return adapter.evaluate(instances, SEED_CANDIDATE).scores

    assert run(1) == run(8)


def test_a_transport_storm_still_aborts_on_the_concurrent_path():
    """The abort is raised inside a worker; swallowing it there would let a
    throttled run keep scoring zeros."""
    instances = [
        instance_from_record(
            {
                "id": f"q{i}",
                "question": "q?",
                "answer": "a",
                "level": "hard",
                "type": "bridge",
                "supporting_facts": {"title": ["Alpha"], "sent_id": [0]},
            }
        )
        for i in range(20)
    ]

    class Throttled:
        def retrieve(self, query, *, k=None):
            raise _Throttled

    program = _program(["s1", "q", "s2", "a"])
    program.retriever = Throttled()
    adapter = HotpotQAAdapter(
        program=program,
        instances=instances_by_id(instances),
        max_workers=4,
        max_transport_errors=3,
    )
    with pytest.raises(RuntimeError, match="failed to reach the model"):
        adapter.evaluate(instances, SEED_CANDIDATE)


# ---------------------------------------------------------------------------
# Shared base-val replay (D009)
# ---------------------------------------------------------------------------


def _seed_cache_for(instances, answers):
    """Build a cache as scripts/build_hotpotqa_base_val.py would."""
    from gepa_taxonomy.seed_cache import SeedEvaluationCache

    entries = {}
    for inst, answer in zip(instances, answers, strict=True):
        entries[inst.task.example_id] = {
            "score": 1.0,
            "trace": {
                "example_id": inst.task.example_id,
                "answer": answer,
                "retrieved_titles": ["Alpha", "Beta"],
                "module_calls": [
                    {"component": c, "prompt": f"{c} prompt", "output": f"{c} out"} for c in COMPONENTS
                ],
            },
        }
    return SeedEvaluationCache.build(dict(SEED_CANDIDATE), entries)


def test_base_candidate_val_rollouts_are_replayed_without_any_lm_call():
    """D009: every seed and both arms must start from identical state. A live
    re-evaluation re-samples it -- two real runs of the same candidate measured
    56.0% and 56.5%."""
    instances = _instances_n(4)
    cache = _seed_cache_for(instances, ["Paris"] * 4)

    program = _program([], passages=[Passage("Alpha", "body")])
    adapter = HotpotQAAdapter(
        program=program, instances=instances_by_id(instances), seed_cache=cache
    )
    batch = adapter.evaluate(instances, SEED_CANDIDATE, capture_traces=True)

    assert adapter.replayed == 4
    assert program.lm.prompts == [], "a replayed rollout must issue no LM call"
    assert batch.scores == [1.0] * 4
    assert adapter.spend_usd == 0.0, "replay must contribute no spend (D013a)"


def test_replayed_traces_keep_full_prompts_so_they_remain_judgeable():
    """The known defect of the SWE-Bench rollout cache was that cached traces
    kept prompt digests, so a replayed rollout could not be judged at all."""
    instances = _instances_n(1)
    cache = _seed_cache_for(instances, ["Paris"])
    adapter = HotpotQAAdapter(
        program=_program([]), instances=instances_by_id(instances), seed_cache=cache
    )
    batch = adapter.evaluate(instances, SEED_CANDIDATE, capture_traces=True)

    calls = extract_calls(batch.trajectories[0])
    assert [c.component for c in calls] == list(COMPONENTS)
    assert all(c.prompt and c.output for c in calls)


def test_a_mutated_candidate_is_never_replayed():
    """Scope is (base candidate) x (val instances). Replaying a mutated
    candidate would return the base program's results for it."""
    instances = _instances_n(2)
    cache = _seed_cache_for(instances, ["Paris"] * 2)
    mutated = {**SEED_CANDIDATE, SUMMARIZE1: "an improved instruction"}

    program = _program(["s1", "q", "s2", "London"] * 2, passages=[Passage("Alpha", "b")])
    adapter = HotpotQAAdapter(
        program=program, instances=instances_by_id(instances), seed_cache=cache
    )
    adapter.evaluate(instances, mutated)
    assert adapter.replayed == 0
    assert program.lm.prompts, "the mutated candidate must run live"


def test_a_train_instance_miss_falls_through_instead_of_raising():
    """The base candidate is legitimately evaluated on TRAIN minibatches during
    reflective mutation. Treating that miss as an incomplete cache killed a run
    (F016); completeness is asserted once at launch instead."""
    val = _instances_n(2)
    cache = _seed_cache_for(val, ["Paris"] * 2)
    train = [
        instance_from_record(
            {
                "id": "train-1",
                "question": "t?",
                "answer": "Paris",
                "level": "hard",
                "type": "bridge",
                "supporting_facts": {"title": ["Alpha"], "sent_id": [0]},
            }
        )
    ]
    program = _program(["s1", "q", "s2", "Paris"], passages=[Passage("Alpha", "b")])
    adapter = HotpotQAAdapter(
        program=program, instances=instances_by_id(train), seed_cache=cache
    )
    batch = adapter.evaluate(train, SEED_CANDIDATE)  # base candidate, TRAIN instance
    assert adapter.replayed == 0
    assert batch.scores == [1.0]


def test_assert_covers_catches_an_incomplete_cache_at_launch():
    instances = _instances_n(3)
    cache = _seed_cache_for(instances[:2], ["Paris"] * 2)
    with pytest.raises(KeyError, match="missing"):
        cache.assert_covers(i.task.example_id for i in instances)


def _instances_n(n):
    return [
        instance_from_record(
            {
                "id": f"q{i}",
                "question": f"question {i}?",
                "answer": "Paris",
                "level": "hard",
                "type": "bridge",
                "supporting_facts": {"title": ["Alpha", "Beta"], "sent_id": [0, 0]},
            }
        )
        for i in range(n)
    ]


def test_retrieval_modules_get_retrieval_feedback_and_answerer_gets_answer_feedback():
    adapter, instances = _adapter(["s1", "q", "s2", "London"])
    batch = adapter.evaluate(list(instances.values()), SEED_CANDIDATE, capture_traces=True)
    dataset = adapter.make_reflective_dataset(SEED_CANDIDATE, batch, list(COMPONENTS))

    assert "still to be retrieved: Beta" in dataset[SUMMARIZE1][0]["feedback"]
    assert "Second-hop search query issued: q" in dataset[CREATE_QUERY_HOP2][0]["feedback"]
    assert "Answer F1" in dataset[FINAL_ANSWER][0]["feedback"]


def test_gold_answer_reaches_reflection_only_for_train_ids():
    adapter, instances = _adapter(["s1", "q", "s2", "London"], reflection_gold_ids=frozenset({"q1"}))
    batch = adapter.evaluate(list(instances.values()), SEED_CANDIDATE, capture_traces=True)
    dataset = adapter.make_reflective_dataset(SEED_CANDIDATE, batch, [FINAL_ANSWER])
    assert "Correct answer: Paris" in dataset[FINAL_ANSWER][0]["feedback"]

    held_out, instances2 = _adapter(["s1", "q", "s2", "London"], reflection_gold_ids=frozenset())
    batch2 = held_out.evaluate(list(instances2.values()), SEED_CANDIDATE, capture_traces=True)
    dataset2 = held_out.make_reflective_dataset(SEED_CANDIDATE, batch2, [FINAL_ANSWER])
    assert "Correct answer" not in dataset2[FINAL_ANSWER][0]["feedback"]


def test_each_component_sees_its_own_output():
    adapter, instances = _adapter(["SUM ONE", "THE QUERY", "SUM TWO", "THE ANSWER"])
    batch = adapter.evaluate(list(instances.values()), SEED_CANDIDATE, capture_traces=True)
    dataset = adapter.make_reflective_dataset(SEED_CANDIDATE, batch, list(COMPONENTS))
    assert dataset[SUMMARIZE1][0]["produced"] == "SUM ONE"
    assert dataset[CREATE_QUERY_HOP2][0]["produced"] == "THE QUERY"
    assert dataset[SUMMARIZE2][0]["produced"] == "SUM TWO"
    assert dataset[FINAL_ANSWER][0]["produced"] == "THE ANSWER"


# ---------------------------------------------------------------------------
# Integration with the optimizer-side enricher
# ---------------------------------------------------------------------------


class ScriptedJudge:
    candidate_key = ""

    def judge(self, traces):
        return {
            t.trace_id: [Occurrence("A.1", "Missed_Second_Hop", "the hop 2 query", CREATE_QUERY_HOP2)] for t in traces
        }


def test_taxonomy_enricher_routes_to_the_right_hotpotqa_component():
    adapter, instances = _adapter(["s1", "q", "s2", "London"])
    batch = adapter.evaluate(list(instances.values()), SEED_CANDIDATE, capture_traces=True)
    baseline = adapter.make_reflective_dataset(SEED_CANDIDATE, batch, list(COMPONENTS))
    dataset = TaxonomyFeedbackEnricher(judge=ScriptedJudge())(
        candidate=SEED_CANDIDATE,
        eval_batch=batch,
        components_to_update=list(COMPONENTS),
        reflective_dataset=baseline,
    )

    assert dataset[CREATE_QUERY_HOP2][0][FAILURE_MODES_KEY] == [
        {"name": "Missed_Second_Hop", "evidence": "the hop 2 query"}
    ]
    assert FAILURE_MODES_KEY not in dataset[SUMMARIZE1][0]


def test_baseline_arm_has_no_taxonomy_feedback_without_the_enricher():
    adapter, instances = _adapter(["s1", "q", "s2", "London"])
    batch = adapter.evaluate(list(instances.values()), SEED_CANDIDATE, capture_traces=True)
    bare = adapter.make_reflective_dataset(SEED_CANDIDATE, batch, list(COMPONENTS))
    assert all(FAILURE_MODES_KEY not in example for examples in bare.values() for example in examples)
