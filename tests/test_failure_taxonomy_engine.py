"""Drive ``TaxonomyFeedbackEnricher`` through the real ``gepa.optimize()``.

The end-to-end tests verify that the optimizer invokes the independent review
after the adapter creates its ordinary records and before reflection sees them.
The adapter itself is never wrapped or replaced.

No model is called: the reflection LM is a fixture returning a fixed
instruction, and the judge is scripted. The whole module is free to run.
"""

from __future__ import annotations

import gepa
import pytest
from gepa.core.adapter import EvaluationBatch

from failure_taxonomy import FAILURE_MODES_KEY, Occurrence, TaxonomyFeedbackEnricher

COMPONENTS = ("solver", "refiner")


class ToyAdapter:
    """A real, minimal GEPAAdapter: scores on a keyword, captures module calls."""

    propose_new_texts = None

    def __init__(self):
        self.reflective_datasets = []

    def evaluate(self, batch, candidate, capture_traces=False):
        outputs, scores, trajectories = [], [], []
        for item in batch:
            # Score rises when the solver instruction mentions the magic word,
            # so the engine has a real gradient to hill-climb on.
            score = 1.0 if "PRECISE" in candidate["solver"] else 0.0
            solver_out = f"answer for {item['id']} (precise={score == 1.0})"
            outputs.append(solver_out)
            scores.append(score)
            if capture_traces:
                trajectories.append(
                    {
                        "instance_id": item["id"],
                        "task": item["question"],
                        "module_calls": [
                            {"component": "solver", "prompt": candidate["solver"], "output": solver_out},
                            {"component": "refiner", "prompt": candidate["refiner"], "output": solver_out},
                        ],
                    }
                )
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        dataset = {
            component: [
                {"instance_id": t["instance_id"], "produced": t["module_calls"][0]["output"]}
                for t in eval_batch.trajectories
            ]
            for component in components_to_update
        }
        self.reflective_datasets.append(dataset)
        return dataset


class ScriptedJudge:
    candidate_key = ""

    def __init__(self):
        self.calls = 0

    def judge(self, traces):
        self.calls += 1
        return {t.trace_id: [Occurrence("A.1", "Vague_Instruction", "answer for ...", "solver")] for t in traces}


def _dataset(n=4):
    return [{"id": f"q{i}", "question": f"question {i}"} for i in range(n)]


def _reflection_lm(_prompt):
    # Whatever reflection is shown, it proposes the improved instruction.
    return "```\nBe PRECISE and answer directly.\n```"


@pytest.fixture
def seed():
    return {"solver": "Answer the question.", "refiner": "Improve the answer."}


def _run(adapter, seed, **kwargs):
    return gepa.optimize(
        seed_candidate=seed,
        trainset=_dataset(),
        valset=_dataset(),
        adapter=adapter,
        reflection_lm=_reflection_lm,
        max_metric_calls=40,
        reflection_minibatch_size=2,
        display_progress_bar=False,
        seed=0,
        **kwargs,
    )


def test_enricher_completes_a_real_optimize_run(seed):
    adapter = ToyAdapter()
    judge = ScriptedJudge()
    result = _run(adapter, seed, reflective_dataset_enricher=TaxonomyFeedbackEnricher(judge=judge))

    assert result.best_candidate["solver"]
    assert judge.calls > 0, "the judge was never reached inside a real run"


def test_the_diagnosis_actually_reaches_the_proposer(seed):
    """The end-to-end claim: a judged failure mode lands in the reflection prompt.

    Everything else could pass while the diagnosis is dropped somewhere between
    the review hook and the proposer, which would make the treatment arm a baseline
    arm wearing the wrong label -- discoverable only after the money is spent.
    """
    prompts: list[str] = []

    def recording_lm(prompt):
        prompts.append(prompt)
        return _reflection_lm(prompt)

    gepa.optimize(
        seed_candidate=seed,
        trainset=_dataset(),
        valset=_dataset(),
        adapter=ToyAdapter(),
        reflection_lm=recording_lm,
        reflective_dataset_enricher=TaxonomyFeedbackEnricher(judge=ScriptedJudge()),
        max_metric_calls=40,
        reflection_minibatch_size=2,
        display_progress_bar=False,
        seed=0,
    )

    assert prompts, "reflection never ran"
    assert any("Vague_Instruction" in p for p in prompts), "the failure mode never reached the proposer"
    assert any(FAILURE_MODES_KEY in p for p in prompts)


def test_the_enricher_does_not_mutate_the_adapter_dataset(seed):
    """Enrichment copies records instead of mutating adapter-owned data."""
    adapter = ToyAdapter()
    _run(adapter, seed, reflective_dataset_enricher=TaxonomyFeedbackEnricher(judge=ScriptedJudge()))

    enriched = [
        example
        for dataset in adapter.reflective_datasets
        for examples in dataset.values()
        for example in examples
        if FAILURE_MODES_KEY in example
    ]
    assert enriched == []


def test_baseline_run_never_calls_the_judge(seed):
    judge = ScriptedJudge()
    _run(ToyAdapter(), seed)
    assert judge.calls == 0


def test_engine_still_finds_the_optimum_with_the_enricher(seed):
    result = _run(
        ToyAdapter(),
        seed,
        reflective_dataset_enricher=TaxonomyFeedbackEnricher(judge=ScriptedJudge()),
    )
    assert "PRECISE" in result.best_candidate["solver"]
    assert max(result.val_aggregate_scores) == 1.0
