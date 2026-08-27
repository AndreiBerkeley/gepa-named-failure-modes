"""The HoVer arm. FREE: no network, no LLM, no index.

Weighted toward the two things that would fail silently rather than loudly:
HoVer's supporting_facts shape differs from HotpotQA's, and title matching has
to survive the corpus and the dataset disagreeing about punctuation.
"""

from __future__ import annotations

import json

import pytest

from gepa_taxonomy.hover.grading import grade, normalize_title, retrieval_feedback, score_feedback
from gepa_taxonomy.hover.program import COMPONENTS, SEED_CANDIDATE
from gepa_taxonomy.hover.splits import DEFAULT_SIZES, build_splits, stratum_of
from gepa_taxonomy.hover.tasks import Gold, instance_from_record

RECORD = {
    "uid": "abc-123",
    "claim": "Skagen Painter Peder Severin Kroyer favored naturalism.",
    "supporting_facts": [["Kristian Zahrtmann", 0], ["Kristian Zahrtmann", 1], ["Ossian Elgstrom", 2]],
    "label": "SUPPORTED",
    "num_hops": 3,
}


class TestTasks:
    def test_hover_supporting_facts_shape_is_list_of_pairs(self):
        """HotpotQA's extractor expects a dict of parallel lists and returns ()
        on HoVer's list-of-pairs. That failure is SILENT -- every rollout would
        score against no gold and look like total retrieval failure."""
        inst = instance_from_record(RECORD)
        assert inst.gold.titles == ("Kristian Zahrtmann", "Ossian Elgstrom"), "titles deduplicated, order preserved"

    def test_task_carries_no_gold(self):
        """Structural gold-blindness: there is no field gold could travel on."""
        inst = instance_from_record(RECORD)
        assert not hasattr(inst.task, "titles")
        assert not hasattr(inst.task, "label")
        with pytest.raises((AttributeError, TypeError)):
            inst.task.titles = ("leak",)  # type: ignore[misc]

    def test_label_is_kept_on_gold_but_is_not_the_metric(self):
        inst = instance_from_record(RECORD)
        assert inst.gold.label == "SUPPORTED"

    def test_normalised_pair_form_also_works(self):
        rec = dict(RECORD, supporting_facts=[{"key": "Some Title", "value": 0}])
        assert instance_from_record(rec).gold.titles == ("Some Title",)


class TestTitleNormalisation:
    @pytest.mark.parametrize(
        "a,b",
        [
            ("Kristian Zahrtmann", "kristian zahrtmann"),
            ("Peder_Severin_Kroyer", "Peder Severin Kroyer"),
            ("Mercury (planet)", "Mercury planet"),
            ("  Spaced   Out  ", "Spaced Out"),
        ],
    )
    def test_corpus_and_dataset_title_forms_match(self, a, b):
        """They disagree on underscores, case and parenthetical punctuation.
        Raw string comparison silently UNDER-counts retrieval, which reads as a
        weak program rather than a normalisation bug."""
        assert normalize_title(a) == normalize_title(b)


class TestGrading:
    def test_strict_score_needs_every_gold_document(self):
        gold = Gold(example_id="x", titles=("A", "B", "C"))
        g = grade(gold, [["A", "B"], ["Z"], []])
        assert g.score == 0.0, "2 of 3 is not partial credit"
        assert g.loose_recall == pytest.approx(2 / 3)
        assert g.missing == ("c",)

    def test_full_retrieval_scores_one(self):
        gold = Gold(example_id="x", titles=("A", "B"))
        g = grade(gold, [["A"], ["B"]])
        assert g.score == 1.0 and g.all_found and g.missing == ()

    def test_a_document_found_early_still_counts(self):
        """The program returns the union of all hops, so hop 3 missing a
        document hop 1 already found must not lose it."""
        gold = Gold(example_id="x", titles=("A",))
        assert grade(gold, [["A"], ["Z"], ["Y"]]).score == 1.0

    def test_per_hop_is_cumulative_for_attribution(self):
        gold = Gold(example_id="x", titles=("A", "B"))
        g = grade(gold, [["A"], ["B"], []])
        assert g.per_hop_found == (("a",), ("a", "b"), ("a", "b"))

    def test_a_record_with_no_gold_scores_zero_and_says_so(self):
        """Scoring 1.0 would reward a broken record; scoring a silent 0.0 would
        look like a program failure. It is surfaced instead."""
        g = grade(Gold(example_id="x", titles=()), [["A"]])
        assert g.score == 0.0 and "no gold titles" in g.missing[0]


class TestFeedbackGoldSafety:
    def test_score_feedback_names_no_document(self):
        """Used on val/test, where naming a missing document hands over the
        answer."""
        g = grade(Gold(example_id="x", titles=("Secret Article", "B")), [["B"]])
        text = score_feedback(g)
        assert "Secret Article" not in text
        assert "1 of 2" in text and "1 still missing" in text

    def test_retrieval_feedback_names_documents_for_train_only(self):
        g = grade(Gold(example_id="x", titles=("Secret Article",)), [[]])
        assert "secret article" in retrieval_feedback(g, "summarize1").lower()

    def test_hop1_bottleneck_is_called_out(self):
        g = grade(Gold(example_id="x", titles=("A",)), [[], [], []])
        assert "Hop 1 retrieved none" in retrieval_feedback(g, "create_query_hop2")


class TestProgram:
    def test_four_components_three_hops(self):
        assert COMPONENTS == ("summarize1", "create_query_hop2", "summarize2", "create_query_hop3")
        assert set(SEED_CANDIDATE) == set(COMPONENTS)

    def test_seed_prompts_are_plain_signature_defaults(self):
        """Seeding from an optimized prompt would start the baseline from an
        already-searched point and destroy the comparison."""
        for text in SEED_CANDIDATE.values():
            assert text.startswith("Given the fields")
            assert len(text) < 120, "a long seed prompt is an already-optimized one"

    def test_last_module_writes_a_query_not_an_answer(self):
        """This is the single difference from our HotpotQA port."""
        assert "query" in SEED_CANDIDATE["create_query_hop3"]
        assert "answer" not in SEED_CANDIDATE["create_query_hop3"]


def _pool(n: int = 900):
    # 50/34/17 across 2/3/4 hops, roughly HoVer's real proportions.
    out = []
    for i in range(n):
        hops = 2 if i % 6 < 3 else (3 if i % 6 < 5 else 4)
        out.append({"uid": f"id-{i:04d}", "num_hops": hops, "claim": f"claim {i}"})
    return out


class TestSplits:
    def test_disjoint_and_exactly_sized(self):
        splits = build_splits(_pool())
        ids = {n: set(s.example_ids) for n, s in splits.items()}
        assert {n: len(s) for n, s in ids.items()} == DEFAULT_SIZES
        assert not (ids["train"] & ids["val"]) and not (ids["val"] & ids["test"]) and not (ids["train"] & ids["test"])

    def test_deterministic_at_a_fixed_seed(self):
        a = build_splits(_pool())["val"].to_json()
        b = build_splits(_pool())["val"].to_json()
        assert json.dumps(a) == json.dumps(b)

    def test_hop_distribution_is_preserved(self):
        """4-hop claims need four specific documents on an all-or-nothing
        metric. If val and test differ on hop mix they measure different things
        and their scores are not comparable."""
        pool = _pool()
        by_id = {r["uid"]: r for r in pool}
        splits = build_splits(pool)
        base = {}
        for h in (2, 3, 4):
            base[h] = sum(1 for r in pool if r["num_hops"] == h) / len(pool)
        for name in ("val", "test"):
            ids = splits[name].example_ids
            for h in (2, 3, 4):
                share = sum(1 for i in ids if by_id[i]["num_hops"] == h) / len(ids)
                assert abs(share - base[h]) < 0.05, f"{name} hop-{h} share {share:.3f} vs pool {base[h]:.3f}"

    def test_manifest_ids_are_sorted(self):
        """gepa keys val subscores positionally, so manifest order is
        load-bearing and must not depend on upstream ordering."""
        ids = build_splits(_pool())["val"].to_json()["example_ids"]
        assert ids == sorted(ids)

    def test_refuses_an_undersized_pool(self):
        with pytest.raises(ValueError, match="need 750"):
            build_splits(_pool(100))

    def test_stratum_key(self):
        assert stratum_of({"num_hops": 3}) == "3"
        assert stratum_of({}) == "unknown"
