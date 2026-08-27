"""IFBench splits. FREE: synthetic records for the generator, manifests for the rest.

The property that matters most here is not a size or a ratio -- it is that
train/val and test come from **different pools with disjoint constraint
vocabularies**. An earlier version split the 300-instance IFBench test set
three ways, which trained the optimizer on the out-of-distribution constraints it
was then scored against. Both arms would have shared that leak, so the A/B would
not have been biased -- it would simply have measured constraint memorisation
instead of the generalisation IFBench exists to test.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from gepa_taxonomy.ifbench.splits import (
    TEST_DATASET,
    TRAIN_DATASET,
    TRAIN_SIZE,
    VAL_SIZE,
    build_test,
    build_train_val,
    n_constraints_of,
)

MANIFESTS = Path(__file__).resolve().parents[1] / "demo" / "manifests"


def _pool(n: int = 2000):
    """IF-RLVR shaped: ``ground_truth`` is a STRING holding a list of dicts."""
    records = []
    for i in range(n):
        k = (i % 5) + 1  # 1..5 constraints, matching the real distribution's spread
        ids = [f"detectable_format:c{j}" for j in range(k)]
        kwargs = [None] + [{"x": 1}] * (k - 1)
        records.append(
            {
                "key": f"row_{i}",
                "messages": [{"role": "user", "content": f"Task {i}."}],
                "ground_truth": str([{"instruction_id": ids, "kwargs": kwargs}]),
            }
        )
    return records


class TestConstraintCount:
    def test_parses_the_stringified_ground_truth(self):
        assert n_constraints_of(_pool(5)[2]) == 3

    def test_a_malformed_ground_truth_counts_zero_rather_than_raising(self):
        assert n_constraints_of({"ground_truth": "not a literal ["}) == 0
        assert n_constraints_of({}) == 0


class TestTrainVal:
    def test_sizes_are_exact(self):
        m = build_train_val(_pool())
        assert (m["train"].n, m["val"].n) == (TRAIN_SIZE, VAL_SIZE)

    def test_train_and_val_are_disjoint(self):
        m = build_train_val(_pool())
        assert not (set(m["train"].example_ids) & set(m["val"].example_ids))

    def test_the_same_seed_gives_byte_identical_manifests(self):
        a, b = build_train_val(_pool(), seed=5), build_train_val(_pool(), seed=5)
        assert {k: v.to_json() for k, v in a.items()} == {k: v.to_json() for k, v in b.items()}

    def test_a_different_seed_gives_a_different_partition(self):
        assert set(build_train_val(_pool(), seed=5)["val"].example_ids) != set(
            build_train_val(_pool(), seed=6)["val"].example_ids
        )

    def test_multi_constraint_share_matches_between_train_and_val(self):
        """Partial credit is what keeps the acceptance gate informative,
        and it lives entirely in the multi-constraint instances."""
        pool = {str(r["key"]): r for r in _pool()}
        m = build_train_val(_pool())

        def share(name):
            ids = m[name].example_ids
            return sum(1 for i in ids if n_constraints_of(pool[i]) > 1) / len(ids)

        assert share("train") == pytest.approx(share("val"), abs=0.05)

    def test_both_manifests_name_the_training_dataset(self):
        m = build_train_val(_pool())
        assert m["train"].dataset == m["val"].dataset == TRAIN_DATASET

    def test_too_small_a_pool_raises_rather_than_silently_shrinking(self):
        with pytest.raises(ValueError, match="need"):
            build_train_val(_pool(100))


class TestTest:
    def test_test_takes_the_whole_ifbench_set_and_names_it(self):
        records = [{"key": i, "instruction_id_list": ["count:x"]} for i in range(300)]
        manifest = build_test(records)
        assert manifest.n == 300
        assert manifest.dataset == TEST_DATASET


class TestCommittedManifests:
    """Guards on the shipped artifacts, not just the generator."""

    def _load(self, name):
        if not MANIFESTS.exists():
            pytest.skip("manifests not built")
        return json.loads((MANIFESTS / f"{name}.json").read_text(encoding="utf-8"))

    def test_train_and_val_come_from_the_training_pool_and_test_does_not(self):
        """The whole point of the recorded requirement. If test were drawn from the same pool, the
        optimizer would see the OOD constraints it is scored on."""
        assert self._load("train")["dataset"] == TRAIN_DATASET
        assert self._load("val")["dataset"] == TRAIN_DATASET
        assert self._load("test")["dataset"] == TEST_DATASET

    def test_sizes_match_the_demo_setup(self):
        # The committed manifests are the runnable demo: 10/10/10, small enough
        # that the from-zero pipeline costs cents. The study used 150/300/300.
        assert self._load("train")["n"] == 10
        assert self._load("val")["n"] == 10
        assert self._load("test")["n"] == 10

    def test_train_and_val_are_disjoint(self):
        train, val = set(self._load("train")["example_ids"]), set(self._load("val")["example_ids"])
        assert not (train & val)

    def test_the_two_constraint_vocabularies_are_disjoint(self):
        """Asserted rather than assumed: if upstream ever adds an IFBench
        constraint to the training set, test stops being out-of-distribution and
        the benchmark quietly stops measuring what it is for."""
        from gepa_taxonomy.ifbench._vendor.ifbench import instructions_registry as ifb
        from gepa_taxonomy.ifbench._vendor.ifevalg import instructions_registry as ifg

        overlap = set(ifb.INSTRUCTION_DICT) & set(ifg.INSTRUCTION_DICT)
        assert not overlap, f"vocabularies overlap: {sorted(overlap)[:5]}"
        assert len(ifb.INSTRUCTION_DICT) == 58
        assert len(ifg.INSTRUCTION_DICT) == 54

    def test_val_carries_real_partial_credit(self):
        """Selection happens on val. The '85% binary' limitation is a property of
        the TEST set only -- val is drawn from IF-RLVR, where most instances carry
        several constraints."""
        pytest.importorskip("datasets")
        from datasets import load_dataset

        ids = set(self._load("val")["example_ids"])
        pool = {str(r["key"]): r for r in load_dataset(TRAIN_DATASET, split="train") if str(r["key"]) in ids}
        counts = collections.Counter(n_constraints_of(pool[i]) for i in ids if i in pool)
        multi = sum(v for k, v in counts.items() if k > 1) / max(1, sum(counts.values()))
        assert multi > 0.5, f"val is mostly single-constraint ({multi:.0%}); the acceptance gate loses granularity"
