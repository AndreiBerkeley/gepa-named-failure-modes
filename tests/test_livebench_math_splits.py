"""LiveBench-Math splits. FREE: synthetic records, no download.

The committed manifests are a stage artifact, so the properties that make
them usable -- disjoint, exactly sized, stratified, and byte-identical at a fixed
seed -- are pinned here rather than trusted.
"""

from __future__ import annotations

import collections

import pytest

from gepa_taxonomy.livebench_math.splits import DEFAULT_SIZES, build_splits, stratum_of
from gepa_taxonomy.livebench_math.tasks import is_included


def _pool():
    """A synthetic stand-in with the real pool's shape: 117 MC / 29 AIME / 72 olympiad."""
    records = []
    for i in range(117):
        records.append({"question_id": f"mc{i}", "task": "math_comp", "subtask": "amc_12a_2023"})
    for i in range(29):
        records.append({"question_id": f"ai{i}", "task": "math_comp", "subtask": "aime_i_2024"})
    for i in range(72):
        records.append({"question_id": f"ol{i}", "task": "olympiad", "subtask": "usamo"})
    return records


class TestStratum:
    def test_amc_variants_share_a_stratum_because_they_share_a_scorer(self):
        keys = {
            stratum_of({"question_id": "x", "task": "math_comp", "subtask": s})
            for s in ("amc_12a_2023", "updated_amc_12b_2023", "smc")
        }
        assert keys == {("math_comp", "multiple_choice")}

    def test_aime_and_olympiad_are_separate(self):
        assert stratum_of({"task": "math_comp", "subtask": "aime_ii_2024"}) == ("math_comp", "aime")
        assert stratum_of({"task": "olympiad", "subtask": "imo"}) == ("olympiad", "proof_reorder")


class TestInclusion:
    def test_amps_hard_is_excluded(self):
        assert not is_included({"task": "AMPS_Hard", "subtask": "amps_hard_gcd"})

    def test_the_two_kept_tasks_are_included(self):
        assert is_included({"task": "math_comp"})
        assert is_included({"task": "olympiad"})


class TestSplits:
    def test_sizes_are_exact(self):
        manifests = build_splits(_pool())
        assert {n: m.n for n, m in manifests.items()} == DEFAULT_SIZES

    def test_splits_are_disjoint(self):
        manifests = build_splits(_pool())
        seen: set[str] = set()
        for manifest in manifests.values():
            ids = set(manifest.example_ids)
            assert not (seen & ids)
            seen |= ids
        assert len(seen) == sum(DEFAULT_SIZES.values())

    def test_the_same_seed_gives_byte_identical_manifests(self):
        a = build_splits(_pool(), seed=7)
        b = build_splits(_pool(), seed=7)
        assert {n: m.to_json() for n, m in a.items()} == {n: m.to_json() for n, m in b.items()}

    def test_a_different_seed_gives_a_different_partition(self):
        a = build_splits(_pool(), seed=7)["val"]
        b = build_splits(_pool(), seed=8)["val"]
        assert set(a.example_ids) != set(b.example_ids)

    def test_val_and_test_have_the_same_composition(self):
        """val drives selection and test is the headline. If their scorer mixes
        differ they measure different things, and val's partial-credit share --
        which is what keeps minibatch comparisons informative -- becomes a
        matter of luck."""
        pool = {str(r["question_id"]): r for r in _pool()}
        manifests = build_splits(_pool())

        def shares(name):
            counts = collections.Counter(stratum_of(pool[i]) for i in manifests[name].example_ids)
            total = sum(counts.values())
            return {k: v / total for k, v in counts.items()}

        val, test = shares("val"), shares("test")
        assert set(val) == set(test)
        for key in val:
            assert val[key] == pytest.approx(test[key], abs=0.02), f"{key} differs: {val[key]:.3f} vs {test[key]:.3f}"

    def test_every_stratum_appears_in_every_split(self):
        pool = {str(r["question_id"]): r for r in _pool()}
        manifests = build_splits(_pool())
        for name, manifest in manifests.items():
            strata = {stratum_of(pool[i]) for i in manifest.example_ids}
            assert len(strata) == 3, f"{name} is missing a stratum: {strata}"

    def test_manifest_ids_are_sorted(self):
        """gepa keys val subscores POSITIONALLY, so the manifest's order is
        the run's order and must not depend on iteration order upstream."""
        for manifest in build_splits(_pool()).values():
            ids = manifest.to_json()["example_ids"]
            assert ids == sorted(ids)

    def test_too_small_a_pool_raises_rather_than_silently_shrinking(self):
        with pytest.raises(ValueError, match="need"):
            build_splits(_pool()[:50])
