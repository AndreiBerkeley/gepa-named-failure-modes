"""Tests for split construction and the committed manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gepa_taxonomy.splits import (
    DEFAULT_SEED,
    DEFAULT_SIZES,
    HARD_LABELS,
    build_verified_splits,
    load_manifest,
)

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests" / "swebench_verified"
SUBSETS = ("val", "test", "train")


@pytest.fixture
def toy():
    """A skewed pool shaped like Verified: few hard instances, one giant repo."""
    ids = [f"r{i // 50}__proj-{i}" for i in range(500)]
    repo = {i: f"repo{int(i.split('__')[0][1:])}/proj" for i in ids}
    # 45 hard, like Verified's real 45/500.
    difficulty = {i: ("1-4 hours" if n < 45 else "<15 min fix") for n, i in enumerate(ids)}
    return ids, difficulty, repo


def _build(toy, **kw):
    ids, difficulty, repo = toy
    return build_verified_splits(ids, difficulty, repo, **kw)


def test_splits_are_disjoint(toy):
    splits = _build(toy)
    seen: set[str] = set()
    for name in SUBSETS:
        s = set(splits[name])
        assert not (seen & s), f"{name} overlaps an earlier subset"
        seen |= s


def test_splits_are_exhaustive(toy):
    splits = _build(toy)
    assert sum(len(v) for v in splits.values()) == len(toy[0])


def test_requested_sizes_are_exact(toy):
    splits = _build(toy)
    assert len(splits["val"]) == DEFAULT_SIZES["val"]
    assert len(splits["test"]) == DEFAULT_SIZES["test"]


def test_val_gets_the_requested_hard_count(toy):
    _ids, difficulty, _repo = toy
    splits = _build(toy, val_hard=30)
    hard = sum(1 for i in splits["val"] if difficulty[i] in HARD_LABELS)
    assert hard == 30


def test_val_hard_cannot_exceed_the_pool(toy):
    with pytest.raises(ValueError, match="only"):
        _build(toy, val_hard=999)


def test_lowering_val_hard_leaves_more_hard_for_test(toy):
    """The tension recorded as F011/O013, pinned as a property."""
    _ids, difficulty, _repo = toy
    a = _build(toy, val_hard=30)
    b = _build(toy, val_hard=15)
    ha = sum(1 for i in a["test"] if difficulty[i] in HARD_LABELS)
    hb = sum(1 for i in b["test"] if difficulty[i] in HARD_LABELS)
    assert hb > ha


def test_deterministic_for_a_given_seed(toy):
    assert _build(toy, seed=7) == _build(toy, seed=7)


def test_different_seeds_give_different_splits(toy):
    assert _build(toy, seed=1) != _build(toy, seed=2)


# --------------------------------------------------------------------------
# Committed manifests
# --------------------------------------------------------------------------


@pytest.mark.skipif(not MANIFEST_DIR.exists(), reason="manifests not built yet")
class TestCommittedManifests:
    def test_all_present(self):
        for name in SUBSETS:
            assert (MANIFEST_DIR / f"{name}.json").exists(), name

    def test_sizes_match_the_approved_plan(self):
        assert len(load_manifest(MANIFEST_DIR / "val.json")) == 60
        assert len(load_manifest(MANIFEST_DIR / "test.json")) == 300
        assert len(load_manifest(MANIFEST_DIR / "train.json")) == 140

    def test_disjoint_and_exhaustive(self):
        seen: set[str] = set()
        for name in SUBSETS:
            ids = set(load_manifest(MANIFEST_DIR / f"{name}.json"))
            assert not (seen & ids), f"{name} overlaps"
            seen |= ids
        assert len(seen) == 500, "manifests must partition Verified exactly"

    def test_test_is_disjoint_from_val(self):
        """The taxonomy comes from val traces; test must stay untouched (D025)."""
        assert not (set(load_manifest(MANIFEST_DIR / "test.json")) & set(load_manifest(MANIFEST_DIR / "val.json")))

    def test_ids_are_sorted_and_unique(self):
        for name in SUBSETS:
            ids = load_manifest(MANIFEST_DIR / f"{name}.json")
            assert ids == sorted(ids), f"{name} not sorted -- file would churn"
            assert len(ids) == len(set(ids)), f"{name} has duplicates"

    def test_provenance_recorded(self):
        for name in SUBSETS:
            data = json.loads((MANIFEST_DIR / f"{name}.json").read_text())
            assert data["seed"] == DEFAULT_SEED
            assert data["dataset"] == "SWE-bench/SWE-bench_Verified"
            assert data["dataset_split"] == "test", "only the test split is gradeable"
            assert data["n"] == len(data["instance_ids"])
