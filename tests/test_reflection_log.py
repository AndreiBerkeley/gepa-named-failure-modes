"""The reflection-dataset logger records exactly what reflection consumed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gepa_taxonomy.reflection_log import ReflectionDatasetLogger


def test_logs_typed_dict_events_line_buffered(tmp_path):
    path = tmp_path / "reflection_datasets.jsonl"
    logger = ReflectionDatasetLogger(path)
    event = {
        "iteration": 3,
        "candidate_idx": 1,
        "components": ["solver"],
        "dataset": {"solver": [{"instance_id": "t9", "failure_modes": [{"name": "X", "evidence": "e"}]}]},
    }
    logger.on_reflective_dataset_built(event)
    # line-buffered: readable before close
    row = json.loads(path.read_text().splitlines()[0])
    assert row["iteration"] == 3
    assert row["dataset"]["solver"][0]["failure_modes"][0]["name"] == "X"
    logger.close()


def test_tolerates_attribute_style_events(tmp_path):
    class Event:
        def __init__(self):
            self.iteration = 1
            self.candidate_idx = 0
            self.components = ("c",)
            self.dataset = {"c": []}

    logger = ReflectionDatasetLogger(tmp_path / "r.jsonl")
    logger.on_reflective_dataset_built(Event())
    row = json.loads((tmp_path / "r.jsonl").read_text())
    assert row["components"] == ["c"] and row["dataset"] == {"c": []}
    logger.close()
