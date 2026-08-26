"""Paid measurements must never be lost.

A previous calibration run lost a PAID solver measurement when the refiner
403'd: results were written only at the very end, so the exception discarded
data we had already been billed for. These tests pin the fix.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_rollout.py"


def load_script():
    spec = importlib.util.spec_from_file_location("calibrate_rollout", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["calibrate_rollout"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def script(tmp_path, monkeypatch):
    mod = load_script()
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "calibration")
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path / "repos")
    return mod


class _FakeLM:
    """Solver succeeds; refiner raises, mimicking the 403 that lost data."""

    calls: list[str] = []

    def __init__(self, model: str, **kw):
        self.model = model
        _FakeLM.calls.append(model)
        if "sonnet" in model:
            raise RuntimeError("AccessDeniedException: You don't have access to the model")

    def complete(self, prompt: str, *, max_tokens: int = 4096):
        return "--- a/x.py\n+++ b/x.py\n@@\n-a\n+b\n", 20_000, 700


def test_solver_measurement_survives_refiner_failure(script, tmp_path, monkeypatch):
    """The regression this fix exists for."""
    _FakeLM.calls = []

    from gepa_taxonomy import bedrock

    monkeypatch.setattr(bedrock, "BedrockLM", _FakeLM)
    monkeypatch.setattr(bedrock, "require_credentials", lambda: "us-east-1")

    # Retrieval is network-heavy; stub it with a realistic-sized context.
    from gepa_taxonomy import retrieval
    from gepa_taxonomy.program import RetrievedFile

    class FakeRetriever:
        def __init__(self, *a, **k):
            pass

        def retrieve(self, task, *, k):
            return [RetrievedFile(path="m.py", content="x = 1\n" * 2000)]

    monkeypatch.setattr(retrieval, "BM25Retriever", FakeRetriever)

    monkeypatch.setattr(sys, "argv", ["calibrate_rollout.py"])
    rc = script.main()

    assert rc == 1, "a refiner failure must be reported as a failure"

    out = script.OUT_DIR / "calibration.json"
    assert out.exists(), "the paid solver measurement must be on disk"
    data = json.loads(out.read_text())

    assert data["solver_tokens_in"] == 20_000
    assert data["solver_tokens_out"] == 700
    assert data["solver_usd"] > 0, "we were billed; the cost must be recorded"
    assert data["complete"] is False, "a partial result must be marked partial"
    assert "AccessDenied" in data["refiner_error"]
    assert "refiner_tokens_in" not in data, "must not fabricate unmeasured values"


def test_provenance_is_written_before_any_spend(script, monkeypatch):
    """The file must exist before the first billed call, not after it."""
    seen: list[bool] = []

    from gepa_taxonomy import bedrock, retrieval
    from gepa_taxonomy.program import RetrievedFile

    out = script.OUT_DIR / "calibration.json"

    class WatchLM(_FakeLM):
        def __init__(self, model: str, **kw):
            seen.append(out.exists())
            super().__init__(model, **kw)

    class FakeRetriever:
        def __init__(self, *a, **k):
            pass

        def retrieve(self, task, *, k):
            return [RetrievedFile(path="m.py", content="x = 1\n" * 100)]

    monkeypatch.setattr(bedrock, "BedrockLM", WatchLM)
    monkeypatch.setattr(bedrock, "require_credentials", lambda: "us-east-1")
    monkeypatch.setattr(retrieval, "BM25Retriever", FakeRetriever)
    monkeypatch.setattr(sys, "argv", ["calibrate_rollout.py"])
    script.main()

    assert seen and seen[0] is True, "provenance must be flushed before the first LM call"


def test_profile_prefix_is_configurable(script, monkeypatch, capsys):
    """A profile-scoped authorization problem must be a one-flag fix."""
    from gepa_taxonomy import retrieval
    from gepa_taxonomy.program import RetrievedFile

    class FakeRetriever:
        def __init__(self, *a, **k):
            pass

        def retrieve(self, task, *, k):
            return [RetrievedFile(path="m.py", content="x = 1\n" * 100)]

    monkeypatch.setattr(retrieval, "BM25Retriever", FakeRetriever)
    monkeypatch.setattr(sys, "argv", ["calibrate_rollout.py", "--dry-run", "--profile-prefix", "us."])
    script.main()
    # us. carries a ~10% premium, so the estimate must differ from global.
    first = capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["calibrate_rollout.py", "--dry-run", "--profile-prefix", "global."])
    script.main()
    second = capsys.readouterr().out

    def est(text: str) -> float:
        line = next(ln for ln in text.splitlines() if "estimated cost" in ln)
        return float(line.split("$")[1].split()[0])

    assert est(first) > est(second), "us. profile must price above global."
