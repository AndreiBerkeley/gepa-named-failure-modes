"""Interpreter resolution for the AdaMAST corpus judge. FREE: no subprocess runs."""

from __future__ import annotations

from pathlib import Path

from gepa_taxonomy import taxonomy_judge


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("ADAMAST_PYTHON", str(tmp_path / "python"))
    assert taxonomy_judge.default_adamast_python() == tmp_path / "python"


def test_sibling_checkout_is_the_default(monkeypatch, tmp_path):
    monkeypatch.delenv("ADAMAST_PYTHON", raising=False)
    sibling = tmp_path / "adamast-public" / ".venv" / "bin" / "python"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("")
    monkeypatch.setattr(taxonomy_judge, "_SIBLING_VENV_PYTHON", sibling)
    monkeypatch.setattr(taxonomy_judge, "_UV_TOOL_PYTHON", tmp_path / "missing")
    assert taxonomy_judge.default_adamast_python() == sibling


def test_sibling_default_matches_generate_taxonomy():
    """Both stages must look in the same place after bootstrap."""
    repo = Path(__file__).resolve().parents[1]
    assert taxonomy_judge._SIBLING_VENV_PYTHON == repo.parent / "adamast-public" / ".venv" / "bin" / "python"
