from __future__ import annotations

from pathlib import Path

from gepa_taxonomy import pipeline


def test_dry_run_orders_prepare_then_optimize(tmp_path, monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(
        pipeline, "_run", lambda command, *, dry_run, pythonpath=None: commands.append(command)
    )

    assert (
        pipeline.main(
            [
                "ifbench",
                "--seed",
                "1",
                "--budget",
                "60",
                "--results",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )

    assert [Path(command[1]).name for command in commands] == [
        "build_ifbench_base_val.py",
        "generate_taxonomy.py",
        "judge_corpus.py",
        "reduce_taxonomy.py",
        "run_ifbench_seed.py",
    ]
    generation = commands[1]
    corpus_judge = commands[2]
    run = commands[-1]
    assert generation[generation.index("--model") + 1] == "us.anthropic.claude-sonnet-4-6"
    assert corpus_judge[corpus_judge.index("--model") + 1] == "us.anthropic.claude-sonnet-4-6"
    assert run[run.index("--reflection-model") + 1] == "us.anthropic.claude-sonnet-4-6"


def test_existing_taxonomy_skips_preparation(tmp_path, monkeypatch):
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text('{"codes": [{"id": "A.1", "name": "A"}]}', encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(
        pipeline, "_run", lambda command, *, dry_run, pythonpath=None: commands.append(command)
    )

    pipeline.main(
        [
            "ifbench",
            "--seed",
            "3",
            "--budget",
            "60",
            "--taxonomy",
            str(taxonomy),
            "--results",
            str(tmp_path / "results"),
            "--dry-run",
        ]
    )

    assert len(commands) == 1
    assert Path(commands[0][1]).name == "run_ifbench_seed.py"
    assert str(taxonomy) in commands[0]
