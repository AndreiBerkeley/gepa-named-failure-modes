"""Local Docker grading backend for SWE-Bench.

Batched by design: one ``swebench.harness.run_evaluation`` invocation grades a
whole evaluation batch, with ``--max-workers`` giving the parallelism inside.
Grading instances one at a time would pay harness startup per rollout, which on
a ~1.9 min emulated evaluation is a large relative overhead.

Gold handling: this module is the *only* place a :class:`Gold` is used, and it
uses it solely to let the harness pick the right tests. Patches come in, scores
come out; nothing here flows back to the program.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gepa_taxonomy.tasks import Gold, Task

MODEL_TAG = "gepa"  # becomes the report filename prefix: gepa.<run_id>.json

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

#: Caps keep the per-instance detail JSON-small: it rides on every trace and
#: reflective-dataset example, so it must stay an excerpt, not a log dump.
MAX_FAILING_TESTS = 5
TEST_OUTPUT_TAIL_CHARS = 600


@dataclass
class LocalDockerGrader:
    """Grades patches with the official SWE-Bench harness on local Docker.

    On arm64 the harness pulls the official x86_64 images and runs them under
    emulation (verified deterministic across 6 instances x 3 rounds).
    """

    work_dir: Path
    dataset_name: str = "SWE-bench/SWE-bench_Verified"
    split: str = "test"
    max_workers: int = 4
    cache_level: str = "env"
    timeout: int = 1800
    run_id_prefix: str = "gepa"
    #: Set True only for a dry configuration check; never for real grading.
    dry_run: bool = False
    calls: int = 0
    _log: list[dict[str, Any]] = field(default_factory=list, repr=False)

    # -- batch API (what the adapter uses) --------------------------------

    def grade_batch(self, items: list[tuple[Task, Gold, str]]) -> dict[str, tuple[float, dict[str, Any]]]:
        """Grade ``(task, gold, patch)`` triples. Returns {instance_id: (score, detail)}.

        A patch that reaches here has already passed the skip gate, so every
        item genuinely needs a container.
        """
        if not items:
            return {}

        self.calls += 1
        run_id = f"{self.run_id_prefix}-{uuid.uuid4().hex[:10]}"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # Absolute: the harness runs with cwd=work_dir, so a relative path
        # would resolve against work_dir a second time and not be found.
        preds_path = (self.work_dir / f"{run_id}-predictions.json").resolve()
        preds_path.write_text(
            json.dumps(
                [
                    {"instance_id": t.instance_id, "model_name_or_path": MODEL_TAG, "model_patch": patch}
                    for t, _g, patch in items
                ],
                indent=2,
            )
        )

        ids = [t.instance_id for t, _g, _p in items]
        cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            self.dataset_name,
            "--split",
            self.split,
            "--predictions_path",
            str(preds_path),
            "--run_id",
            run_id,
            "--max_workers",
            str(self.max_workers),
            "--cache_level",
            self.cache_level,
            "--timeout",
            str(self.timeout),
            "--instance_ids",
            *ids,
        ]

        if self.dry_run:
            return {i: (0.0, {"resolved": False, "dry_run": True}) for i in ids}

        proc = subprocess.run(cmd, cwd=str(self.work_dir), capture_output=True, text=True)
        report = self._read_report(run_id)

        if report is None:
            # No report: treat every instance as unresolved and say so loudly
            # rather than silently scoring 0 as if it had been evaluated.
            detail = {
                "resolved": False,
                "harness_error": True,
                "returncode": proc.returncode,
                "stderr_tail": (proc.stderr or "")[-500:],
            }
            self._log.append({"run_id": run_id, "error": "no report", "ids": ids})
            return {i: (0.0, dict(detail)) for i in ids}

        resolved = set(report.get("resolved_ids", []))
        errored = set(report.get("error_ids", []))
        empty = set(report.get("empty_patch_ids", []))

        out: dict[str, tuple[float, dict[str, Any]]] = {}
        for i in ids:
            score = 1.0 if i in resolved else 0.0
            detail: dict[str, Any] = {
                "resolved": i in resolved,
                "errored": i in errored,
                "empty_patch": i in empty,
                "run_id": run_id,
            }
            detail.update(self._instance_substance(run_id, i))
            out[i] = (score, detail)
        self._log.append(
            {
                "run_id": run_id,
                "n": len(ids),
                "resolved": len(resolved & set(ids)),
                "errored": len(errored & set(ids)),
            }
        )
        return out

    # -- single-instance API (protocol compatibility) ---------------------

    def grade(self, task: Task, gold: Gold, patch: str) -> tuple[float, dict[str, Any]]:
        return self.grade_batch([(task, gold, patch)])[task.instance_id]

    # -- internals --------------------------------------------------------

    def _instance_substance(self, run_id: str, instance_id: str) -> dict[str, Any]:
        """Per-instance harness artifacts, surfaced so reflection sees *why* a
        patch failed rather than a bare 0.

        Best-effort by design: a missing or malformed artifact yields absent
        fields, never an exception -- grading must not fail because a log did.
        """
        detail: dict[str, Any] = {}
        # The harness writes logs/run_evaluation/ under its cwd; like
        # _read_report, both plausible bases are checked.
        inst_dir: Path | None = None
        for base in (self.work_dir, Path.cwd()):
            d = base / "logs" / "run_evaluation" / run_id / MODEL_TAG / instance_id
            if d.is_dir():
                inst_dir = d
                break
        if inst_dir is None:
            return detail

        try:
            report = json.loads((inst_dir / "report.json").read_text())
            status = report[instance_id]["tests_status"]
            # Tests that should have passed but did not: the fix's own target
            # tests first, then regressions.
            failing = [
                *status["FAIL_TO_PASS"]["failure"],
                *status["PASS_TO_PASS"]["failure"],
            ]
            if failing:
                detail["failing_tests"] = [str(t) for t in failing[:MAX_FAILING_TESTS]]
        except Exception:
            pass

        try:
            text = (inst_dir / "test_output.txt").read_text(errors="replace")
            tail = _ANSI_RE.sub("", text)[-TEST_OUTPUT_TAIL_CHARS:]
            if tail.strip():
                detail["test_output_tail"] = tail
        except Exception:
            pass

        return detail

    def _read_report(self, run_id: str) -> dict[str, Any] | None:
        """The harness writes ``<model>.<run_id>.json`` into its working dir."""
        for base in (self.work_dir, Path.cwd()):
            p = base / f"{MODEL_TAG}.{run_id}.json"
            if p.exists():
                try:
                    return json.loads(p.read_text())
                except json.JSONDecodeError:
                    return None
        return None

    def summary(self) -> dict[str, Any]:
        return {"harness_invocations": self.calls, "batches": list(self._log)}
