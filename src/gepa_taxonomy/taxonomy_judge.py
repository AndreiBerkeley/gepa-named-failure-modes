"""Taxonomy-conditioned failure diagnosis -- the TREATMENT arm's only addition.

What this does
--------------
For each FAILED instance in the parent's reflection minibatch, ask a certified
AdaMAST taxonomy which failure codes the rollout's evidence supports, and hand
those codes to GEPA's reflection prompt beside the feedback the baseline
already sees. Nothing else about the run changes: with no judge attached the
adapter emits byte-identical reflective datasets (``tests/test_taxonomy_judge.py``).

Scope, and why it is this narrow
--------------------------------
*Parent minibatch only.* ``make_reflective_dataset(task.parent_candidate,
eval_curr, predictor_names)`` is called at ``reflective_mutation.py:420`` with
the PARENT's evaluation. Child evaluations (``_batch_evaluate`` at ~:553) never
reach reflection, so judging them would be pure spend with no path to a prompt.

*Failed instances only.* A resolved rollout has no failure to name, and paying
a judge to say so competes with rollouts for the same dollars.

*One subagent per judgement.* The reflective dataset is built per component,
and reflection that rewrites ``solver_instruction`` must not be shown failures
that belong to the refiner. Each judgement is therefore scoped to the component
under update: its role statement, its input, its prompt and its output. The
taxonomy is built for this -- its category-B codes carry ``applies_to_role``,
and every one of them is ``"solver"``.

*Full traces, no truncation.* AdaMAST's default ``max_trace_chars`` is 6000,
which drops ~96% of one of our trajectories (measured: the base-val trajectories
run 36 k-153 k chars, mean 115 k). ``MAX_TRACE_CHARS`` is set so
``_format_trace`` never reaches its start/tail sampling branch.

Cost
----
Judging is metered into the same ``CostMeter`` family as solver, refiner and
reflection, so it competes for the same per-seed dollar budget. That is the
experiment's design: the treatment arm may buy fewer rollouts than the baseline
with the same money, and that trade is exactly what the comparison measures.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gepa_taxonomy.adamast_trace import AdamastRecord
from gepa_taxonomy.cost import Phase
from gepa_taxonomy.program import REFINER, SOLVER
from gepa_taxonomy.seed_cache import candidate_hash

#: AdaMAST judge mode. ``"default"`` runs the SelectionJudge, which makes
#: exactly ONE model call per trace (``adamast/judges/simple.py:147-191``) and
#: returns every supported code rather than forcing a single label. There is no
#: second/verification pass and none is wanted: a second pass would double the
#: judge's share of a fixed budget.
JUDGE_MODE = "default"

#: Bedrock, so judging uses the same account, bearer credential and price sheet
#: as every other call in this experiment.
JUDGE_PROVIDER = "openai"

#: Trace budget handed to AdaMAST. Above the largest formatted trace we have
#: measured (153,638 chars for ``results/traces/base_val.adamast.jsonl``) with
#: headroom for the longer prompts an optimized candidate can produce, and
#: still ~110 k tokens -- inside Sonnet's context window. It is a ceiling that
#: prevents a runaway trace from being rejected outright, NOT a summarisation
#: target: a trace under it is passed through whole.
MAX_TRACE_CHARS = 400_000

#: The selection judge emits a short JSON object; AdaMAST's 8192 default only
#: buys unused headroom.
MAX_OUTPUT_TOKENS = 4096

#: Characters per token for the spend estimate. Deliberately LOW (English prose
#: is ~4), because a low divisor over-estimates tokens: under-metering judge
#: spend would let the dollar stopper fire late and overshoot a real budget,
#: which is the same failure mode ``cost.UnpricedModelError`` exists to prevent.
CHARS_PER_TOKEN = 3.5

#: Role names must match the taxonomy's own ``applies_to_role`` values, or
#: role-scoped codes cannot be routed. The pruned taxonomy uses ``"solver"``.
ROLE_BY_COMPONENT: dict[str, str] = {SOLVER: "solver", REFINER: "refiner"}

#: What each subagent is for, stated to the judge in its own trace. Taken from
#: the program's actual structure (``program.py``), not aspirationally: the
#: solver sees no feedback, and the refiner never sees test results.
ROLE_PURPOSE: dict[str, str] = {
    "solver": (
        "First of the pipeline's two model calls. Reads the issue text and the "
        "BM25-retrieved source files and must output a unified diff that fixes the "
        "issue. It receives no feedback of any kind and gets exactly one attempt."
    ),
    "refiner": (
        "Second of the pipeline's two model calls. Reads the issue text, the same "
        "retrieved source files, the solver's candidate patch, and cheap static "
        "checks on that patch (is it a well-formed diff, does it apply to the "
        "repository, do the modified files parse). It must output a corrected "
        "unified diff. It never sees test results, and its output is what gets "
        "graded."
    ),
}

#: Where a uv-installed ``adamast`` tool keeps its interpreter.
_UV_TOOL_PYTHON = Path.home() / ".local" / "share" / "uv" / "tools" / "adamast" / "bin" / "python"

_WORKER = Path(__file__).with_name("_adamast_worker.py")


def default_adamast_python() -> Path | None:
    """Locate the interpreter that has ``adamast`` installed.

    ``ADAMAST_PYTHON`` wins so a differently-installed AdaMAST can be pointed
    at without a code change; otherwise the uv tool layout, then the sibling of
    whatever ``adamast`` is on PATH.
    """
    override = os.environ.get("ADAMAST_PYTHON")
    if override:
        return Path(override)
    if _UV_TOOL_PYTHON.exists():
        return _UV_TOOL_PYTHON
    exe = shutil.which("adamast")
    if exe:
        sibling = Path(exe).resolve().parent / "python"
        if sibling.exists():
            return sibling
    return None


def taxonomy_fingerprint(path: str | Path) -> str:
    """Content hash of the taxonomy file.

    Part of the cache key so that editing or re-pruning the taxonomy invalidates
    every judgement made under the old one, rather than silently mixing two
    code sets inside one run.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Role-scoped trace records
# ---------------------------------------------------------------------------


def build_role_record(
    *,
    component: str,
    rollout: Any,
    task: Any,
    grading: dict[str, Any] | None = None,
) -> AdamastRecord:
    """Build the AdaMAST-native record for ONE subagent of one rollout.

    Sections, in order: the task formulation, what this subagent is for, what it
    was given, and its full untruncated trace (prompt then output).

    Section 3 names the subagent's input rather than repeating it. The prompt in
    section 4 already embeds that input verbatim -- for the solver that is ~60 KB
    of retrieved source -- and duplicating it would double the judge's token bill
    for no new evidence. The refiner's input (candidate patch, static checks) is
    small and IS repeated, because it is the thing its failure usually turns on.

    Outcome (score, resolved, harness detail) stays in ``metadata`` and out of
    the trajectory: AdaMAST's own checklist warns against leaking oracle
    outcomes to the judge, and ``adamast_trace.build_record`` holds the same line.
    """
    role = ROLE_BY_COMPONENT[component]
    problem_statement = getattr(task, "problem_statement", "") or ""
    solver_prompt, refiner_prompt = rollout.prompts()

    if role == "solver":
        given = (
            f"Repository: {getattr(task, 'repo', '')}\n"
            f"Issue: the task formulation above.\n"
            "Retrieved source files (contents embedded verbatim in the prompt below):\n"
            + ("\n".join(f"  - {p}" for p in rollout.retrieved_paths) or "  (none retrieved)")
        )
        prompt, output = solver_prompt, rollout.solver_patch
    else:
        given = (
            f"Repository: {getattr(task, 'repo', '')}\n"
            "Issue: the task formulation above.\n"
            "Retrieved source files (contents embedded verbatim in the prompt below):\n"
            + ("\n".join(f"  - {p}" for p in rollout.retrieved_paths) or "  (none retrieved)")
            + "\n\nCandidate patch produced by the solver:\n"
            + (rollout.solver_patch or "(the solver produced no patch)")
            + "\n\nAutomated checks reported on that candidate patch:\n"
            + (rollout.feedback.render() or "(none)")
        )
        prompt, output = refiner_prompt, rollout.refiner_patch

    trajectory = "\n\n".join(
        (
            f"[TASK]\n{problem_statement}",
            f"[ROLE: {role}]\n{ROLE_PURPOSE[role]}",
            f"[INPUT GIVEN TO THE {role.upper()}]\n{given}",
            f"[{role.upper()} PROMPT]\n{prompt}",
            f"[{role.upper()} OUTPUT]\n{output or '(no output)'}",
        )
    )

    return AdamastRecord(
        problem_id=f"{rollout.instance_id}::{role}",
        task=problem_statement,
        raw_trajectory=trajectory,
        metadata={
            "system": "gepa-swebench-solver-refiner",
            "benchmark": "SWE-bench_Verified",
            "instance_id": rollout.instance_id,
            "component": component,
            "role": role,
            "repo": getattr(task, "repo", None),
            "retrieved_paths": list(rollout.retrieved_paths),
            "grading": dict(grading or {}),
        },
    )


# ---------------------------------------------------------------------------
# Durable cache
# ---------------------------------------------------------------------------


@dataclass
class JudgeCache:
    """Write-through cache of judgements, keyed the way they were paid for.

    Same shape and durability argument as ``rollout_cache.RolloutCache``: append
    a JSONL line and fsync it the instant a judgement completes, so an
    interruption re-pays nothing, and drop a truncated final line at load rather
    than refusing to start.

    The key carries the COMPONENT as well as the candidate and instance. A
    solver judgement and a refiner judgement of the same rollout are different
    diagnoses of different evidence; serving one for the other would put
    ``Solver_*`` codes in front of the refiner's reflection.
    """

    path: Path
    _entries: dict[tuple[str, str, str, str], dict[str, Any]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _fh: Any = field(default=None, repr=False)
    #: Judgements served from cache this session -- i.e. not paid for twice.
    hits: int = 0
    #: Malformed trailing records discarded at load (an interrupted append).
    truncated_records: int = 0

    @classmethod
    def open(cls, path: str | Path) -> JudgeCache:
        cache = cls(path=Path(path))
        cache.load()
        cache._fh = cache.path.open("a", buffering=1)  # line-buffered
        return cache

    @staticmethod
    def _key(taxonomy: str, candidate: str, component: str, instance_id: str) -> tuple[str, str, str, str]:
        return (taxonomy, candidate, component, instance_id)

    def load(self) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            return 0
        loaded = 0
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = self._key(
                        rec["taxonomy"], rec["candidate_hash"], rec["component"], rec["instance_id"]
                    )
                except (json.JSONDecodeError, KeyError):
                    self.truncated_records += 1
                    continue
                self._entries[key] = rec
                loaded += 1
        return loaded

    def _write(self, rec: dict[str, Any]) -> None:
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def get(
        self, *, taxonomy: str, candidate: dict[str, str], component: str, instance_id: str
    ) -> list[dict[str, Any]] | None:
        key = self._key(taxonomy, candidate_hash(candidate), component, instance_id)
        with self._lock:
            rec = self._entries.get(key)
            if rec is None:
                return None
            self.hits += 1
            return [dict(m) for m in rec.get("failure_modes", [])]

    def put(
        self,
        *,
        taxonomy: str,
        candidate: dict[str, str],
        component: str,
        instance_id: str,
        failure_modes: list[dict[str, Any]],
        cost_usd: float = 0.0,
    ) -> None:
        rec = {
            "taxonomy": taxonomy,
            "candidate_hash": candidate_hash(candidate),
            "component": component,
            "instance_id": instance_id,
            "failure_modes": failure_modes,
            "cost_usd": cost_usd,
        }
        with self._lock:
            self._entries[self._key(taxonomy, rec["candidate_hash"], component, instance_id)] = rec
            self._write(rec)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def recovered_usd(self) -> float:
        """Judge spend a restart does not re-pay."""
        return sum(r.get("cost_usd", 0.0) for r in self._entries.values())


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------


class JudgeError(RuntimeError):
    """The judge subprocess could not produce diagnoses."""


@dataclass
class TaxonomyJudge:
    """Diagnose failed rollouts against a certified AdaMAST taxonomy.

    Fail-soft is a hard requirement, not politeness: this runs inside a paid
    optimization loop, and a lost diagnosis must never cost the run. Every
    error path returns "no codes for these instances" and logs once.
    """

    taxonomy_path: Path
    meter: Any
    model: str
    cache: JudgeCache | None = None
    provider: str = JUDGE_PROVIDER
    aws_region: str | None = None
    aws_profile: str | None = None
    max_trace_chars: int = MAX_TRACE_CHARS
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    phase: Phase = "optimization"
    allow_review_required: bool = False
    python: Path | None = None
    timeout_s: int = 900
    #: Injection point for tests and for anyone wiring a different judge
    #: transport. Takes the request dict, returns the response dict.
    runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    log: Callable[[str], None] = print

    calls: int = 0
    judged: int = 0
    failures: int = 0
    role_mismatch_dropped: int = 0
    spend_usd: float = 0.0
    _warned: bool = field(default=False, repr=False)
    _fingerprint: str = field(default="", repr=False)
    _codes: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.taxonomy_path = Path(self.taxonomy_path)
        self._fingerprint = taxonomy_fingerprint(self.taxonomy_path)
        document = json.loads(self.taxonomy_path.read_text(encoding="utf-8-sig"))
        self._codes = {str(c["id"]): c for c in document.get("codes", []) if c.get("id")}
        if not self._codes:
            raise ValueError(f"taxonomy {self.taxonomy_path} contains no codes")
        if self.python is None:
            self.python = default_adamast_python()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def preflight(self) -> dict[str, Any]:
        """Prove the judge can run, before a launch spends anything.

        A treatment run whose judge is broken is a baseline run wearing the
        wrong label -- and it would only be discovered after the money was gone.
        Raises rather than degrading.
        """
        if self.runner is not None:
            return {"transport": "injected", "codes": len(self._codes), "taxonomy": self._fingerprint}
        if self.python is None or not Path(self.python).exists():
            raise JudgeError(
                "no interpreter with adamast installed. Expected the uv tool at "
                f"{_UV_TOOL_PYTHON}, or set ADAMAST_PYTHON."
            )
        # Fixed argv, no shell: the only variable is the interpreter path.
        probe = subprocess.run(
            [
                str(self.python),
                "-c",
                "import adamast, importlib.metadata as m; print(m.version('adamast'))",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if probe.returncode != 0:
            raise JudgeError(f"adamast is not importable by {self.python}: {probe.stderr.strip()[:200]}")
        return {
            "transport": str(self.python),
            "adamast": probe.stdout.strip(),
            "codes": len(self._codes),
            "taxonomy": self._fingerprint,
        }

    # -- API --------------------------------------------------------------

    def judge(
        self,
        candidate: dict[str, str],
        component: str,
        subjects: dict[str, dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Diagnose one component of each failed rollout in ``subjects``.

        ``subjects`` maps instance id to ``{"rollout": Rollout, "task": Task,
        "grading": dict}``. Returns a mapping from instance id to that
        component's failure codes. An instance is ABSENT from the result only
        when its judgement failed -- a judged-but-clean instance maps to ``[]``,
        which is a diagnosis ("no code in this taxonomy fired") and is worth
        showing to reflection.
        """
        if component not in ROLE_BY_COMPONENT:
            return {}

        results: dict[str, list[dict[str, Any]]] = {}
        pending: dict[str, AdamastRecord] = {}

        for instance_id, subject in subjects.items():
            if self.cache is not None:
                cached = self.cache.get(
                    taxonomy=self._fingerprint,
                    candidate=candidate,
                    component=component,
                    instance_id=instance_id,
                )
                if cached is not None:
                    results[instance_id] = cached
                    continue
            try:
                pending[instance_id] = build_role_record(
                    component=component,
                    rollout=subject["rollout"],
                    task=subject["task"],
                    grading=subject.get("grading"),
                )
            except Exception as exc:
                self._warn(f"could not build a {component} trace for {instance_id}: {exc}")

        if not pending:
            return results

        try:
            response = self._run(list(pending.values()))
            # Metering first: a malformed response still cost money, and the
            # budget must know about it even when the diagnosis is unusable.
            cost = self._meter_usage(response.get("usage") or {})
            by_trace = {str(d.get("trace_id", "")): d for d in response.get("diagnoses") or []}
        except Exception as exc:
            self.failures += 1
            self._warn(f"judging failed for {sorted(pending)}: {exc}")
            return results

        for instance_id, record in pending.items():
            diagnosis = by_trace.get(record.problem_id)
            if diagnosis is None:
                self._warn(f"judge returned no diagnosis for {record.problem_id}")
                continue
            modes = self._normalise(diagnosis.get("failure_modes") or [], role=ROLE_BY_COMPONENT[component])
            results[instance_id] = modes
            self.judged += 1
            if self.cache is not None:
                self.cache.put(
                    taxonomy=self._fingerprint,
                    candidate=candidate,
                    component=component,
                    instance_id=instance_id,
                    failure_modes=modes,
                    cost_usd=cost / max(1, len(pending)),
                )
        return results

    # -- internals --------------------------------------------------------

    def _run(self, records: list[AdamastRecord]) -> dict[str, Any]:
        """ONE judge_traces call per minibatch: the judge is built once and the
        whole batch goes through it, so taxonomy loading and provider setup are
        paid once rather than per instance."""
        request = {
            "taxonomy_path": str(self.taxonomy_path),
            "traces": [r.to_dict() for r in records],
            "provider": self.provider,
            "model": self.model,
            "mode": JUDGE_MODE,
            "max_trace_chars": self.max_trace_chars,
            "max_output_tokens": self.max_output_tokens,
            "aws_region": self.aws_region,
            "aws_profile": self.aws_profile,
            "allow_review_required": self.allow_review_required,
        }
        self.calls += 1
        if self.runner is not None:
            return self.runner(request)

        if self.python is None:
            raise JudgeError("no interpreter with adamast installed; set ADAMAST_PYTHON")
        # Fixed argv, no shell: the only variable is the interpreter path.
        proc = subprocess.run(
            [str(self.python), str(_WORKER)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
        if proc.returncode != 0:
            raise JudgeError(proc.stderr.strip()[:400] or f"worker exited {proc.returncode}")
        return json.loads(proc.stdout)

    def _meter_usage(self, usage: dict[str, Any]) -> float:
        """Book judge spend to the shared budget.

        APPROXIMATION, and a deliberate one. AdaMAST's provider adapters return
        text only -- ``BedrockProvider.complete`` reads
        ``response["output"]["message"]["content"]`` and drops the ``usage``
        block -- so no token counts come back from a judge call. The worker
        measures the exact prompt and response CHARACTERS at the transport
        boundary and we convert at ``CHARS_PER_TOKEN``, then price the result
        with the same ``cost.price_call`` table every other call uses. The
        divisor is set low on purpose so the estimate errs high; see its comment.
        """
        prompt_tokens = math.ceil(int(usage.get("prompt_chars", 0)) / CHARS_PER_TOKEN)
        response_tokens = math.ceil(int(usage.get("response_chars", 0)) / CHARS_PER_TOKEN)
        cost = self.meter.record(
            model=self.model,
            input_tokens=prompt_tokens,
            output_tokens=response_tokens,
            phase=self.phase,
        )
        self.spend_usd += cost
        return cost

    def _normalise(self, modes: list[dict[str, Any]], *, role: str) -> list[dict[str, Any]]:
        """Reduce the judge's output to {code, name, evidence, severity}.

        ``name`` and ``severity`` are taken from the TAXONOMY, not from the
        judge. AdaMAST's judge catalog is built from ``_normalize_code``, which
        keeps id/name/description/category/when_to_use/when_not_to_use and drops
        ``severity`` and ``applies_to_role`` -- so a judge-reported severity is
        invented, while the taxonomy's is the certified one.

        Role-scoped codes that name a different subagent are dropped for the
        same reason the cache key carries the component: ``Solver_Malformed_Diff``
        in front of the refiner's reflection is a false accusation.
        """
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for mode in modes:
            code_id = str(mode.get("code") or "").strip()
            spec = self._codes.get(code_id)
            if spec is None or code_id in seen:
                continue
            applies_to = str(spec.get("applies_to_role") or "").strip()
            if applies_to and applies_to != role:
                self.role_mismatch_dropped += 1
                continue
            seen.add(code_id)
            out.append(
                {
                    "code": code_id,
                    "name": str(spec.get("name") or code_id),
                    "evidence": str(mode.get("evidence") or ""),
                    "severity": str(spec.get("severity") or ""),
                }
            )
        return out

    def _warn(self, message: str) -> None:
        """Log once. A judge that is broken is broken for the whole run, and a
        message per minibatch would bury the run log it shares with gepa."""
        if self._warned:
            return
        self._warned = True
        self.log(f"  [taxonomy-judge] {message} -- continuing without codes (logged once)")

    def summary(self) -> dict[str, Any]:
        return {
            "taxonomy": str(self.taxonomy_path),
            "taxonomy_fingerprint": self._fingerprint,
            "judge_model": self.model,
            "judge_batches": self.calls,
            "instances_judged": self.judged,
            "judge_failures": self.failures,
            "role_mismatch_dropped": self.role_mismatch_dropped,
            "judge_cache_hits": self.cache.hits if self.cache is not None else 0,
            "judge_usd": round(self.spend_usd, 4),
        }
