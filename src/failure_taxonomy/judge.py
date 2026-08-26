"""The failure judge: one call per rollout, occurrences attributed to components.

Why this is not AdaMAST's judge
-------------------------------
AdaMAST's ``SelectionJudge`` returns a closed record -- a code and an evidence
string -- with no notion of components and no extension point. Attribution had
to be smuggled in by *asking separately per component*: one call per component
per rollout, with the component encoded in which call was made. That costs N
calls per rollout and resends the whole code catalog N times, and it shows each
call only its own slice, so no judgement can ever say "this step failed because
the previous one handed it something unusable".

Judging the whole trace once and having the judge attribute each occurrence
fixes all three at the same time, and it drops the runtime dependency on any
particular taxonomy tool -- this package needs a code list, not a vendor.

Occurrences, not codes
----------------------
The unit is an *occurrence*: one observation of one failure mode at one place in
one trace. The same code may legitimately fire several times in a rollout --
two malformed sections, two dropped entities -- and those are separate evidence
for reflection, so they are not collapsed. Deduplication would throw away
exactly the multiplicity that tells reflection how bad a problem is.

Attribution
-----------
Each occurrence carries the component whose behaviour the evidence is about, or
``None`` when the evidence is genuinely cross-component or grounded in shared
task text. ``None`` routes to *every* component: an unattributable failure is
still information, and hiding it from all components would lose it entirely.
Nothing in the taxonomy restricts which codes may apply to which component --
placement is decided by observation, not by declaration.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from failure_taxonomy.schema import Taxonomy
from failure_taxonomy.trace import SegmentedTrace

#: Sentinel the judge is told to use when an occurrence belongs to no single
#: component. Also what an unrecognised component name degrades to.
GENERAL = None

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True, slots=True)
class Occurrence:
    """One observation of one failure mode in one trace."""

    code: str
    name: str
    evidence: str
    component: str | None = GENERAL

    def for_reflection(self) -> dict[str, str]:
        """What reflection sees.

        The code id is deliberately absent. ``B.4`` carries no meaning to a
        language model; the name does. The id stays in the cache and the logs,
        where it is what cross-run analysis joins on.
        """
        return {"name": self.name, "evidence": self.evidence}


class FailureJudge(Protocol):
    """Diagnoses rollouts against a taxonomy.

    Implement this to plug in a different judge -- a local model, a rules
    engine, a human-in-the-loop queue. The adapter only needs ``judge``.
    """

    def judge(self, traces: Sequence[SegmentedTrace]) -> dict[str, list[Occurrence]]:
        """Map trace id -> occurrences. A trace id is absent only on failure."""
        ...


PROMPT_TEMPLATE = """You are diagnosing a failure in a multi-step AI system, using a fixed taxonomy of failure modes.

## Failure mode taxonomy

Select only from these codes. Do not invent codes.

{catalog}

## The system's components

{components}

## The execution trace

{trace}

## Your task

Identify every failure mode from the taxonomy that this trace provides concrete evidence for.

Rules:
- Quote evidence VERBATIM from the trace. Never paraphrase, summarise, or invent a quote.
- Report each distinct occurrence separately. If the same failure mode occurs more than once
  with different evidence, report it once per occurrence.
- Attribute each occurrence to the component whose OWN behaviour the evidence is about.
  A later component's prompt often quotes an earlier component's output verbatim -- attribute
  such evidence to the component that PRODUCED the text, not the one that received it.
- Use {general} for the component when the evidence is genuinely about the system as a whole,
  spans several components, or comes from the shared task description.
- If the trace shows no failure from this taxonomy, return an empty list. That is a valid answer.

Return JSON only, in exactly this form:

{{"occurrences": [{{"code": "<id>", "component": "<component name or {general}>", "evidence": "<verbatim quote>"}}]}}"""


@dataclass
class LLMFailureJudge:
    """Default judge: one language-model call per trace.

    ``lm`` is any callable taking a prompt and returning text, which is the same
    shape GEPA already uses for ``reflection_lm``. Keeping to that convention
    means a caller who has wired a model for GEPA has already wired one for this.

    Fail-soft is a hard requirement, not politeness: this runs inside a paid
    optimization loop. Every error path yields no occurrences for the affected
    trace and logs once. A lost diagnosis must never cost a run.
    """

    taxonomy: Taxonomy
    lm: Callable[[str], str]
    cache: Any | None = None
    #: Identifies the candidate being judged, for cache keying. Set per call by
    #: the adapter; judgements of different candidates are different judgements.
    candidate_key: str = ""
    log: Callable[[str], None] = print
    #: Traces judged concurrently. Judging is network-bound and each trace is
    #: independent, so this is close to a linear speedup in wall-clock and
    #: changes nothing about the result -- same prompts, same model, same
    #: occurrences, just not one-at-a-time.
    #:
    #: It matters because the judge runs on EVERY trace in a reflective
    #: minibatch: at HotpotQA's minibatch of 15, serial judging adds fifteen
    #: sequential model calls to every iteration, roughly doubling a run's
    #: wall-clock.
    #:
    #: Default 1 so the shipped behaviour is unchanged for anyone who does not
    #: opt in; concurrency is the caller's decision because it is the caller's
    #: rate limit.
    max_workers: int = 1

    calls: int = 0
    judged: int = 0
    failures: int = 0
    unknown_codes_dropped: int = 0
    unknown_components_generalised: int = 0
    _warned: bool = field(default=False, repr=False)
    #: Guards the counters, the warn-once flag, and the results map, all of which
    #: worker threads mutate.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def judge(self, traces: Sequence[SegmentedTrace]) -> dict[str, list[Occurrence]]:
        """Judge every trace, serving cache hits first.

        Cache lookups happen up front and serially: they are cheap, and doing
        them before spawning workers means a fully-cached batch costs no threads
        at all. Only the misses are dispatched.

        Results are keyed by ``trace_id``, so completion order is irrelevant.
        That is deliberate: GEPA keys valset subscores and the Pareto frontier
        by POSITION, so anything that assembles results out of order silently
        mis-attaches them to the wrong instance.
        """
        results: dict[str, list[Occurrence]] = {}
        pending: list[SegmentedTrace] = []

        for trace in traces:
            cached = None
            if self.cache is not None:
                cached = self.cache.get(
                    taxonomy=self.taxonomy.fingerprint,
                    candidate_key=self.candidate_key,
                    trace_id=trace.trace_id,
                )
            if cached is not None:
                results[trace.trace_id] = cached
            else:
                pending.append(trace)

        if not pending:
            return results

        if self.max_workers <= 1 or len(pending) == 1:
            for trace in pending:
                self._judge_into(trace, results)
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(pending))) as pool:
                futures = [pool.submit(self._judge_into, trace, results) for trace in pending]
                for future in futures:
                    # _judge_into is fail-soft and swallows per-trace errors, so
                    # anything reaching here is a genuine defect in this method
                    # rather than a judging failure -- surface it.
                    future.result()

        return results

    def _judge_into(self, trace: SegmentedTrace, results: dict[str, list[Occurrence]]) -> None:
        """Judge one trace and store it. Fail-soft: a lost diagnosis never costs a run."""
        try:
            occurrences = self._judge_one(trace)
        except Exception as exc:
            with self._lock:
                self.failures += 1
            self._warn(f"judging failed for {trace.trace_id}: {type(exc).__name__}: {exc}")
            return

        with self._lock:
            results[trace.trace_id] = occurrences
            self.judged += 1

        if self.cache is not None:
            self.cache.put(
                taxonomy=self.taxonomy.fingerprint,
                candidate_key=self.candidate_key,
                trace_id=trace.trace_id,
                occurrences=occurrences,
            )

    # -- internals --------------------------------------------------------

    def build_prompt(self, trace: SegmentedTrace) -> str:
        components = trace.components
        if components:
            vocabulary = "\n".join(f"- {c}" for c in components)
        else:
            # Unsegmented trajectory: there is no vocabulary to attribute to, so
            # say so plainly rather than inviting the model to invent names.
            vocabulary = f"(The trace is not segmented by component. Attribute every occurrence to {_general_token()}.)"
        return PROMPT_TEMPLATE.format(
            catalog=self.taxonomy.catalog_text(),
            components=vocabulary,
            trace=trace.render(),
            general=_general_token(),
        )

    def _judge_one(self, trace: SegmentedTrace) -> list[Occurrence]:
        with self._lock:
            self.calls += 1
        raw = self.lm(self.build_prompt(trace))
        payload = _extract_json_object(raw)
        return self._normalise(payload.get("occurrences") or [], trace)

    def _normalise(self, raw_occurrences: Any, trace: SegmentedTrace) -> list[Occurrence]:
        """Validate the judge's output against the taxonomy and the trace.

        Unknown codes are dropped: a judge that invents a code has produced
        something no taxonomy defines, and passing it to reflection would give
        it a diagnosis nobody certified. Unknown component names degrade to
        general rather than being dropped, because the observation may still be
        sound even when the attribution is not.
        """
        if not isinstance(raw_occurrences, Sequence) or isinstance(raw_occurrences, str | bytes):
            return []
        known = set(trace.components)
        out: list[Occurrence] = []
        for raw in raw_occurrences:
            if not isinstance(raw, Mapping):
                continue
            code_id = str(raw.get("code") or "").strip()
            spec = self.taxonomy.get(code_id)
            if spec is None:
                with self._lock:
                    self.unknown_codes_dropped += 1
                continue
            component = str(raw.get("component") or "").strip()
            if not component or component.lower() in _GENERAL_ALIASES:
                component = GENERAL
            elif component not in known:
                with self._lock:
                    self.unknown_components_generalised += 1
                component = GENERAL
            out.append(
                Occurrence(
                    code=spec.id,
                    # From the taxonomy, never the judge: a judge-supplied name
                    # would let a relabelled code reach reflection.
                    name=spec.name,
                    evidence=str(raw.get("evidence") or "").strip(),
                    component=component,
                )
            )
        return out

    def _warn(self, message: str) -> None:
        """Log once. A broken judge is broken for the whole run, and one message
        per minibatch would bury the run log it shares with GEPA."""
        with self._lock:
            if self._warned:
                return
            self._warned = True
        self.log(f"  [failure-judge] {message} -- continuing without codes (logged once)")

    def summary(self) -> dict[str, Any]:
        return {
            "taxonomy_fingerprint": self.taxonomy.fingerprint,
            "taxonomy_codes": len(self.taxonomy),
            "judge_calls": self.calls,
            "traces_judged": self.judged,
            "judge_failures": self.failures,
            "unknown_codes_dropped": self.unknown_codes_dropped,
            "unknown_components_generalised": self.unknown_components_generalised,
        }


_GENERAL_ALIASES = frozenset({"general", "none", "null", "n/a", "system", "all"})


def _general_token() -> str:
    return "general"


def _extract_json_object(text: str) -> dict[str, Any]:
    """Recover a JSON object from a model response.

    Models wrap JSON in prose or fences even when told not to, and a judge that
    fails on a stray "Here is the result:" would silently cost diagnoses. Three
    attempts: a fenced block, the whole string, then a raw decode from the first
    brace, which tolerates trailing commentary.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty judge response")

    fenced = _JSON_FENCE.search(text)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            if start == -1:
                continue
            try:
                parsed, _ = json.JSONDecoder().raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, Mapping):
            return dict(parsed)
        if isinstance(parsed, list):
            # A model that answered with the bare array rather than the object.
            return {"occurrences": parsed}

    raise ValueError(f"no JSON object in judge response: {text[:200]!r}")
