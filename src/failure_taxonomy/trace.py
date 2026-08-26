"""Segmented traces: the contract that lets this work on any GEPA program.

GEPA treats trajectories as opaque -- "the engine never inspects them"
(``gepa/core/adapter.py``) -- so their contents are entirely adapter-defined.
That freedom is why taxonomy conditioning needs a contract of its own: to
attribute a failure to the component that caused it, something has to know
which component produced which text.

The contract
------------
A trajectory may expose a list of per-component calls under ``module_calls``
(or ``component_calls``), as either a mapping key or an attribute. Each entry
carries the component name and any of ``input`` / ``prompt`` / ``output``::

    {"module_calls": [
        {"component": "summarize1", "input": ..., "prompt": ..., "output": ...},
        {"component": "create_query_hop2", ...},
    ]}

Adapters that do not supply it still work. The trajectory is then rendered
whole as a single unattributed segment, the judge has no component vocabulary
to attribute to, and every occurrence comes back general -- which routes it to
every component. Precision degrades; nothing breaks. That is deliberate: the
feature has to be usable against an existing adapter without rewriting it, with
per-component precision as the reward for adding the capture.

Why segmentation matters more than it looks
-------------------------------------------
Downstream prompts routinely quote upstream outputs -- a refiner's prompt
embeds the solver's candidate output verbatim. A judge shown flat trace text
cannot tell which component *authored* a span from which merely *received* it.
Showing the structure explicitly is what makes attribution answerable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Trajectory keys/attributes checked for per-component calls, in order.
CALLS_KEYS = ("module_calls", "component_calls")

#: Per-call keys, in the order they are rendered for the judge.
_CALL_TEXT_FIELDS = (("input", "INPUT"), ("prompt", "PROMPT"), ("output", "OUTPUT"))


@dataclass(frozen=True, slots=True)
class ComponentCall:
    """One component's turn within a rollout.

    ``prompt`` is what the component was actually sent and ``output`` is what it
    produced. ``input`` is optional context for cases where the prompt embeds a
    large payload verbatim (retrieved sources, a candidate patch) and naming it
    separately is clearer than making the judge find it.
    """

    component: str
    input: str = ""
    prompt: str = ""
    output: str = ""

    @classmethod
    def from_any(cls, raw: Any) -> ComponentCall | None:
        """Build from a mapping or an object, or return None if unusable."""

        def pick(key: str) -> Any:
            if isinstance(raw, Mapping):
                return raw.get(key)
            return getattr(raw, key, None)

        component = pick("component") or pick("name") or pick("module")
        if not component:
            return None
        return cls(
            component=str(component),
            input=_as_text(pick("input")),
            prompt=_as_text(pick("prompt")),
            output=_as_text(pick("output")),
        )

    def render(self) -> str:
        body = [f"[COMPONENT: {self.component}]"]
        for attr, label in _CALL_TEXT_FIELDS:
            value = getattr(self, attr)
            if value:
                body.append(f"--- {self.component} {label} ---\n{value}")
        return "\n".join(body)


@dataclass(frozen=True, slots=True)
class SegmentedTrace:
    """One rollout, split into the components that produced it."""

    trace_id: str
    task: str = ""
    calls: tuple[ComponentCall, ...] = ()
    #: Whole-trajectory text, used when no per-component calls were supplied.
    fallback_text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def components(self) -> tuple[str, ...]:
        """Component names in first-appearance order, deduplicated.

        A component may legitimately appear more than once (a loop, a retry);
        the vocabulary offered to the judge lists each name once.
        """
        seen: dict[str, None] = {}
        for call in self.calls:
            seen.setdefault(call.component, None)
        return tuple(seen)

    @property
    def is_segmented(self) -> bool:
        return bool(self.calls)

    def render(self) -> str:
        """The trace as shown to the judge.

        Steps are numbered so the judge can refer to order, and each carries its
        component name so attribution has something to attribute to.
        """
        parts: list[str] = []
        if self.task:
            parts.append(f"[TASK]\n{self.task}")
        if self.calls:
            for i, call in enumerate(self.calls, start=1):
                parts.append(f"[STEP {i} of {len(self.calls)}]\n{call.render()}")
        elif self.fallback_text:
            parts.append(f"[TRAJECTORY]\n{self.fallback_text}")
        return "\n\n".join(parts)

    def to_generation_record(self) -> dict[str, Any]:
        """An AdaMAST-native record for the taxonomy-generation stage.

        The same segmentation feeds both stages. That is not tidiness for its
        own sake: a generator that has to recover component structure from trace
        prose can latch onto whatever the trace happens to embed -- source code,
        payloads -- and invent agents that do not exist. Handing it the real
        component names removes the guess.
        """
        return {
            "problem_id": self.trace_id,
            "task": self.task,
            "raw_trajectory": self.render(),
            "metadata": {**dict(self.metadata), "components": list(self.components)},
        }


def _as_text(value: Any) -> str:
    """Render an arbitrary trajectory value as text a judge can read.

    Structured values are JSON-dumped rather than ``str()``-ed. Many adapters
    store a trajectory as a plain dict of fields, and Python's repr of a dict is
    noticeably worse to read than indented JSON -- single quotes, no line
    breaks, nested reprs. Empty containers render as the empty string so the
    caller can distinguish "no trajectory" from "a trajectory that says {}".
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping | Sequence):
        if not value:
            return ""
        try:
            return json.dumps(value, indent=2, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def extract_calls(trajectory: Any) -> list[ComponentCall]:
    """Pull per-component calls out of an adapter-defined trajectory.

    Returns an empty list when the trajectory does not follow the contract --
    that is the ordinary unsegmented case, not an error.
    """
    raw_calls: Any = None
    for key in CALLS_KEYS:
        if isinstance(trajectory, Mapping):
            raw_calls = trajectory.get(key)
        else:
            raw_calls = getattr(trajectory, key, None)
        if raw_calls:
            break
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, str | bytes):
        return []
    calls = [ComponentCall.from_any(entry) for entry in raw_calls]
    return [c for c in calls if c is not None]


def build_trace(
    trajectory: Any,
    *,
    trace_id: str,
    task: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> SegmentedTrace:
    """Build a :class:`SegmentedTrace` from one adapter trajectory."""
    calls = extract_calls(trajectory)
    fallback = "" if calls else _as_text(trajectory)
    if not task and isinstance(trajectory, Mapping):
        task = _as_text(trajectory.get("task") or trajectory.get("problem_statement"))
    return SegmentedTrace(
        trace_id=trace_id,
        task=task,
        calls=tuple(calls),
        fallback_text=fallback,
        metadata=dict(metadata or {}),
    )
