"""HotpotQA task and gold types, with the same gold-blindness discipline as D008.

Gold blindness is enforced *structurally*, not by care: :class:`Task` is frozen
with ``slots`` and has no field a gold answer or supporting-fact title could
travel on. Gold is split off at load time into a separate :class:`Gold` that the
program never receives. On SWE-Bench that discipline caught a real leak; there
is no reason to relax it here just because the task is cheaper.

What counts as gold on HotpotQA
-------------------------------
Both the ``answer`` and the ``supporting_facts`` titles. The titles matter as
much as the answer: they are what the *feedback function* is computed from, and
handing them to a retrieval module at inference time would let it retrieve the
gold documents by name -- scoring the oracle rather than the program.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Task:
    """What the program is allowed to see. No gold can be attached: frozen+slots."""

    example_id: str
    question: str
    level: str = ""
    type: str = ""

    def to_prompt_context(self) -> dict[str, str]:
        return {"question": self.question}


@dataclass(frozen=True, slots=True)
class Gold:
    """What only the grader and the feedback function may see."""

    example_id: str
    answer: str
    #: Distinct supporting-fact titles, order-insensitive. HotpotQA lists one
    #: entry per supporting SENTENCE, so titles repeat; retrieval is scored at
    #: document level, so they are deduplicated here rather than at every use.
    titles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Instance:
    task: Task
    gold: Gold


def _titles_from(supporting_facts: Mapping[str, object] | None) -> tuple[str, ...]:
    if not supporting_facts:
        return ()
    raw = supporting_facts.get("title") if isinstance(supporting_facts, Mapping) else None
    if not isinstance(raw, Iterable) or isinstance(raw, str):
        return ()
    seen: dict[str, None] = {}
    for title in raw:
        seen.setdefault(str(title), None)
    return tuple(seen)


def instance_from_record(record: Mapping[str, object]) -> Instance:
    """Split one dataset row into the gold-free half and the gold half."""
    example_id = str(record["id"])
    return Instance(
        task=Task(
            example_id=example_id,
            question=str(record.get("question") or "").strip(),
            level=str(record.get("level") or ""),
            type=str(record.get("type") or ""),
        ),
        gold=Gold(
            example_id=example_id,
            answer=str(record.get("answer") or "").strip(),
            titles=_titles_from(record.get("supporting_facts")),  # type: ignore[arg-type]
        ),
    )


class GoldLeakError(AssertionError):
    """A gold value reached a prompt. The run fails rather than scoring itself."""


def assert_gold_free(text: str, gold: Gold, *, where: str) -> None:
    """Value-based leak check. **NOT used by the HotpotQA program. Do not add it.**

    Kept because it is a useful check for benchmarks whose gold is distinctive
    text, and because deleting it would invite someone to reinvent it here.

    It does not work for open-domain QA, and the reason is structural rather
    than fixable. Gold supporting-fact titles are the Wikipedia article names of
    the entities a question asks about, so they legitimately appear:

    * in the question itself -- measured at **66% of val instances**;
    * in every retrieved passage that retrieval correctly finds (hop-1 recall
      is 64.5%);
    * in any summary a module writes from such a passage.

    Applied to this pipeline it does not detect leaks, it detects retrieval
    working -- and because the adapter scores a raised rollout 0.0, it converts
    a healthy run into an all-zero one that looks exactly like a real negative
    result. That cost a killed run to learn.

    Gold blindness here is enforced by construction instead: :class:`Task` has
    no field gold can travel on, and the program is handed nothing else.
    """
    haystack = text.lower()
    for title in gold.titles:
        if title and title.lower() in haystack:
            raise GoldLeakError(f"gold supporting-fact title {title!r} appeared in the {where}")
    answer = gold.answer.strip()
    if len(answer) >= 12 and answer.lower() in haystack:
        raise GoldLeakError(f"gold answer {answer!r} appeared in the {where}")
