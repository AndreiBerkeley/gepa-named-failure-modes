"""HoVer task and gold types, with the same structural gold-blindness as HotpotQA.

HoVer is claim verification, but the benchmark as GEPA uses it is a **retrieval**
task: given a claim, find the 2-4 Wikipedia articles that support or refute it.
The SUPPORTED / NOT_SUPPORTED label is carried on :class:`Gold` for analysis but
is deliberately **not** what the program is scored on -- see ``grading.py``.

Gold blindness is enforced structurally, not by care: :class:`Task` is frozen
with ``slots`` and has no field a supporting-fact title could travel on. Gold is
split off at load time into a separate :class:`Gold` the program never receives.

Why ``num_hops`` is on the Task
-------------------------------
It is the number of reasoning hops the claim requires (2, 3 or 4), and it is a
property of the *claim*, not an answer -- a solver could in principle infer it
from the claim text. It is kept on the Task because splits stratify on it, and
because a per-hop-count score breakdown is the most informative view of where a
candidate improves. It leaks nothing: knowing a claim needs three hops does not
tell you which documents they are.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Task:
    """What the program is allowed to see. No gold can be attached: frozen+slots."""

    example_id: str
    claim: str
    num_hops: int = 0

    def to_prompt_context(self) -> dict[str, str]:
        return {"claim": self.claim}


@dataclass(frozen=True, slots=True)
class Gold:
    """What only the grader and the feedback function may see."""

    example_id: str
    #: Distinct supporting-article titles, order-insensitive. HoVer lists one
    #: entry per supporting SENTENCE, so titles repeat; retrieval is scored at
    #: document level, so they are deduplicated here rather than at every use.
    titles: tuple[str, ...]
    #: SUPPORTED / NOT_SUPPORTED. Carried for analysis only -- the metric is
    #: retrieval, so this never enters scoring. See grading.py.
    label: str = ""
    num_hops: int = 0


@dataclass(frozen=True, slots=True)
class Instance:
    task: Task
    gold: Gold


def _titles_from(supporting_facts: object) -> tuple[str, ...]:
    """Pull distinct article titles out of HoVer's supporting_facts.

    HoVer's shape is a list of ``[title, sentence_index]`` pairs -- NOT
    HotpotQA's dict-of-parallel-lists. Handing HotpotQA's extractor a HoVer
    record returns an empty tuple silently, which would score every rollout
    against no gold at all and look like total retrieval failure.
    """
    if not isinstance(supporting_facts, Iterable) or isinstance(supporting_facts, str | bytes):
        return ()
    seen: dict[str, None] = {}
    for fact in supporting_facts:
        title: object = None
        if isinstance(fact, Mapping):
            # Some exports normalise the pair into {"key": title, "value": idx}.
            title = fact.get("key") or fact.get("title")
        elif isinstance(fact, Sequence) and not isinstance(fact, str | bytes) and fact:
            title = fact[0]
        if title:
            seen.setdefault(str(title), None)
    return tuple(seen)


def instance_from_record(record: Mapping[str, object]) -> Instance:
    """Split one HoVer row into the gold-free half and the gold half."""
    example_id = str(record.get("uid") or record.get("id") or "")
    num_hops = int(record.get("num_hops") or 0)
    return Instance(
        task=Task(
            example_id=example_id,
            claim=str(record.get("claim") or "").strip(),
            num_hops=num_hops,
        ),
        gold=Gold(
            example_id=example_id,
            titles=_titles_from(record.get("supporting_facts")),
            label=str(record.get("label") or ""),
            num_hops=num_hops,
        ),
    )


def instances_by_id(instances: Iterable[Instance]) -> dict[str, Instance]:
    return {i.task.example_id: i for i in instances}
