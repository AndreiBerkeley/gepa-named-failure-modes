"""The HoVer candidate program: a 4-module, 3-hop retrieval chain.

This is the published ``HoverMultiHop`` program from LangProBe, the one GEPA's
paper uses. Our HotpotQA program was already a port of it with the last hop
answering instead of writing another query -- so this restores that hop rather
than inventing anything:

    retrieve(claim)     ->  summarize1
                        ->  create_query_hop2  ->  retrieve(query2)
                        ->  summarize2
                        ->  create_query_hop3  ->  retrieve(query3)

Four LM calls and three retrievals per rollout, fixed. Cost predictability is
load-bearing: three seeds under a fixed dollar budget must be comparable, and a
variable-cost program would give one seed more iterations than another.

Note the program returns **no answer**. The output is the union of retrieved
documents, and the score is whether the gold set is inside it. The SUPPORTED /
NOT_SUPPORTED label is never predicted.

``k`` follows the pilot: 7 on the first two hops, 10 on the last. The wider
final hop is deliberate -- it is the only chance to recover documents the
earlier hops missed, and it is not followed by a summarisation step whose
prompt would grow with it.

Gold blindness
--------------
Structural, and structurally only: :class:`Task` is frozen with ``slots`` and
carries no field a supporting-fact title could travel on. The value-based audit
is NOT applied, for the same reason as HotpotQA (F027): on this task gold titles
are the article names of entities the claim names outright, so they legitimately
appear in the claim, in every correctly retrieved passage, and in any summary
written from one. Auditing for them detects retrieval working, not leaking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from gepa_taxonomy.cost import CostMeter, Phase

# Shared with HotpotQA on purpose: same 2017 Wikipedia abstracts corpus, same
# BM25 index on disk. Duplicating the retriever would risk the two benchmarks
# silently drifting apart on tokenisation or k-handling, which would make their
# retrieval numbers incomparable.
from gepa_taxonomy.hotpotqa.retrieval import Passage, WikiRetriever, render_passages
from gepa_taxonomy.hover.tasks import Task

SUMMARIZE1 = "summarize1"
CREATE_QUERY_HOP2 = "create_query_hop2"
SUMMARIZE2 = "summarize2"
CREATE_QUERY_HOP3 = "create_query_hop3"
COMPONENTS = (SUMMARIZE1, CREATE_QUERY_HOP2, SUMMARIZE2, CREATE_QUERY_HOP3)


class LMClient(Protocol):
    def complete(self, prompt: str, *, max_tokens: int = 1024) -> tuple[str, int, int]:
        """Return ``(text, input_tokens, output_tokens)``."""
        ...


# DSPy signature defaults, transcribed from the LangProBe HoverMultiHop program
# (`claim, passages -> summary`, `claim, summary_1 -> query`, and so on). They
# are deliberately plain: GEPA's job is to improve them, and seeding from an
# already-optimized prompt would start the baseline from a searched point and
# destroy the comparison.
SEED_CANDIDATE: dict[str, str] = {
    SUMMARIZE1: "Given the fields `claim`, `passages`, produce the fields `summary`.",
    CREATE_QUERY_HOP2: "Given the fields `claim`, `summary_1`, produce the fields `query`.",
    SUMMARIZE2: "Given the fields `claim`, `context`, `passages`, produce the fields `summary`.",
    CREATE_QUERY_HOP3: "Given the fields `claim`, `summary_1`, `summary_2`, produce the fields `query`.",
}

SUMMARIZE1_PROMPT = """{instruction}

claim: {claim}

passages:
{passages}

Respond with the summary only."""

CREATE_QUERY_HOP2_PROMPT = """{instruction}

claim: {claim}

summary_1: {summary_1}

Respond with the search query only."""

SUMMARIZE2_PROMPT = """{instruction}

claim: {claim}

context: {context}

passages:
{passages}

Respond with the summary only."""

CREATE_QUERY_HOP3_PROMPT = """{instruction}

claim: {claim}

summary_1: {summary_1}

summary_2: {summary_2}

Respond with the search query only."""


@dataclass
class ModuleCall:
    """One component's turn. Consumed by the failure-taxonomy trace contract."""

    component: str
    prompt: str
    output: str
    input: str = ""
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "input": self.input,
            "prompt": self.prompt,
            "output": self.output,
        }


@dataclass
class Rollout:
    """One complete pass through the program; also the trace record."""

    example_id: str
    claim: str
    passages_hop1: list[Passage] = field(default_factory=list)
    passages_hop2: list[Passage] = field(default_factory=list)
    passages_hop3: list[Passage] = field(default_factory=list)
    summary_1: str = ""
    query_hop2: str = ""
    summary_2: str = ""
    query_hop3: str = ""
    calls: list[ModuleCall] = field(default_factory=list)
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def hop_titles(self) -> tuple[tuple[str, ...], ...]:
        """Titles retrieved at each hop, in order. Grading needs the per-hop
        split to attribute a miss to the hop that should have caught it."""
        return tuple(
            tuple(p.title for p in hop)
            for hop in (self.passages_hop1, self.passages_hop2, self.passages_hop3)
        )

    @property
    def retrieved_titles(self) -> tuple[str, ...]:
        """Distinct titles across all hops, in first-seen order."""
        seen: dict[str, None] = {}
        for passage in [*self.passages_hop1, *self.passages_hop2, *self.passages_hop3]:
            seen.setdefault(passage.title, None)
        return tuple(seen)

    @property
    def tokens(self) -> tuple[int, int]:
        return (sum(c.tokens_in for c in self.calls), sum(c.tokens_out for c in self.calls))

    def to_trace(self) -> dict[str, Any]:
        """The trajectory handed to the adapter and the taxonomy wrapper.

        ``module_calls`` carries FULL prompts and outputs, not digests -- a
        trace of digests cannot be judged and cannot seed taxonomy generation
        (F012).
        """
        return {
            "example_id": self.example_id,
            "instance_id": self.example_id,
            "task": self.claim,
            "module_calls": [c.to_dict() for c in self.calls],
            "retrieved_titles": list(self.retrieved_titles),
            # Per-hop, not just the union: grading attributes a miss to the hop
            # that should have caught it, and a replayed rollout has to be
            # regradable from the trace alone (the shared base-val cache stores
            # exactly this and nothing else).
            "hop_titles": [list(h) for h in self.hop_titles],
            "query_hop2": self.query_hop2,
            "query_hop3": self.query_hop3,
            "cost_usd": self.cost_usd,
            "error": self.error,
        }


@dataclass
class HoverMultiHopProgram:
    """Four LM calls around a fixed three-hop BM25 retriever."""

    retriever: WikiRetriever
    lm: LMClient
    meter: CostMeter
    model: str
    k: int = 7
    #: The last hop retrieves wider; it is the final chance to recover a miss.
    k_final: int = 10
    max_tokens: int = 1024

    def run(self, task: Task, candidate: dict[str, str], *, phase: Phase = "optimization") -> Rollout:
        rollout = Rollout(example_id=task.example_id, claim=task.claim)

        def call(component: str, prompt: str, *, input_label: str) -> str:
            text, tin, tout = self.lm.complete(prompt, max_tokens=self.max_tokens)
            rollout.cost_usd += self.meter.record(model=self.model, input_tokens=tin, output_tokens=tout, phase=phase)
            rollout.calls.append(
                ModuleCall(
                    component=component,
                    prompt=prompt,
                    output=text.strip(),
                    input=input_label,
                    tokens_in=tin,
                    tokens_out=tout,
                )
            )
            return text.strip()

        # Hop 1 -----------------------------------------------------------
        rollout.passages_hop1 = self.retriever.retrieve(task.claim, k=self.k)
        rollout.summary_1 = call(
            SUMMARIZE1,
            SUMMARIZE1_PROMPT.format(
                instruction=candidate[SUMMARIZE1],
                claim=task.claim,
                passages=render_passages(rollout.passages_hop1),
            ),
            input_label=f"claim + {len(rollout.passages_hop1)} hop-1 passages",
        )

        # Hop 2 query -----------------------------------------------------
        rollout.query_hop2 = call(
            CREATE_QUERY_HOP2,
            CREATE_QUERY_HOP2_PROMPT.format(
                instruction=candidate[CREATE_QUERY_HOP2],
                claim=task.claim,
                summary_1=rollout.summary_1,
            ),
            input_label="claim + summary_1",
        )

        # Hop 2 -----------------------------------------------------------
        rollout.passages_hop2 = self.retriever.retrieve(rollout.query_hop2, k=self.k)
        rollout.summary_2 = call(
            SUMMARIZE2,
            SUMMARIZE2_PROMPT.format(
                instruction=candidate[SUMMARIZE2],
                claim=task.claim,
                context=rollout.summary_1,
                passages=render_passages(rollout.passages_hop2),
            ),
            input_label=f"claim + summary_1 as context + {len(rollout.passages_hop2)} hop-2 passages",
        )

        # Hop 3 query -----------------------------------------------------
        rollout.query_hop3 = call(
            CREATE_QUERY_HOP3,
            CREATE_QUERY_HOP3_PROMPT.format(
                instruction=candidate[CREATE_QUERY_HOP3],
                claim=task.claim,
                summary_1=rollout.summary_1,
                summary_2=rollout.summary_2,
            ),
            input_label="claim + summary_1 + summary_2",
        )

        # Hop 3 -- retrieval only; nothing summarises it, it IS the output ---
        rollout.passages_hop3 = self.retriever.retrieve(rollout.query_hop3, k=self.k_final)
        return rollout
