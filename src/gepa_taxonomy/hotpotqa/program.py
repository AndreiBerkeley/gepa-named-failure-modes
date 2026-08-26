"""The HotpotQA candidate program: a 4-module multi-hop retrieval chain.

Ported from the published GEPA program (appendix E.1 -- the HoVerMultiHop
program from LangProBe with the last hop answering instead of writing another
query). Four optimizable components, two retrieval hops:

    retrieve(question)  ->  summarize1
                        ->  create_query_hop2  ->  retrieve(query)
                        ->  summarize2
                        ->  final_answer

Exactly four LM calls per rollout, fixed. Cost predictability is load-bearing:
three seeds under a fixed dollar budget must be comparable, and a variable-cost
agent would give one seed far more iterations than another, confounding the
comparison we are running.

Gold blindness
--------------
Enforced **structurally, and structurally only**: :class:`Task` is frozen with
``slots`` and has no field a gold answer or supporting-fact title could travel
on, :class:`Gold` is a separate object, and ``run()`` receives only a ``Task``.
There is no code path by which gold can reach a prompt.

The value-based audit used on SWE-Bench (``assert_gold_free`` on every prompt)
is deliberately NOT applied here, and that is a considered decision rather than
an omission. On SWE-Bench a gold patch is distinctive text that could never
legitimately appear in a prompt. On open-domain QA the opposite holds: gold
supporting-fact titles are the Wikipedia article names of the entities the
question asks about, so they appear **in the question itself in 66% of val
instances** (measured), in every retrieved passage that retrieval correctly
finds, and in any summary derived from one. Auditing for them does not detect
leaks; it detects retrieval working. See F027 -- it cost a killed run to learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from gepa_taxonomy.cost import CostMeter, Phase
from gepa_taxonomy.hotpotqa.retrieval import Passage, WikiRetriever, render_passages
from gepa_taxonomy.hotpotqa.tasks import Task

SUMMARIZE1 = "summarize1"
CREATE_QUERY_HOP2 = "create_query_hop2"
SUMMARIZE2 = "summarize2"
FINAL_ANSWER = "final_answer"
COMPONENTS = (SUMMARIZE1, CREATE_QUERY_HOP2, SUMMARIZE2, FINAL_ANSWER)


class LMClient(Protocol):
    def complete(self, prompt: str, *, max_tokens: int = 1024) -> tuple[str, int, int]:
        """Return ``(text, input_tokens, output_tokens)``."""
        ...


# Seed instructions, transcribed VERBATIM from GEPA appendix L ("Base Prompt"
# blocks, HotpotQA / GPT-4.1 Mini). They are DSPy signature defaults, and they
# are deliberately plain: GEPA's job is to improve them, and seeding from the
# paper's *optimized* prompts would start the baseline from an already-searched
# point and destroy the comparison.
SEED_CANDIDATE: dict[str, str] = {
    SUMMARIZE1: "Given the fields `question`, `passages`, produce the fields `summary`.",
    CREATE_QUERY_HOP2: "Given the fields `question`, `summary_1`, produce the fields `query`.",
    SUMMARIZE2: "Given the fields `question`, `context`, `passages`, produce the fields `summary`.",
    FINAL_ANSWER: "Given the fields `question`, `summary_1`, `summary_2`, produce the fields `answer`.",
}

SUMMARIZE1_PROMPT = """{instruction}

question: {question}

passages:
{passages}

Respond with the summary only."""

CREATE_QUERY_HOP2_PROMPT = """{instruction}

question: {question}

summary_1: {summary_1}

Respond with the search query only."""

SUMMARIZE2_PROMPT = """{instruction}

question: {question}

context: {context}

passages:
{passages}

Respond with the summary only."""

FINAL_ANSWER_PROMPT = """{instruction}

question: {question}

summary_1: {summary_1}

summary_2: {summary_2}

Respond with the answer only, as briefly as possible."""


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
    question: str
    passages_hop1: list[Passage] = field(default_factory=list)
    passages_hop2: list[Passage] = field(default_factory=list)
    summary_1: str = ""
    query_hop2: str = ""
    summary_2: str = ""
    answer: str = ""
    calls: list[ModuleCall] = field(default_factory=list)
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def retrieved_titles(self) -> tuple[str, ...]:
        """Distinct titles across both hops, in first-seen order."""
        seen: dict[str, None] = {}
        for passage in [*self.passages_hop1, *self.passages_hop2]:
            seen.setdefault(passage.title, None)
        return tuple(seen)

    @property
    def tokens(self) -> tuple[int, int]:
        return (sum(c.tokens_in for c in self.calls), sum(c.tokens_out for c in self.calls))

    def to_trace(self) -> dict[str, Any]:
        """The trajectory handed to the adapter and to the taxonomy wrapper.

        ``module_calls`` carries the FULL prompts and outputs, not digests. A
        trace that keeps only digests cannot be judged and cannot be used for
        taxonomy generation -- a mistake this project has already paid for once
        (F012), and the reason a cached rollout could not be judged at all.
        """
        return {
            "example_id": self.example_id,
            "instance_id": self.example_id,
            "task": self.question,
            "module_calls": [c.to_dict() for c in self.calls],
            "retrieved_titles": list(self.retrieved_titles),
            "query_hop2": self.query_hop2,
            "answer": self.answer,
            "cost_usd": self.cost_usd,
            "error": self.error,
        }


@dataclass
class MultiHopProgram:
    """Four LM calls around a fixed two-hop BM25 retriever."""

    retriever: WikiRetriever
    lm: LMClient
    meter: CostMeter
    model: str
    k: int = 10
    max_tokens: int = 1024

    def run(self, task: Task, candidate: dict[str, str], *, phase: Phase = "optimization") -> Rollout:
        rollout = Rollout(example_id=task.example_id, question=task.question)

        def call(component: str, prompt: str, *, input_label: str, retrieved: str = "") -> str:
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
        rollout.passages_hop1 = self.retriever.retrieve(task.question, k=self.k)
        hop1_text = render_passages(rollout.passages_hop1)
        rollout.summary_1 = call(
            SUMMARIZE1,
            SUMMARIZE1_PROMPT.format(
                instruction=candidate[SUMMARIZE1],
                question=task.question,
                passages=hop1_text,
            ),
            input_label=f"question + {len(rollout.passages_hop1)} hop-1 passages",
            retrieved=hop1_text,
        )

        # Hop 2 query -----------------------------------------------------
        rollout.query_hop2 = call(
            CREATE_QUERY_HOP2,
            CREATE_QUERY_HOP2_PROMPT.format(
                instruction=candidate[CREATE_QUERY_HOP2],
                question=task.question,
                summary_1=rollout.summary_1,
            ),
            input_label="question + summary_1",
        )

        # Hop 2 -----------------------------------------------------------
        rollout.passages_hop2 = self.retriever.retrieve(rollout.query_hop2, k=self.k)
        hop2_text = render_passages(rollout.passages_hop2)
        rollout.summary_2 = call(
            SUMMARIZE2,
            SUMMARIZE2_PROMPT.format(
                instruction=candidate[SUMMARIZE2],
                question=task.question,
                context=rollout.summary_1,
                passages=hop2_text,
            ),
            input_label=f"question + summary_1 as context + {len(rollout.passages_hop2)} hop-2 passages",
            retrieved=hop2_text,
        )

        # Answer ----------------------------------------------------------
        rollout.answer = call(
            FINAL_ANSWER,
            FINAL_ANSWER_PROMPT.format(
                instruction=candidate[FINAL_ANSWER],
                question=task.question,
                summary_1=rollout.summary_1,
                summary_2=rollout.summary_2,
            ),
            input_label="question + summary_1 + summary_2",
        )
        return rollout
