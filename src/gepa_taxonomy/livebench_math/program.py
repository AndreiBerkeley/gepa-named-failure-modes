"""The LiveBench-Math candidate program: a two-module solve -> review pipeline.

    solve(problem)                 -> draft_answer
    review(problem, draft_answer)  -> answer      <- this is what gets graded

Exactly two LM calls per rollout, fixed. Cost predictability is load-bearing:
three seeds under one dollar budget must be comparable, and a variable-cost
agent gives one seed more iterations than another -- which is what ruled
AppWorld out (D049).

Why two modules and not one
---------------------------
A single ``solve`` module would make failure attribution unary, and the whole
premise of the treatment arm is that a taxonomy routes failures to the component
responsible. Two modules is also the shape GEPA's own published IFBench program
uses (``generate_response_module`` -> ``ensure_correct_response_module``), so it
is a precedented pipeline rather than one invented for this experiment.

``review`` is deliberately the interesting module. Self-correction on math is
known to be double-edged -- models talk themselves out of correct answers as
often as they repair wrong ones -- so when to override and when to leave alone
is a genuinely prompt-addressable decision, and "review discarded a correct
draft" is a failure mode a taxonomy can name and a prompt can fix. It is also
where the two arms are most likely to diverge.

Answer format
-------------
Each LiveBench problem statement carries its own output-format instruction
("put your final answer in a $\\boxed{}$", "repeat the letter five times").
Those are part of the benchmark and are passed through verbatim; every scorer
parses the answer out of the format they specify. The candidate instructions
wrap that text, they never replace it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from gepa_taxonomy.cost import CostMeter, Phase
from gepa_taxonomy.livebench_math.tasks import Task

SOLVE = "solve"
REVIEW = "review"
COMPONENTS = (SOLVE, REVIEW)


class LMClient(Protocol):
    def complete(self, prompt: str, *, max_tokens: int = 1024) -> tuple[str, int, int]:
        """Return ``(text, input_tokens, output_tokens)``."""
        ...


#: Seed instructions in the DSPy signature style GEPA's published base prompts
#: use (appendix L, "Base Prompt" blocks). Deliberately plain: GEPA's job is to
#: improve them, and seeding from anything already tuned would start the
#: baseline from a searched point and destroy the comparison.
SEED_CANDIDATE: dict[str, str] = {
    SOLVE: "Given the fields `problem`, produce the fields `answer`.",
    REVIEW: "Given the fields `problem`, `draft_answer`, produce the fields `answer`.",
}

SOLVE_PROMPT = """{instruction}

problem:
{problem}"""

REVIEW_PROMPT = """{instruction}

problem:
{problem}

draft_answer:
{draft_answer}"""


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
    question: str = ""
    subtask: str = ""
    draft_answer: str = ""
    answer: str = ""
    calls: list[ModuleCall] = field(default_factory=list)
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def tokens(self) -> tuple[int, int]:
        return (sum(c.tokens_in for c in self.calls), sum(c.tokens_out for c in self.calls))

    def to_trace(self) -> dict[str, Any]:
        """The trajectory handed to the adapter and to the taxonomy wrapper.

        ``module_calls`` carries FULL prompts and outputs, not digests: a trace
        of digests cannot be judged and cannot seed taxonomy generation (F012).
        """
        return {
            "example_id": self.example_id,
            "instance_id": self.example_id,
            "task": self.question,
            "module_calls": [c.to_dict() for c in self.calls],
            "subtask": self.subtask,
            "draft_answer": self.draft_answer,
            "answer": self.answer,
            "cost_usd": self.cost_usd,
            "error": self.error,
        }


@dataclass
class SolveReviewProgram:
    """Two LM calls: draft an answer, then review it."""

    lm: LMClient
    meter: CostMeter
    model: str
    #: A ceiling, not a reservation -- billing is on tokens actually produced, so
    #: a generous cap costs nothing and avoids scoring a truncated-mid-reasoning
    #: response as a wrong answer. Competition solutions run long.
    max_tokens: int = 4096

    def run(self, task: Task, candidate: dict[str, str], *, phase: Phase = "optimization") -> Rollout:
        rollout = Rollout(example_id=task.example_id, question=task.question, subtask=task.subtask)

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

        rollout.draft_answer = call(
            SOLVE,
            SOLVE_PROMPT.format(instruction=candidate[SOLVE], problem=task.question),
            input_label="problem",
        )
        rollout.answer = call(
            REVIEW,
            REVIEW_PROMPT.format(
                instruction=candidate[REVIEW],
                problem=task.question,
                draft_answer=rollout.draft_answer,
            ),
            input_label="problem + draft_answer",
        )
        return rollout
