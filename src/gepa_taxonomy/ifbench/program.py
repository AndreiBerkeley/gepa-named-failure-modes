"""The IFBench candidate program: generate_response -> ensure_correct_response.

    generate_response(prompt)                  -> draft
    ensure_correct_response(prompt, draft)     -> response   <- this is graded

Exactly two LM calls per rollout, fixed, and no retrieval. Cost predictability is
load-bearing: three seeds under one dollar budget must be comparable, and a
variable-cost program gives one seed more iterations than another.

Faithful to the published program
---------------------------------
Module names and seed instructions are GEPA's own, transcribed **verbatim** from
the paper's LaTeX source (Appendix L, "IFBench, GPT-4.1 Mini", the ``Base
Prompt`` blocks -- not the MIPROv2 or GEPA-optimised blocks printed alongside
them). They are deliberately terse; GEPA's job is to improve them, and seeding
from anything already searched would destroy the comparison. This is the same
discipline used for HotpotQA.

Gold blindness
--------------
Structural: :class:`Task` is frozen with ``slots`` and carries only the prompt,
:class:`Gold` holds the verifier ids and their arguments, and ``run()`` receives
only a ``Task``. The natural-language constraint is in the prompt by design --
that is the task -- but the machine-readable spec the grader uses never reaches
a module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from gepa_taxonomy.cost import CostMeter, Phase
from gepa_taxonomy.ifbench.tasks import Task

GENERATE = "generate_response_module"
ENSURE = "ensure_correct_response_module"
COMPONENTS = (GENERATE, ENSURE)


class LMClient(Protocol):
    def complete(self, prompt: str, *, max_tokens: int = 1024) -> tuple[str, int, int]:
        """Return ``(text, input_tokens, output_tokens)``."""
        ...


#: Verbatim from GEPA Appendix L, "IFBench, GPT-4.1 Mini" Base Prompt blocks.
SEED_CANDIDATE: dict[str, str] = {
    GENERATE: "Respond to the query",
    ENSURE: (
        "Ensure the response is correct and adheres to the given constraints. "
        "Your response will be used as the final response."
    ),
}

GENERATE_PROMPT = """{instruction}

query:
{query}"""

ENSURE_PROMPT = """{instruction}

query:
{query}

response:
{response}"""


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
    query: str = ""
    draft: str = ""
    response: str = ""
    calls: list[ModuleCall] = field(default_factory=list)
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def tokens(self) -> tuple[int, int]:
        return (sum(c.tokens_in for c in self.calls), sum(c.tokens_out for c in self.calls))

    def to_trace(self) -> dict[str, Any]:
        """The trajectory handed to the adapter and to the taxonomy wrapper.

        ``module_calls`` carries FULL prompts and outputs, not digests: a trace of
        digests cannot be judged and cannot seed taxonomy generation.
        """
        return {
            "example_id": self.example_id,
            "instance_id": self.example_id,
            "task": self.query,
            "module_calls": [c.to_dict() for c in self.calls],
            "draft": self.draft,
            "response": self.response,
            "cost_usd": self.cost_usd,
            "error": self.error,
        }


@dataclass
class GenerateEnsureProgram:
    """Two LM calls: draft a response, then check it against the constraints."""

    lm: LMClient
    meter: CostMeter
    model: str
    #: A ceiling, not a reservation -- billing is on tokens produced, so a
    #: generous cap costs nothing and avoids scoring a truncated response as
    #: non-compliant. Several IFBench constraints demand long outputs
    #: (word-count ranges, per-sentence rules), so this is not academic.
    max_tokens: int = 4096

    def run(self, task: Task, candidate: dict[str, str], *, phase: Phase = "optimization") -> Rollout:
        rollout = Rollout(example_id=task.example_id, query=task.prompt)

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

        rollout.draft = call(
            GENERATE,
            GENERATE_PROMPT.format(instruction=candidate[GENERATE], query=task.prompt),
            input_label="query",
        )
        rollout.response = call(
            ENSURE,
            ENSURE_PROMPT.format(
                instruction=candidate[ENSURE],
                query=task.prompt,
                response=rollout.draft,
            ),
            input_label="query + draft response",
        )
        return rollout
