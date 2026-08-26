"""The AppWorld candidate program: a single-component ReAct agent.

One optimizable instruction (D046), wrapped around a fixed multi-step loop:
the model writes Python, the environment executes it, the output is appended,
and the model writes the next step -- until it calls
``apis.supervisor.complete_task()`` or the step budget runs out.

Cost is bounded, not fixed
--------------------------
Unlike the HotpotQA program's exactly-four calls, a ReAct rollout takes as many
steps as it takes. That is inherent to the benchmark, but it is capped:
``max_steps`` bounds the worst case so a single pathological task cannot consume
a seed's budget. Steps taken are recorded per rollout, because "the optimizer
learned to be terser" and "the optimizer learned to give up early" look the same
in the score alone.

Trace shape
-----------
Every step is a ``ModuleCall`` under the same component name. ``SegmentedTrace``
de-duplicates the vocabulary, so the judge is offered one component and every
occurrence attributes to it -- attribution is unary here rather than degraded,
which is exactly the ablation D046 wants against HotpotQA's four components.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from gepa_taxonomy.appworld.client import AppWorldClient, TaskResult
from gepa_taxonomy.appworld.prompts import DEMONSTRATION, REACT, TASK_PROMPT

#: Hard ceiling on interaction steps. AppWorld's own baselines use a comparable
#: bound; without one, a task that never calls complete_task() runs forever.
DEFAULT_MAX_STEPS = 30

#: Fenced python block, which is what the published prompt asks the model for.
_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


class LMClient(Protocol):
    def complete(self, prompt: str, *, max_tokens: int = 2048) -> tuple[str, int, int]: ...


def extract_code(text: str) -> str:
    """Pull the first fenced python block out of a model turn.

    An unfenced response yields no code rather than executing the prose: running
    arbitrary model output as Python because it *might* be code is how a
    formatting failure becomes an environment mutation.
    """
    match = _CODE_FENCE.search(text or "")
    return match.group(1).strip() if match else ""


@dataclass
class ModuleCall:
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
    """One complete pass through the ReAct loop; also the trace record."""

    task_id: str
    instruction_text: str = ""
    steps: int = 0
    completed: bool = False
    #: True when the loop ended because it hit ``max_steps`` rather than because
    #: the agent said it was done. A distinct failure mode worth naming.
    exhausted_steps: bool = False
    empty_code_steps: int = 0
    calls: list[ModuleCall] = field(default_factory=list)
    result: TaskResult | None = None
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def score(self) -> float:
        return self.result.score if self.result is not None else 0.0

    def to_trace(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instance_id": self.task_id,
            "task": self.instruction_text,
            "module_calls": [c.to_dict() for c in self.calls],
            "steps": self.steps,
            "completed": self.completed,
            "exhausted_steps": self.exhausted_steps,
            "empty_code_steps": self.empty_code_steps,
            "cost_usd": self.cost_usd,
            "error": self.error,
        }


@dataclass
class ReActProgram:
    client: AppWorldClient
    lm: LMClient
    meter: Any
    model: str
    max_steps: int = DEFAULT_MAX_STEPS
    max_tokens: int = 2048

    def run(self, task_id: str, candidate: dict[str, str], *, phase: str = "optimization") -> Rollout:
        rollout = Rollout(task_id=task_id)
        try:
            info = self.client.initialize(task_id)
        except Exception as exc:
            rollout.error = f"{type(exc).__name__}: {exc}"
            return rollout

        supervisor = info.get("supervisor") or {}
        rollout.instruction_text = str(info.get("instruction") or "")
        base_prompt = TASK_PROMPT.format(
            instruction=candidate[REACT],
            demonstration=DEMONSTRATION,
            first_name=supervisor.get("first_name", ""),
            last_name=supervisor.get("last_name", ""),
            email=supervisor.get("email", ""),
            phone_number=supervisor.get("phone_number", ""),
            task=rollout.instruction_text,
        )

        transcript: list[str] = []
        previous_output = ""
        try:
            for _ in range(self.max_steps):
                prompt = base_prompt + "".join(transcript) + "\nASSISTANT:\n"
                text, tin, tout = self.lm.complete(prompt, max_tokens=self.max_tokens)
                rollout.cost_usd += self.meter.record(
                    model=self.model, input_tokens=tin, output_tokens=tout, phase=phase
                )
                rollout.steps += 1

                code = extract_code(text)
                if not code:
                    rollout.empty_code_steps += 1
                    output = "No code block found in your response. Reply with a ```python block."
                else:
                    output = self.client.execute(task_id, code)

                # Store the step's INCREMENT, not the cumulative prompt.
                #
                # Each step's prompt already contains every earlier turn, so
                # recording it whole makes the trace quadratic: a 30-step rollout
                # came to ~860 KB (~252k tokens) of overlapping copies. That is
                # past the judge's context window, and because judging fails soft
                # the treatment arm would have silently degraded to the baseline
                # on exactly the longest, most interesting rollouts. It also made
                # judging cost $0.76/rollout instead of $0.045.
                #
                # Rendering the calls in order reconstructs the full conversation
                # exactly, so nothing is lost: step 0 carries the base prompt,
                # and each later step carries only the environment output that
                # prompted it.
                rollout.calls.append(
                    ModuleCall(
                        component=REACT,
                        prompt=base_prompt if rollout.steps == 1 else previous_output,
                        output=text.strip(),
                        input=f"step {rollout.steps}",
                        tokens_in=tin,
                        tokens_out=tout,
                    )
                )
                previous_output = f"USER:\nOutput:\n{output}"
                transcript.append(f"{text.strip()}\n\nUSER:\nOutput:\n{output}\n")

                if code and self.client.task_completed(task_id):
                    rollout.completed = True
                    break
            else:
                rollout.exhausted_steps = True

            rollout.result = self.client.evaluate(task_id)
        except Exception as exc:
            rollout.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.client.close(task_id)
        return rollout
