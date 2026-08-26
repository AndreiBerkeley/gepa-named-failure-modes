"""Seed candidate and fixed scaffolding for the AppWorld ReAct agent.

Both are sliced out of the **published** prompt GEPA was applied to
(``assets/react_published.txt``, taken verbatim from ACE's AppWorld repo), so
the starting point is the real one rather than something we wrote.

What is optimizable, and why only this
--------------------------------------
The published file is 267 lines: 30 of intro and API guidance, **215 of a single
worked demonstration**, 18 of numbered key instructions, and a 5-line task
block. GEPA optimizes **one** component here (D046), and we scope it to the
guidance — intro plus key instructions, ~48 lines — leaving the demonstration
and the task block as fixed scaffolding.

That follows GEPA's convention everywhere else in the paper: the *instruction* is
the component; demonstrations and inputs are templated in. The literal
alternative would hand GEPA the whole 7.6 KB file including the demonstration,
which is an order of magnitude larger than any component in the paper, costs far
more per reflection, and invites it to rewrite a worked example whose value is
that it is correct.

The trade is stated plainly because it weakens one thing: ACE ran GEPA over the
whole template, so our 46.4% reference point is now approximate rather than
exact. It was already approximate — they used a smaller open-source model and we
run Haiku 4.5 (F037).
"""

from __future__ import annotations

from pathlib import Path

PUBLISHED = Path(__file__).with_name("assets") / "react_published.txt"

#: Component name. One component, so this is the whole optimizable surface.
REACT = "react_instruction"
COMPONENTS = (REACT,)

#: Section boundaries in the published file, verified against its role turns:
#: the first USER turn ends where the first ASSISTANT turn begins (line 31), and
#: the key instructions open at the last USER turn before the task block.
_INTRO_END = 30
_KEY_INSTRUCTIONS_MARKER = "**Key instructions**:"
_TASK_MARKER = "Using these APIs, now generate code to solve the actual task:"


def _sections() -> tuple[str, str]:
    """Return (instruction, demonstration) from the published prompt."""
    text = PUBLISHED.read_text(encoding="utf-8")
    lines = text.splitlines()

    key_at = next(i for i, line in enumerate(lines) if _KEY_INSTRUCTIONS_MARKER in line)
    task_at = next(i for i, line in enumerate(lines) if _TASK_MARKER in line)

    intro = "\n".join(lines[:_INTRO_END]).strip()
    demonstration = "\n".join(lines[_INTRO_END:key_at]).strip()
    # The key-instructions block runs from its header to the task block, minus
    # the "USER:" line that introduces the task.
    key_instructions = "\n".join(lines[key_at : task_at - 1]).strip()

    return f"{intro}\n\n{key_instructions}", demonstration


SEED_INSTRUCTION, DEMONSTRATION = _sections()

#: The seed candidate GEPA starts from. Deliberately the published text: a
#: hand-improved seed would confound the baseline, and the whole point of the
#: comparison is what reflection adds on top of the published starting point.
SEED_CANDIDATE: dict[str, str] = {REACT: SEED_INSTRUCTION}

#: Rendered once per task. ``{instruction}`` is the optimizable component;
#: everything else is scaffolding the optimizer never sees the inside of.
TASK_PROMPT = """{instruction}

{demonstration}

USER:
Using these APIs, now generate code to solve the actual task:

My name is: {first_name} {last_name}. My personal email is {email} and phone number is {phone_number}.
Task: {task}
"""
