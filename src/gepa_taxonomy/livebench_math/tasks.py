"""LiveBench-Math task and gold types, with the same gold-blindness as D008.

:class:`Task` is frozen with ``slots`` and has no field a ground-truth answer
could travel on; gold is split off at load time into a separate :class:`Gold`
the program never receives.

Unlike HotpotQA, a value-based audit *would* work here -- a LiveBench
ground truth is a bare letter, a three-digit string, or a comma-separated
ordering, none of which legitimately appear in a problem statement. It is still
not applied, for a different reason: single letters and short digit strings
occur constantly inside ordinary mathematical prose ("case (C)", "= 025"), so
an audit would fire on the program working rather than on a leak. Structural
blindness is both sufficient and free.

What the grader needs
---------------------
Routing to one of three LiveBench scorers depends on ``task``/``subtask``, and
the AMC scorer additionally needs the *problem statement* to resolve a letter to
its answer text. Neither is gold -- both are public metadata -- so they live on
:class:`Task` and the grader reads them from there.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

#: LiveBench's three math tasks. ``AMPS_Hard`` is deliberately excluded from
#: this arm and is not routable here.
MATH_COMP = "math_comp"
OLYMPIAD = "olympiad"
INCLUDED_TASKS = (MATH_COMP, OLYMPIAD)


@dataclass(frozen=True, slots=True)
class Task:
    """What the program is allowed to see. No gold can be attached: frozen+slots.

    ``question`` is LiveBench's ``turns[0]`` verbatim, including its own output
    format instruction ("put your final answer in a $\\boxed{}$", "repeat the
    letter five times"). Those instructions are part of the benchmark, not part
    of the candidate prompt, so they must not be paraphrased or stripped: every
    scorer parses the answer out of the format they specify.
    """

    example_id: str
    question: str
    task: str = ""
    subtask: str = ""


@dataclass(frozen=True, slots=True)
class Gold:
    """What only the grader may see."""

    example_id: str
    #: A letter (AMC/SMC), a three-digit string (AIME), or a comma-separated
    #: ordering (olympiad). The scorer is chosen by task/subtask, not by shape.
    ground_truth: str


@dataclass(frozen=True, slots=True)
class Instance:
    task: Task
    gold: Gold


def instance_from_record(record: Mapping[str, object]) -> Instance:
    """Split one dataset row into the gold-free half and the gold half."""
    example_id = str(record["question_id"])
    turns = record.get("turns") or []
    question = str(turns[0]) if turns else ""
    return Instance(
        task=Task(
            example_id=example_id,
            question=question,
            task=str(record.get("task") or ""),
            subtask=str(record.get("subtask") or ""),
        ),
        gold=Gold(
            example_id=example_id,
            ground_truth=str(record.get("ground_truth") or "").strip(),
        ),
    )


def is_included(record: Mapping[str, object]) -> bool:
    """True for the rows this arm uses: math_comp and olympiad only."""
    return str(record.get("task") or "") in INCLUDED_TASKS
