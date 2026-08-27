"""SWE-Bench task types, with gold-blindness enforced structurally.

HARD INVARIANT
--------------------------------------
No program component -- retrieval, solver, feedback builder, or refiner -- may
ever receive gold patches, FAIL_TO_PASS / PASS_TO_PASS test IDs or their output,
or harness results, on any split.

This module enforces that by *type*, not by convention: :class:`Task` has no
field in which gold data could travel. A dataset row is split at load time into
a ``Task`` (given to the program) and a :class:`Gold` (given only to the
grader). A component that never receives a ``Gold`` cannot leak one.

Optimizer-level reflection seeing train-instance metric feedback is fine --
that is GEPA doing its job. Program-level gold blindness is absolute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Dataset columns that carry gold information. Never place these on a Task.
GOLD_FIELDS: frozenset[str] = frozenset(
    {
        "patch",  # the reference solution diff
        "test_patch",  # the diff that introduces the grading tests
        "FAIL_TO_PASS",  # the grading signal
        "PASS_TO_PASS",  # the regression signal
    }
)


@dataclass(frozen=True, slots=True)
class Task:
    """What the program is allowed to see. Deliberately has no gold fields.

    ``slots=True`` means attributes cannot be added at runtime, so gold data
    cannot be attached to a Task after construction either.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    # Needed to build the execution environment; carries no solution information.
    environment_setup_commit: str
    version: str

    def to_prompt_context(self) -> dict[str, str]:
        """The exact fields that may be interpolated into an LM prompt."""
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "problem_statement": self.problem_statement,
        }


@dataclass(frozen=True, slots=True)
class Gold:
    """Grading-only data. Never passed to a program component.

    Held by the evaluation harness. If you find yourself threading a ``Gold``
    into anything under ``program.py``, the invariant is being violated.
    """

    instance_id: str
    patch: str
    test_patch: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Instance:
    """A dataset row, split into its blind and gold halves."""

    task: Task
    gold: Gold


def _as_tuple(value: Any) -> tuple[str, ...]:
    """FAIL_TO_PASS / PASS_TO_PASS ship as JSON-encoded lists in some revisions."""
    if isinstance(value, str):
        import json

        value = json.loads(value)
    return tuple(str(v) for v in value)


def split_row(row: dict[str, Any]) -> Instance:
    """Split a raw SWE-Bench dataset row into (Task, Gold).

    This is the *only* place a raw row should be touched. Everything downstream
    consumes ``Instance.task`` or ``Instance.gold``, never the row.
    """
    task = Task(
        instance_id=row["instance_id"],
        repo=row["repo"],
        base_commit=row["base_commit"],
        problem_statement=row["problem_statement"],
        environment_setup_commit=row["environment_setup_commit"],
        version=str(row["version"]),
    )
    gold = Gold(
        instance_id=row["instance_id"],
        patch=row["patch"],
        test_patch=row["test_patch"],
        fail_to_pass=_as_tuple(row["FAIL_TO_PASS"]),
        pass_to_pass=_as_tuple(row["PASS_TO_PASS"]),
    )
    return Instance(task=task, gold=gold)


@dataclass
class GoldLeakError(AssertionError):
    """Raised when gold data is detected somewhere it must never appear."""

    where: str
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - message formatting
        return f"gold data leaked into {self.where}: {self.detail}"


def _normalise(text: str) -> str:
    """Collapse all whitespace, so detection survives reformatting.

    Deliberately not JSON: serializing escapes newlines (``\\n`` -> ``\\\\n``),
    which silently prevents any multi-line gold patch from ever matching. That
    failure mode is exactly what this module exists to prevent, so we compare
    on raw text with whitespace normalized instead.
    """
    return " ".join(text.split())


def _walk(payload: Any) -> tuple[list[str], list[str]]:
    """Return ``(dict_keys, string_leaves)`` found anywhere in ``payload``."""
    keys: list[str] = []
    strings: list[str] = []
    stack = [payload]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, str):
            strings.append(node)
        elif isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str):
                    keys.append(k)
                stack.append(v)
        elif isinstance(node, (list, tuple, set, frozenset)):
            stack.extend(node)
        elif hasattr(node, "__dict__"):
            stack.append(vars(node))
        elif hasattr(node, "__slots__"):
            for s in node.__slots__:
                if hasattr(node, s):
                    keys.append(s)
                    stack.append(getattr(node, s))
        elif node is not None and not isinstance(node, (int, float, bool)):
            strings.append(str(node))
    return keys, strings


#: Gold test ids shorter than this collide too readily with ordinary source
#: identifiers to be meaningful evidence of a leak.
_MIN_DISTINCTIVE_TEST_ID = 12


def assert_gold_free(
    payload: Any, *, where: str, gold: Gold | None = None, public_text: str = ""
) -> None:
    """Assert that ``payload`` contains no gold data.

    Two checks, because either alone is insufficient:

    1. **Structural** -- no key named after a gold field appears anywhere in
       the payload. Catches a dict accidentally built from a raw dataset row.
    2. **Value-based** -- if ``gold`` is supplied, no non-trivial gold *value*
       (patch text, distinctive test id) appears in any string within the
       payload. Catches gold that arrives under an innocent key name.

    Called at every program boundary, not just in tests, so a violation fails
    the run rather than silently producing an inflated score.
    """
    keys, strings = _walk(payload)

    for key in keys:
        if key in GOLD_FIELDS:
            raise GoldLeakError(where=where, detail=f"field name {key!r} present")
    # A gold field name can also appear inside a rendered prompt string.
    for text in strings:
        for name in GOLD_FIELDS:
            if name in text and name.isupper():  # FAIL_TO_PASS / PASS_TO_PASS
                raise GoldLeakError(where=where, detail=f"field name {name!r} present in text")

    if gold is None:
        return

    haystacks = [_normalise(s) for s in strings]

    for label, value in (("patch", gold.patch), ("test_patch", gold.test_patch)):
        needle = _normalise(value)
        if len(needle) < 20:  # too short to be distinctive
            continue
        for hay in haystacks:
            if needle in hay:
                raise GoldLeakError(where=where, detail=f"gold {label} text present")

    # SWE-Bench problem statements are issue text, and issue text often names
    # the failing test. That name is published WITH the task, so finding it in
    # a prompt is not a leak -- the benchmark put it there. Anything already in
    # ``public_text`` therefore cannot leak information the solver lacked.
    public = _normalise(public_text)

    for test_id in (*gold.fail_to_pass, *gold.pass_to_pass):
        if len(test_id) < _MIN_DISTINCTIVE_TEST_ID:
            continue
        if public and test_id in public:
            continue
        for hay in haystacks:
            if test_id in hay:
                raise GoldLeakError(where=where, detail=f"grading test id {test_id!r} present")
