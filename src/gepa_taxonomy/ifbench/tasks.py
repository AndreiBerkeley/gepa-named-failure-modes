"""IFBench task and gold types, spanning TWO datasets with disjoint vocabularies.

Following GEPA's published setup:

* **train / val** -- ``allenai/IF_multi_constraints_upto5`` (IF-RLVR Train),
  IFEval-style constraints, verified by the ``ifevalg`` registry.
* **test** -- ``allenai/IFBench_test``, 58 new out-of-distribution constraints,
  verified by the ``ifbench`` registry.

Measured overlap between the two constraint vocabularies is **zero**, which is
the entire point: the paper splits this way "to ensure that the optimizers do not
access the new, unseen constraints being tested in IFBench". Every
:class:`Gold` therefore carries the registry its constraints belong to, and the
grader routes on it rather than guessing from the id.

Gold blindness, and what is actually secret here
------------------------------------------------
:class:`Task` is frozen with ``slots`` and carries only the prompt. The
constraint is *stated in the prompt* -- "respond using between 5 and 10 words" is
text the model must read, and that is the task. What is withheld is its
**structured** form: the verifier id and the exact arguments it will be scored
with. Handing those over would let the program satisfy the checker rather than
the instruction.

Parsing the constraint out of prose is a real part of the benchmark, and it is
exactly where a taxonomy has something to say.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: Which vendored verifier registry a gold's constraints belong to.
IFEVALG = "ifevalg"
IFBENCH = "ifbench"


@dataclass(frozen=True, slots=True)
class Task:
    """What the program is allowed to see. No gold can be attached: frozen+slots."""

    example_id: str
    prompt: str


@dataclass(frozen=True, slots=True)
class Gold:
    """What only the grader may see: which verifiers run, with what arguments."""

    example_id: str
    instruction_ids: tuple[str, ...]
    #: One kwargs mapping per instruction id, positionally aligned. Upstream
    #: stores ``None`` for argument-less constraints and nulls inside the maps;
    #: both are normalised away here rather than at every use, matching what
    #: upstream's own scoring loop does before calling build_description.
    kwargs: tuple[Mapping[str, object], ...]
    #: ``ifevalg`` (train/val) or ``ifbench`` (test). Routing is explicit because
    #: the two vocabularies are disjoint -- an id in the wrong registry is a
    #: KeyError, not a wrong score, but only if we never guess.
    registry: str = IFBENCH

    @property
    def n_constraints(self) -> int:
        return len(self.instruction_ids)


@dataclass(frozen=True, slots=True)
class Instance:
    task: Task
    gold: Gold


def _clean_kwargs(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        return {}
    return {k: v for k, v in raw.items() if v is not None}


def instance_from_ifbench(record: Mapping[str, object]) -> Instance:
    """One row of ``allenai/IFBench_test`` -> the TEST half of the arm."""
    example_id = str(record["key"])
    ids: Sequence[str] = record.get("instruction_id_list") or []  # type: ignore[assignment]
    raw: Sequence[object] = record.get("kwargs") or []  # type: ignore[assignment]
    return Instance(
        task=Task(example_id=example_id, prompt=str(record.get("prompt") or "")),
        gold=Gold(
            example_id=example_id,
            instruction_ids=tuple(str(i) for i in ids),
            kwargs=tuple(_clean_kwargs(k) for k in raw),
            registry=IFBENCH,
        ),
    )


def instance_from_ifrlvr(record: Mapping[str, object]) -> Instance:
    """One row of ``allenai/IF_multi_constraints_upto5`` -> TRAIN/VAL.

    The prompt is the single user turn; ``ground_truth`` is a *string* holding a
    one-element list of ``{instruction_id: [...], kwargs: [...]}``. It is parsed
    with ``literal_eval`` rather than ``json.loads`` because it is a Python repr
    (single quotes, ``None``), not JSON.
    """
    example_id = str(record["key"])
    messages: Sequence[Mapping[str, object]] = record.get("messages") or []  # type: ignore[assignment]
    prompt = str(messages[0].get("content") or "") if messages else ""

    raw = record.get("ground_truth")
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = []
    entry = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], Mapping) else {}
    ids: Sequence[str] = entry.get("instruction_id") or []  # type: ignore[assignment]
    kwargs: Sequence[object] = entry.get("kwargs") or []  # type: ignore[assignment]

    # kwargs can be SHORTER than ids, and individual entries can be None for
    # argument-less constraints. Pad rather than zip-truncate: a missing entry
    # would silently drop the last constraint from scoring.
    padded = [_clean_kwargs(kwargs[i]) if i < len(kwargs) else {} for i in range(len(ids))]

    return Instance(
        task=Task(example_id=example_id, prompt=prompt),
        gold=Gold(
            example_id=example_id,
            instruction_ids=tuple(str(i) for i in ids),
            kwargs=tuple(padded),
            registry=IFEVALG,
        ),
    )


def constraint_family(instruction_id: str) -> str:
    """The prefix of an id (``count``, ``keywords``, ``detectable_format`` ...).

    Used for stratification and for reporting which families a candidate fails --
    a headline score hides that entirely.
    """
    return (instruction_id or "").split(":", 1)[0] or "unknown"
