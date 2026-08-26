"""LiveBench-Math scoring, ported from the upstream scorers.

Three tasks, three scorers, routed by ``subtask``:

======================  ==========================  ==============
subtask                 scorer                      score
======================  ==========================  ==============
``amc_*``, ``smc``      multiple choice, letter     0 or 1
``aime_*``              exact three-digit string    0 or 1
olympiad (``imo``,      expression re-ordering      **fractional**
``usamo``)
======================  ==========================  ==============

Ported rather than imported: ``livebench`` is not on PyPI, and its
``AMPS_Hard`` scorer pulls in sympy, latex2sympy2, lark and antlr4 and *raises*
without ``OPENAI_API_KEY``. This arm excludes AMPS_Hard (D050), so vendoring the
two scorers it does use avoids four dependencies and a second vendor entirely.

Source: ``livebench/process_results/math/{math_competitions,olympiad}/utils.py``
and ``livebench/process_results/util.py`` at LiveBench ``main``.

One deliberate deviation, and why
---------------------------------
Upstream's shipped olympiad path divides by the number of expressions the
*model emitted*::

    match = [(completions[i] == ground_truth[i]) if i < len(ground_truth) else 0
             for i in range(len(completions))]
    frac_matches = sum(match)/len(match)

So a model that emits two confident numbers where seven are required scores
2/2 = **1.0**. That is harmless when scoring fixed model outputs, which is what
LiveBench does. It is not harmless here: GEPA *optimizes* this number, and
"name only the positions you are sure of" is a prompt-level change that raises
the score without solving anything -- and it would raise it for both arms,
flattening exactly the partial-credit signal olympiad was kept for (D050).

We therefore divide by ``len(ground_truth)``, which is what upstream's own
``match_expression_completions_to_ground_truth`` does -- a function defined at
the top of the same file that the shipped code path never calls. Emitting fewer
answers can now only lower the score, and a full correct ordering still scores
1.0, so the metric is unchanged for honest outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# -- upstream helpers (livebench/process_results/util.py) --------------------


def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1].replace("$", "").replace("fbox", "boxed")


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] != left:
            return s
        return s[len(left) :]
    left = "\\boxed{"
    # Upstream asserts here. A malformed box is a model formatting failure, not a
    # harness bug, so it returns the raw string and scores 0 rather than killing
    # the rollout -- gepa's contract wants a failure score plus a trajectory.
    if s[: len(left)] != left or not s.endswith("}"):
        return s
    return s[len(left) : -1]


# -- multiple choice: amc_*, smc ---------------------------------------------

_CHOICE_RE = re.compile(r"\\textbf\{\(([A-E])\)\s?\}(.*?)(?:\\qquad|\$)")


def extract_answer(statement: str, letter: str) -> str:
    """Resolve a choice letter to its answer TEXT, from the problem statement."""
    answers = {m[0]: m[1].strip() for m in _CHOICE_RE.findall(statement)}
    answer = answers.get(letter) or ""
    if not answer:
        # Upstream: happens for one question that is too long for models to repeat.
        return "FAILURE"
    return answer.strip().strip("$").strip("~")


def mathcontest_process_results(ground_truth: str, llm_answer: str, question_text: str) -> float:
    if not (isinstance(ground_truth, str) and len(ground_truth) == 1 and "A" <= ground_truth <= "E"):
        raise ValueError(f"multiple-choice ground truth must be a single letter A-E, got {ground_truth!r}")

    score = 0.0

    solution_matches = re.findall(r"<solution>(.*?)</solution>", llm_answer)
    if solution_matches:
        last = solution_matches[-1]
        if len(set(last)) == 1 and next(iter(set(last))).lower() == ground_truth.lower():
            score = 1.0

    # The prompt asks the model to repeat the letter five times.
    if ground_truth * 4 in llm_answer:
        score = 1.0

    if score == 0.0:
        last_boxed = last_boxed_only_string(llm_answer)
        if last_boxed:
            cleaned = remove_boxed(last_boxed).replace("\\text{", "").replace("}", "").replace("\\", "").lower()
            if cleaned in {"a", "b", "c", "d", "e"} and cleaned == ground_truth.lower():
                score = 1.0

    if score == 0.0:
        value = extract_answer(question_text, ground_truth)
        if value and value != "FAILURE" and value in llm_answer[-(20 + len(value)) :]:
            score = 1.0

    if score == 0.0:
        last_line = llm_answer.strip().split("\n")[-1] if llm_answer.strip() else ""
        if last_line.strip().replace("*", "").lower() == ground_truth.lower():
            score = 1.0
        elif "(" in last_line and ")" in last_line:
            val = last_line.split("(")[1].split(")")[0]
            if val.lower() == ground_truth.lower():
                score = 1.0

    return score


# -- exact answer: aime_* -----------------------------------------------------


def aime_process_results(ground_truth: str, llm_answer: str) -> float:
    solution_matches = re.findall(r"<solution>(.*?)</solution>", llm_answer)
    if solution_matches:
        last = solution_matches[-1]
        if len(set(last)) == 1 and next(iter(set(last))).lower() == ground_truth.lower():
            return 1.0
    return 1.0 if ground_truth in llm_answer[-50:] else 0.0


# -- expression re-ordering: imo, usamo --------------------------------------


def _trim_to_digits(s: str) -> tuple[str, int]:
    start = 0
    while start < len(s) and not s[start].isdigit():
        start += 1
    end = start
    while end < len(s) and s[end].isdigit():
        end += 1
    return s[start:end], len(s) - (end - start)


def extract_expression_completions(generation: str) -> list[object]:
    """Pull the emitted ordering out of a model turn. Ported verbatim.

    Four fallbacks in upstream order: an ``Answer:`` line, a ``\\boxed{}``, the
    last line, then a final ``Answer:``-split pass. ``'NO ANSWER'`` sentinels are
    kept because they must not silently match a ground-truth integer.
    """
    numbers: list[object] | None = None

    if "answer:" in generation.lower():
        lines = generation.lower().strip().split("\n")
        answer_line, answer_index = None, None
        for i, line in enumerate(lines):
            if "answer:" in line:
                answer_line, answer_index = line, i
        answer_str = ""
        if answer_line is not None:
            answer_str = answer_line.split("answer:")[1].replace("answer:", "").replace("**", "").replace(".", "")
            answer_str = answer_str.strip()
        if answer_str == "" and answer_index is not None and answer_index < len(lines) - 1:
            answer_str = lines[answer_index + 1].replace("answer:", "").replace("**", "").replace(".", "").strip()
        numbers = []
        for n in answer_str.split(","):
            token = n.strip().split(" ")[-1]
            for junk in ("$", "{", "}", "\\", "boxed", "<", ">"):
                token = token.replace(junk, "")
            try:
                numbers.append(int(token))
            except ValueError:
                numbers.append("NO ANSWER")
        if not numbers or set(numbers) == {"NO ANSWER"}:
            numbers = None

    if numbers is None and "\\boxed" in generation:
        boxed = last_boxed_only_string(generation)
        string = remove_boxed(boxed) if boxed is not None else generation
        string = string.replace("\\text{", "").replace("}", "").replace("\\", "")
        numbers = []
        for n in string.strip().split(","):
            try:
                numbers.append(int(n.strip()))
            except ValueError:
                numbers.append("NO ANSWER")
        if not numbers or set(numbers) == {"NO ANSWER"}:
            numbers = None

    if numbers is None:
        last_line = generation.strip().lower().split("\n")[-1] if generation.strip() else ""
        numbers = []
        for n in last_line.strip().split(","):
            token, _ = _trim_to_digits(n)
            if not token.strip():
                continue
            try:
                numbers.append(int(token.strip()))
            except ValueError:
                numbers.append("NO ANSWER")
        if not numbers or set(numbers) == {"NO ANSWER"}:
            numbers = None

    if numbers is None:
        tail = [k.strip() for k in generation.lower().split("answer:")[-1].split(",")]
        rebuilt: list[object] = []
        for i, n in enumerate(tail):
            token, removed = _trim_to_digits(n)
            if token != "" and token != "\u2082":
                rebuilt.append(int(token))
            if i > 0 and removed > 0:
                break
        numbers = rebuilt

    return numbers


def proof_rearrangement_process_results(ground_truth: str, llm_answer: str) -> float:
    """Fraction of ground-truth positions filled correctly.

    Denominator is ``len(ground_truth)``, NOT ``len(completions)`` -- see the
    module docstring. Upstream's shipped path uses the latter, which lets a
    short answer score 1.0 and is directly exploitable by an optimizer.
    """
    truth = [int(n) for n in ground_truth.split(",") if n.strip()]
    if not truth:
        return 0.0
    completions = extract_expression_completions(llm_answer)
    matches = sum(1 for i, want in enumerate(truth) if i < len(completions) and completions[i] == want)
    return matches / len(truth)


# -- routing ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Grade:
    score: float
    #: Which scorer ran. Recorded so a score can always be explained.
    scorer: str
    #: What the scorer parsed out of the model turn, for feedback and diagnosis.
    parsed: str = ""
    #: Olympiad only: (correct positions, total positions).
    positions: tuple[int, int] = (0, 0)


def scorer_for(subtask: str) -> str:
    sub = (subtask or "").lower()
    if sub.startswith("aime"):
        return "aime"
    if sub in {"imo", "usamo"}:
        return "olympiad"
    return "multiple_choice"


_REPEATED_LETTER = re.compile(r"([A-E])\1{3,}")
_AIME_DIGITS = re.compile(r"\b(\d{1,3})\b")


def extract_choice(llm_answer: str) -> str:
    """The choice letter a response actually committed to, or ``""``.

    Mirrors the paths ``mathcontest_process_results`` scores on, and exists so
    feedback can distinguish "answered D, which is wrong" from "never emitted a
    choice at all". Reporting the last LINE instead -- which an earlier version
    of this did -- makes every response look parsed, and those two failures are
    the ones a prompt edit treats completely differently.
    """
    text = llm_answer or ""

    for match in re.findall(r"<solution>(.*?)</solution>", text):
        stripped = match.strip()
        if stripped and len(set(stripped.upper())) == 1 and stripped[0].upper() in "ABCDE":
            return stripped[0].upper()

    repeated = _REPEATED_LETTER.search(text)
    if repeated:
        return repeated.group(1)

    boxed = last_boxed_only_string(text)
    if boxed:
        cleaned = remove_boxed(boxed).replace("\\text{", "").replace("}", "").replace("\\", "").strip().upper()
        if cleaned in {"A", "B", "C", "D", "E"}:
            return cleaned

    last_line = text.strip().split("\n")[-1].strip() if text.strip() else ""
    bare = last_line.replace("*", "").strip().upper()
    if bare in {"A", "B", "C", "D", "E"}:
        return bare
    if "(" in last_line and ")" in last_line:
        val = last_line.split("(")[1].split(")")[0].strip().upper()
        if val in {"A", "B", "C", "D", "E"}:
            return val
    return ""


def extract_aime_answer(llm_answer: str) -> str:
    """The integer a response committed to in its tail, or ``""``.

    Scoped to the last 50 characters because that is the window
    ``aime_process_results`` scores on -- reporting a number from elsewhere in
    the response would explain a score the scorer never saw.
    """
    text = (llm_answer or "")[-50:]
    for match in re.findall(r"<solution>(.*?)</solution>", llm_answer or ""):
        stripped = match.strip()
        if stripped.isdigit():
            return stripped
    found = _AIME_DIGITS.findall(text)
    return found[-1] if found else ""


def grade(answer: str, ground_truth: str, *, subtask: str, question: str) -> Grade:
    """Score one model answer. Deterministic, offline, no LM call."""
    answer = answer or ""
    which = scorer_for(subtask)

    if which == "aime":
        return Grade(
            score=aime_process_results(ground_truth, answer),
            scorer=which,
            parsed=extract_aime_answer(answer),
        )

    if which == "olympiad":
        truth = [int(n) for n in ground_truth.split(",") if n.strip()]
        completions = extract_expression_completions(answer)
        matches = sum(1 for i, want in enumerate(truth) if i < len(completions) and completions[i] == want)
        return Grade(
            score=(matches / len(truth)) if truth else 0.0,
            scorer=which,
            parsed=",".join(str(c) for c in completions),
            positions=(matches, len(truth)),
        )

    return Grade(
        score=mathcontest_process_results(ground_truth, answer, question),
        scorer=which,
        parsed=extract_choice(answer),
    )


# -- feedback for the reflective dataset --------------------------------------


def answer_feedback(graded: Grade, ground_truth: str) -> str:
    """Gold-revealing feedback. TRAIN ids only (D028)."""
    if graded.scorer == "olympiad":
        correct, total = graded.positions
        return (
            f"You placed {correct} of {total} expressions correctly.\n"
            f"Your ordering: {graded.parsed or '(nothing parsed)'}\n"
            f"Correct ordering: {ground_truth}"
        )
    verdict = "correct" if graded.score >= 1.0 else "incorrect"
    return (
        f"Your final answer was {verdict}.\n"
        f"Parsed from your response: {graded.parsed or '(nothing parsed)'}\n"
        f"Correct answer: {ground_truth}"
    )


def score_feedback(graded: Grade) -> str:
    """Gold-free feedback, used on val and test ids.

    Says what the scorer could and could not parse, which is the one thing a
    prompt can actually fix without seeing the answer. A response that reasons
    correctly but never emits the required format scores 0, and without this the
    optimizer cannot tell that apart from being wrong.
    """
    if graded.scorer == "olympiad":
        correct, total = graded.positions
        return (
            f"You placed {correct} of {total} expressions correctly.\n"
            f"Your ordering was parsed as: {graded.parsed or '(nothing parsed)'}"
        )
    if not graded.parsed:
        # Guarded on the score too: the multiple-choice scorer can also credit an
        # answer given as TEXT rather than as a letter, and telling a response
        # that just scored 1.0 that nothing was parsed would be simply false.
        if graded.score <= 0.0:
            return "Score: 0.00. No answer could be parsed from your response in the required format."
        return f"Score: {graded.score:.2f}. Credited from your answer text, though no choice letter was emitted."
    return f"Score: {graded.score:.2f}. Parsed from your response: {graded.parsed}"


__all__ = [
    "Grade",
    "aime_process_results",
    "answer_feedback",
    "extract_expression_completions",
    "grade",
    "mathcontest_process_results",
    "proof_rearrangement_process_results",
    "score_feedback",
    "scorer_for",
]
