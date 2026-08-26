"""Scoring and textual feedback for HotpotQA.

The metric
----------
Answer F1 over normalised tokens -- the standard HotpotQA answer metric
(Yang et al. 2018), and the one RLMOpt reports ("word-level F1"). It is a
*partial-credit* metric, which is the property that matters most here: the
SWE-Bench round failed because a binary metric at a low base rate made a
6-instance minibatch a 0-0 tie about 75% of the time, so GEPA was hill-climbing
on variance. A continuous score makes every minibatch comparison informative.

Retrieval recall is computed too, but only as a *diagnostic*, never as the
optimisation target. Optimising recall directly would reward retrieving the
gold documents rather than answering the question.

The feedback function
---------------------
Reproduces the published GEPA setup verbatim in intent: "The textual feedback
module identifies the set of relevant documents remaining to be retrieved at
each stage of the program, and provides that as feedback to the modules at that
stage" (GEPA appendix E.1).

This is the **baseline arm's** feedback, so getting it right is what makes the
comparison honest. It is deliberately strong: it names the missing gold
documents by title. Any advantage the taxonomy arm shows has to be earned on
top of that, not by comparison with a weakened baseline.
"""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from dataclasses import dataclass

from gepa_taxonomy.hotpotqa.tasks import Gold

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """HotpotQA's official normalisation: lowercase, strip punctuation/articles."""
    lowered = text.lower()
    no_punct = lowered.translate(_PUNCT)
    no_articles = _ARTICLES.sub(" ", no_punct)
    return _WHITESPACE.sub(" ", no_articles).strip()


def answer_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer.

    Yes/no/noanswer are compared exactly: for those, partial token overlap is
    meaningless ("yes" vs "yes it is" should not be 0.5 correct when the gold
    label is a boolean).
    """
    pred_norm, gold_norm = normalize_answer(prediction), normalize_answer(gold)
    if gold_norm in {"yes", "no", "noanswer"} or pred_norm in {"yes", "no", "noanswer"}:
        return float(pred_norm == gold_norm)

    pred_tokens, gold_tokens = pred_norm.split(), gold_norm.split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    for token in pred_tokens:
        if gold_counts.get(token, 0) > 0:
            gold_counts[token] -= 1
            common[token] = common.get(token, 0) + 1
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


@dataclass(frozen=True, slots=True)
class Grade:
    """One graded rollout."""

    score: float
    f1: float
    em: float
    #: Fraction of gold supporting-fact documents that retrieval surfaced.
    retrieval_recall: float
    retrieved_titles: tuple[str, ...]
    missing_titles: tuple[str, ...]


def grade(prediction: str, retrieved_titles: Sequence[str], gold: Gold) -> Grade:
    f1 = answer_f1(prediction, gold.answer)
    retrieved = {t.strip().lower() for t in retrieved_titles}
    missing = tuple(t for t in gold.titles if t.strip().lower() not in retrieved)
    found = len(gold.titles) - len(missing)
    recall = found / len(gold.titles) if gold.titles else 1.0
    return Grade(
        score=f1,
        f1=f1,
        em=exact_match(prediction, gold.answer),
        retrieval_recall=recall,
        retrieved_titles=tuple(retrieved_titles),
        missing_titles=missing,
    )


def retrieval_feedback(retrieved_titles: Sequence[str], gold: Gold) -> str:
    """The baseline arm's per-stage feedback: which gold documents are still missing.

    Reproduces the published setup. Named documents are the strong signal here,
    and the taxonomy arm has to beat it rather than substitute for it.
    """
    retrieved = {t.strip().lower() for t in retrieved_titles}
    found = [t for t in gold.titles if t.strip().lower() in retrieved]
    missing = [t for t in gold.titles if t.strip().lower() not in retrieved]
    lines = [
        f"Gold documents retrieved so far ({len(found)}/{len(gold.titles)}): "
        + (", ".join(found) if found else "none"),
    ]
    if missing:
        lines.append("Gold documents still to be retrieved: " + ", ".join(missing))
    else:
        lines.append("All gold documents have been retrieved.")
    return "\n".join(lines)


def answer_feedback(prediction: str, gold: Gold) -> str:
    """Feedback for the answering stage: the score and the correct answer.

    The gold answer is shown because this is optimizer-level feedback on TRAIN
    instances, which is standard GEPA practice (D028). The program itself never
    sees it -- ``assert_gold_free`` guards every prompt.
    """
    f1 = answer_f1(prediction, gold.answer)
    verdict = "correct" if f1 == 1.0 else ("partially correct" if f1 > 0 else "incorrect")
    return (
        f"Predicted answer: {prediction or '(empty)'}\nCorrect answer: {gold.answer}\nAnswer F1: {f1:.2f} ({verdict})"
    )
