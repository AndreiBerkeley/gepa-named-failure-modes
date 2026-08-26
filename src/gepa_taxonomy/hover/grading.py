"""HoVer scoring: strict document retrieval, plus per-hop feedback.

The metric
----------
``retrieval_score`` is **1.0 only if every gold article was retrieved**, else
0.0 -- the published HoVer metric, and the one the prior pilot used. Partial
credit is deliberately not given: a claim needing 3 documents is not two-thirds
verified by 2 of them, because the missing hop is usually the one that makes the
claim decidable.

That makes it a **binary per-example metric**, which is the shape that broke the
SWE-Bench round (F036: ~75% of minibatch comparisons were 0-0 ties, so GEPA
hill-climbed on variance). It is workable here for one measured reason: the seed
program scores **~46.7%** on this task (prior pilot, three measurements:
46.67 / 46.67 / 49.6). Near 50% is where a binary metric discriminates *best* --
at a minibatch of 6 the subsample score spreads across 0-6 instead of collapsing
to zero. SWE-Bench's problem was a 14-21% base rate, where most minibatches
scored 0 and there was nothing to compare. Watch the tie rate anyway: if
accepted-candidate scores start clustering, this assumption is what to re-check.

``loose_recall`` is computed and reported but **never selected on**. It exists so
a run that is genuinely improving from 1-of-3 to 2-of-3 documents is visible in
the logs rather than looking flat.

The feedback function
---------------------
Adapted from the prior pilot: it names which gold documents are still missing
and, critically, *at which hop* they became findable. That per-hop attribution
is what lets reflection tell "hop 1 never surfaced the entity" apart from "hop 3
had the query but retrieved the wrong article". It is also exactly the kind of
hand-authored diagnostic the taxonomy arm replaces, which makes HoVer a third
site for that comparison.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from gepa_taxonomy.hover.tasks import Gold

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_title(title: str) -> str:
    """Match titles the way the corpus and the dataset disagree about them.

    Wikipedia titles differ between HoVer's supporting_facts and the 2017
    abstracts dump in casing, underscores-vs-spaces, and parenthetical
    disambiguators' punctuation. Comparing raw strings silently under-counts
    retrieval -- it looks like a weak program rather than a normalisation bug.
    """
    text = str(title).replace("_", " ").lower().strip()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


@dataclass(frozen=True, slots=True)
class Grade:
    score: float
    #: Fraction of gold documents retrieved. Reported, never selected on.
    loose_recall: float
    found: tuple[str, ...]
    missing: tuple[str, ...]
    #: Cumulative retrieved-title sets after each hop, for per-hop attribution.
    per_hop_found: tuple[tuple[str, ...], ...] = field(default=())

    @property
    def all_found(self) -> bool:
        return self.score >= 1.0


def grade(gold: Gold, hop_titles: Sequence[Sequence[str]]) -> Grade:
    """Score one rollout.

    ``hop_titles`` is the titles retrieved at each hop, in order. The union
    across hops is what the strict metric is computed against -- a document
    found at hop 1 still counts if hop 3 misses it, because the program returns
    everything it retrieved.
    """
    wanted = {normalize_title(t) for t in gold.titles if str(t).strip()}
    if not wanted:
        # No gold to match. Scoring 1.0 would silently reward a broken record;
        # scoring 0.0 would punish the program for a data defect. Neither is
        # right, so this is surfaced as a zero WITH an explicit missing marker
        # that the feedback function reports.
        return Grade(score=0.0, loose_recall=0.0, found=(), missing=("<no gold titles in record>",))

    cumulative: set[str] = set()
    per_hop: list[tuple[str, ...]] = []
    for titles in hop_titles:
        cumulative |= {normalize_title(t) for t in titles}
        per_hop.append(tuple(sorted(cumulative & wanted)))

    found = cumulative & wanted
    missing = wanted - cumulative
    return Grade(
        score=1.0 if not missing else 0.0,
        loose_recall=len(found) / len(wanted),
        found=tuple(sorted(found)),
        missing=tuple(sorted(missing)),
        per_hop_found=tuple(per_hop),
    )


def score_feedback(graded: Grade) -> str:
    """Outcome text safe for any split -- names no gold document."""
    if graded.all_found:
        return "All supporting documents were retrieved."
    n_missing = len(graded.missing)
    return (
        f"Retrieval INCOMPLETE: {len(graded.found)} of "
        f"{len(graded.found) + n_missing} supporting documents found; "
        f"{n_missing} still missing. The claim cannot be verified without them."
    )


def retrieval_feedback(graded: Grade, component: str) -> str:
    """Per-hop diagnostic. **Names gold titles -- train split only.**

    The caller gates this on split membership, exactly as the HotpotQA and
    IFBench adapters do. Naming missing documents to a val or test rollout would
    hand the program the answer.
    """
    if graded.all_found:
        return "All supporting documents were retrieved; preserve whatever this instruction is doing."

    lines = [f"Still missing after every hop: {list(graded.missing)}."]
    gained_at: list[str] = []
    previous = 0
    for hop_index, found in enumerate(graded.per_hop_found, start=1):
        gained = len(found) - previous
        previous = len(found)
        gained_at.append(f"hop {hop_index}: +{gained}")
    if gained_at:
        lines.append(f"Documents recovered per hop -- {', '.join(gained_at)}.")

    if graded.per_hop_found and not graded.per_hop_found[0]:
        lines.append(
            "Hop 1 retrieved none of the supporting documents, so every later hop "
            "started from an unhelpful summary. The first query is the bottleneck."
        )
    lines.append(
        f"Revise `{component}` to name the concrete entities and relationships in the claim "
        "rather than referring to them indirectly, so the retriever can match them."
    )
    return " ".join(lines)
