#!/usr/bin/env python
"""Build the BM25 index over the 2017 Wikipedia abstracts corpus.

Free and offline: no model is called. Run once; the index is reused by every
HotpotQA seed and both arms.

This is the retrieval backend the published GEPA/DSPy multi-hop programs use --
local BM25 over ~5.2M Wikipedia abstracts, no hosted retriever and no Docker.
Keeping the same backend is what makes our numbers comparable to the published
ones at all; swapping in a dense retriever would change the task, not just the
implementation.

    uv run python scripts/build_wiki_index.py

Memory note: the tokenised corpus is held in RAM while the index is built.
Peak usage is several GB. The saved index is memory-mapped at query time, so
serving is cheap even though building is not.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO / "data" / "wiki" / "wiki.abstracts.2017.jsonl"
DEFAULT_INDEX = REPO / "data" / "wiki" / "bm25s_index"

#: Matches the DSPy multi-hop tutorial's retriever settings, so retrieval
#: behaviour is the published one rather than a tuned variant of it.
K1 = 0.9
B = 0.4


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    """Return (titles, documents).

    A document is ``title | abstract``: the title is prepended because HotpotQA
    gold supporting facts are identified BY TITLE, and a query naming an entity
    should match the article about it even when the abstract never repeats the
    name in full.
    """
    titles: list[str] = []
    documents: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            title = (record.get("title") or "").strip()
            if not title:
                # pid 0 is an empty placeholder row in the published corpus.
                continue
            text = record.get("text") or []
            body = "".join(text).strip() if isinstance(text, list) else str(text).strip()
            titles.append(title)
            documents.append(f"{title} | {body}")
    return titles, documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--limit", type=int, default=0, help="index only the first N docs (smoke tests)")
    args = parser.parse_args()

    if not args.corpus.exists():
        parser.error(f"corpus not found: {args.corpus}\nDownload it first (see docs).")

    import bm25s
    import Stemmer

    started = time.time()
    print(f"reading {args.corpus} ...", flush=True)
    titles, documents = load_corpus(args.corpus)
    if args.limit:
        titles, documents = titles[: args.limit], documents[: args.limit]
    print(f"  {len(documents):,} documents in {time.time() - started:.0f}s", flush=True)

    stemmer = Stemmer.Stemmer("english")
    print("tokenising ...", flush=True)
    tokens = bm25s.tokenize(documents, stopwords="en", stemmer=stemmer, show_progress=True)

    print("indexing ...", flush=True)
    retriever = bm25s.BM25(k1=K1, b=B)
    retriever.index(tokens, show_progress=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Titles are saved beside the index: retrieval returns corpus positions, and
    # grading needs to map those back to titles to score supporting-fact recall.
    retriever.save(str(args.out), corpus=None)
    (args.out / "titles.json").write_text(json.dumps(titles), encoding="utf-8")

    print(f"\nindex written to {args.out}")
    print(f"documents: {len(documents):,}   elapsed: {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
