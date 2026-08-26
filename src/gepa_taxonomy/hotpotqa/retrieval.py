"""BM25 retrieval over the 2017 Wikipedia abstracts corpus.

Fixed scaffolding: GEPA optimizes the four prompt components, never the
retriever. Keeping retrieval fixed and deterministic is what makes two arms
comparable -- a retriever that varied between runs would put noise directly
into the metric the acceptance test reads.

The index is built once by ``scripts/build_wiki_index.py`` and memory-mapped
here, so the ~5.2M-document corpus costs a few hundred MB of RSS at query time
rather than being re-tokenised per run.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DEFAULT_INDEX = Path(__file__).resolve().parents[3] / "data" / "wiki" / "bm25s_index"

#: Passages per hop. The DSPy multi-hop tutorial's default.
DEFAULT_K = 10

#: Characters kept per retrieved passage when rendered into a prompt. Wikipedia
#: abstracts are short; this only clips the rare very long one, and clipping by
#: whole characters (not tokens) keeps rendering deterministic.
MAX_PASSAGE_CHARS = 1200


@dataclass(frozen=True, slots=True)
class Passage:
    title: str
    text: str

    def render(self) -> str:
        body = self.text
        if len(body) > MAX_PASSAGE_CHARS:
            body = body[:MAX_PASSAGE_CHARS].rstrip() + " ... [truncated]"
        return f"[{self.title}] {body}"


def render_passages(passages: list[Passage]) -> str:
    if not passages:
        return "(no passages retrieved)"
    return "\n".join(f"{i}. {p.render()}" for i, p in enumerate(passages, start=1))


@dataclass
class WikiRetriever:
    """Deterministic BM25 retriever over Wikipedia abstracts."""

    index_dir: Path = DEFAULT_INDEX
    k: int = DEFAULT_K
    _retriever: object | None = field(default=None, repr=False)
    _titles: list[str] = field(default_factory=list, repr=False)
    _corpus: list[str] = field(default_factory=list, repr=False)
    _thread_local: threading.local = field(default_factory=threading.local, repr=False)

    def load(self) -> WikiRetriever:
        import bm25s

        if not Path(self.index_dir).exists():
            raise FileNotFoundError(
                f"BM25 index not found at {self.index_dir}. Build it first: uv run python scripts/build_wiki_index.py"
            )
        self._retriever = bm25s.BM25.load(str(self.index_dir), mmap=True, load_corpus=False)
        self._titles = json.loads((Path(self.index_dir) / "titles.json").read_text(encoding="utf-8"))
        # Warm the offset table here rather than lazily inside a worker thread:
        # several threads racing to build the same 5.2M-entry table would each
        # scan 1.8 GB, and the first rollouts would stall behind it.
        _line_offsets(self._corpus_path())
        return self

    @property
    def _stemmer(self):
        """One Stemmer per thread.

        PyStemmer does not document its Stemmer objects as thread-safe, and a
        corrupted stem would not raise -- it would silently change which
        documents are retrieved, which is precisely the kind of quiet wrongness
        this project keeps paying for. A Stemmer is cheap to construct, so each
        thread gets its own rather than relying on a race test that happened to
        pass.
        """
        import Stemmer

        local = self._thread_local
        stemmer = getattr(local, "stemmer", None)
        if stemmer is None:
            stemmer = local.stemmer = Stemmer.Stemmer("english")
        return stemmer

    def retrieve(self, query: str, *, k: int | None = None) -> list[Passage]:
        """Top-k passages for ``query``.

        An empty or unmatchable query returns an empty list rather than raising:
        a module that produced no query is a failure to diagnose, not a crash --
        and crashing here would lose the whole rollout's trace, which is the
        evidence the taxonomy is built from.
        """
        import bm25s

        if self._retriever is None:
            self.load()
        query = (query or "").strip()
        if not query:
            return []
        k = k or self.k
        tokens = bm25s.tokenize(query, stopwords="en", stemmer=self._stemmer, show_progress=False)
        try:
            results, _scores = self._retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
        except ValueError:
            # bm25s raises when no query term survives tokenisation.
            return []
        return [self._passage_at(int(idx)) for idx in results[0]]

    def _passage_at(self, index: int) -> Passage:
        title = self._titles[index] if 0 <= index < len(self._titles) else ""
        return Passage(title=title, text=self._text_for(index))

    def _text_for(self, index: int) -> str:
        """Abstract text for a corpus position.

        The index is stored without its corpus to keep it small, so text is read
        back from the source JSONL on demand and cached. Retrieval is top-10 per
        hop over a handful of hops, so the working set stays tiny.
        """
        return _corpus_line(self._corpus_path(), index)

    def _corpus_path(self) -> Path:
        return Path(self.index_dir).parent / "wiki.abstracts.2017.jsonl"


@lru_cache(maxsize=1)
def _line_offsets(corpus_path: Path) -> list[int]:
    """Byte offset of every line, so a passage can be read without a full scan.

    Built once (~5.2M entries, a few hundred MB of ints) and reused. The
    alternative -- holding the whole 1.8 GB corpus in memory -- costs more, and
    re-scanning per lookup would make each rollout quadratic in corpus size.
    """
    offsets: list[int] = []
    with corpus_path.open("rb") as fh:
        offset = 0
        for line in fh:
            offsets.append(offset)
            offset += len(line)
    return offsets


def _corpus_line(corpus_path: Path, index: int) -> str:
    """Read one abstract by corpus position.

    ``index`` counts NON-EMPTY-TITLE rows, matching what the index was built
    over, while the file also contains the empty pid-0 placeholder -- so the
    mapping is maintained by the titles file, and this reads by file line.
    """
    offsets = _line_offsets(corpus_path)
    # +1: the builder skips the empty pid-0 row, so corpus position i is file
    # line i+1. Verified against the published corpus, whose first row is empty.
    line_no = index + 1
    if not (0 <= line_no < len(offsets)):
        return ""
    with corpus_path.open("rb") as fh:
        fh.seek(offsets[line_no])
        raw = fh.readline()
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    text = record.get("text") or []
    return "".join(text).strip() if isinstance(text, list) else str(text).strip()
