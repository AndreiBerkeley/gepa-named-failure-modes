"""BM25 retrieval over a repository checkout at ``base_commit``.

Fixed scaffolding, not an optimized component. Deterministic by construction:
same task + same commit -> same files in the same order, every time. That
matters twice over -- run-to-run comparability, and a stable prompt prefix.

Gold blindness: this module receives a :class:`~gepa_taxonomy.tasks.Task` and a
checkout. It never sees a ``Gold``, and it explicitly refuses to index test
files, which is where the grading tests live.
"""

from __future__ import annotations

import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from gepa_taxonomy.program import RetrievedFile
from gepa_taxonomy.tasks import Task

#: Directories and filename patterns excluded from the index. Retrieving a test
#: file risks surfacing the grading tests themselves, so tests are excluded
#: outright rather than filtered later.
EXCLUDED_DIR_PARTS = frozenset({"tests", "test", "testing", ".git", "docs", "doc", "examples", "build", "dist"})
EXCLUDED_NAME_RE = re.compile(r"(^test_|_test\.py$|^conftest\.py$)")

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

#: Skip files too large to be plausible single-fix targets; they would also
#: blow the context budget on their own.
MAX_FILE_BYTES = 200_000


def tokenize(text: str) -> list[str]:
    """Identifier-aware tokenization, with snake_case/camelCase split."""
    out: list[str] = []
    for raw in TOKEN_RE.findall(text):
        low = raw.lower()
        out.append(low)
        if "_" in low:
            out.extend(p for p in low.split("_") if len(p) > 2)
        else:
            parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", raw)
            if len(parts) > 1:
                out.extend(p.lower() for p in parts if len(p) > 2)
    return out


@dataclass
class BM25Index:
    """Minimal BM25-Okapi over file contents. No external dependency."""

    k1: float = 1.5
    b: float = 0.75
    paths: list[str] = field(default_factory=list)
    freqs: list[Counter] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    df: Counter = field(default_factory=Counter)

    def add(self, path: str, text: str) -> None:
        toks = tokenize(text)
        if not toks:
            return
        tf = Counter(toks)
        self.paths.append(path)
        self.freqs.append(tf)
        self.lengths.append(len(toks))
        self.df.update(tf.keys())

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self.paths:
            return []
        n = len(self.paths)
        avgdl = sum(self.lengths) / n
        q = tokenize(query)
        scored: list[tuple[str, float]] = []
        for i, path in enumerate(self.paths):
            tf, dl, score = self.freqs[i], self.lengths[i], 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            if score > 0:
                scored.append((path, score))
        # Sort by score, then path, so ties break deterministically.
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


def ensure_checkout(task: Task, cache_dir: Path) -> Path:
    """Clone ``task.repo`` and check out ``base_commit``. Free (git only).

    Cached per repo; the checkout is moved to the right commit per task.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = cache_dir / task.repo.replace("/", "__")
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--quiet", f"https://github.com/{task.repo}.git", str(repo_dir)],
            check=True,
        )
    subprocess.run(["git", "-C", str(repo_dir), "checkout", "--quiet", "--force", task.base_commit], check=True)
    return repo_dir


def _indexable(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in EXCLUDED_DIR_PARTS for part in rel.parts):
        return False
    if EXCLUDED_NAME_RE.search(path.name):
        return False
    try:
        return path.stat().st_size <= MAX_FILE_BYTES
    except OSError:
        return False


@dataclass
class BM25Retriever:
    """Retrieves top-k source files for a task from a local checkout."""

    cache_dir: Path
    max_file_chars: int = 40_000

    def retrieve(self, task: Task, *, k: int) -> list[RetrievedFile]:
        root = ensure_checkout(task, self.cache_dir)
        index = BM25Index()
        contents: dict[str, str] = {}

        for path in sorted(root.rglob("*.py")):
            if not _indexable(path, root):
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            rel = str(path.relative_to(root))
            contents[rel] = text
            index.add(rel, text)

        hits = index.search(task.problem_statement, k)
        return [RetrievedFile(path=p, content=contents[p][: self.max_file_chars]) for p, _ in hits]
