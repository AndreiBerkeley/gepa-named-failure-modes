"""The taxonomy: a set of named failure codes, and nothing more.

The schema is deliberately minimal. A code needs an ``id`` and a ``name``; a
``description`` is strongly recommended and everything else is optional. That
floor is the point: this package must work with an AdaMAST taxonomy, with MAST,
and with a taxonomy someone wrote by hand in a JSON file, because the whole
value of taxonomy-conditioned reflection disappears if using it requires
adopting one particular generator.

Fields this package does NOT interpret
--------------------------------------
``category``, ``severity``, ``applies_to_role`` and friends are preserved in
``FailureCode.extra`` and carried through to logs, but they never affect
routing. Routing is decided by *where a failure was observed* -- the judge
attributes each occurrence to a component -- not by what the taxonomy declares
a code is allowed to apply to. A taxonomy with no role information at all is
therefore fully supported, which is the common case outside AdaMAST.

Optional guidance fields (``when_to_use``, ``when_not_to_use``) ARE passed to
the judge when present, because they measurably sharpen code selection. They
are read opportunistically: absent, the catalog simply omits them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Keys lifted into first-class attributes. Everything else lands in ``extra``.
_PRIMARY_KEYS = frozenset({"id", "code", "name", "label", "description", "definition"})

#: Optional per-code guidance shown to the judge when a taxonomy provides it.
_GUIDANCE_KEYS = ("when_to_use", "when_not_to_use")


class TaxonomyError(ValueError):
    """The taxonomy document could not be read as a set of failure codes."""


@dataclass(frozen=True, slots=True)
class FailureCode:
    """One failure mode.

    ``id`` is the stable key used in caches and logs. ``name`` is what reflection
    actually sees -- an opaque id like ``B.4`` tells a language model nothing, so
    the name has to carry the meaning on its own.
    """

    id: str
    name: str
    description: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> FailureCode:
        code_id = str(raw.get("id") or raw.get("code") or "").strip()
        if not code_id:
            raise TaxonomyError(f"failure code has no 'id': {dict(raw)!r}")
        name = str(raw.get("name") or raw.get("label") or code_id).strip()
        description = str(raw.get("description") or raw.get("definition") or "").strip()
        extra = {k: v for k, v in raw.items() if k not in _PRIMARY_KEYS}
        return cls(id=code_id, name=name, description=description, extra=extra)

    def catalog_entry(self) -> str:
        """Render this code for the judge's prompt."""
        lines = [f"- {self.id} | {self.name}"]
        if self.description:
            lines.append(f"    {self.description}")
        for key in _GUIDANCE_KEYS:
            value = str(self.extra.get(key) or "").strip()
            if value:
                lines.append(f"    {key.replace('_', ' ')}: {value}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Taxonomy:
    """An immutable set of failure codes plus a content fingerprint.

    The fingerprint keys the judge cache. Editing or re-pruning a taxonomy must
    invalidate every judgement made under the old one, rather than silently
    mixing two code sets inside a single run.
    """

    codes: tuple[FailureCode, ...]
    fingerprint: str
    source: str | None = None
    #: Trace ids of the generation corpus, when the file records them (written
    #: by evidence-based reduction). Consumers use this to assert, in code, that
    #: the taxonomy never scores a trace it was generated from.
    generation_trace_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.codes:
            raise TaxonomyError("taxonomy contains no codes")

    def __len__(self) -> int:
        return len(self.codes)

    def get(self, code_id: str) -> FailureCode | None:
        return self._index().get(code_id)

    def _index(self) -> dict[str, FailureCode]:
        # Rebuilt on demand; the code count is small and the object is frozen.
        return {c.id: c for c in self.codes}

    def catalog_text(self) -> str:
        """The full code list, as shown to the judge."""
        return "\n".join(c.catalog_entry() for c in self.codes)

    @classmethod
    def from_codes(cls, raw_codes: Iterable[Mapping[str, Any]], *, source: str | None = None) -> Taxonomy:
        # Repeated ids are tolerated only when the entries are identical --
        # exports and hand-merged files do duplicate rows harmlessly. A repeated
        # id with DIFFERENT content is a conflict, not a duplicate, and silently
        # keeping the first would mean the judge emits that id and reflection
        # receives a name the taxonomy's author never wrote. Since bringing your
        # own taxonomy is a supported entry point, that file may not have come
        # from a generator that deduplicates.
        seen: dict[str, FailureCode] = {}
        codes: list[FailureCode] = []
        for raw in raw_codes:
            if not isinstance(raw, Mapping):
                continue
            code = FailureCode.from_mapping(raw)
            previous = seen.get(code.id)
            if previous is not None:
                if previous == code:
                    continue
                raise TaxonomyError(
                    f"code id {code.id!r} appears twice with different content: "
                    f"{previous.name!r} vs {code.name!r}. Ids are what the judge emits "
                    f"and what cross-run analysis joins on, so they must be unique."
                )
            seen[code.id] = code
            codes.append(code)
        payload = json.dumps(
            [{"id": c.id, "name": c.name, "description": c.description} for c in codes],
            sort_keys=True,
        )
        return cls(
            codes=tuple(codes),
            fingerprint=hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
            source=source,
        )


def _codes_from_document(document: Any) -> Sequence[Mapping[str, Any]]:
    """Pull a flat code list out of the shapes a taxonomy is published in.

    Three are accepted: a bare list of codes; ``{"codes": [...]}`` (AdaMAST's
    public schema); and the layered ``category_a``/``category_b``/``category_c``
    form, which AdaMAST also emits and which nests codes either as a list or as
    an id-keyed mapping.
    """
    if isinstance(document, list):
        return document
    if not isinstance(document, Mapping):
        raise TaxonomyError(f"taxonomy must be a list or object, got {type(document).__name__}")

    codes = document.get("codes")
    if isinstance(codes, list):
        return codes

    # Layered form. Look through wrapper layers before giving up.
    for layer in (document.get("full_layer"), document.get("annotation_layer"), document):
        if not isinstance(layer, Mapping):
            continue
        collected: list[Mapping[str, Any]] = []
        for key in ("category_a", "category_b", "category_c"):
            values = layer.get(key)
            if isinstance(values, Mapping):
                for fallback_id, raw in values.items():
                    if isinstance(raw, Mapping):
                        collected.append({"id": fallback_id, **raw})
            elif isinstance(values, list):
                collected.extend(r for r in values if isinstance(r, Mapping))
        if collected:
            return collected

    raise TaxonomyError("no codes found: expected a list, a 'codes' key, or category_a/b/c layers")


def load_taxonomy(path: str | Path) -> Taxonomy:
    """Read a taxonomy from a JSON file.

    ``utf-8-sig`` because taxonomies are routinely produced by tools that emit a
    BOM, and a BOM makes ``json.loads`` fail with an error that reads like
    corruption.
    """
    p = Path(path)
    try:
        document = json.loads(p.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise TaxonomyError(f"taxonomy file not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise TaxonomyError(f"taxonomy {p} is not valid JSON: {exc}") from exc
    taxonomy = Taxonomy.from_codes(_codes_from_document(document), source=str(p))
    source_ids = (document.get("reduction") or {}).get("source_trace_ids") or ()
    if source_ids:
        object.__setattr__(taxonomy, "generation_trace_ids", frozenset(str(i) for i in source_ids))
    return taxonomy
