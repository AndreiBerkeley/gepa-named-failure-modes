"""Taxonomy feedback for GEPA's ``optimize_anything`` API.

Separate from the ``GEPAAdapter`` path (``failure_taxonomy.adapter``) because
the injection point is different in kind: there we wrap an adapter and rewrite
its reflective dataset; here the evaluator's return value *is* the feedback
channel, so an arm is chosen by swapping one callable.
"""

from gepa_taxonomy.oa.asi import (
    ARMS,
    FAILURE_MODES_KEY,
    PRESERVED_KEYS,
    SCORE_ONLY,
    STOCK,
    TAXONOMY,
    ArmedEvaluator,
    TraceSink,
    trace_from_side_info,
)

__all__ = [
    "ARMS",
    "FAILURE_MODES_KEY",
    "PRESERVED_KEYS",
    "SCORE_ONLY",
    "STOCK",
    "TAXONOMY",
    "ArmedEvaluator",
    "TraceSink",
    "trace_from_side_info",
]
