"""Optimizer-side taxonomy feedback for GEPA reflection.

Adds a structured failure diagnosis after the task adapter builds its ordinary
reflective dataset and before GEPA proposes a revision. The task adapter remains
unwrapped and unchanged.

    from failure_taxonomy import LLMFailureJudge, TaxonomyFeedbackEnricher, load_taxonomy

    taxonomy = load_taxonomy("taxonomy.json")
    judge = LLMFailureJudge(taxonomy=taxonomy, lm=my_lm)
    enricher = TaxonomyFeedbackEnricher(judge=judge)

The taxonomy is any JSON file whose codes carry an ``id`` and a ``name``. The
stage boundary is that file: bring your own and the generation stage is
skippable entirely.
"""

from failure_taxonomy.cache import JudgeCache, candidate_key
from failure_taxonomy.enricher import FAILURE_MODES_KEY, TaxonomyFeedbackEnricher
from failure_taxonomy.generation import harvest_traces, trace_report, write_generation_traces
from failure_taxonomy.judge import GENERAL, FailureJudge, LLMFailureJudge, Occurrence
from failure_taxonomy.schema import FailureCode, Taxonomy, TaxonomyError, load_taxonomy
from failure_taxonomy.trace import ComponentCall, SegmentedTrace, build_trace, extract_calls

__all__ = [
    "FAILURE_MODES_KEY",
    "GENERAL",
    "ComponentCall",
    "FailureCode",
    "FailureJudge",
    "JudgeCache",
    "LLMFailureJudge",
    "Occurrence",
    "SegmentedTrace",
    "Taxonomy",
    "TaxonomyError",
    "TaxonomyFeedbackEnricher",
    "build_trace",
    "candidate_key",
    "extract_calls",
    "harvest_traces",
    "load_taxonomy",
    "trace_report",
    "write_generation_traces",
]
