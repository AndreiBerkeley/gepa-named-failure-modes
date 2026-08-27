"""IFBench arm: a two-module generate -> ensure pipeline over verifiable constraints.

The second benchmark alongside HotpotQA (O018). Chosen for being genuinely
*different* from it -- instruction following rather than multi-hop RAG, two
modules rather than four, no retrieval, and grading that is deterministic and
free. HoVer has larger headroom above GEPA but is a near-sibling of HotpotQA
, so it would measure one benchmark twice.
"""
