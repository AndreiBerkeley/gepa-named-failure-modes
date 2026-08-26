"""Judge concurrency. FREE: the LM is a local callable, no network.

Parallelism here is a wall-clock optimisation that must be invisible in the
result. At HotpotQA's minibatch of 15 the judge adds fifteen sequential model
calls to every iteration, roughly doubling a run; but a concurrency bug that
dropped or duplicated a diagnosis would corrupt what reflection sees, which is
the treatment arm's entire input.
"""

from __future__ import annotations

import threading
import time

from failure_taxonomy.judge import LLMFailureJudge
from failure_taxonomy.schema import Taxonomy
from failure_taxonomy.trace import SegmentedTrace

TAXONOMY = Taxonomy.from_codes(
    [
        {"id": "A.1", "name": "Bad_Thing", "description": "a bad thing"},
        {"id": "A.2", "name": "Other_Thing", "description": "another"},
    ]
)


def _traces(n: int) -> list[SegmentedTrace]:
    return [SegmentedTrace(trace_id=f"t{i}", task=f"task {i}", fallback_text=f"trace body {i}") for i in range(n)]


def _lm(prompt: str) -> str:
    # Echo a code back so every trace yields exactly one occurrence.
    time.sleep(0.01)
    return '{"occurrences": [{"code": "A.1", "component": null, "evidence": "trace body"}]}'


class TestParallelMatchesSerial:
    def test_same_results_either_way(self):
        traces = _traces(12)
        serial = LLMFailureJudge(taxonomy=TAXONOMY, lm=_lm, max_workers=1).judge(traces)
        parallel = LLMFailureJudge(taxonomy=TAXONOMY, lm=_lm, max_workers=6).judge(traces)
        assert set(serial) == set(parallel) == {t.trace_id for t in traces}
        for tid in serial:
            assert [o.code for o in serial[tid]] == [o.code for o in parallel[tid]]

    def test_counters_are_not_lost_under_concurrency(self):
        traces = _traces(40)
        j = LLMFailureJudge(taxonomy=TAXONOMY, lm=_lm, max_workers=8)
        j.judge(traces)
        assert j.calls == 40, "a lost increment would understate judge spend"
        assert j.judged == 40

    def test_every_trace_is_judged_exactly_once(self):
        """A duplicated dispatch would double-charge and double-count evidence."""
        seen: list[str] = []
        lock = threading.Lock()

        def counting_lm(prompt: str) -> str:
            with lock:
                seen.append(prompt)
            return '{"occurrences": []}'

        traces = _traces(20)
        LLMFailureJudge(taxonomy=TAXONOMY, lm=counting_lm, max_workers=8).judge(traces)
        assert len(seen) == 20
        assert len(set(seen)) == 20, "the same trace was judged twice"


class TestFailSoftUnderConcurrency:
    def test_one_bad_trace_does_not_lose_the_others(self):
        def flaky(prompt: str) -> str:
            if "trace body 3" in prompt:
                raise RuntimeError("model said no")
            return '{"occurrences": []}'

        j = LLMFailureJudge(taxonomy=TAXONOMY, lm=flaky, max_workers=8, log=lambda _m: None)
        results = j.judge(_traces(10))
        assert j.failures == 1
        assert "t3" not in results, "a failed trace yields no entry"
        assert len(results) == 9, "the other nine must survive"


class TestCacheInteraction:
    def test_cached_traces_are_not_re_judged(self):
        calls: list[str] = []

        def counting_lm(prompt: str) -> str:
            calls.append(prompt)
            return '{"occurrences": []}'

        class Cache:
            def __init__(self):
                self.store = {}

            def get(self, *, taxonomy, candidate_key, trace_id):
                return self.store.get(trace_id)

            def put(self, *, taxonomy, candidate_key, trace_id, occurrences):
                self.store[trace_id] = occurrences

        cache = Cache()
        traces = _traces(6)
        j = LLMFailureJudge(taxonomy=TAXONOMY, lm=counting_lm, cache=cache, max_workers=4)
        j.judge(traces)
        assert len(calls) == 6
        j2 = LLMFailureJudge(taxonomy=TAXONOMY, lm=counting_lm, cache=cache, max_workers=4)
        results = j2.judge(traces)
        assert len(calls) == 6, "second pass must be served entirely from cache"
        assert len(results) == 6
