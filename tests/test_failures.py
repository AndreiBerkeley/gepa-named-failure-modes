"""Shared failure classifier. FREE: pure functions, no network.

These tests encode two incidents rather than a specification, because the
specification was wrong twice and the incidents are what corrected it.
"""

from __future__ import annotations

import threading

import pytest

from gepa_taxonomy.failures import (
    PROGRAM,
    TRANSPORT,
    FailureLog,
    classify,
    describe,
    is_recognised_transport,
)


class TestClassify:
    def test_an_unrecognised_exception_defaults_to_transport(self):
        """The inversion, and the reason this module exists.

        Under the old allow-list an unmatched exception was counted as a PROGRAM
        error, which does not count toward the abort threshold. IFBench seed 2
        accumulated 273 of them and the guard never fired (F056). Transport is
        the safe default: mistaking a program fault for transport costs one loud
        abort; the reverse silently corrupts a paid run.
        """
        assert classify(RuntimeError("something nobody anticipated")) == TRANSPORT

    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("RateLimitError: too many requests"),
            RuntimeError("litellm.APIConnectionError: connection failed"),
            RuntimeError("Internal Server Error"),  # F040: space, not 'internalserver'
            RuntimeError("httpcore.ConnectError: [Errno 11001] getaddrinfo failed"),
            RuntimeError("ServiceUnavailableError"),
            RuntimeError("litellm.APIError: upstream problem"),
        ],
    )
    def test_known_network_failures_are_transport(self, exc):
        assert classify(exc) == TRANSPORT
        assert is_recognised_transport(exc)

    def test_f040_regression_internal_server_error_with_a_space(self):
        """The exact string that slipped past the old allow-list."""
        assert is_recognised_transport(RuntimeError("Internal Server Error"))

    def test_f056_regression_connect_error_and_getaddrinfo(self):
        """'ConnectError' does not contain 'connectionerror'; DNS failures say
        'getaddrinfo failed'. Neither matched before."""
        assert is_recognised_transport(RuntimeError("ConnectError: getaddrinfo failed"))

    def test_unrecognised_is_transport_but_not_recognised(self):
        """Both facts matter: it counts toward the abort AND is flagged as
        something TRANSPORT_MARKERS does not yet name."""
        exc = ValueError("a novel failure")
        assert classify(exc) == TRANSPORT
        assert not is_recognised_transport(exc)


class TestDescribe:
    def test_includes_type_and_message(self):
        assert describe(ValueError("boom")) == "ValueError: boom"


class TestFailureLog:
    def test_counts_and_samples(self):
        log = FailureLog()
        log.record(RuntimeError("RateLimitError: slow down"))
        log.record(RuntimeError("RateLimitError: slow down"))
        s = log.summary()
        assert s["transport_errors"] == 2
        assert s["error_samples"][TRANSPORT] == ["RuntimeError: RateLimitError: slow down"], (
            "duplicates are not re-sampled"
        )

    def test_records_the_message_that_f053_discarded(self):
        """The whole point: a count without a cause made 273 failures in a
        finished, paid-for seed permanently undiagnosable."""
        log = FailureLog()
        log.record(ValueError("the thing that actually went wrong"))
        assert "the thing that actually went wrong" in log.summary()["error_samples"][TRANSPORT][0]

    def test_samples_are_bounded(self):
        log = FailureLog(max_samples=3)
        for i in range(50):
            log.record(RuntimeError(f"distinct failure {i}"))
        assert log.summary()["transport_errors"] == 50
        assert len(log.summary()["error_samples"][TRANSPORT]) == 3

    def test_unrecognised_transport_is_counted_separately(self):
        log = FailureLog()
        log.record(RuntimeError("RateLimitError: known"))
        log.record(RuntimeError("wholly novel"))
        s = log.summary()
        assert s["transport_errors"] == 2
        assert s["unrecognised_transport"] == 1, "a rising count here means the marker list is stale"

    def test_aborting_count_is_transport_only(self):
        log = FailureLog()
        log.record(RuntimeError("RateLimitError: x"))
        assert log.aborting_count == 1

    def test_empty_summary_omits_empty_sample_buckets(self):
        assert FailureLog().summary()["error_samples"] == {}

    def test_is_thread_safe(self):
        """Adapters record from worker threads; a lost increment would understate
        a failure storm."""
        log = FailureLog()

        def hammer():
            for _ in range(200):
                log.record(RuntimeError("RateLimitError: concurrent"))

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert log.summary()["transport_errors"] == 1600


class TestProgramBucket:
    def test_program_markers_is_empty_by_design(self):
        """Not an oversight. For these programs a rollout is a model call plus
        glue, so nearly every exception is model- or network-related, and the
        ones that are genuinely our bug are systematic and SHOULD abort. Entries
        get added here only with a recorded sample as evidence."""
        from gepa_taxonomy.failures import PROGRAM_MARKERS

        assert PROGRAM_MARKERS == ()

    def test_a_program_marker_would_route_to_the_program_bucket(self, monkeypatch):
        """The mechanism works when a marker is added -- verified without
        committing to one."""
        import gepa_taxonomy.failures as f

        monkeypatch.setattr(f, "PROGRAM_MARKERS", ("definitelyourbug",))
        exc = RuntimeError("DefinitelyOurBug: candidate missing a component")
        assert f.classify(exc) == PROGRAM
        log = f.FailureLog()
        log.record(exc)
        assert log.summary()["program_errors"] == 1
        assert log.aborting_count == 0
