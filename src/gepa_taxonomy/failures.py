"""Rollout failure classification and recording, shared by every adapter.

Why this module exists
----------------------
Four adapters carried private copies of the same classifier and they had already
**drifted into three different versions** -- HotpotQA's was missing the
``"internal server"`` marker added for F040, so the same failure would be
classified differently depending on which benchmark hit it.

Why the default is TRANSPORT
----------------------------
The original classifier used an allow-list: match a known transport marker, or be
counted as a **program** error. That default is backwards, and it has now failed
twice.

* **F040** -- ``"Internal Server Error"`` (with a space) did not match the
  ``internalserver`` marker.
* **F056** -- IFBench seed 2 recorded **273** program errors and **0** transport
  errors. Those 273 rollouts scored 0.0, entered minibatch gates and full-val
  evaluations, and depressed whichever candidates hit them -- while the
  abort-at-N guard, whose entire purpose is to stop exactly that, never fired
  because the failures landed in the bucket it does not watch.

The two misclassification directions are not symmetric:

* transport mistaken for program -> **silent corruption** of a paid run;
* program mistaken for transport -> a **loud abort**, which costs one run and
  loses nothing.

So anything not positively identified as a program fault is treated as
transport. ``PROGRAM_MARKERS`` is deliberately empty: for these programs a
rollout is a model call plus a little glue, so essentially every exception is
model- or network-related, and the exceptions that genuinely are our bug are
systematic and *should* abort. Entries are added here only with evidence from a
recorded sample -- which is what ``samples`` is for.

Recording the message
---------------------
The old adapters counted failures and discarded the exception text (F053).
When 273 of them appeared in a finished, paid-for seed, the cause was
unrecoverable: nothing reached the run log either. A bounded sample of the
messages makes the count explicable without unbounded memory or log spam.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

TRANSPORT = "transport"
PROGRAM = "program"

#: Substrings identifying a failure to reach the model. Matched against the
#: exception's type name and message, lowercased, because litellm raises a wide
#: family of provider-specific classes and importing them to isinstance-check
#: would couple every adapter to litellm's internals.
#:
#: This list is now advisory rather than load-bearing: an unmatched exception is
#: treated as transport anyway. It is kept so the summary can distinguish
#: "recognised network failure" from "unrecognised, assumed transport", which is
#: the signal that tells us what to add here.
TRANSPORT_MARKERS: tuple[str, ...] = (
    "ratelimit",
    "throttl",
    "timeout",
    "serviceunavailable",
    "service unavailable",
    "internalserver",
    "internal server",
    "apiconnection",
    "connectionerror",
    "connecterror",
    "getaddrinfo",
    "overloaded",
    "toomanyrequests",
    "too many requests",
    "bedrockexception",
    "apierror",
    "badgateway",
    "remotedisconnected",
    "incompleteread",
    "ssl",
)

#: Exceptions that are positively OUR fault and must not count toward the
#: transport abort. Deliberately empty -- see the module docstring. Add only with
#: a recorded sample as evidence.
PROGRAM_MARKERS: tuple[str, ...] = ()


def describe(exc: BaseException) -> str:
    """``TypeName: message``, which is what gets sampled and shown."""
    return f"{type(exc).__name__}: {exc}"


def classify(exc: BaseException) -> str:
    """``TRANSPORT`` or ``PROGRAM``, defaulting to TRANSPORT.

    The default is the whole point: an unrecognised failure is assumed to be a
    failure to reach the model, so it counts toward the abort threshold rather
    than silently scoring a candidate down.
    """
    blob = describe(exc).lower()
    if any(marker in blob for marker in PROGRAM_MARKERS):
        return PROGRAM
    return TRANSPORT


def is_recognised_transport(exc: BaseException) -> bool:
    """True when a known marker matched, as opposed to defaulting to transport."""
    blob = describe(exc).lower()
    return any(marker in blob for marker in TRANSPORT_MARKERS)


@dataclass
class FailureLog:
    """Counts rollout failures and keeps a bounded sample of their messages.

    Thread-safe: adapters call ``record`` from worker threads.
    """

    #: Messages kept per bucket. Enough to diagnose, small enough that a run
    #: bleeding thousands of failures does not bloat the summary.
    max_samples: int = 5

    transport: int = 0
    program: int = 0
    #: Transport failures that did NOT match a known marker. A rising count here
    #: means TRANSPORT_MARKERS is missing something; the samples say what.
    unrecognised: int = 0

    samples: dict[str, list[str]] = field(default_factory=lambda: {TRANSPORT: [], PROGRAM: []})
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, exc: BaseException) -> str:
        """Classify, count, sample. Returns the bucket."""
        kind = classify(exc)
        text = describe(exc)
        with self._lock:
            if kind == TRANSPORT:
                self.transport += 1
                if not is_recognised_transport(exc):
                    self.unrecognised += 1
            else:
                self.program += 1
            bucket = self.samples[kind]
            if len(bucket) < self.max_samples and text not in bucket:
                bucket.append(text)
        return kind

    @property
    def aborting_count(self) -> int:
        """Failures that count toward the abort threshold."""
        return self.transport

    def summary(self) -> dict[str, object]:
        with self._lock:
            return {
                "transport_errors": self.transport,
                "program_errors": self.program,
                "unrecognised_transport": self.unrecognised,
                # The field whose absence made F056 undiagnosable.
                "error_samples": {k: list(v) for k, v in self.samples.items() if v},
            }


__all__ = [
    "PROGRAM",
    "PROGRAM_MARKERS",
    "TRANSPORT",
    "TRANSPORT_MARKERS",
    "FailureLog",
    "classify",
    "describe",
    "is_recognised_transport",
]
