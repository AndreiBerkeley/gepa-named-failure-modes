"""Metered language-model calls for the pipeline and the demo.

One thin wrapper over litellm: every call returns its exact token counts so
the dollar budget is enforceable, and retries, timeouts, and logging are
uniform. Model ids are plain litellm ids -- ``gpt-5-mini``,
``gemini/gemini-2.5-flash-lite``, ``anthropic/claude-sonnet-4-6``,
``bedrock/...`` -- and route exactly as litellm routes them. Credentials are
whatever the chosen provider reads from the environment; a missing credential
surfaces as that provider's own error on the first call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MeteredLM:
    """Thin litellm wrapper returning ``(text, input_tokens, output_tokens)``.

    Token counts come from the API response, not an estimate, because they feed
    the dollar-budget meter directly.
    """

    model: str
    temperature: float | None = None
    timeout: int = 600
    max_retries: int = 3

    def complete(self, prompt: str, *, max_tokens: int = 4096) -> tuple[str, int, int]:
        import litellm

        # On a retried failure litellm prints a four-line feedback banner per
        # attempt; a throttled harvest turns that into a wall. Keep the errors,
        # drop the banner.
        litellm.suppress_debug_info = True

        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "timeout": self.timeout,
            "num_retries": self.max_retries,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        response = litellm.completion(**kwargs)
        text = response.choices[0].message.content or ""
        usage = response.usage
        return text, int(usage.prompt_tokens), int(usage.completion_tokens)


@dataclass
class MeteredReflectionLM:
    """The reflection LM in the shape gepa actually invokes it.

    gepa's contract, read from source rather than assumed:

    * ``api.py:246`` wraps any non-``total_cost`` object in ``TrackingLM``.
    * ``TrackingLM.__call__`` (``lm.py:225-235``) does ``self._fn(prompt)`` --
      so the object must be **callable, taking one positional prompt and
      returning a string**.
    * ``StatelessReflectionLM`` then calls that callable per component.

    Passing a ``MeteredLM`` directly failed here: it exposes ``.complete()`` but
    is not callable, so ``self._fn(prompt)`` raised ``TypeError``. gepa catches
    that, logs "Reflective mutation did not propose a new candidate", and keeps
    iterating -- burning minibatch rollouts forever while never proposing
    anything.

    This wrapper also fixes a second, quieter bug: reflection spend was not
    being metered at all. ``cost.py`` states the per-seed budget covers
    "minibatch rollouts, reflection calls, and val evaluations", but nothing
    recorded reflection. Every call is now booked to a meter, so the budget
    means what it says.

    ``batch_complete`` is deliberately NOT provided. ``TrackingLM.__getattr__``
    exposes it only when the wrapped object has it, and its absence simply
    routes gepa to the per-task reflection path. With the default
    single-mutation sampling there is one reflection job per iteration, so
    batching would buy nothing and adds a second call shape to keep conformant.
    """

    lm: MeteredLM
    meter: Any
    model: str
    phase: str = "optimization"
    max_tokens: int = 8192
    calls: int = 0
    #: Append-only spend log. The out-of-process watchdog enforces a hard dollar
    #: ceiling and can only read what is on disk; without this, reflection spend
    #: is invisible to it and the ceiling under-counts.
    spend_log: Any = None
    #: Append-only prompt/response archive. Nothing else persists the raw
    #: reflection bodies -- gepa's state and run logs store only ids and scores
    #: -- so auditing "did the reflection prompt actually contain X" previously
    #: required triangulating renders, billed token counts and lineage diffs.
    #: One JSONL line per call makes that a grep.
    prompt_log: Any = None

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        """One positional prompt in, a string out. This is the whole contract."""
        text = prompt if isinstance(prompt, str) else str(prompt)
        out, tin, tout = self.lm.complete(text, max_tokens=self.max_tokens)
        self.calls += 1
        before = getattr(self.meter, "budgeted_usd", 0.0)
        self.meter.record(model=self.model, input_tokens=tin, output_tokens=tout, phase=self.phase)
        after = getattr(self.meter, "budgeted_usd", 0.0)
        self._append(after - before, tin, tout)
        if self.prompt_log is not None:
            import json
            try:
                with open(self.prompt_log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"call": self.calls, "prompt": text, "response": out}) + "\n")
            except OSError:
                pass  # archive loss must never cost a paid run
        return out

    def _append(self, cost_usd: float, tin: int, tout: int) -> None:
        """Write through immediately: an interrupted run must not hide spend."""
        if self.spend_log is None:
            return
        import json
        import os

        try:
            with open(self.spend_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "call": self.calls, "cost_usd": cost_usd,
                    "input_tokens": tin, "output_tokens": tout, "phase": self.phase,
                }) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError:
            # Losing the log must never cost a paid run; the in-process stopper
            # still meters this call.
            pass


class ReflectionConformanceError(RuntimeError):
    """Raised when a reflection LM would not survive gepa's invocation path."""


def verify_reflection_lm(obj: Any) -> dict[str, Any]:
    """Free preflight: check conformance the way the engine will invoke it.

    Mirrors gepa's own wrapping and call shape without making a network call,
    so a mismatch fails at launch-time preflight instead of being swallowed
    mid-run as "did not propose a new candidate".
    """
    import inspect

    from gepa.lm import TrackingLM

    report: dict[str, Any] = {"callable": callable(obj), "type": type(obj).__name__}

    if not report["callable"]:
        raise ReflectionConformanceError(
            f"reflection LM {type(obj).__name__} is not callable. gepa invokes it as "
            "`self._fn(prompt)` (lm.py:231); an object exposing only `.complete()` "
            "raises TypeError, which gepa SWALLOWS -- the run then burns its whole "
            "budget without ever proposing a candidate."
        )

    call = obj if inspect.isfunction(obj) or inspect.ismethod(obj) else type(obj).__call__
    try:
        sig = inspect.signature(call)
        params = [
            p for p in sig.parameters.values()
            if p.name != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        ]
        report["positional_params"] = [p.name for p in params]
        if not params:
            raise ReflectionConformanceError(
                f"reflection LM {type(obj).__name__} takes no positional argument; "
                "gepa calls it with exactly one (the prompt)."
            )
    except (TypeError, ValueError):  # builtins / C callables
        report["positional_params"] = "unknown"

    # gepa wraps it exactly like this; make sure that survives too.
    wrapped = TrackingLM(obj) if not hasattr(obj, "total_cost") else obj
    report["wrapped_as"] = type(wrapped).__name__
    report["exposes_total_cost"] = hasattr(wrapped, "total_cost")
    report["batched_path"] = hasattr(obj, "batch_complete")
    report["ok"] = True
    return report
