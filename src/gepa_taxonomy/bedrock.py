"""Bedrock LM client and credential handling.

Auth is the native bearer-token variable ``AWS_BEARER_TOKEN_BEDROCK``, which
both boto3/botocore and litellm read directly. There is no ``.env`` file and
none should be created.

Routing invariant
-----------------
Every call must go to **Bedrock** using bearer auth. It must never fall back to
direct-Anthropic routing -- that would hit a different account, a different
price sheet, and (silently) a different set of credentials. The ``bedrock/``
prefix on the model id is what pins litellm's provider resolution; the guard
below makes an unprefixed id a hard error rather than a silent reroute.
Locked in by ``tests/test_bedrock_routing.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

BEARER_ENV = "AWS_BEARER_TOKEN_BEDROCK"
REGION_ENV = "AWS_REGION"
DEFAULT_REGION = "us-east-1"

#: litellm resolves the provider from this prefix. Without it, an
#: ``anthropic.*`` model id routes to the direct Anthropic API instead.
BEDROCK_PREFIX = "bedrock/"

CREDENTIAL_HELP = (
    f"{BEARER_ENV} is not set.\n"
    "Non-interactive shells do not auto-load ~/.zshrc. Either run from an "
    "interactive shell where it is already exported, or wrap the command:\n"
    "  zsh -c 'source ~/.zshrc >/dev/null 2>&1; <command>'"
)


class CredentialsMissingError(RuntimeError):
    """Raised at startup when Bedrock credentials are absent.

    Deliberately raised *before* any work begins, so a long run cannot die
    halfway through on an auth error.
    """


class RoutingError(RuntimeError):
    """Raised when a model id would not route to Bedrock."""


def require_credentials() -> str:
    """Fail fast if Bedrock credentials are missing. Returns the region.

    Never returns, logs, or otherwise exposes the token value.
    """
    if not os.environ.get(BEARER_ENV):
        raise CredentialsMissingError(CREDENTIAL_HELP)
    return os.environ.get(REGION_ENV) or DEFAULT_REGION


def bedrock_model_id(model: str) -> str:
    """Return ``model`` with the ``bedrock/`` provider prefix, validating it.

    Rejects ids that carry another provider's prefix, and ids that are not
    Bedrock inference-profile / foundation-model ids.
    """
    if model.startswith(BEDROCK_PREFIX):
        bare = model[len(BEDROCK_PREFIX) :]
    else:
        bare = model

    if "/" in bare:
        raise RoutingError(
            f"model id {model!r} carries a non-Bedrock provider prefix. All calls must route to Bedrock."
        )
    if not any(
        bare.startswith(p)
        for p in ("anthropic.", "us.anthropic.", "global.anthropic.", "eu.anthropic.", "apac.anthropic.")
    ):
        raise RoutingError(
            f"model id {model!r} is not a Bedrock Anthropic id. Expected an "
            "'anthropic.*' foundation model or a '<region>.anthropic.*' "
            "inference profile."
        )
    return BEDROCK_PREFIX + bare


@dataclass
class BedrockLM:
    """Thin litellm wrapper returning ``(text, input_tokens, output_tokens)``.

    Token counts come from the API response, not an estimate, because they feed
    the dollar-budget meter directly.
    """

    model: str
    region: str | None = None
    temperature: float | None = None
    timeout: int = 600
    max_retries: int = 3

    def __post_init__(self) -> None:
        self.region = self.region or require_credentials()
        self.routed_model = bedrock_model_id(self.model)

    def complete(self, prompt: str, *, max_tokens: int = 4096) -> tuple[str, int, int]:
        import litellm

        kwargs: dict[str, object] = {
            "model": self.routed_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "aws_region_name": self.region,
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

    Passing a ``BedrockLM`` directly failed here: it exposes ``.complete()`` but
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

    lm: BedrockLM
    meter: Any
    model: str
    phase: str = "optimization"
    max_tokens: int = 8192
    calls: int = 0
    #: Append-only spend log. The out-of-process watchdog enforces a hard dollar
    #: ceiling and can only read what is on disk; without this, reflection spend
    #: is invisible to it and the ceiling under-counts (D030).
    spend_log: Any = None
    #: Append-only prompt/response archive. Nothing else persists the raw
    #: reflection bodies -- gepa's state and run logs store only ids and scores
    #: -- so auditing "did the reflection prompt actually contain X" previously
    #: required triangulating renders, billed token counts and lineage diffs
    #: (TAXONOMY_AUDIT.md §2.5). One JSONL line per call makes that a grep.
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
