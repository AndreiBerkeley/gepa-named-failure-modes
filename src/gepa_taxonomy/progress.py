"""Periodic progress line for long batch evaluations."""

from __future__ import annotations

import threading


def report_rollouts(adapter, total: int, *, label: str = "harvest", interval: float = 30.0) -> threading.Event:
    """Print ``label n/total`` every ``interval`` seconds until the returned event is set.

    Reads the adapter's ``rollouts`` counter, which every benchmark adapter
    maintains under its own lock; evaluation itself is untouched.
    """
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(interval):
            done = getattr(adapter, "rollouts", 0)
            print(f"{label}: {min(done, total)}/{total} instances", flush=True)

    threading.Thread(target=_loop, daemon=True).start()
    return stop


def report_spend(meters, *, label: str = "spend so far", interval: float = 60.0) -> threading.Event:
    """Print total metered spend every ``interval`` seconds until stopped.

    A run-long heartbeat: phases that write no log lines (the base-program
    evaluation before iteration 0, a slow single-worker stretch) still show
    movement, because spend always moves when rollouts do.
    """
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(interval):
            total = sum(getattr(m, "budgeted_usd", 0.0) + getattr(m, "excluded_usd", 0.0) for m in meters)
            print(f"{label}: ${total:.3f}", flush=True)

    threading.Thread(target=_loop, daemon=True).start()
    return stop
