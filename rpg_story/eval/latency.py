"""Latency helpers shared by automated evaluation scripts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import fmean, median, pstdev
from typing import Any, Callable, TypeVar
import time


T = TypeVar("T")


@dataclass(frozen=True)
class TimedCall:
    """Result metadata for one timed call."""

    seconds: float
    started_at: str
    ended_at: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timed_call(func: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[TimedCall, T]:
    """Execute ``func`` once and return timing metadata plus its result."""

    started_at = utc_now_iso()
    started = time.perf_counter()
    result = func(*args, **kwargs)
    seconds = max(0.0, time.perf_counter() - started)
    ended_at = utc_now_iso()
    return TimedCall(seconds=round(seconds, 6), started_at=started_at, ended_at=ended_at), result


def summarize_durations(values: list[float]) -> dict[str, float | int | None]:
    """Summarize a duration sample list for reporting."""

    cleaned = [float(value) for value in values if isinstance(value, (int, float))]
    if not cleaned:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }
    return {
        "count": len(cleaned),
        "mean": round(fmean(cleaned), 6),
        "median": round(median(cleaned), 6),
        "min": round(min(cleaned), 6),
        "max": round(max(cleaned), 6),
        "std": round(pstdev(cleaned), 6) if len(cleaned) >= 2 else 0.0,
    }
