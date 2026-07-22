"""TEMPORARY per-process phase-timing utility.

Usage:
    from agent.sampler._phase_timer import phase, timed, reset, format_summary

    @timed("pose_solve")           # decorate a hot function
    def solve(self, ...): ...

    with phase("mcts_search"):     # or wrap inline
        ...

    reset()                        # at the start of an episode
    print(format_summary())        # at the end

Each Ray worker has its own counters (module-global within its process).
Remove with: `grep -rn "_phase_timer\\|TIMING" .` and delete those lines + this file.
"""

import functools
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Dict


_acc: Dict[str, float] = defaultdict(float)
_counts: Dict[str, int] = defaultdict(int)


@contextmanager
def phase(name: str):
    t = time.perf_counter()
    try:
        yield
    finally:
        _acc[name] += time.perf_counter() - t
        _counts[name] += 1


def timed(name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with phase(name):
                return fn(*args, **kwargs)

        return wrapper

    return deco


def increment(name: str, value: float = 1.0) -> None:
    """Accumulate an arbitrary value (not wall-clock time) into a named bucket."""
    _acc[name] += value
    _counts[name] += 1


def reset() -> None:
    _acc.clear()
    _counts.clear()


def format_summary(header: str = "") -> str:
    if not _acc:
        return f"{header} (no timing data)" if header else "(no timing data)"
    rows = [header] if header else []
    name_w = max(len(n) for n in _acc)
    for name in sorted(_acc, key=lambda n: -_acc[n]):
        total = _acc[name]
        n = _counts[name]
        avg_ms = (total / n) * 1000.0 if n else 0.0
        rows.append(
            f"  {name:>{name_w}s}  total={total:7.2f}s  n={n:>5d}  avg={avg_ms:7.2f}ms"
        )
    return "\n".join(rows)
