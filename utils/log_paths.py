"""Helpers for separating desktop and NUC execution log directories."""

from __future__ import annotations

import os
from pathlib import Path


DESKTOP_LOG_SUFFIX = "_desktop"
NUC_LOG_SUFFIX = "_nuc"
LOG_MACHINE_SUFFIXES = (DESKTOP_LOG_SUFFIX, NUC_LOG_SUFFIX)


def _replace_machine_suffix(name: str, suffix: str) -> str:
    for existing in LOG_MACHINE_SUFFIXES:
        if name.endswith(existing):
            return name[: -len(existing)] + suffix
    return name + suffix


def unique_suffixed_dir(base_dir: str, prefix: str, suffix: str) -> str:
    counter = 1
    while True:
        path = os.path.join(base_dir, f"{prefix}_{counter}{suffix}")
        if not os.path.exists(path):
            return path
        counter += 1


def with_log_machine_suffix(path: str, suffix: str) -> str:
    parts = list(Path(str(path)).parts)
    if not parts:
        return path

    for idx, part in enumerate(parts):
        if part.startswith("exec_"):
            parts[idx] = _replace_machine_suffix(part, suffix)
            return str(Path(*parts))

    parts[-1] = _replace_machine_suffix(parts[-1], suffix)
    return str(Path(*parts))


def equivalent_log_paths(a: str, b: str) -> bool:
    if not a or not b:
        return False

    def variants(path: str) -> set[str]:
        raw = str(path)
        return {
            os.path.normcase(os.path.abspath(raw)),
            os.path.normcase(
                os.path.abspath(with_log_machine_suffix(raw, DESKTOP_LOG_SUFFIX))
            ),
            os.path.normcase(
                os.path.abspath(with_log_machine_suffix(raw, NUC_LOG_SUFFIX))
            ),
        }

    return not variants(a).isdisjoint(variants(b))
