"""Helper types not available in the stdlib."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

_T_contra = TypeVar("_T_contra", contravariant=True)

type TSeq[T] = tuple[T, ...]
"""Variable-length tuple alias."""


class _SupportsLT(Protocol[_T_contra]):
    """A type that supports less-than comparison (`__lt__` method)."""

    def __lt__(self, other: _T_contra, /) -> bool:
        """Less-than comparison."""
        ...


type SupportsLT = _SupportsLT[Any]
