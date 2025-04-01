"""Helper types not available in the stdlib."""

from __future__ import annotations

from typing import Any, Protocol, TypeGuard, TypeVar, cast

_T_contra = TypeVar("_T_contra", contravariant=True)


class _SupportsLT(Protocol[_T_contra]):
    """A type that supports less-than comparison (`__lt__` method)."""

    def __lt__(self, other: _T_contra, /) -> bool:
        """Less-than comparison."""
        ...


type SupportsLT = _SupportsLT[Any]


def isdict[K, V](
    value: object, valtype: type[V] = type[Any], keytype: type[K] = str
) -> TypeGuard[dict[K, V]]:
    """Return True is `value` is of type `dict[K, V]`.

    Checks the first key/value pair to see if they are the correct type. If the dict is
    empty, it's considered of being any type.
    """
    if not isinstance(value, dict):
        return False

    if not value:  # Empty dicts can be of any type
        return True

    value = cast(dict[Any, Any], value)

    if not isinstance(next(iter(value.keys())), keytype):
        return False

    return isinstance(next(iter(value.values())), valtype)


def islist[T](value: object, type_: type[T]) -> TypeGuard[list[T]]:
    """Return True is `value` is of type `list[T]`.

    Checks the first item of the list to see if it's the correct type. If the list is
    empty, it's considered of being of any type.
    """
    if not isinstance(value, list):
        return False

    if not value:  # Empty lists can be of any type
        return True

    return isinstance(value[0], type_)
