"""Helper types not available in the stdlib."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol, TypeGuard, TypeVar, cast, override

from typing_extensions import TypeIs

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


type MaybeFunc[T, U] = Callable[[T], Maybe[U]] | Callable[[T], U | None]
"""Function that takes a naked T and creates a new Maybe[U] or new U|None object."""


class MaybeError(Exception):
    """Error raised when a Nothing is wrapped."""


class Maybe[T](ABC):
    """A type representing an optional value.

    - `maybe` to construct a new value from `T | None`.
    - `map` (fmap) and `bind` (>>=) to manipulate the inner value. `bind` supports both
        function that return another Maybe and `T | None`.
    - `unwrap` to force the underlying value, with an exception if it's None.
    - `unwrap_or` to get a default value if Nothing.
    - `isjust`, `isnothing`, `ismaybe` and `ismaybe_t` for type narrowing.

    Pattern matching works too, but you must have wildcard pattern since we can't tell
    the type system that only Just/Nothing are allowed.
    """

    @abstractmethod
    def map[U](self, f: Callable[[T], U]) -> Maybe[U]:
        """Apply a function to the contained value if it exists."""

    @abstractmethod
    def bind[U](self, f: MaybeFunc[T, U]) -> Maybe[U]:
        """Apply a function that returns a Maybe to the contained value if it exists.

        The function can return both an actual Maybe, or a `T | None` object that's
        wrapped into a Maybe.
        """

    @abstractmethod
    def unwrap(self, msg: str | None = None) -> T:
        """Extract the contained value, or raises an error if Nothing.

        Raises:
            MaybeError: if the object is Nothing. If `msg` is given, it will be used as
            the exception message.
        """

    @abstractmethod
    def unwrap_or[U](self, default: T | U) -> T | U:
        """Extract the contained value, or return the default value if Nothing."""

    @abstractmethod
    def unwrap_f[U](self, f: Callable[[], U]) -> T | U:
        """Extract the contained value, or use function result if Nothing."""


class Just[T](Maybe[T]):
    """A Maybe that contains a value."""

    value: T

    __match_args__ = ("value",)

    def __init__(self, value: T) -> None:
        self.value = value

    @override
    def map[U](self, f: Callable[[T], U]) -> Maybe[U]:
        return Just(f(self.value))

    @override
    def bind[U](self, f: MaybeFunc[T, U]) -> Maybe[U]:
        result = f(self.value)
        if ismaybe(result):
            return result
        return maybe(result)

    @override
    def unwrap(self, msg: str | None = None) -> T:
        return self.value

    @override
    def unwrap_or[U](self, default: T | U) -> T | U:
        return self.value

    @override
    def unwrap_f[U](self, f: Callable[[], U]) -> T | U:
        return self.value

    def __eq__(self, other: object) -> bool:
        """Items are equal if they're both Just with the same inner value."""
        return (
            ismaybe_t(other, type(self.value))
            and isjust(other)
            and self.value == other.value
        )

    def __repr__(self) -> str:
        """Represent object as `Just(val)`."""
        return f"Just({self.value!r})"


class Nothing[T](Maybe[T]):
    """A Maybe that contains no value."""

    __match_args__ = ()

    @override
    def map[U](self, f: Callable[[T], U]) -> Maybe[U]:
        return Nothing()

    @override
    def bind[U](self, f: MaybeFunc[T, U]) -> Maybe[U]:
        return Nothing()

    @override
    def unwrap(self, msg: str | None = None) -> T:
        raise MaybeError(msg or "Cannot unwrap Nothing")

    @override
    def unwrap_or[U](self, default: T | U) -> T | U:
        return default

    @override
    def unwrap_f[U](self, f: Callable[[], U]) -> T | U:
        return f()

    def __eq__(self, other: object) -> bool:
        """Items are equal if they're both Nothing."""
        return ismaybe(other) and isnothing(other)

    def __repr__(self) -> str:
        """Represent object as `Nothing`."""
        return "Nothing"


def isjust[T](maybe: Maybe[T]) -> TypeIs[Just[T]]:
    """Return true if the object is Just, with type narrowing."""
    return isinstance(maybe, Just)


def isnothing[T](maybe: Maybe[T]) -> TypeIs[Nothing[T]]:
    """Return true if the object is Nothing, with type narrowing."""
    return isinstance(maybe, Nothing)


def ismaybe(maybe: Any) -> TypeIs[Maybe[Any]]:
    """Return true if the object is Maybe, with type narrowing.

    Does not assert generic type. Use `ismaybe_t` for that.
    """
    return isinstance(maybe, Maybe)


def ismaybe_t[T](maybe: Any, t: type[T]) -> TypeIs[Maybe[T]]:
    """Return true if the object is Maybe, with type narrowing. Generic.

    If `maybe` is Just, we check the value against `t`. If it's Nothing, it's considered
    to be of any type.

    If you don't know the underlying type, use `ismaybe`.
    """
    print(maybe, t)
    if ismaybe(maybe):
        if isjust(maybe) and isinstance(maybe.value, t):
            return True

        if isnothing(maybe):
            return True

    return False


def maybe[T](val: T | None, t: type[T] | None = None) -> Maybe[T]:
    """Construct Maybe from optional value.

    If `val` is None, specify the type `t` for proper type inference.
    """
    if val is None:
        return Nothing()
    return Just(val)
