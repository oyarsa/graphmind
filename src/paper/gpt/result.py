"""GPTResult monad and related utility functions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeGuard, TypeVar, cast, overload

if TYPE_CHECKING:
    from paper.gpt.model import PromptResult

T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True, kw_only=True)
class GPTResult(Generic[T_co]):  # noqa: UP046
    """Result of a GPT request and its full API cost."""

    result: T_co
    cost: float

    def map[U](self, func: Callable[[T_co], U]) -> GPTResult[U]:
        """Apply `func` to inner value and return new result."""
        return GPTResult(result=func(self.result), cost=self.cost)

    def then[U](self, other: GPTResult[U]) -> GPTResult[U]:
        """Combine two request costs with the second result."""
        return GPTResult(result=other.result, cost=self.cost + other.cost)

    def bind[U](self, func: Callable[[T_co], GPTResult[U]]) -> GPTResult[U]:
        """Apply monadic function to inner value and sum the costs."""
        return self.then(func(self.result))

    async def abind[U](
        self, func: Callable[[T_co], Awaitable[GPTResult[U]]]
    ) -> GPTResult[U]:
        """Apply monadic function to inner value and sum the costs (async version)."""
        return self.then(await func(self.result))

    async def amap[U](self, func: Callable[[T_co], Awaitable[U]]) -> GPTResult[U]:
        """Apply `func` to inner value and return new result (async version)."""
        return GPTResult(result=await func(self.result), cost=self.cost)

    @staticmethod
    def unit[T](value: T) -> GPTResult[T]:
        """New result with cost 0."""
        return GPTResult(result=value, cost=0)

    def fix[X](self: GPTResult[X | None], default: Callable[[], X] | X) -> GPTResult[X]:
        """Fix the result of the GPTResult by replacing None with the result of `default`.

        Args:
            self: GPTResult to fix containing a value or None.
            default: Function that returns the default value or the value itself.

        Returns:
            GPTResult with the result replaced by the default if it was None.
            If the result is already valid (not None), it returns itself.
        """
        if gpt_is_valid(self):
            return self

        if callable(default):
            value = cast(X, default())
        else:
            value: X = default

        return self.map(lambda _: value)

    @overload
    def lift[U, V](
        self, other: GPTResult[U], func: Callable[[T_co, U], V]
    ) -> GPTResult[V]: ...

    @overload
    def lift[U1, U2, V](
        self,
        other1: GPTResult[U1],
        other2: GPTResult[U2],
        func: Callable[[T_co, U1, U2], V],
    ) -> GPTResult[V]: ...

    @overload
    def lift[U1, U2, U3, V](
        self,
        other1: GPTResult[U1],
        other2: GPTResult[U2],
        other3: GPTResult[U3],
        func: Callable[[T_co, U1, U2, U3], V],
    ) -> GPTResult[V]: ...

    def lift(self, *args: Any, **_: Any) -> GPTResult[Any]:
        """Combine multiple results with an n-ary function."""
        *others, func = args

        values = [self, *others]
        results = (v.result for v in values)
        total_cost = sum(v.cost for v in values)

        return GPTResult(result=func(*results), cost=total_cost)


def gpt_is_valid[T](result: GPTResult[T | None]) -> TypeGuard[GPTResult[T]]:
    """Check if the GPTResult is valid, i.e. has a non-None result."""
    return result.result is not None


def gpt_is_none[T](result: GPTResult[T | None]) -> TypeGuard[GPTResult[None]]:
    """Check if the GPTResult is empty, i.e. has a None result."""
    return result.result is None


def gpt_is_type[T, U](result: GPTResult[T], type_: type[U]) -> TypeGuard[GPTResult[U]]:
    """Check if the GPTResult is of a specific type."""
    return isinstance(result.result, type_)


def gpt_unit[T](value: T) -> GPTResult[T]:
    """Create a unit GPTResult with the given value and cost 0.

    Use this instead of GPTResult.unit for type inference.
    """
    return GPTResult[T].unit(value)


def gpt_sequence[T](results: Iterable[GPTResult[T]]) -> GPTResult[Sequence[T]]:
    """Convert sequence of results to result of sequence, aggregating costs."""
    items: list[T] = []
    total_cost = 0.0

    for result in results:
        items.append(result.result)
        total_cost += result.cost

    return GPTResult(result=tuple(items), cost=total_cost)


def gpr_traverse[T, U](
    results: Iterable[GPTResult[PromptResult[T]]], f: Callable[[T], U]
) -> GPTResult[Sequence[PromptResult[U]]]:
    """Sequence results and map function over values in GPTResult+PromptResult stack."""
    return gpt_sequence(results).map(lambda items: tuple(item.map(f) for item in items))


def gpr_map[T, U](
    result: GPTResult[PromptResult[T]], f: Callable[[T], U]
) -> GPTResult[PromptResult[U]]:
    """Map functions through GPTResult+PromptResult stack."""
    return result.map(lambda r: r.map(f))
