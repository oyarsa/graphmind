"""Utilities for handling async tasks with progress bars."""

from collections.abc import Awaitable, Iterable
from typing import Any

from tqdm.asyncio import tqdm_asyncio


def as_completed[T](
    tasks: Iterable[Awaitable[T]], **kwargs: Any
) -> Iterable[Awaitable[T]]:
    """Returns iterator overs `tasks` as they are completed, showing a progress bar.

    Tasks returned by the iterator still need to be `await`ed.
    Type-safe wrapper around `tqdm.asyncio.as_completed`. `kwargs` are forwarded to it.

    See also `asyncio.as_completed.`
    """
    return tqdm_asyncio.as_completed(tasks, **kwargs)  # type: ignore


async def gather[T](*tasks: Awaitable[T], **kwargs: Any) -> Iterable[T]:
    """Wait for tasks to complete with a progress bar. Returns an iterator the results.

    Type-safe wrapper around `tqdm.asyncio.as_completed`. `kwargs` are forwarded to it.

    See also `asyncio.gather`.
    """
    # There's no safe way to type-check this function, so we ignore the type error
    return await tqdm_asyncio.gather(*tasks, **kwargs)  # type: ignore
