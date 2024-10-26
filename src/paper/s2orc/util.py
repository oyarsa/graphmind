import urllib.parse
from collections.abc import Iterable
from typing import Any

import tqdm


def parse_url(url: str) -> urllib.parse.ParseResult:
    """Wrapper around urllib.parse.urlparse for type-checking."""
    return urllib.parse.urlparse(url)


async def progress_gather(*awaitable: Any, **kwargs: Any) -> Iterable[Any]:
    """Run asyncio.gather with tqdm progress bar.

    Arguments are exactly the same as asyncio.gather.
    """
    # There's no safe way to type-check this function, so we ignore the type error
    return await tqdm.asyncio.gather(*awaitable, **kwargs)  # type: ignore
