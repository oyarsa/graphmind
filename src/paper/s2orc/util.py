"""URL manipulation functions.

Wraps existing functions from libraries that might not integrate well with strict typing.
"""

import urllib.parse


def parse_url(url: str) -> urllib.parse.ParseResult:
    """Wrapper around urllib.parse.urlparse for type-checking."""
    return urllib.parse.urlparse(url)
