import urllib.parse


def parse_url(url: str) -> urllib.parse.ParseResult:
    """Wrapper around urllib.parse.urlparse for type-checking."""
    return urllib.parse.urlparse(url)
