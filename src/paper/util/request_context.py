"""Request context management using contextvars for request ID propagation."""

import contextvars
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class RequestContext:
    """Context information for a request, used for error logging."""

    request_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    arxiv_id: str | None = None

    def format(self) -> str:
        """Format context for log output."""
        parts = [self.request_id, self.timestamp.strftime("%Y-%m-%d %H:%M:%S")]
        if self.arxiv_id:
            parts.append(self.arxiv_id)
        return " | ".join(parts)


_request_context_var: contextvars.ContextVar[RequestContext | None] = (
    contextvars.ContextVar("request_context", default=None)
)


def get_request_context() -> RequestContext | None:
    """Get the current request context, or None if not set."""
    return _request_context_var.get()


def set_request_context(request_id: str) -> RequestContext:
    """Create and set a new request context. Returns the created context."""
    ctx = RequestContext(request_id=request_id)
    _request_context_var.set(ctx)
    return ctx


def update_request_context(*, arxiv_id: str | None = None) -> None:
    """Update fields on the current request context.

    Does nothing if no context is set. `request_id` and `timestamp` cannot be set after
    creation.

    Args:
        arxiv_id: The arXiv ID for the paper being processed.
    """
    ctx = _request_context_var.get()
    if ctx is None:
        return

    if arxiv_id is not None:
        ctx.arxiv_id = arxiv_id
