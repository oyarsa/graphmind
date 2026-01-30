"""Request context management using contextvars for request ID propagation."""

import contextvars

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def get_request_id() -> str | None:
    """Get the current request ID from context, or None if not set."""
    return request_id_var.get()


def set_request_id(request_id: str) -> contextvars.Token[str | None]:
    """Set the request ID in context. Returns a token for resetting."""
    return request_id_var.set(request_id)
