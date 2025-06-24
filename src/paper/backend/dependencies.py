"""Dependency injection items."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

import dotenv
from fastapi import Depends, FastAPI, Request

from paper import single_paper
from paper.backend.db import DatabaseManager
from paper.gpt.run_gpt import LLMClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events.

    Handles startup and shutdown operations including:
    - Loading environment variables from .env file
    - Setting up rate limiter for external API calls
    - Opening and closing database connections

    Args:
        app: FastAPI application instance.

    Yields:
        None during application runtime.
    """
    dotenv.load_dotenv()
    app.state.limiter = single_paper.get_limiter(use_semaphore=False)
    app.state.llm_registry = LLMClientRegistry()

    async with DatabaseManager(
        dbname=os.environ["XP_DB_NAME"],
        user=os.environ["XP_DB_USER"],
        password=os.environ["XP_DB_PASSWORD"],
        host=os.environ["XP_DB_HOST"],
        port=os.environ["XP_DB_PORT"],
    ) as db:
        app.state.db = db
        yield


def get_limiter(request: Request) -> single_paper.Limiter:
    """Dependency injection for the request rate limiter.

    Args:
        request: FastAPI request object containing application state.

    Returns:
        Rate limiter instance from application state.
    """
    return request.app.state.limiter


LimiterDep = Annotated[single_paper.Limiter, Depends(get_limiter)]


class LLMClientRegistry:
    """Registry of LLMClient for multiple models.

    We need to share the LLMClients across request, but we need separate objects for
    different models.

    Uses `LLMClient.new_env` to create the LLMs with seed=0.

    Attributes:
        _register: Mapping of model name to LLMClient.
    """

    def __init__(self) -> None:
        self._register: dict[str, LLMClient] = {}

    def get_client(self, model: str) -> LLMClient:
        """Get client for model if it exists, or creates a new one."""
        if model not in self._register:
            self._register[model] = LLMClient.new_env(model, seed=0)
        return self._register[model]


def get_llm_clients(request: Request) -> LLMClientRegistry:
    """Dependency injection for the LLM clients.

    Args:
        request: FastAPI request object containing application state.

    Returns:
        LLM Client registry.
    """
    return request.app.state.llm_registry


LLMRegistryDep = Annotated[LLMClientRegistry, Depends(get_llm_clients)]
