"""Dependency injection items."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

import dotenv
from fastapi import Depends, FastAPI, Request

from paper import single_paper
from paper.backend.db import DatabaseManager


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
