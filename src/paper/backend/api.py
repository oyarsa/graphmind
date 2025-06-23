"""FastAPI application with PostgreSQL backend.

Main application module that sets up the FastAPI server, database connections,
CORS middleware, and routes for the Paper Explorer API.
"""

import datetime as dt
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from paper import single_paper
from paper.backend.db import DatabaseManager
from paper.backend.model import HealthCheck
from paper.backend.routers import mind, network


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


def _setup_cors(app: FastAPI) -> None:
    """Configure CORS middleware for the application.

    Sets up different CORS policies for production and development:
    - Production: Uses ALLOWED_ORIGINS environment variable
    - Development: Allows all origins for easier testing

    Args:
        app: FastAPI application instance to configure.

    Raises:
        ValueError: If ALLOWED_ORIGINS is not set in production.
    """
    # Configure CORS
    if os.getenv("API_ENV") == "production":
        allowed_origins = os.getenv("ALLOWED_ORIGINS")
        if not allowed_origins:
            raise ValueError(
                "ALLOWED_ORIGINS environment variable is not set in production!\n"
                "Set it to a comma-separated list of allowed origin hosts."
            )
        cors_origins = allowed_origins.split(",")
    else:
        # In development, allow all origins
        cors_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


app = FastAPI(
    lifespan=lifespan,
    title="Paper Explorer",
    description="API for the Paper Explorer tool. Allows exploration of papers and"
    " their relationships to others, including citations and semantic similarity.",
)
_setup_cors(app)


app.include_router(network.router)
app.include_router(mind.router)


@app.get("/health")
async def health() -> HealthCheck:
    """Health check endpoint.

    Returns:
        HealthCheck response with status and timestamp.
    """
    return HealthCheck(status="ok", timestamp=dt.datetime.now(dt.UTC).isoformat())
