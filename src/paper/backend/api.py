"""FastAPI application with PostgreSQL backend.

Main application module that sets up the FastAPI server, database connections,
CORS middleware, and routes for the Paper Explorer API.
"""

import datetime as dt
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from paper.backend.dependencies import ENABLE_NETWORK, lifespan
from paper.backend.model import HealthCheck
from paper.backend.routers import mind
from paper.util import VERSION, setup_logging

logger = logging.getLogger(__name__)


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


setup_logging()
app = FastAPI(
    lifespan=lifespan,
    title="Paper Explorer",
    description="API for the Paper Explorer tool. Allows exploration of papers and"
    " their relationships to others, including citations and semantic similarity.",
    version=VERSION,
)
_setup_cors(app)


if ENABLE_NETWORK:
    from paper.backend.routers import network

    app.include_router(network.router)

app.include_router(mind.router)


@app.get("/health")
async def health() -> HealthCheck:
    """Health check endpoint.

    Returns HealthCheck response with status, timestamp, and version.
    """
    return HealthCheck(
        status="ok",
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
        version=VERSION,
    )
