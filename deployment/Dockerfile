FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app
RUN uv sync --frozen --no-dev --no-cache --extra cpu

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "paper.backend.api:app", \
     "--host", "0.0.0.0", "--port", "8000"]
