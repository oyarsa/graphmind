"""Client for processing Server-Sent Events from endpoint and extracting completion data."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

import aiohttp
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def main(
    url: Annotated[str, typer.Argument(help="The URL to connect to for SSE stream.")],
    output: Annotated[
        Path,
        typer.Argument(help="Output file path for the completion JSON data."),
    ],
) -> None:
    """Connect to an SSE endpoint and extract the completion JSON data.

    Processes Server-Sent Events from the specified URL, displays progress messages, and
    saves the final JSON data from the 'complete' event.
    """
    if not url.startswith("http"):
        url = f"http://{url}"

    if completion_data := asyncio.run(fetch_sse_completion(url)):
        save_completion_data(completion_data, output)
    else:
        console.print("[red]No completion data received.[/red]")
        sys.exit(1)


async def fetch_sse_completion(url: str) -> dict[str, Any] | None:
    """Fetch SSE stream and extract completion data.

    Connects to the SSE endpoint and processes events until a 'complete' event is
    received, then returns its data payload.

    Args:
        url: The SSE endpoint URL to connect to.

    Returns:
        The JSON data from the 'complete' event, or None if not received.

    Raises:
        aiohttp.ClientError: If the HTTP request fails.
        json.JSONDecodeError: If the completion data is not valid JSON.
    """
    console.print(f"[cyan]Connecting to: {url}[/cyan]")

    async with aiohttp.ClientSession() as session:
        headers = {"Accept": "text/event-stream"}

        async with session.get(url, headers=headers, raise_for_status=True) as response:
            with Progress(
                SpinnerColumn(),
                TimeElapsedColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Waiting for events...", total=None)

                async for event in parse_sse_stream(response):
                    if data := handle_event(event, progress, task):
                        return data

    return None


def handle_event(
    event: dict[str, str], progress: Progress, task: TaskID
) -> dict[str, Any] | None:
    """Handle a single SSE event and update progress.

    Args:
        event: The SSE event dictionary with 'event' and 'data' keys.
        progress: The Rich Progress instance to update.
        task: The TaskID of the current progress task.

    Returns:
        The JSON data from a 'complete' event, or None for other events.
    """
    match event["event"]:
        # Skip keep-alive messages
        case "keep-alive":
            pass

        case "connected":
            data = parse_event_data(event["data"])
            message = data.get("message", "Connected")
            progress.update(task, description=f"[green]{message}[/green]")

        case "progress":
            data = parse_event_data(event["data"])
            message = data.get("message", "Processing...")
            progress.update(task, description=f"[yellow]{message}[/yellow]")

        case "complete":
            progress.update(task, description="[green]Complete![/green]")
            return parse_event_data(event["data"])

        case "error":
            data = parse_event_data(event["data"])
            error_msg = data.get("message", "Unknown error")
            progress.update(task, description=f"[red]Error: {error_msg}[/red]")
            raise RuntimeError(f"Server error: {error_msg}")

        case _:
            pass

    return None


async def parse_sse_stream(
    response: aiohttp.ClientResponse,
) -> AsyncIterator[dict[str, str]]:
    """Parse Server-Sent Events from an HTTP response stream.

    Implements the SSE protocol parsing, handling event types and data fields.
    This version reads the stream in chunks to avoid "Chunk too big" errors.

    Args:
        response: The aiohttp response object to read from.

    Yields:
        Parsed SSE events as dictionaries with 'event' and 'data' keys.
    """
    event_type = None
    event_data: list[str] = []
    buffer = ""

    # Read the stream in chunks instead of line-by-line
    async for chunk in response.content.iter_any():
        buffer += chunk.decode()

        # Process any complete lines within the buffer
        while "\n" in buffer:
            line_, _, buffer = buffer.partition("\n")
            line = line_.rstrip("\r")

            # Empty line signals end of event
            if not line:
                if event_data:
                    yield {
                        "event": event_type or "message",
                        "data": "\n".join(event_data),
                    }
                    event_type = None
                    event_data = []
                continue

            # Keep-alive comment
            if line.startswith(":"):
                if line == ": keep-alive":
                    yield {"event": "keep-alive", "data": ""}
                continue

            # Parse field and value
            if ":" in line:
                field, _, value_ = line.partition(":")
                value = value_.lstrip()

                if field == "event":
                    event_type = value
                elif field == "data":
                    event_data.append(value)
            # Line without colon is treated as field name with empty value
            elif line == "data":
                event_data.append("")

    # Handle any remaining event data after the stream has closed
    if event_data:
        yield {"event": event_type or "message", "data": "\n".join(event_data)}


def parse_event_data(data: str) -> dict[str, Any]:
    """Parse JSON data from an SSE event.

    Args:
        data: The raw data string from the SSE event.

    Returns:
        The parsed JSON object as a dictionary.

    Raises:
        json.JSONDecodeError: If the data is not valid JSON.
    """
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse event data: {e}[/red]")
        raise


def save_completion_data(data: dict[str, Any], output_path: Path) -> None:
    """Save completion data to a file or display it.

    Args:
        data: The completion data to save.
        output_path: Optional path to save the data to.
    """
    data_json = json.dumps(data)

    output_path.write_text(data_json)
    console.print(f"[green]✓[/green] Saved completion data to: {output_path}")
    console.print(f"[dim]File size: {len(data_json):,} bytes[/dim]")


if __name__ == "__main__":
    app()
