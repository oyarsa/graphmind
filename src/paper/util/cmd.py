"""Functions to run external commands and scripts."""

import concurrent.futures
import subprocess
from collections.abc import Iterable, Sequence


def title(message: str) -> None:
    """Print highlighted message as a title."""
    red_bg = "\033[41m"
    normal_color = "\033[0m"
    print(f"\n{red_bg}>>> {message}{normal_color}")


def run(*args: object) -> None:
    """Run command with uv. Quit on error."""
    subprocess.run(["uv", "run", *(str(x) for x in args)], check=True)


def run_parallel_commands(commands: Iterable[Sequence[object]]) -> None:
    """Run multiple commands at the same time. If any returns an error, quit."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run, *cmd) for cmd in commands]
        for future in futures:
            future.result()
