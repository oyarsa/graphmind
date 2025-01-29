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


def run_many(cmds: Iterable[Sequence[object]]) -> None:
    """Run many commands in sequence and stop on first error."""
    for cmd in cmds:
        run(*cmd)


def run_parallel_commands(
    commands: Iterable[Sequence[object]], seq: bool = False
) -> None:
    """Run multiple commands at the same time. If any returns an error, quit.

    Args:
        commands: Commands to run.
        seq: If True, the commands are run sequentially. Otherwise, runs them in multiple
            threads.
    """
    if seq:
        run_many(commands)
        return

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run, *cmd) for cmd in commands]
        for future in futures:
            future.result()
