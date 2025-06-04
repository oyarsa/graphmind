"""Compress all JSON files in a directory using zstd compression.

This script finds all .json files in a given directory (recursively) and compresses
them using zstd, creating .json.zst files. Optionally, it removes the original files
after successful compression.
"""

import logging
from pathlib import Path
from typing import Annotated

import typer
import zstandard as zstd
from tqdm import tqdm

from paper.util import setup_logging
from paper.util.cli import die

logger = logging.getLogger("paper.scripts.compress_json_files")


def compress_json_file(json_path: Path, keep_original: bool = True) -> bool:
    """Compress a single JSON file using zstd.

    Args:
        json_path: Path to the JSON file to compress
        keep_original: If True, keep the original file after compression

    Returns:
        True if compression was successful, False otherwise
    """
    # Skip if already compressed
    if json_path.suffix == ".zst":
        return True

    # Skip if compressed version already exists
    zst_path = json_path.with_suffix(json_path.suffix + ".zst")
    if zst_path.exists():
        logger.info("Skipping %s - compressed version already exists", json_path)
        return True

    try:
        # Read original file
        original_data = json_path.read_bytes()

        # Compress
        cctx = zstd.ZstdCompressor()
        compressed_data = cctx.compress(original_data)

        # Write compressed file
        zst_path.write_bytes(compressed_data)

        # Verify by decompressing and comparing
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(zst_path.read_bytes())

        if decompressed != original_data:
            # Verification failed, remove compressed file
            zst_path.unlink()
            logger.error("Verification failed for %s", json_path)
            return False

    except Exception:
        logger.exception("Error compressing %s", json_path)
        return False
    else:
        # Remove original if requested
        if not keep_original:
            json_path.unlink()

        return True


def compress_directory(
    directory: Path,
    recursive: bool = True,
    keep_original: bool = True,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Compress all JSON files in a directory.

    Args:
        directory: Directory to process
        recursive: If True, process subdirectories recursively
        keep_original: If True, keep original files after compression
        dry_run: If True, only show what would be done without actually doing it

    Returns:
        Tuple of (successful_count, failed_count)
    """
    # Find all JSON files
    glob_fn = Path.rglob if recursive else Path.glob
    json_files = [f for f in glob_fn(directory, "*.json") if f.is_file()]

    if not json_files:
        logger.info("No JSON files found to compress")
        return 0, 0

    logger.info("Found %d JSON files to compress", len(json_files))

    if dry_run:
        logger.info("Dry run - files that would be compressed:")
        for f in json_files:
            logger.info("%s", f)
        return len(json_files), 0

    successful = 0
    failed = 0
    total_original_size = 0
    total_compressed_size = 0

    # Process each file
    for json_path in tqdm(json_files, desc="Compressing files"):
        total_original_size += json_path.stat().st_size

        if compress_json_file(json_path, keep_original):
            successful += 1
            # Get compressed size
            zst_path = json_path.with_suffix(json_path.suffix + ".zst")
            if zst_path.exists():
                total_compressed_size += zst_path.stat().st_size
        else:
            failed += 1

    # Print summary
    logger.info("Successfully compressed: %d", successful)
    if failed > 0:
        logger.error("Failed: %d", failed)

    if successful > 0 and total_original_size > 0:
        overall_ratio = (1 - total_compressed_size / total_original_size) * 100
        logger.info("Compression statistics:")
        logger.info("  Original size: %.2f MB", total_original_size / 1024 / 1024)
        logger.info("  Compressed size: %.2f MB", total_compressed_size / 1024 / 1024)
        logger.info("  Space saved: %.1f%%", overall_ratio)

    return successful, failed


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def compress(
    directory: Annotated[
        Path,
        typer.Argument(help="Directory containing JSON files to compress"),
    ],
    keep_original: Annotated[
        bool,
        typer.Option(help="Keep original files after compression"),
    ] = True,
    no_recursive: Annotated[
        bool,
        typer.Option("--no-recursive", help="Don't process subdirectories recursively"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", "-n", help="Show what would be done without actually doing it"
        ),
    ] = False,
) -> None:
    """Compress all JSON files in a directory using zstd compression."""
    setup_logging()

    if not directory.exists():
        die(f"Directory '{directory}' does not exist")

    if not directory.is_dir():
        die(f"'{directory}' is not a directory")

    successes, failed = compress_directory(
        directory,
        recursive=not no_recursive,
        keep_original=keep_original,
        dry_run=dry_run,
    )

    if failed > 0:
        die(f"Failed to compress {failed} files")

    print(f"{successes} files successfully compressed")


if __name__ == "__main__":
    app()
