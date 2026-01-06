"""Batch processing utilities."""

from dataclasses import dataclass
from typing import Callable, Any, Optional

from rdkit_cli.io.readers import MoleculeReader, MoleculeRecord
from rdkit_cli.io.writers import MoleculeWriter
from rdkit_cli.progress.ninja import NinjaProgress
from rdkit_cli.parallel.executor import ParallelExecutor


@dataclass
class BatchResult:
    """Result of batch processing."""

    total_processed: int
    successful: int
    failed: int
    elapsed_time: float


def process_molecules(
    reader: MoleculeReader,
    writer: MoleculeWriter,
    processor: Callable[[MoleculeRecord], Optional[dict[str, Any]]],
    n_workers: int = -1,
    quiet: bool = False,
    batch_size: int = 1000,
) -> BatchResult:
    """
    Process molecules from reader through processor and write to writer.

    This is the main batch processing function used by most commands.

    Args:
        reader: MoleculeReader to read from
        writer: MoleculeWriter to write to
        processor: Function that takes MoleculeRecord and returns dict or None
        n_workers: Number of worker processes (-1 for all)
        quiet: Suppress progress output
        batch_size: Number of records to process in each batch

    Returns:
        BatchResult with processing statistics
    """
    total = len(reader)
    progress = NinjaProgress(total=total, quiet=quiet)

    successful = 0
    failed = 0
    write_buffer: list[dict[str, Any]] = []
    write_buffer_size = 1000

    progress.start()

    try:
        if n_workers == 1:
            # Sequential processing
            for record in reader:
                result = processor(record)
                if result is not None:
                    write_buffer.append(result)
                    successful += 1
                else:
                    failed += 1

                progress.update()

                if len(write_buffer) >= write_buffer_size:
                    writer.write_batch(write_buffer)
                    write_buffer = []
        else:
            # Parallel processing - collect batch, process in parallel, write
            executor = ParallelExecutor(processor, n_workers=n_workers)
            batch: list[MoleculeRecord] = []

            for record in reader:
                batch.append(record)

                if len(batch) >= batch_size:
                    # Process batch in parallel
                    results = executor.map_ordered(batch)
                    for result in results:
                        if result is not None:
                            write_buffer.append(result)
                            successful += 1
                        else:
                            failed += 1
                        progress.update()

                    if len(write_buffer) >= write_buffer_size:
                        writer.write_batch(write_buffer)
                        write_buffer = []

                    batch = []

            # Process remaining batch
            if batch:
                results = executor.map_ordered(batch)
                for result in results:
                    if result is not None:
                        write_buffer.append(result)
                        successful += 1
                    else:
                        failed += 1
                    progress.update()

        # Write remaining buffer
        if write_buffer:
            writer.write_batch(write_buffer)

    finally:
        progress.finish()

    return BatchResult(
        total_processed=total,
        successful=successful,
        failed=failed,
        elapsed_time=progress.elapsed_time,
    )


def process_molecules_simple(
    reader: MoleculeReader,
    processor: Callable[[MoleculeRecord], Optional[dict[str, Any]]],
    n_workers: int = -1,
    quiet: bool = False,
) -> tuple[list[dict[str, Any]], BatchResult]:
    """
    Process molecules and return results in memory (for small datasets).

    Args:
        reader: MoleculeReader to read from
        processor: Function that takes MoleculeRecord and returns dict or None
        n_workers: Number of worker processes (-1 for all)
        quiet: Suppress progress output

    Returns:
        Tuple of (results list, BatchResult)
    """
    total = len(reader)
    progress = NinjaProgress(total=total, quiet=quiet)

    results: list[dict[str, Any]] = []
    successful = 0
    failed = 0

    progress.start()

    try:
        if n_workers == 1:
            for record in reader:
                result = processor(record)
                if result is not None:
                    results.append(result)
                    successful += 1
                else:
                    failed += 1
                progress.update()
        else:
            executor = ParallelExecutor(processor, n_workers=n_workers)
            records = list(reader)
            progress.set_total(len(records))

            for result in executor.map_ordered(records):
                if result is not None:
                    results.append(result)
                    successful += 1
                else:
                    failed += 1
                progress.update()

    finally:
        progress.finish()

    return results, BatchResult(
        total_processed=total,
        successful=successful,
        failed=failed,
        elapsed_time=progress.elapsed_time,
    )
