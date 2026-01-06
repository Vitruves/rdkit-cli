"""Parallel processing executor."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterator, TypeVar, Optional, Any
from dataclasses import dataclass

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""

    n_workers: int = -1  # -1 means auto-detect
    chunk_size: int = 100

    def __post_init__(self):
        if self.n_workers == -1:
            self.n_workers = os.cpu_count() or 1


def get_worker_count(n_requested: int) -> int:
    """
    Get actual worker count based on request and system.

    Args:
        n_requested: Requested number of workers (-1 for all, 0 for 1)

    Returns:
        Actual number of workers to use
    """
    max_workers = os.cpu_count() or 1
    if n_requested <= 0:
        return max_workers
    return min(n_requested, max_workers)


# Global worker function storage for pickling
_worker_func: Optional[Callable] = None
_worker_args: tuple = ()


def _init_worker(func: Callable, args: tuple):
    """Initialize worker process with function and extra args."""
    global _worker_func, _worker_args
    _worker_func = func
    _worker_args = args


def _worker_wrapper(item: Any) -> Any:
    """Wrapper that calls the stored worker function."""
    global _worker_func, _worker_args
    if _worker_func is None:
        raise RuntimeError("Worker function not initialized")
    return _worker_func(item, *_worker_args)


class ParallelExecutor:
    """
    Generic parallel executor for batch processing.

    Uses ProcessPoolExecutor since RDKit operations are CPU-bound
    and benefit from true parallelism (bypassing GIL).
    """

    def __init__(
        self,
        func: Callable[[T], R],
        n_workers: int = -1,
        initializer: Optional[Callable] = None,
        initargs: tuple = (),
    ):
        """
        Initialize parallel executor.

        Args:
            func: Function to apply to each item
            n_workers: Number of worker processes (-1 for all CPUs)
            initializer: Optional initializer for worker processes
            initargs: Arguments for initializer
        """
        self.func = func
        self.n_workers = get_worker_count(n_workers)
        self.initializer = initializer
        self.initargs = initargs

    def map_unordered(
        self,
        items: list[T],
        chunk_size: int = 100,
    ) -> Iterator[R]:
        """
        Process items in parallel, yielding results as they complete.

        Results may be returned in any order.

        Args:
            items: Items to process
            chunk_size: Number of items per chunk

        Yields:
            Results as they complete
        """
        if not items:
            return

        # For single item or single worker, just run sequentially
        if len(items) == 1 or self.n_workers == 1:
            for item in items:
                yield self.func(item)
            return

        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=self.initializer,
            initargs=self.initargs,
        ) as executor:
            # Submit all tasks
            futures = {executor.submit(self.func, item): i for i, item in enumerate(items)}

            # Yield results as they complete
            for future in as_completed(futures):
                try:
                    yield future.result()
                except Exception as e:
                    # Yield None for failed items, let caller handle
                    yield None

    def map_ordered(
        self,
        items: list[T],
        chunk_size: int = 100,
    ) -> list[R]:
        """
        Process items and return results in original order.

        Args:
            items: Items to process
            chunk_size: Number of items per chunk (unused, for API compatibility)

        Returns:
            Results in same order as input
        """
        if not items:
            return []

        # For single item or single worker, just run sequentially
        if len(items) == 1 or self.n_workers == 1:
            return [self.func(item) for item in items]

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            return list(executor.map(self.func, items, chunksize=max(1, len(items) // (self.n_workers * 4))))


def parallel_map(
    func: Callable[[T], R],
    items: list[T],
    n_workers: int = -1,
    ordered: bool = True,
) -> list[R]:
    """
    Simple parallel map with default settings.

    Args:
        func: Function to apply to each item
        items: Items to process
        n_workers: Number of workers (-1 for all CPUs)
        ordered: If True, preserve order; if False, return as completed

    Returns:
        List of results
    """
    executor = ParallelExecutor(func, n_workers=n_workers)

    if ordered:
        return executor.map_ordered(items)
    else:
        return list(executor.map_unordered(items))
