"""Ninja-style progress monitoring."""

import sys
import time
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProgressStats:
    """Statistics for progress display."""

    completed: int
    total: int
    elapsed: float
    rate: float
    eta: Optional[float]
    percentage: float


class NinjaProgress:
    """
    Ninja-style progress reporter.

    Format: [42/100] 42% | 15.3 it/s | ETA: 3.8s | Elapsed: 2.8s

    Features:
    - No progress bar (just stats)
    - Updates in-place on single line
    - Thread-safe updates
    """

    def __init__(
        self,
        total: int,
        quiet: bool = False,
        update_interval: float = 0.1,
        file=None,
    ):
        """
        Initialize progress reporter.

        Args:
            total: Total number of items to process
            quiet: If True, suppress all output
            update_interval: Minimum seconds between display updates
            file: File to write progress to (default: stderr)
        """
        self.total = total
        self.quiet = quiet
        self.update_interval = update_interval
        self._file = file or sys.stderr

        self._completed = 0
        self._start_time: Optional[float] = None
        self._last_update_time: float = 0
        self._lock = threading.Lock()
        self._finished = False
        self._last_line_length = 0

    def start(self):
        """Start the progress tracker."""
        self._start_time = time.perf_counter()
        self._display()

    def update(self, n: int = 1):
        """
        Update progress by n items.

        Args:
            n: Number of items completed
        """
        with self._lock:
            self._completed += n

            # Throttle display updates
            now = time.perf_counter()
            if now - self._last_update_time >= self.update_interval:
                self._display()
                self._last_update_time = now

    def set_total(self, total: int):
        """Update the total count (useful when count is discovered during processing)."""
        with self._lock:
            self.total = total

    def finish(self):
        """Complete the progress display."""
        with self._lock:
            self._finished = True
            self._display(final=True)
            if not self.quiet:
                self._file.write("\n")
                self._file.flush()

    @property
    def elapsed_time(self) -> float:
        """Return elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    @property
    def completed(self) -> int:
        """Return number of completed items."""
        return self._completed

    def _calculate_stats(self) -> ProgressStats:
        """Calculate current progress statistics."""
        elapsed = self.elapsed_time
        completed = self._completed

        # Calculate rate (items per second)
        rate = completed / elapsed if elapsed > 0 else 0.0

        # Calculate percentage
        percentage = (completed / self.total * 100) if self.total > 0 else 0.0

        # Calculate ETA
        remaining = self.total - completed
        eta = remaining / rate if rate > 0 and remaining > 0 else None

        return ProgressStats(
            completed=completed,
            total=self.total,
            elapsed=elapsed,
            rate=rate,
            eta=eta,
            percentage=percentage,
        )

    def _display(self, final: bool = False):
        """Display the progress line."""
        if self.quiet:
            return

        stats = self._calculate_stats()

        # Format: [42/100] 42% | 15.3 it/s | ETA: 3.8s | Elapsed: 2.8s
        parts = [
            f"[{stats.completed}/{stats.total}]",
            f"{stats.percentage:.0f}%",
            f"{stats.rate:.1f} it/s",
        ]

        if stats.eta is not None and not final:
            parts.append(f"ETA: {self._format_time(stats.eta)}")

        parts.append(f"Elapsed: {self._format_time(stats.elapsed)}")

        line = " | ".join(parts)

        # Clear previous line and write new one
        clear = " " * self._last_line_length
        self._file.write(f"\r{clear}\r{line}")
        self._file.flush()
        self._last_line_length = len(line)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


class progress_context:
    """Context manager for progress tracking."""

    def __init__(self, total: int, quiet: bool = False, description: str = ""):
        """
        Initialize progress context.

        Args:
            total: Total number of items
            quiet: Suppress output
            description: Optional description (currently unused, for future)
        """
        self.progress = NinjaProgress(total=total, quiet=quiet)
        self._description = description

    def __enter__(self) -> NinjaProgress:
        self.progress.start()
        return self.progress

    def __exit__(self, *args):
        self.progress.finish()
