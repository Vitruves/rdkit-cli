"""File splitting engine for molecular datasets."""

from pathlib import Path
from typing import Iterator, Optional

from rdkit_cli.io.readers import MoleculeRecord


class FileSplitter:
    """Split molecular datasets into smaller files."""

    def __init__(
        self,
        n_chunks: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ):
        """
        Initialize file splitter.

        Args:
            n_chunks: Number of output files to create
            chunk_size: Number of molecules per output file

        Note: Exactly one of n_chunks or chunk_size must be specified.
        """
        if n_chunks is None and chunk_size is None:
            raise ValueError("Either n_chunks or chunk_size must be specified")
        if n_chunks is not None and chunk_size is not None:
            raise ValueError("Only one of n_chunks or chunk_size can be specified")

        self.n_chunks = n_chunks
        self.chunk_size = chunk_size

    def calculate_chunk_assignments(
        self,
        total_records: int,
    ) -> list[tuple[int, int]]:
        """
        Calculate start/end indices for each chunk.

        Args:
            total_records: Total number of records

        Returns:
            List of (start_idx, end_idx) tuples for each chunk
        """
        if self.n_chunks is not None:
            # Split into n equal-ish chunks
            chunk_size = total_records // self.n_chunks
            remainder = total_records % self.n_chunks

            assignments = []
            start = 0
            for i in range(self.n_chunks):
                # Distribute remainder across first chunks
                size = chunk_size + (1 if i < remainder else 0)
                if size > 0:
                    assignments.append((start, start + size))
                    start += size
            return assignments
        else:
            # Fixed chunk size
            assignments = []
            start = 0
            while start < total_records:
                end = min(start + self.chunk_size, total_records)
                assignments.append((start, end))
                start = end
            return assignments

    def split_records(
        self,
        records: list[MoleculeRecord],
    ) -> Iterator[tuple[int, list[MoleculeRecord]]]:
        """
        Split records into chunks.

        Args:
            records: List of molecule records

        Yields:
            Tuples of (chunk_index, chunk_records)
        """
        assignments = self.calculate_chunk_assignments(len(records))

        for chunk_idx, (start, end) in enumerate(assignments):
            yield chunk_idx, records[start:end]

    @staticmethod
    def generate_output_path(
        output_dir: Path,
        base_name: str,
        chunk_idx: int,
        extension: str,
        total_chunks: int,
    ) -> Path:
        """
        Generate output file path for a chunk.

        Args:
            output_dir: Output directory
            base_name: Base name for output files
            chunk_idx: Chunk index (0-based)
            extension: File extension (with or without dot)
            total_chunks: Total number of chunks (for padding)

        Returns:
            Path for the output file
        """
        # Determine padding width
        width = len(str(total_chunks))

        # Ensure extension has dot
        if not extension.startswith("."):
            extension = f".{extension}"

        filename = f"{base_name}_{chunk_idx + 1:0{width}d}{extension}"
        return output_dir / filename
