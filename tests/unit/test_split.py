"""Unit tests for split module."""

import pytest
from pathlib import Path


class TestFileSplitter:
    """Test FileSplitter class."""

    def test_init_requires_one_param(self):
        """Test that either n_chunks or chunk_size must be specified."""
        from rdkit_cli.core.split import FileSplitter

        with pytest.raises(ValueError, match="Either n_chunks or chunk_size"):
            FileSplitter()

        with pytest.raises(ValueError, match="Only one of"):
            FileSplitter(n_chunks=2, chunk_size=10)

    def test_calculate_assignments_by_chunks(self):
        """Test chunk assignment calculation by number of chunks."""
        from rdkit_cli.core.split import FileSplitter

        splitter = FileSplitter(n_chunks=3)
        assignments = splitter.calculate_chunk_assignments(10)

        assert len(assignments) == 3
        # 10 / 3 = 3 with remainder 1, so chunks are 4, 3, 3
        assert assignments[0] == (0, 4)
        assert assignments[1] == (4, 7)
        assert assignments[2] == (7, 10)

    def test_calculate_assignments_by_size(self):
        """Test chunk assignment calculation by chunk size."""
        from rdkit_cli.core.split import FileSplitter

        splitter = FileSplitter(chunk_size=3)
        assignments = splitter.calculate_chunk_assignments(10)

        assert len(assignments) == 4
        assert assignments[0] == (0, 3)
        assert assignments[1] == (3, 6)
        assert assignments[2] == (6, 9)
        assert assignments[3] == (9, 10)

    def test_calculate_assignments_exact_division(self):
        """Test when records divide evenly."""
        from rdkit_cli.core.split import FileSplitter

        splitter = FileSplitter(n_chunks=4)
        assignments = splitter.calculate_chunk_assignments(12)

        assert len(assignments) == 4
        for i, (start, end) in enumerate(assignments):
            assert end - start == 3

    def test_generate_output_path(self):
        """Test output path generation."""
        from rdkit_cli.core.split import FileSplitter

        path = FileSplitter.generate_output_path(
            output_dir=Path("/tmp"),
            base_name="molecules",
            chunk_idx=0,
            extension="csv",
            total_chunks=5,
        )
        assert path == Path("/tmp/molecules_1.csv")

        path = FileSplitter.generate_output_path(
            output_dir=Path("/tmp"),
            base_name="data",
            chunk_idx=9,
            extension=".smi",
            total_chunks=100,
        )
        assert path == Path("/tmp/data_010.smi")

    def test_split_records(self):
        """Test splitting records."""
        from rdkit_cli.core.split import FileSplitter
        from rdkit_cli.io.readers import MoleculeRecord

        records = [MoleculeRecord(None, smiles=f"mol{i}") for i in range(10)]
        splitter = FileSplitter(n_chunks=3)

        chunks = list(splitter.split_records(records))

        assert len(chunks) == 3
        total = sum(len(chunk_records) for _, chunk_records in chunks)
        assert total == 10

    def test_more_chunks_than_records(self):
        """Test when requesting more chunks than records."""
        from rdkit_cli.core.split import FileSplitter

        splitter = FileSplitter(n_chunks=10)
        assignments = splitter.calculate_chunk_assignments(3)

        # Should only create 3 chunks (one per record)
        assert len(assignments) == 3
