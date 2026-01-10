"""Unit tests for merge module."""

import pytest
from pathlib import Path


class TestMoleculeMerger:
    """Test MoleculeMerger class."""

    def test_merge_single_file(self, tmp_dir, sample_csv):
        """Test merging a single file."""
        from rdkit_cli.core.merge import MoleculeMerger

        merger = MoleculeMerger(deduplicate=False)
        results = list(merger.merge_files([sample_csv]))

        assert len(results) > 0
        assert all("smiles" in r for r in results)

    def test_merge_multiple_files(self, tmp_dir):
        """Test merging multiple files."""
        from rdkit_cli.core.merge import MoleculeMerger

        # Create two CSV files
        file1 = tmp_dir / "file1.csv"
        file2 = tmp_dir / "file2.csv"

        file1.write_text("smiles,name\nCCO,ethanol\nCCC,propane\n")
        file2.write_text("smiles,name\nc1ccccc1,benzene\nCC(=O)C,acetone\n")

        merger = MoleculeMerger(deduplicate=False)
        results = list(merger.merge_files([file1, file2]))

        assert len(results) == 4

    def test_merge_with_deduplication(self, tmp_dir):
        """Test merging with deduplication."""
        from rdkit_cli.core.merge import MoleculeMerger

        # Create files with duplicates
        file1 = tmp_dir / "file1.csv"
        file2 = tmp_dir / "file2.csv"

        file1.write_text("smiles,name\nCCO,ethanol1\nCCC,propane\n")
        file2.write_text("smiles,name\nCCO,ethanol2\nc1ccccc1,benzene\n")

        merger = MoleculeMerger(deduplicate=True, dedupe_key="smiles")
        results = list(merger.merge_files([file1, file2]))

        # CCO should appear only once
        smiles_list = [r["smiles"] for r in results]
        assert smiles_list.count("CCO") == 1
        assert len(results) == 3

    def test_merge_add_source(self, tmp_dir):
        """Test adding source file column."""
        from rdkit_cli.core.merge import MoleculeMerger

        file1 = tmp_dir / "molecules.csv"
        file1.write_text("smiles,name\nCCO,ethanol\n")

        merger = MoleculeMerger(add_source=True)
        results = list(merger.merge_files([file1]))

        assert len(results) == 1
        assert results[0]["source_file"] == "molecules.csv"

    def test_merge_stats(self, tmp_dir):
        """Test merge statistics."""
        from rdkit_cli.core.merge import MoleculeMerger

        file1 = tmp_dir / "file1.csv"
        file1.write_text("smiles,name\nCCO,ethanol\nCCO,ethanol2\nCCC,propane\n")

        merger = MoleculeMerger(deduplicate=True)
        list(merger.merge_files([file1]))

        stats = merger.get_stats()
        assert stats["unique_molecules"] == 2
