# tests/test_io_ops.py
"""Tests for I/O operations module."""

import pytest
from pathlib import Path
from rdkit import Chem

from rdkit_cli.commands import io_ops
from rdkit_cli.core.common import read_molecules


class TestConvert:
    """Test molecular file conversion."""
    
    def test_sdf_to_smiles(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test SDF to SMILES conversion."""
        output_file = temp_output_dir / "output.smi"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            format="smiles"
        )
        
        result = io_ops.convert(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 0
            assert '\t' in lines[0]  # SMILES\tID format
    
    def test_smiles_to_sdf(self, sample_smiles_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test SMILES to SDF conversion."""
        output_file = temp_output_dir / "output.sdf"
        
        args = mock_args(
            input_file=str(sample_smiles_file),
            output=str(output_file),
            format="sdf"
        )
        
        result = io_ops.convert(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        molecules = read_molecules(output_file)
        assert len(molecules) > 0


class TestValidate:
    """Test molecular validation."""
    
    def test_validate_molecules(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test molecular structure validation."""
        output_file = temp_output_dir / "valid.sdf"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            log_errors=None,
            strict=False
        )
        
        result = io_ops.validate(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        molecules = read_molecules(output_file)
        assert len(molecules) > 0
        
        # All molecules should be valid
        for mol in molecules:
            assert mol is not None
            assert mol.GetNumAtoms() > 0


class TestSplit:
    """Test dataset splitting."""
    
    def test_split_dataset(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test splitting molecular dataset."""
        args = mock_args(
            input_file=str(sample_sdf_file),
            output_dir=str(temp_output_dir),
            chunk_size=3,
            prefix="test_chunk"
        )
        
        result = io_ops.split(args, graceful_exit_mock)
        
        assert result == 0
        
        # Check that chunk files were created
        chunk_files = list(temp_output_dir.glob("test_chunk_*.sdf"))
        assert len(chunk_files) > 0
        
        # Verify molecules in chunks
        total_molecules = 0
        for chunk_file in chunk_files:
            molecules = read_molecules(chunk_file)
            total_molecules += len(molecules)
            assert len(molecules) <= 3  # Chunk size
        
        # Total should match original
        original_molecules = read_molecules(sample_sdf_file)
        assert total_molecules == len(original_molecules)


class TestMerge:
    """Test file merging."""
    
    def test_merge_files(self, sample_molecules, temp_output_dir, graceful_exit_mock, mock_args):
        """Test merging multiple molecular files."""
        # Create two input files
        file1 = temp_output_dir / "file1.sdf"
        file2 = temp_output_dir / "file2.sdf"
        
        from rdkit_cli.core.common import write_molecules
        write_molecules(sample_molecules[:5], file1)
        write_molecules(sample_molecules[5:], file2)
        
        output_file = temp_output_dir / "merged.sdf"
        
        args = mock_args(
            input_files=[str(file1), str(file2)],
            output=str(output_file),
            validate=False
        )
        
        result = io_ops.merge(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        merged_molecules = read_molecules(output_file)
        assert len(merged_molecules) == len(sample_molecules)


class TestDeduplicate:
    """Test deduplication."""
    
    def test_deduplicate_inchi(self, temp_output_dir, graceful_exit_mock, mock_args):
        """Test deduplication using InChI keys."""
        # Create file with duplicates
        duplicate_smiles = ["CCO", "CCO", "CC(C)O", "CCO", "c1ccccc1"]
        molecules = []
        
        for i, smiles in enumerate(duplicate_smiles):
            mol = Chem.MolFromSmiles(smiles)
            mol.SetProp("_Name", f"mol_{i+1}")
            molecules.append(mol)
        
        input_file = temp_output_dir / "duplicates.sdf"
        output_file = temp_output_dir / "unique.sdf"
        
        from rdkit_cli.core.common import write_molecules
        write_molecules(molecules, input_file)
        
        args = mock_args(
            input_file=str(input_file),
            output=str(output_file),
            method="inchi-key",
            keep_first=True
        )
        
        result = io_ops.deduplicate(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        unique_molecules = read_molecules(output_file)
        assert len(unique_molecules) == 3  # CCO, isopropanol, benzene