# tests/test_conformers.py
import pytest
import tempfile
from pathlib import Path
from rdkit_cli.commands import conformers


class TestConformers:
    """Test conformer generation."""

    def test_generate_conformers(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test basic conformer generation."""
        output_file = temp_output_dir / "conformers.sdf"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            num_confs=5,
            method='etkdg',
            optimize=True,
            ff='uff',
            max_iters=200,
            energy_window=10.0
        )
        
        result = conformers.generate(args, graceful_exit_mock)
        assert result == 0
        assert output_file.exists()
        assert output_file.stat().st_size > 0


class TestAlignment:
    """Test molecule alignment."""

    def test_align_molecules(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test molecule alignment to template."""
        output_file = temp_output_dir / "aligned.sdf"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            template="CCO",  # ethanol as template
            align_mode="mcs"
        )
        
        result = conformers.align(args, graceful_exit_mock)
        assert result == 0
        assert output_file.exists()


class TestShapeSimilarity:
    """Test 3D shape similarity."""

    def test_shape_similarity(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test 3D shape similarity calculation."""
        output_file = temp_output_dir / "shape_similarity.csv"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            reference="CCO",  # ethanol as reference
            threshold=0.3
        )
        
        result = conformers.shape_similarity(args, graceful_exit_mock)
        assert result == 0
        assert output_file.exists()
        assert output_file.stat().st_size > 0


class TestPharmacophoreScreen:
    """Test pharmacophore screening."""

    def test_pharmacophore_screen(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test pharmacophore screening."""
        output_file = temp_output_dir / "pharmacophore_hits.csv"
        
        # Create a simple pharmacophore pattern file
        pattern_file = temp_output_dir / "pharmacophore.json"
        pattern_file.write_text('{"features": [{"type": "Donor", "radius": 1.0}]}')
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            pharmacophore=str(pattern_file),
            tolerance=1.0
        )
        
        result = conformers.pharmacophore_screen(args, graceful_exit_mock)
        assert result == 0
        assert output_file.exists()