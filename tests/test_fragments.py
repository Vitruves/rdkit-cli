# tests/test_fragments.py
import pytest
from pathlib import Path
from rdkit_cli.commands import fragments


class TestFragments:
    """Test molecule fragmentation."""

    def test_fragment_molecules(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test basic molecule fragmentation."""
        output_file = temp_output_dir / "fragments.sdf"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            method="brics",
            min_fragment_size=3,
            max_fragment_size=50,
            include_parent=True
        )
        
        result = fragments.fragment_molecules(args, graceful_exit_mock)
        assert result == 0
        assert output_file.exists()

    def test_fragment_molecules_recap(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test RECAP fragmentation."""
        output_file = temp_output_dir / "recap_fragments.sdf"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            method="recap",
            min_fragment_size=3,
            max_fragment_size=50,
            include_parent=False
        )
        
        result = fragments.fragment_molecules(args, graceful_exit_mock)
        assert result == 0
        assert output_file.exists()


class TestFragmentSimilarity:
    """Test fragment-based similarity."""

    def test_fragment_similarity(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test fragment-based similarity calculation."""
        output_file = temp_output_dir / "fragment_similarity.csv"
        
        # Create a reference fragments file
        ref_frags_file = temp_output_dir / "ref_frags.smi"
        ref_frags_file.write_text("CCO\tethanol\nCC\tmethyl\n")
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            reference_frags=str(ref_frags_file),
            method="tanimoto"
        )
        
        result = fragments.fragment_similarity(args, graceful_exit_mock)
        assert result == 0
        assert output_file.exists()


class TestLeadOptimization:
    """Test lead optimization through fragment replacement."""

    def test_lead_optimization(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test lead optimization."""
        output_file = temp_output_dir / "optimized.sdf"
        
        # Create a fragment library file
        fragment_lib = temp_output_dir / "fragments.smi"
        fragment_lib.write_text("CC\tethyl\nCCC\tpropyl\nCCCC\tbutyl\n")
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            fragment_library=str(fragment_lib),
            max_products=10,
            similarity_threshold=0.7
        )
        
        result = fragments.lead_optimization(args, graceful_exit_mock)
        assert result == 0
        assert output_file.exists()