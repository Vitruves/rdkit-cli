# tests/test_substructure.py
import pytest
import tempfile
from pathlib import Path
from rdkit_cli.commands import substructure


class TestSubstructureSearch:
    """Test substructure searching."""

    def test_substructure_search(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test basic substructure search."""
        output_file = temp_output_dir / "hits.csv"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            query="C",  # simple carbon pattern
            count_matches=False,
            unique_matches=False
        )
        
        result = substructure.search(args, graceful_exit_mock)
        assert result == 0
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_substructure_search_with_count(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test substructure search with match counting."""
        output_file = temp_output_dir / "hits_count.csv"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            query="C",
            count_matches=True,
            unique_matches=True
        )
        
        result = substructure.search(args, graceful_exit_mock)
        assert result == 0
        assert output_file.exists()


class TestSMARTSFilter:
    """Test SMARTS pattern filtering."""

    def test_smarts_filter(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test filtering with SMARTS patterns."""
        output_file = temp_output_dir / "filtered.sdf"
        
        # Create SMARTS pattern file
        pattern_file = temp_output_dir / "patterns.txt"
        pattern_file.write_text("C\nO\n")  # carbon and oxygen patterns
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            smarts_file=str(pattern_file),
            mode='include'
        )
        
        result = substructure.filter_smarts(args, graceful_exit_mock)
        assert result == 0
        assert output_file.exists()


class TestScaffoldAnalysis:
    """Test scaffold analysis."""

    def test_scaffold_analysis(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test scaffold frequency analysis."""
        output_file = temp_output_dir / "scaffold_analysis.csv"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            include_counts=True,
            min_frequency=1
        )
        
        result = substructure.scaffold_analysis(args, graceful_exit_mock)
        assert result == 0
        assert output_file.exists()


class TestMurckoScaffolds:
    """Test Murcko scaffold extraction."""

    def test_murcko_scaffolds(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test Murcko scaffold extraction."""
        output_file = temp_output_dir / "scaffolds.sdf"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            generic=True,
            unique_only=False
        )
        
        result = substructure.murcko_scaffolds(args, graceful_exit_mock)
        assert result == 0
        assert output_file.exists()


class TestFunctionalGroups:
    """Test functional group identification."""

    def test_functional_groups(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test functional group identification."""
        output_file = temp_output_dir / "functional_groups.csv"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            hierarchy="ifg"
        )
        
        result = substructure.functional_groups(args, graceful_exit_mock)
        assert result == 0
        assert output_file.exists()