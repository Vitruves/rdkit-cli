"""Unit tests for stats module."""

import pytest
from rdkit import Chem


class TestDatasetStatistics:
    """Test DatasetStatistics class."""

    def test_calculate_basic_stats(self):
        """Test basic statistics calculation."""
        from rdkit_cli.core.stats import DatasetStatistics

        mols = [
            Chem.MolFromSmiles("C"),  # methane
            Chem.MolFromSmiles("CC"),  # ethane
            Chem.MolFromSmiles("CCC"),  # propane
            Chem.MolFromSmiles("CCCC"),  # butane
        ]

        stats = DatasetStatistics()
        result = stats.calculate(mols)

        assert result["total_molecules"] == 4
        assert result["valid_molecules"] == 4
        assert result["invalid_molecules"] == 0
        assert result["validity_rate"] == 1.0

    def test_calculate_with_invalid_molecules(self):
        """Test statistics with invalid molecules."""
        from rdkit_cli.core.stats import DatasetStatistics

        mols = [
            Chem.MolFromSmiles("C"),
            None,  # Invalid
            Chem.MolFromSmiles("CC"),
            None,  # Invalid
        ]

        stats = DatasetStatistics()
        result = stats.calculate(mols)

        assert result["total_molecules"] == 4
        assert result["valid_molecules"] == 2
        assert result["invalid_molecules"] == 2
        assert result["validity_rate"] == 0.5

    def test_calculate_specific_properties(self):
        """Test calculating specific properties."""
        from rdkit_cli.core.stats import DatasetStatistics

        mols = [
            Chem.MolFromSmiles("c1ccccc1"),  # benzene
            Chem.MolFromSmiles("CCO"),  # ethanol
        ]

        stats = DatasetStatistics(properties=["MolWt", "LogP"])
        result = stats.calculate(mols)

        assert "MolWt_mean" in result
        assert "LogP_mean" in result
        assert "TPSA_mean" not in result  # Not requested

    def test_empty_input(self):
        """Test with empty molecule list."""
        from rdkit_cli.core.stats import DatasetStatistics

        stats = DatasetStatistics()
        result = stats.calculate([])

        assert result["total_molecules"] == 0
        assert result["validity_rate"] == 0.0

    def test_all_invalid(self):
        """Test with all invalid molecules."""
        from rdkit_cli.core.stats import DatasetStatistics

        stats = DatasetStatistics()
        result = stats.calculate([None, None, None])

        assert result["total_molecules"] == 3
        assert result["valid_molecules"] == 0
        assert result["validity_rate"] == 0.0

    def test_available_properties(self):
        """Test available properties list."""
        from rdkit_cli.core.stats import DatasetStatistics

        props = DatasetStatistics.available_properties()
        assert "MolWt" in props
        assert "LogP" in props
        assert "TPSA" in props

    def test_property_statistics(self):
        """Test that property statistics are reasonable."""
        from rdkit_cli.core.stats import DatasetStatistics

        mols = [
            Chem.MolFromSmiles("C"),  # MW ~16
            Chem.MolFromSmiles("CCCCCCCCCC"),  # MW ~142
        ]

        stats = DatasetStatistics(properties=["MolWt"])
        result = stats.calculate(mols)

        # Check that min < mean < max
        assert result["MolWt_min"] < result["MolWt_mean"]
        assert result["MolWt_mean"] < result["MolWt_max"]
