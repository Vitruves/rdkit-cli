# tests/test_descriptors.py
"""Tests for molecular descriptors module."""

import pytest
import pandas as pd
from pathlib import Path

from rdkit_cli.commands import descriptors


class TestDescriptors:
    """Test descriptor calculation."""
    
    def test_basic_descriptors(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test basic descriptor calculation."""
        output_file = temp_output_dir / "descriptors.csv"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            descriptor_set="basic",
            descriptors=None,
            include_3d=False,
            skip_errors=True
        )
        
        result = descriptors.calculate(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        df = pd.read_csv(output_file)
        assert len(df) > 0
        assert 'ID' in df.columns
        assert 'SMILES' in df.columns
        assert 'MolWt' in df.columns
        assert 'LogP' in df.columns
        assert 'TPSA' in df.columns
        
        # Check that molecular weights are reasonable
        assert df['MolWt'].min() > 0
        assert df['MolWt'].max() < 1000
    
    def test_lipinski_descriptors(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test Lipinski descriptor set."""
        output_file = temp_output_dir / "lipinski.csv"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            descriptor_set="lipinski",
            descriptors=None,
            include_3d=False,
            skip_errors=True
        )
        
        result = descriptors.calculate(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        df = pd.read_csv(output_file)
        assert len(df) > 0
        
        # Check Lipinski descriptors are present
        lipinski_descriptors = ['MolWt', 'LogP', 'NumHBD', 'NumHBA', 'NumRotatableBonds']
        for desc in lipinski_descriptors:
            assert desc in df.columns
    
    def test_custom_descriptors(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test custom descriptor list."""
        output_file = temp_output_dir / "custom.csv"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            descriptor_set="basic",
            descriptors="MolWt,LogP,TPSA",
            include_3d=False,
            skip_errors=True
        )
        
        result = descriptors.calculate(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        df = pd.read_csv(output_file)
        assert len(df) > 0
        
        # Should only have specified descriptors plus ID and SMILES
        expected_cols = {'ID', 'SMILES', 'MolWt', 'LogP', 'TPSA'}
        assert set(df.columns) == expected_cols


class TestPhysicochemical:
    """Test physicochemical properties calculation."""
    
    def test_physicochemical_basic(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test basic physicochemical properties."""
        output_file = temp_output_dir / "physico.csv"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            include_druglike_filters=False,
            include_qed=False
        )
        
        result = descriptors.physicochemical(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        df = pd.read_csv(output_file)
        assert len(df) > 0
        
        # Check expected columns
        expected_cols = ['ID', 'SMILES', 'MolWt', 'LogP', 'TPSA', 'NumHBD', 'NumHBA']
        for col in expected_cols:
            assert col in df.columns
    
    def test_physicochemical_with_filters(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test physicochemical properties with drug-like filters."""
        output_file = temp_output_dir / "physico_filters.csv"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            include_druglike_filters=True,
            include_qed=True
        )
        
        result = descriptors.physicochemical(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        df = pd.read_csv(output_file)
        assert len(df) > 0
        
        # Check filter columns
        filter_cols = ['Lipinski_Pass', 'Veber_Pass', 'Egan_Pass', 'QED']
        for col in filter_cols:
            assert col in df.columns
        
        # Check boolean filter columns
        assert df['Lipinski_Pass'].dtype == bool
        assert df['Veber_Pass'].dtype == bool
        assert df['Egan_Pass'].dtype == bool


class TestADMET:
    """Test ADMET properties calculation."""
    
    def test_admet_basic(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test basic ADMET properties."""
        output_file = temp_output_dir / "admet.csv"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            models="basic"
        )
        
        result = descriptors.admet(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        df = pd.read_csv(output_file)
        assert len(df) > 0
        assert 'ID' in df.columns
        assert 'SMILES' in df.columns
        
        # Check some ADMET properties are calculated
        admet_cols = [col for col in df.columns if col not in ['ID', 'SMILES']]
        assert len(admet_cols) > 0
    
    def test_admet_all_models(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test all ADMET models."""
        output_file = temp_output_dir / "admet_all.csv"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            models="all"
        )
        
        result = descriptors.admet(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        df = pd.read_csv(output_file)
        assert len(df) > 0
        
        # Should have more columns with all models
        assert len(df.columns) > 5