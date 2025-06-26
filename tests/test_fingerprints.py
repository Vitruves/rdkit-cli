# tests/test_fingerprints.py
"""Tests for fingerprints module."""

import pytest
import pandas as pd
import pickle
from pathlib import Path

from rdkit_cli.commands import fingerprints


class TestFingerprints:
    """Test fingerprint generation."""
    
    def test_morgan_fingerprints(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test Morgan fingerprint generation."""
        output_file = temp_output_dir / "fps.pkl"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            fp_type="morgan",
            radius=2,
            n_bits=2048,
            use_features=False,
            use_chirality=False
        )
        
        result = fingerprints.generate(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        # Load and check fingerprint data
        with open(output_file, 'rb') as f:
            fp_data = pickle.load(f)
        
        assert 'fingerprints' in fp_data
        assert 'mol_ids' in fp_data
        assert 'fp_type' in fp_data
        assert len(fp_data['fingerprints']) > 0
        assert len(fp_data['mol_ids']) == len(fp_data['fingerprints'])
        assert fp_data['fp_type'] == 'morgan'
    
    def test_rdkit_fingerprints(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test RDKit fingerprint generation."""
        output_file = temp_output_dir / "rdkit_fps.pkl"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            fp_type="rdkit",
            radius=2,
            n_bits=2048,
            use_features=False,
            use_chirality=False
        )
        
        result = fingerprints.generate(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
    
    def test_maccs_fingerprints(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test MACCS fingerprint generation."""
        output_file = temp_output_dir / "maccs_fps.pkl"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            fp_type="maccs",
            radius=2,
            n_bits=2048,
            use_features=False,
            use_chirality=False
        )
        
        result = fingerprints.generate(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()


class TestSimilarity:
    """Test similarity search."""
    
    def test_similarity_search(self, sample_sdf_file, sample_smiles_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test molecular similarity search."""
        output_file = temp_output_dir / "similar.csv"
        
        args = mock_args(
            query=str(sample_smiles_file),
            database=str(sample_sdf_file),
            output=str(output_file),
            threshold=0.3,
            metric="tanimoto",
            fp_type="morgan"
        )
        
        result = fingerprints.similarity(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        df = pd.read_csv(output_file)
        assert len(df) >= 0  # May be empty if no matches
        
        if len(df) > 0:
            assert 'ID' in df.columns
            assert 'SMILES' in df.columns
            assert 'Similarity' in df.columns
            
            # Similarities should be above threshold
            assert (df['Similarity'] >= 0.3).all()
            assert (df['Similarity'] <= 1.0).all()


class TestSimilarityMatrix:
    """Test similarity matrix calculation."""
    
    def test_similarity_matrix(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test pairwise similarity matrix calculation."""
        output_file = temp_output_dir / "sim_matrix.csv"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            fp_type="morgan",
            metric="tanimoto"
        )
        
        result = fingerprints.similarity_matrix(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        df = pd.read_csv(output_file, index_col=0)
        
        # Should be square matrix
        assert df.shape[0] == df.shape[1]
        assert len(df) > 0
        
        # Diagonal should be 1.0 (self-similarity)
        for i in range(len(df)):
            assert abs(df.iloc[i, i] - 1.0) < 1e-6
        
        # Matrix should be symmetric
        for i in range(len(df)):
            for j in range(len(df)):
                assert abs(df.iloc[i, j] - df.iloc[j, i]) < 1e-6


class TestClustering:
    """Test molecular clustering."""
    
    def test_butina_clustering(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test Butina clustering."""
        output_file = temp_output_dir / "clusters.csv"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            method="butina",
            threshold=0.6,
            fp_type="morgan"
        )
        
        result = fingerprints.cluster(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        df = pd.read_csv(output_file)
        assert len(df) > 0
        assert 'ID' in df.columns
        assert 'SMILES' in df.columns
        assert 'Cluster' in df.columns
        
        # Cluster IDs should be non-negative integers
        assert (df['Cluster'] >= 0).all()
        assert df['Cluster'].dtype in ['int64', 'int32']
    
    def test_hierarchical_clustering(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test hierarchical clustering."""
        output_file = temp_output_dir / "hier_clusters.csv"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            method="hierarchical",
            threshold=0.6,
            fp_type="morgan"
        )
        
        result = fingerprints.cluster(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        df = pd.read_csv(output_file)
        assert len(df) > 0
        assert 'Cluster' in df.columns


class TestDiversityPicking:
    """Test diversity picking."""
    
    def test_maxmin_diversity(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test MaxMin diversity picking."""
        output_file = temp_output_dir / "diverse.sdf"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            method="maxmin",
            count=5,
            fp_type="morgan"
        )
        
        result = fingerprints.diversity_pick(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        from rdkit_cli.core.common import read_molecules
        diverse_molecules = read_molecules(output_file)
        assert len(diverse_molecules) <= 5
        assert len(diverse_molecules) > 0
    
    def test_sphere_exclusion_diversity(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test sphere exclusion diversity picking."""
        output_file = temp_output_dir / "diverse_sphere.sdf"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            method="sphere-exclusion",
            count=3,
            fp_type="morgan"
        )
        
        result = fingerprints.diversity_pick(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()