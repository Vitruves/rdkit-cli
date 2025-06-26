# tests/test_utils.py
"""Tests for utilities module."""

import pytest
import json
import pandas as pd
from pathlib import Path

from rdkit_cli.commands import utils


class TestInfo:
    """Test file information command."""
    
    def test_info_command(self, sample_sdf_file, graceful_exit_mock, mock_args, capsys):
        """Test file info display."""
        args = mock_args(
            input_file=str(sample_sdf_file)
        )
        
        result = utils.info(args, graceful_exit_mock)
        
        assert result == 0
        
        captured = capsys.readouterr()
        assert "File:" in captured.out
        assert "Valid molecules:" in captured.out
        assert "Molecular statistics:" in captured.out


class TestStats:
    """Test statistics calculation."""
    
    def test_basic_stats(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test basic statistics calculation."""
        output_file = temp_output_dir / "stats.json"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            include_descriptors=False
        )
        
        result = utils.stats(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            stats = json.load(f)
        
        assert 'file_info' in stats
        assert 'basic_statistics' in stats
        
        file_info = stats['file_info']
        assert file_info['total_entries'] > 0
        assert file_info['valid_molecules'] > 0
        
        basic_stats = stats['basic_statistics']
        assert 'atom_counts' in basic_stats
        assert 'heavy_atom_counts' in basic_stats
    
    def test_stats_with_descriptors(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test statistics with descriptor calculation."""
        output_file = temp_output_dir / "stats_desc.json"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            include_descriptors=True
        )
        
        result = utils.stats(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            stats = json.load(f)
        
        assert 'descriptor_statistics' in stats
        
        desc_stats = stats['descriptor_statistics']
        assert 'molecular_weight' in desc_stats
        assert 'logp' in desc_stats
        assert 'tpsa' in desc_stats


class TestSample:
    """Test molecular sampling."""
    
    def test_random_sampling(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test random sampling."""
        output_file = temp_output_dir / "sample.sdf"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            count=5,
            method="random",
            seed=42
        )
        
        result = utils.sample(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        from rdkit_cli.core.common import read_molecules
        sampled_molecules = read_molecules(output_file)
        assert len(sampled_molecules) <= 5
        assert len(sampled_molecules) > 0
    
    def test_systematic_sampling(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test systematic sampling."""
        output_file = temp_output_dir / "systematic.sdf"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            count=3,
            method="systematic",
            seed=42
        )
        
        result = utils.sample(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        from rdkit_cli.core.common import read_molecules
        sampled_molecules = read_molecules(output_file)
        assert len(sampled_molecules) <= 3
    
    def test_diverse_sampling(self, sample_sdf_file, temp_output_dir, graceful_exit_mock, mock_args):
        """Test diverse sampling."""
        output_file = temp_output_dir / "diverse.sdf"
        
        args = mock_args(
            input_file=str(sample_sdf_file),
            output=str(output_file),
            count=4,
            method="diverse",
            seed=42
        )
        
        result = utils.sample(args, graceful_exit_mock)
        
        assert result == 0
        assert output_file.exists()
        
        from rdkit_cli.core.common import read_molecules
        sampled_molecules = read_molecules(output_file)
        assert len(sampled_molecules) <= 4


class TestBenchmark:
    """Test benchmarking functionality."""
    
    def test_descriptor_benchmark(self, sample_sdf_file, graceful_exit_mock, mock_args, capsys):
        """Test descriptor benchmarking."""
        args = mock_args(
            input_file=str(sample_sdf_file),
            operation="descriptors",
            jobs=1,
            iterations=1
        )
        
        result = utils.benchmark(args, graceful_exit_mock)
        
        assert result == 0
        
        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "descriptors" in captured.out
    
    def test_fingerprint_benchmark(self, sample_sdf_file, graceful_exit_mock, mock_args, capsys):
        """Test fingerprint benchmarking."""
        args = mock_args(
            input_file=str(sample_sdf_file),
            operation="fingerprints",
            jobs=1,
            iterations=1
        )
        
        result = utils.benchmark(args, graceful_exit_mock)
        
        assert result == 0
        
        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "fingerprints" in captured.out


class TestConfig:
    """Test configuration management."""
    
    def test_config_set_get(self, graceful_exit_mock, mock_args, capsys):
        """Test setting and getting config values."""
        # Test setting a value
        args = mock_args(
            set=["test_key", "test_value"],
            get=None,
            list=False,
            reset=False
        )
        
        result = utils.config_cmd(args, graceful_exit_mock)
        assert result == 0
        
        captured = capsys.readouterr()
        assert "Set test_key = test_value" in captured.out
        
        # Test getting the value
        args = mock_args(
            set=None,
            get="test_key",
            list=False,
            reset=False
        )
        
        result = utils.config_cmd(args, graceful_exit_mock)
        assert result == 0
        
        captured = capsys.readouterr()
        assert "test_key = test_value" in captured.out
    
    def test_config_list(self, graceful_exit_mock, mock_args, capsys):
        """Test listing all config values."""
        args = mock_args(
            set=None,
            get=None,
            list=True,
            reset=False
        )
        
        result = utils.config_cmd(args, graceful_exit_mock)
        assert result == 0
        
        captured = capsys.readouterr()
        assert "Configuration settings:" in captured.out
    
    def test_config_reset(self, graceful_exit_mock, mock_args, capsys):
        """Test resetting config to defaults."""
        args = mock_args(
            set=None,
            get=None,
            list=False,
            reset=True
        )
        
        result = utils.config_cmd(args, graceful_exit_mock)
        assert result == 0
        
        captured = capsys.readouterr()
        assert "Configuration reset to defaults" in captured.out


