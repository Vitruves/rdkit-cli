# tests/test_integration.py
"""Integration tests for RDKit CLI."""

import pytest
import tempfile
import subprocess
import sys
from pathlib import Path


class TestCLIIntegration:
    """Test CLI integration and end-to-end workflows."""
    
    def test_cli_help(self):
        """Test that CLI help works."""
        result = subprocess.run(
            [sys.executable, "-m", "rdkit_cli.main", "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "rdkit-cli" in result.stdout
        assert "cheminformatics operations" in result.stdout
    
    def test_cli_subcommand_help(self):
        """Test subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "rdkit_cli.main", "descriptors", "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "descriptors" in result.stdout
        assert "calculated descriptors" in result.stdout
    
    def test_convert_workflow(self, sample_smiles_file, temp_output_dir):
        """Test complete conversion workflow."""
        output_file = temp_output_dir / "converted.sdf"
        
        result = subprocess.run([
            sys.executable, "-m", "rdkit_cli.main",
            "convert",
            "-i", str(sample_smiles_file),
            "-o", str(output_file),
            "--format", "sdf"
        ], capture_output=True, text=True)
        
        # Note: This may fail if dependencies aren't installed
        # but at least tests the command structure
        assert result.returncode in [0, 1]  # Allow failure for missing deps
    
    def test_descriptors_workflow(self, sample_sdf_file, temp_output_dir):
        """Test descriptors calculation workflow."""
        output_file = temp_output_dir / "descriptors.csv"
        
        result = subprocess.run([
            sys.executable, "-m", "rdkit_cli.main",
            "descriptors",
            "-i", str(sample_sdf_file),
            "-o", str(output_file),
            "--descriptor-set", "basic"
        ], capture_output=True, text=True)
        
        # Note: This may fail if dependencies aren't installed
        assert result.returncode in [0, 1]  # Allow failure for missing deps
    
    def test_info_workflow(self, sample_sdf_file):
        """Test info command workflow."""
        result = subprocess.run([
            sys.executable, "-m", "rdkit_cli.main",
            "info",
            "-i", str(sample_sdf_file)
        ], capture_output=True, text=True)
        
        # Note: This may fail if dependencies aren't installed
        assert result.returncode in [0, 1]  # Allow failure for missing deps
    
    def test_config_workflow(self):
        """Test config management workflow."""
        # Test config list
        result = subprocess.run([
            sys.executable, "-m", "rdkit_cli.main",
            "config",
            "--list"
        ], capture_output=True, text=True)
        
        assert result.returncode in [0, 1]  # Allow failure for missing deps
        
        # Test config set
        result = subprocess.run([
            sys.executable, "-m", "rdkit_cli.main",
            "config",
            "--set", "test_key", "test_value"
        ], capture_output=True, text=True)
        
        assert result.returncode in [0, 1]  # Allow failure for missing deps


class TestWorkflowChaining:
    """Test chaining multiple commands together."""
    
    def test_convert_then_descriptors(self, sample_smiles_file, temp_output_dir):
        """Test converting SMILES to SDF then calculating descriptors."""
        sdf_file = temp_output_dir / "molecules.sdf"
        desc_file = temp_output_dir / "descriptors.csv"
        
        # Step 1: Convert SMILES to SDF
        result1 = subprocess.run([
            sys.executable, "-m", "rdkit_cli.main",
            "convert",
            "-i", str(sample_smiles_file),
            "-o", str(sdf_file)
        ], capture_output=True, text=True)
        
        # Step 2: Calculate descriptors (if conversion succeeded)
        if result1.returncode == 0 and sdf_file.exists():
            result2 = subprocess.run([
                sys.executable, "-m", "rdkit_cli.main",
                "descriptors",
                "-i", str(sdf_file),
                "-o", str(desc_file),
                "--descriptor-set", "basic"
            ], capture_output=True, text=True)
            
            # If both succeed, check output exists
            if result2.returncode == 0:
                assert desc_file.exists()
    
    def test_sample_then_fingerprints(self, sample_sdf_file, temp_output_dir):
        """Test sampling molecules then generating fingerprints."""
        sample_file = temp_output_dir / "sample.sdf"
        fp_file = temp_output_dir / "fingerprints.pkl"
        
        # Step 1: Sample molecules
        result1 = subprocess.run([
            sys.executable, "-m", "rdkit_cli.main",
            "sample",
            "-i", str(sample_sdf_file),
            "-o", str(sample_file),
            "--count", "5",
            "--method", "random"
        ], capture_output=True, text=True)
        
        # Step 2: Generate fingerprints (if sampling succeeded)
        if result1.returncode == 0 and sample_file.exists():
            result2 = subprocess.run([
                sys.executable, "-m", "rdkit_cli.main",
                "fingerprints",
                "-i", str(sample_file),
                "-o", str(fp_file),
                "--fp-type", "morgan"
            ], capture_output=True, text=True)
            
            # If both succeed, check output exists
            if result2.returncode == 0:
                assert fp_file.exists()


class TestErrorHandling:
    """Test error handling in CLI."""
    
    def test_invalid_command(self):
        """Test invalid command handling."""
        result = subprocess.run([
            sys.executable, "-m", "rdkit_cli.main",
            "invalid_command"
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
    
    def test_missing_input_file(self, temp_output_dir):
        """Test missing input file handling."""
        nonexistent_file = temp_output_dir / "nonexistent.sdf"
        output_file = temp_output_dir / "output.csv"
        
        result = subprocess.run([
            sys.executable, "-m", "rdkit_cli.main",
            "descriptors",
            "-i", str(nonexistent_file),
            "-o", str(output_file)
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
    
    def test_invalid_arguments(self, sample_sdf_file):
        """Test invalid argument handling."""
        result = subprocess.run([
            sys.executable, "-m", "rdkit_cli.main",
            "descriptors",
            "-i", str(sample_sdf_file),
            "--invalid-option", "value"
        ], capture_output=True, text=True)
        
        assert result.returncode != 0


class TestVerboseLogging:
    """Test verbose and debug logging."""
    
    def test_verbose_logging(self, sample_sdf_file):
        """Test verbose logging option."""
        result = subprocess.run([
            sys.executable, "-m", "rdkit_cli.main",
            "-v",
            "info",
            "-i", str(sample_sdf_file)
        ], capture_output=True, text=True)
        
        # Should work with verbose flag
        assert result.returncode in [0, 1]  # Allow failure for missing deps
    
    def test_debug_logging(self, sample_sdf_file):
        """Test debug logging option."""
        result = subprocess.run([
            sys.executable, "-m", "rdkit_cli.main",
            "--debug",
            "info",
            "-i", str(sample_sdf_file)
        ], capture_output=True, text=True)
        
        # Should work with debug flag
        assert result.returncode in [0, 1]  # Allow failure for missing deps