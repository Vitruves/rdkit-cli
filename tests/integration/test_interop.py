"""Interoperability tests for CLI command chaining.

Tests that verify commands can work together - output of one command
can be used as input to another.
"""

import pytest
import subprocess
import sys
from pathlib import Path


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    """Run rdkit-cli command and return result."""
    cmd = [sys.executable, "-m", "rdkit_cli"] + args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


class TestStandardizeThenDescriptors:
    """Test standardize → descriptors pipeline."""

    def test_standardize_then_compute_descriptors(self, sample_csv, tmp_dir):
        """Test computing descriptors on standardized molecules."""
        standardized = tmp_dir / "standardized.csv"
        final = tmp_dir / "descriptors.csv"

        # Step 1: Standardize
        result1 = run_cli([
            "standardize",
            "-i", str(sample_csv),
            "-o", str(standardized),
            "-q",
        ])
        assert result1.returncode == 0
        assert standardized.exists()

        # Step 2: Compute descriptors
        result2 = run_cli([
            "descriptors", "compute",
            "-i", str(standardized),
            "-o", str(final),
            "-d", "MolWt,MolLogP,TPSA",
            "-q",
        ])
        assert result2.returncode == 0
        assert final.exists()

        content = final.read_text()
        assert "MolWt" in content
        assert "MolLogP" in content
        assert "TPSA" in content


class TestFilterThenFingerprints:
    """Test filter → fingerprints pipeline."""

    def test_filter_then_compute_fingerprints(self, sample_csv, tmp_dir):
        """Test computing fingerprints on filtered molecules."""
        filtered = tmp_dir / "filtered.csv"
        final = tmp_dir / "fingerprints.csv"

        # Step 1: Filter for druglike molecules
        result1 = run_cli([
            "filter", "druglike",
            "-i", str(sample_csv),
            "-o", str(filtered),
            "--rule", "lipinski",
            "-q",
        ])
        assert result1.returncode == 0
        assert filtered.exists()

        # Step 2: Compute fingerprints
        result2 = run_cli([
            "fingerprints", "compute",
            "-i", str(filtered),
            "-o", str(final),
            "--type", "morgan",
            "-q",
        ])
        assert result2.returncode == 0
        assert final.exists()


class TestConvertFormats:
    """Test format conversion chains."""

    def test_csv_to_smi_to_csv(self, sample_csv, tmp_dir):
        """Test CSV → SMI → CSV conversion."""
        smi_file = tmp_dir / "converted.smi"
        final = tmp_dir / "back_to_csv.csv"

        # Step 1: Convert to SMI
        result1 = run_cli([
            "convert",
            "-i", str(sample_csv),
            "-o", str(smi_file),
            "-n", "1",
            "-q",
        ])
        assert result1.returncode == 0
        assert smi_file.exists()

        # Step 2: Convert back to CSV
        result2 = run_cli([
            "convert",
            "-i", str(smi_file),
            "-o", str(final),
            "-n", "1",
            "-q",
        ])
        assert result2.returncode == 0
        assert final.exists()


class TestScaffoldThenDiversity:
    """Test scaffold → diversity pipeline."""

    def test_scaffold_then_diversity_analysis(self, sample_csv, tmp_dir):
        """Test diversity analysis on scaffolds."""
        scaffolds = tmp_dir / "scaffolds.csv"

        # Step 1: Extract Murcko scaffolds
        result1 = run_cli([
            "scaffold", "murcko",
            "-i", str(sample_csv),
            "-o", str(scaffolds),
            "-q",
        ])
        assert result1.returncode == 0
        assert scaffolds.exists()

        # Step 2: Analyze diversity (prints to stdout, no -o needed)
        result2 = run_cli([
            "diversity", "analyze",
            "-i", str(scaffolds),
            "--smiles-column", "scaffold",
            "-q",
        ])
        assert result2.returncode == 0


class TestEnumerateThenFilter:
    """Test enumerate → filter pipeline."""

    def test_enumerate_tautomers_then_filter(self, sample_csv, tmp_dir):
        """Test filtering enumerated tautomers."""
        tautomers = tmp_dir / "tautomers.csv"
        filtered = tmp_dir / "filtered.csv"

        # Step 1: Enumerate tautomers
        result1 = run_cli([
            "enumerate", "tautomers",
            "-i", str(sample_csv),
            "-o", str(tautomers),
            "--max-tautomers", "3",
            "-q",
        ])
        assert result1.returncode == 0
        assert tautomers.exists()

        # Step 2: Filter druglike
        result2 = run_cli([
            "filter", "druglike",
            "-i", str(tautomers),
            "-o", str(filtered),
            "--rule", "lipinski",
            "-q",
        ])
        assert result2.returncode == 0
        assert filtered.exists()


class TestFragmentThenAnalyze:
    """Test fragment → analyze pipeline."""

    def test_brics_then_analyze_fragments(self, sample_csv, tmp_dir):
        """Test analyzing BRICS fragments."""
        fragments = tmp_dir / "fragments.csv"
        analysis = tmp_dir / "analysis.csv"

        # Step 1: BRICS fragmentation
        result1 = run_cli([
            "fragment", "brics",
            "-i", str(sample_csv),
            "-o", str(fragments),
            "-q",
        ])
        assert result1.returncode == 0
        assert fragments.exists()

        # Step 2: Analyze fragments
        result2 = run_cli([
            "fragment", "analyze",
            "-i", str(fragments),
            "-o", str(analysis),
            "--fragment-column", "fragment_smiles",
        ])
        assert result2.returncode == 0
        assert analysis.exists()


class TestSimilarityPipeline:
    """Test similarity search → further processing."""

    def test_similarity_search_then_descriptors(self, sample_csv, tmp_dir):
        """Test computing descriptors on similarity search results."""
        similar = tmp_dir / "similar.csv"
        descriptors = tmp_dir / "descriptors.csv"

        # Step 1: Similarity search
        result1 = run_cli([
            "similarity", "search",
            "-i", str(sample_csv),
            "-o", str(similar),
            "--query", "c1ccccc1",
            "--threshold", "0.1",
            "-q",
        ])
        assert result1.returncode == 0
        assert similar.exists()

        # Step 2: Compute descriptors on results
        result2 = run_cli([
            "descriptors", "compute",
            "-i", str(similar),
            "-o", str(descriptors),
            "-d", "MolWt,MolLogP",
            "-q",
        ])
        assert result2.returncode == 0
        assert descriptors.exists()


class TestDiversityPipeline:
    """Test diversity picking → downstream processing."""

    def test_diversity_pick_then_fingerprints(self, sample_csv, tmp_dir):
        """Test computing fingerprints on diverse subset."""
        diverse = tmp_dir / "diverse.csv"
        fingerprints = tmp_dir / "fingerprints.csv"

        # Step 1: Pick diverse subset
        result1 = run_cli([
            "diversity", "pick",
            "-i", str(sample_csv),
            "-o", str(diverse),
            "-k", "3",
            "-q",
        ])
        assert result1.returncode == 0
        assert diverse.exists()

        # Step 2: Compute fingerprints
        result2 = run_cli([
            "fingerprints", "compute",
            "-i", str(diverse),
            "-o", str(fingerprints),
            "--type", "maccs",
            "-q",
        ])
        assert result2.returncode == 0
        assert fingerprints.exists()


class TestFullPipeline:
    """Test complete processing pipeline."""

    def test_standardize_filter_descriptors_fingerprints(self, sample_csv, tmp_dir):
        """Test full pipeline: standardize → filter → descriptors + fingerprints."""
        standardized = tmp_dir / "standardized.csv"
        filtered = tmp_dir / "filtered.csv"
        final = tmp_dir / "final.csv"

        # Step 1: Standardize
        result1 = run_cli([
            "standardize",
            "-i", str(sample_csv),
            "-o", str(standardized),
            "-q",
        ])
        assert result1.returncode == 0

        # Step 2: Filter druglike
        result2 = run_cli([
            "filter", "druglike",
            "-i", str(standardized),
            "-o", str(filtered),
            "--rule", "lipinski",
            "-q",
        ])
        assert result2.returncode == 0

        # Step 3: Compute descriptors
        result3 = run_cli([
            "descriptors", "compute",
            "-i", str(filtered),
            "-o", str(final),
            "-d", "MolWt,MolLogP,TPSA,HeavyAtomCount",
            "-q",
        ])
        assert result3.returncode == 0

        # Verify final output
        content = final.read_text()
        assert "MolWt" in content
        assert "HeavyAtomCount" in content


class TestParallelizationConsistency:
    """Test that results are consistent across parallelization settings."""

    def test_descriptors_single_vs_multi_cpu(self, sample_csv, tmp_dir):
        """Test that descriptors are same with -n 1 and -n 2."""
        single = tmp_dir / "single.csv"
        multi = tmp_dir / "multi.csv"

        # Single process
        result1 = run_cli([
            "descriptors", "compute",
            "-i", str(sample_csv),
            "-o", str(single),
            "-d", "MolWt,MolLogP",
            "-n", "1",
            "-q",
        ])
        assert result1.returncode == 0

        # Multi process
        result2 = run_cli([
            "descriptors", "compute",
            "-i", str(sample_csv),
            "-o", str(multi),
            "-d", "MolWt,MolLogP",
            "-n", "2",
            "-q",
        ])
        assert result2.returncode == 0

        # Compare results (order may differ due to parallelization)
        import pandas as pd
        df1 = pd.read_csv(single).sort_values("smiles").reset_index(drop=True)
        df2 = pd.read_csv(multi).sort_values("smiles").reset_index(drop=True)

        # Values should be approximately equal
        assert len(df1) == len(df2)
        for col in ["MolWt", "MolLogP"]:
            diff = (df1[col] - df2[col]).abs()
            assert diff.max() < 0.001, f"Values differ in column {col}"
