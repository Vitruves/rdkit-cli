"""Integration tests for CLI commands."""

import pytest
import subprocess
import sys
from pathlib import Path


def run_cli(args: list[str], input_file: Path = None) -> subprocess.CompletedProcess:
    """Run rdkit-cli command and return result."""
    cmd = [sys.executable, "-m", "rdkit_cli"] + args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


class TestDescriptorsCommand:
    """Test descriptors command."""

    def test_list_descriptors(self):
        """Test listing descriptors."""
        result = run_cli(["descriptors", "list"])
        assert result.returncode == 0
        assert "MolWt" in result.stdout

    def test_list_descriptors_all(self):
        """Test listing all descriptors."""
        result = run_cli(["descriptors", "list", "--all"])
        assert result.returncode == 0
        # Should have many lines
        assert len(result.stdout.split("\n")) > 50

    def test_compute_descriptors(self, sample_csv, output_csv):
        """Test computing descriptors."""
        result = run_cli([
            "descriptors", "compute",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-d", "MolWt,MolLogP",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

        content = output_csv.read_text()
        assert "MolWt" in content
        assert "MolLogP" in content


class TestFingerprintsCommand:
    """Test fingerprints command."""

    def test_list_fingerprints(self):
        """Test listing fingerprints."""
        result = run_cli(["fingerprints", "list"])
        assert result.returncode == 0
        assert "morgan" in result.stdout.lower()

    def test_compute_fingerprints(self, sample_csv, output_csv):
        """Test computing fingerprints."""
        result = run_cli([
            "fingerprints", "compute",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--type", "morgan",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()


class TestFilterCommand:
    """Test filter command."""

    def test_filter_substructure(self, sample_csv, output_csv):
        """Test substructure filtering."""
        result = run_cli([
            "filter", "substructure",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--smarts", "c1ccccc1",  # benzene ring
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_filter_druglike(self, sample_csv, output_csv):
        """Test drug-likeness filtering."""
        result = run_cli([
            "filter", "druglike",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--rule", "lipinski",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()


class TestStandardizeCommand:
    """Test standardize command."""

    def test_standardize_basic(self, sample_csv, output_csv):
        """Test basic standardization."""
        result = run_cli([
            "standardize",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_standardize_with_options(self, sample_csv, output_csv):
        """Test standardization with options."""
        result = run_cli([
            "standardize",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--fragment-parent",
            "-n", "1",  # Single process to avoid pickling issues
            "-q",
        ])
        assert result.returncode == 0


class TestConvertCommand:
    """Test convert command."""

    def test_convert_csv_to_smi(self, sample_csv, output_smi):
        """Test converting CSV to SMI."""
        result = run_cli([
            "convert",
            "-i", str(sample_csv),
            "-o", str(output_smi),
            "-n", "1",  # Single process to avoid pickling issues
            "-q",
        ])
        assert result.returncode == 0
        assert output_smi.exists()


class TestSimilarityCommand:
    """Test similarity command."""

    def test_similarity_search(self, sample_csv, output_csv):
        """Test similarity search."""
        result = run_cli([
            "similarity", "search",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--query", "c1ccccc1",
            "--threshold", "0.1",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()


class TestScaffoldCommand:
    """Test scaffold command."""

    def test_scaffold_murcko(self, sample_csv, output_csv):
        """Test Murcko scaffold extraction."""
        result = run_cli([
            "scaffold", "murcko",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()


class TestHelpCommand:
    """Test help output."""

    def test_main_help(self):
        """Test main help."""
        result = run_cli(["--help"])
        assert result.returncode == 0
        assert "rdkit-cli" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_descriptors_help(self):
        """Test descriptors help."""
        result = run_cli(["descriptors", "--help"])
        assert result.returncode == 0
        assert "descriptors" in result.stdout.lower()

    def test_version(self):
        """Test version output."""
        result = run_cli(["--version"])
        assert result.returncode == 0


class TestErrorHandling:
    """Test error handling."""

    def test_missing_input_file(self, output_csv):
        """Test error when input file missing."""
        result = run_cli([
            "descriptors", "compute",
            "-i", "/nonexistent/file.csv",
            "-o", str(output_csv),
        ])
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "not found" in result.stderr.lower()

    def test_invalid_smarts(self, sample_csv, output_csv):
        """Test error with invalid SMARTS."""
        result = run_cli([
            "filter", "substructure",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--smarts", "invalid(((",
        ])
        assert result.returncode != 0


class TestEnumerateCommand:
    """Test enumerate command."""

    def test_enumerate_stereoisomers(self, sample_csv, output_csv):
        """Test stereoisomer enumeration."""
        result = run_cli([
            "enumerate", "stereoisomers",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--max-isomers", "5",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_enumerate_tautomers(self, sample_csv, output_csv):
        """Test tautomer enumeration."""
        result = run_cli([
            "enumerate", "tautomers",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--max-tautomers", "5",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_canonical_tautomer(self, sample_csv, output_csv):
        """Test canonical tautomer generation."""
        result = run_cli([
            "enumerate", "canonical-tautomer",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()


class TestFragmentCommand:
    """Test fragment command."""

    def test_fragment_brics(self, sample_csv, output_csv):
        """Test BRICS fragmentation."""
        result = run_cli([
            "fragment", "brics",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_fragment_recap(self, sample_csv, output_csv):
        """Test RECAP fragmentation."""
        result = run_cli([
            "fragment", "recap",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_functional_groups(self, sample_csv, output_csv):
        """Test functional group extraction."""
        result = run_cli([
            "fragment", "functional-groups",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()


class TestDiversityCommand:
    """Test diversity command."""

    def test_diversity_pick(self, sample_csv, output_csv):
        """Test diversity picking."""
        result = run_cli([
            "diversity", "pick",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-k", "3",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_diversity_analyze(self, sample_csv, output_csv):
        """Test diversity analysis."""
        result = run_cli([
            "diversity", "analyze",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()


class TestMCSCommand:
    """Test mcs command."""

    def test_mcs_find(self, sample_csv, output_csv):
        """Test MCS finding."""
        result = run_cli([
            "mcs",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_mcs_with_options(self, sample_csv, output_csv):
        """Test MCS with options."""
        result = run_cli([
            "mcs",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--timeout", "30",
            "--atom-compare", "elements",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()


class TestDepictCommand:
    """Test depict command."""

    def test_depict_single(self, output_svg):
        """Test single molecule depiction."""
        result = run_cli([
            "depict", "single",
            "--smiles", "c1ccccc1",
            "-o", str(output_svg),
        ])
        assert result.returncode == 0
        assert output_svg.exists()
        content = output_svg.read_text()
        assert "<svg" in content

    def test_depict_single_png(self, output_png):
        """Test single molecule depiction as PNG."""
        result = run_cli([
            "depict", "single",
            "--smiles", "CCO",
            "-o", str(output_png),
            "-f", "png",
        ])
        assert result.returncode == 0
        assert output_png.exists()
        # PNG magic bytes
        content = output_png.read_bytes()
        assert content[:4] == b'\x89PNG'

    def test_depict_batch(self, sample_csv, output_dir):
        """Test batch molecule depiction."""
        result = run_cli([
            "depict", "batch",
            "-i", str(sample_csv),
            "-o", str(output_dir),
            "-f", "svg",
            "-q",
        ])
        assert result.returncode == 0
        # Should have created some SVG files
        svg_files = list(output_dir.glob("*.svg"))
        assert len(svg_files) > 0

    def test_depict_grid(self, sample_csv, output_svg):
        """Test grid molecule depiction."""
        result = run_cli([
            "depict", "grid",
            "-i", str(sample_csv),
            "-o", str(output_svg),
            "--mols-per-row", "3",
        ])
        assert result.returncode == 0
        assert output_svg.exists()


class TestConformersCommand:
    """Test conformers command."""

    def test_conformers_generate(self, sample_csv, output_sdf):
        """Test conformer generation."""
        result = run_cli([
            "conformers", "generate",
            "-i", str(sample_csv),
            "-o", str(output_sdf),
            "--num", "2",
            "-n", "1",  # Single process
            "-q",
        ])
        assert result.returncode == 0
        assert output_sdf.exists()


class TestReactionsCommand:
    """Test reactions command."""

    def test_reactions_transform(self, sample_csv, output_csv):
        """Test SMIRKS transformation."""
        result = run_cli([
            "reactions", "transform",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--smirks", "[OH:1]>>[O-:1]",  # Deprotonate hydroxyl
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()
