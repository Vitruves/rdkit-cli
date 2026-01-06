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

    def test_filter_substructure_exclude(self, sample_csv, output_csv):
        """Test substructure filtering with exclude mode."""
        result = run_cli([
            "filter", "substructure",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--smarts", "c1ccccc1",
            "--exclude",
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

    def test_filter_druglike_veber(self, sample_csv, output_csv):
        """Test drug-likeness filtering with Veber rule."""
        result = run_cli([
            "filter", "druglike",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--rule", "veber",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_filter_druglike_with_violations(self, sample_csv, output_csv):
        """Test drug-likeness filtering with max violations."""
        result = run_cli([
            "filter", "druglike",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--rule", "lipinski",
            "--max-violations", "1",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_filter_pains(self, sample_csv, output_csv):
        """Test PAINS filtering."""
        result = run_cli([
            "filter", "pains",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_filter_pains_keep(self, sample_csv, output_csv):
        """Test PAINS filtering with keep-pains mode."""
        result = run_cli([
            "filter", "pains",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--keep-pains",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_filter_elements(self, sample_csv, output_csv):
        """Test element filtering."""
        result = run_cli([
            "filter", "elements",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--allowed", "C,H,N,O",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_filter_elements_required(self, sample_csv, output_csv):
        """Test element filtering with required elements."""
        result = run_cli([
            "filter", "elements",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--required", "N",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_filter_elements_forbidden(self, sample_csv, output_csv):
        """Test element filtering with forbidden elements."""
        result = run_cli([
            "filter", "elements",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--forbidden", "S,P",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_filter_complexity(self, sample_csv, output_csv):
        """Test complexity filtering."""
        result = run_cli([
            "filter", "complexity",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--min-atoms", "5",
            "--max-atoms", "50",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_filter_complexity_rings(self, sample_csv, output_csv):
        """Test complexity filtering by ring count."""
        result = run_cli([
            "filter", "complexity",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--min-rings", "1",
            "--max-rings", "3",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_filter_complexity_rotatable(self, sample_csv, output_csv):
        """Test complexity filtering by rotatable bonds."""
        result = run_cli([
            "filter", "complexity",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--min-rotatable", "0",
            "--max-rotatable", "5",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_filter_property(self, sample_csv, output_csv):
        """Test property filtering."""
        result = run_cli([
            "filter", "property",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--rule", "MolWt<500",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_filter_property_multiple_rules(self, sample_csv, output_csv):
        """Test property filtering with multiple rules."""
        result = run_cli([
            "filter", "property",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--rule", "MolWt<500",
            "--rule", "MolLogP<5",
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


class TestStatsCommand:
    """Test stats command."""

    def test_stats_basic(self, sample_csv, output_csv):
        """Test basic stats calculation."""
        result = run_cli([
            "stats",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_stats_list_properties(self):
        """Test listing available properties."""
        result = run_cli(["stats", "-i", "dummy.csv", "--list-properties"])
        assert result.returncode == 0
        assert "MolWt" in result.stdout

    def test_stats_specific_properties(self, sample_csv, output_csv):
        """Test with specific properties."""
        result = run_cli([
            "stats",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-p", "MolWt,LogP",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_stats_json_format(self, sample_csv, output_csv):
        """Test JSON output format."""
        result = run_cli([
            "stats",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--format", "json",
            "-q",
        ])
        assert result.returncode == 0
        content = output_csv.read_text()
        assert "{" in content  # JSON format


class TestSplitCommand:
    """Test split command."""

    def test_split_by_chunks(self, sample_csv, output_dir):
        """Test splitting by number of chunks."""
        result = run_cli([
            "split",
            "-i", str(sample_csv),
            "-o", str(output_dir),
            "-c", "2",
            "-q",
        ])
        assert result.returncode == 0
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) == 2

    def test_split_by_size(self, sample_csv, output_dir):
        """Test splitting by chunk size."""
        result = run_cli([
            "split",
            "-i", str(sample_csv),
            "-o", str(output_dir),
            "-s", "2",
            "-q",
        ])
        assert result.returncode == 0
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) >= 2

    def test_split_with_prefix(self, sample_csv, output_dir):
        """Test splitting with custom prefix."""
        result = run_cli([
            "split",
            "-i", str(sample_csv),
            "-o", str(output_dir),
            "-c", "2",
            "--prefix", "molecules",
            "-q",
        ])
        assert result.returncode == 0
        assert any("molecules" in f.name for f in output_dir.glob("*.csv"))


class TestSampleCommand:
    """Test sample command."""

    def test_sample_by_count(self, sample_csv, output_csv):
        """Test sampling by count."""
        result = run_cli([
            "sample",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-k", "3",
            "--seed", "42",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()
        # Count lines (minus header)
        lines = output_csv.read_text().strip().split("\n")
        assert len(lines) == 4  # 1 header + 3 samples

    def test_sample_by_fraction(self, sample_csv, output_csv):
        """Test sampling by fraction."""
        result = run_cli([
            "sample",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-f", "0.5",
            "--seed", "42",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_sample_stream_mode(self, sample_csv, output_csv):
        """Test stream (reservoir) sampling."""
        result = run_cli([
            "sample",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-k", "2",
            "--stream",
            "--seed", "42",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()


class TestDeduplicateCommand:
    """Test deduplicate command."""

    def test_deduplicate_basic(self, sample_csv, output_csv):
        """Test basic deduplication."""
        result = run_cli([
            "deduplicate",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_deduplicate_by_inchikey(self, sample_csv, output_csv):
        """Test deduplication by InChIKey."""
        result = run_cli([
            "deduplicate",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-b", "inchikey",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_deduplicate_list_keys(self):
        """Test listing available key types."""
        result = run_cli(["deduplicate", "-i", "dummy.csv", "-o", "out.csv", "--list-keys"])
        assert result.returncode == 0
        assert "smiles" in result.stdout
        assert "inchikey" in result.stdout


class TestValidateCommand:
    """Test validate command."""

    def test_validate_basic(self, sample_csv, output_csv):
        """Test basic validation."""
        result = run_cli([
            "validate",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()
        content = output_csv.read_text()
        assert "is_valid" in content

    def test_validate_valid_only(self, sample_csv, output_csv):
        """Test outputting only valid molecules."""
        result = run_cli([
            "validate",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--valid-only",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()
        content = output_csv.read_text()
        assert "is_valid" not in content  # No validation columns

    def test_validate_with_constraints(self, sample_csv, output_csv):
        """Test validation with constraints."""
        result = run_cli([
            "validate",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--max-atoms", "50",
            "--max-rings", "5",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_validate_allowed_elements(self, sample_csv, output_csv):
        """Test validation with element constraints."""
        result = run_cli([
            "validate",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--allowed-elements", "C,H,N,O,S",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()

    def test_validate_with_invalid_molecules(self, sample_csv_with_invalid, output_csv):
        """Test validation with some invalid molecules."""
        result = run_cli([
            "validate",
            "-i", str(sample_csv_with_invalid),
            "-o", str(output_csv),
            "--summary",
            "-q",
        ])
        assert result.returncode == 0
        assert output_csv.exists()
        # Check summary was printed
        assert "Invalid:" in result.stderr
