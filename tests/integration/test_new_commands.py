"""Integration tests for new CLI commands."""

import pytest
from pathlib import Path


class TestInfoCommand:
    """Test info command."""

    def test_info_basic(self, cli_runner):
        """Test basic info command."""
        result = cli_runner(["info", "CCO"])
        assert result == 0

    def test_info_json(self, cli_runner):
        """Test info command with JSON output."""
        result = cli_runner(["info", "CCO", "--json"])
        assert result == 0

    def test_info_invalid_smiles(self, cli_runner):
        """Test info with invalid SMILES."""
        result = cli_runner(["info", "invalid_smiles"])
        assert result == 1


class TestMergeCommand:
    """Test merge command."""

    def test_merge_two_files(self, cli_runner, tmp_dir):
        """Test merging two files."""
        # Create two input files
        file1 = tmp_dir / "file1.csv"
        file2 = tmp_dir / "file2.csv"
        output = tmp_dir / "merged.csv"

        file1.write_text("smiles,name\nCCO,ethanol\n")
        file2.write_text("smiles,name\nc1ccccc1,benzene\n")

        result = cli_runner([
            "merge",
            "-i", str(file1), str(file2),
            "-o", str(output),
            "-q",
        ])

        assert result == 0
        assert output.exists()

        content = output.read_text()
        assert "CCO" in content
        assert "c1ccccc1" in content or "benzene" in content

    def test_merge_with_dedupe(self, cli_runner, tmp_dir):
        """Test merging with deduplication."""
        file1 = tmp_dir / "file1.csv"
        file2 = tmp_dir / "file2.csv"
        output = tmp_dir / "merged.csv"

        file1.write_text("smiles,name\nCCO,ethanol1\n")
        file2.write_text("smiles,name\nCCO,ethanol2\nc1ccccc1,benzene\n")

        result = cli_runner([
            "merge",
            "-i", str(file1), str(file2),
            "-o", str(output),
            "--dedupe",
            "-q",
        ])

        assert result == 0

        content = output.read_text()
        # CCO should appear only once
        assert content.count("CCO") == 1


class TestSAScorerCommand:
    """Test sascorer command."""

    def test_sascorer_basic(self, cli_runner, sample_csv, output_csv):
        """Test basic SA score calculation."""
        result = cli_runner([
            "sascorer",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])

        assert result == 0
        assert output_csv.exists()

        content = output_csv.read_text()
        assert "sa_score" in content

    def test_sascorer_with_qed(self, cli_runner, sample_csv, output_csv):
        """Test SA score with QED."""
        result = cli_runner([
            "sascorer",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--qed",
            "-q",
        ])

        assert result == 0

        content = output_csv.read_text()
        assert "qed_score" in content


class TestRGroupCommand:
    """Test rgroup command."""

    def test_rgroup_basic(self, cli_runner, tmp_dir, output_csv):
        """Test basic R-group decomposition."""
        # Create input with benzene derivatives
        input_file = tmp_dir / "benzenes.csv"
        input_file.write_text("smiles,name\nc1ccc(C)cc1,toluene\nc1ccc(CC)cc1,ethylbenzene\n")

        result = cli_runner([
            "rgroup",
            "-i", str(input_file),
            "-o", str(output_csv),
            "--core", "c1ccc([*:1])cc1",
            "-q",
        ])

        assert result == 0
        assert output_csv.exists()


class TestRingsCommand:
    """Test rings command."""

    def test_rings_extract(self, cli_runner, sample_csv, output_csv):
        """Test ring extraction."""
        result = cli_runner([
            "rings", "extract",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])

        assert result == 0
        assert output_csv.exists()

    def test_rings_info(self, cli_runner, sample_csv, output_csv):
        """Test ring info."""
        result = cli_runner([
            "rings", "info",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])

        assert result == 0

        content = output_csv.read_text()
        assert "num_rings" in content


class TestMMPCommand:
    """Test mmp command."""

    def test_mmp_fragment(self, cli_runner, sample_csv, output_csv):
        """Test MMP fragmentation."""
        result = cli_runner([
            "mmp", "fragment",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-q",
        ])

        assert result == 0
        assert output_csv.exists()

    def test_mmp_transform(self, cli_runner, tmp_dir, output_csv):
        """Test MMP transformation."""
        input_file = tmp_dir / "mols.csv"
        input_file.write_text("smiles,name\nc1ccc(C)cc1,toluene\n")

        result = cli_runner([
            "mmp", "transform",
            "-i", str(input_file),
            "-o", str(output_csv),
            "-t", "[c:1][CH3]>>[c:1][NH2]",
            "-q",
        ])

        assert result == 0


class TestProtonateCommand:
    """Test protonate command."""

    def test_protonate_basic(self, cli_runner, sample_csv, output_csv):
        """Test basic protonation."""
        result = cli_runner([
            "protonate",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--ph", "7.4",
            "-q",
        ])

        assert result == 0
        assert output_csv.exists()

    def test_protonate_neutralize(self, cli_runner, tmp_dir, output_csv):
        """Test neutralization."""
        input_file = tmp_dir / "charged.csv"
        input_file.write_text("smiles,name\nCC(=O)[O-],acetate\nCC[NH3+],ethylammonium\n")

        result = cli_runner([
            "protonate",
            "-i", str(input_file),
            "-o", str(output_csv),
            "--neutralize",
            "-q",
        ])

        assert result == 0


class TestPropsCommand:
    """Test props command."""

    def test_props_add(self, cli_runner, sample_csv, output_csv):
        """Test adding a property column."""
        result = cli_runner([
            "props", "add",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-c", "series",
            "-v", "test_series",
        ])

        assert result == 0

        content = output_csv.read_text()
        assert "series" in content
        assert "test_series" in content

    def test_props_rename(self, cli_runner, sample_csv, output_csv):
        """Test renaming a column."""
        result = cli_runner([
            "props", "rename",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "--from", "name",
            "--to", "molecule_name",
        ])

        assert result == 0

        content = output_csv.read_text()
        assert "molecule_name" in content

    def test_props_drop(self, cli_runner, sample_csv, output_csv):
        """Test dropping a column."""
        result = cli_runner([
            "props", "drop",
            "-i", str(sample_csv),
            "-o", str(output_csv),
            "-c", "name",
        ])

        assert result == 0

    def test_props_list(self, cli_runner, sample_csv):
        """Test listing columns."""
        result = cli_runner([
            "props", "list",
            "-i", str(sample_csv),
        ])

        assert result == 0


class TestAlignCommand:
    """Test align command."""

    def test_align_basic(self, cli_runner, tmp_dir, output_sdf):
        """Test basic 3D alignment."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Create reference SDF
        ref_mol = Chem.MolFromSmiles("c1ccccc1")
        ref_mol = Chem.AddHs(ref_mol)
        AllChem.EmbedMolecule(ref_mol, randomSeed=42)

        ref_file = tmp_dir / "ref.sdf"
        writer = Chem.SDWriter(str(ref_file))
        writer.write(ref_mol)
        writer.close()

        # Create input SDF
        probe_mol = Chem.MolFromSmiles("c1ccc(C)cc1")
        probe_mol = Chem.AddHs(probe_mol)
        AllChem.EmbedMolecule(probe_mol, randomSeed=42)

        input_file = tmp_dir / "probe.sdf"
        writer = Chem.SDWriter(str(input_file))
        writer.write(probe_mol)
        writer.close()

        result = cli_runner([
            "align",
            "-i", str(input_file),
            "-o", str(output_sdf),
            "-r", str(ref_file),
            "-q",
        ])

        assert result == 0


class TestRMSDCommand:
    """Test rmsd command."""

    def test_rmsd_compare(self, cli_runner, tmp_dir, output_csv):
        """Test RMSD comparison."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Create reference
        ref_mol = Chem.MolFromSmiles("CCCC")
        ref_mol = Chem.AddHs(ref_mol)
        AllChem.EmbedMolecule(ref_mol, randomSeed=42)

        ref_file = tmp_dir / "ref.sdf"
        writer = Chem.SDWriter(str(ref_file))
        writer.write(ref_mol)
        writer.close()

        # Create input
        probe_mol = Chem.MolFromSmiles("CCCC")
        probe_mol = Chem.AddHs(probe_mol)
        AllChem.EmbedMolecule(probe_mol, randomSeed=43)

        input_file = tmp_dir / "probe.sdf"
        writer = Chem.SDWriter(str(input_file))
        writer.write(probe_mol)
        writer.close()

        result = cli_runner([
            "rmsd", "compare",
            "-i", str(input_file),
            "-o", str(output_csv),
            "-r", str(ref_file),
            "-q",
        ])

        assert result == 0
