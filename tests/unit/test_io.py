"""Unit tests for IO module."""

import pytest
from pathlib import Path


class TestFormatDetection:
    """Test format detection."""

    def test_detect_csv(self):
        """Test CSV format detection."""
        from rdkit_cli.io.formats import detect_format, FileFormat

        assert detect_format(Path("file.csv")) == FileFormat.CSV

    def test_detect_tsv(self):
        """Test TSV format detection."""
        from rdkit_cli.io.formats import detect_format, FileFormat

        assert detect_format(Path("file.tsv")) == FileFormat.TSV

    def test_detect_smi(self):
        """Test SMI format detection."""
        from rdkit_cli.io.formats import detect_format, FileFormat

        assert detect_format(Path("file.smi")) == FileFormat.SMI
        assert detect_format(Path("file.smiles")) == FileFormat.SMI

    def test_detect_sdf(self):
        """Test SDF format detection."""
        from rdkit_cli.io.formats import detect_format, FileFormat

        assert detect_format(Path("file.sdf")) == FileFormat.SDF
        assert detect_format(Path("file.mol")) == FileFormat.SDF

    def test_detect_parquet(self):
        """Test Parquet format detection."""
        from rdkit_cli.io.formats import detect_format, FileFormat

        assert detect_format(Path("file.parquet")) == FileFormat.PARQUET

    def test_unknown_format(self):
        """Test unknown format raises error."""
        from rdkit_cli.io.formats import detect_format

        with pytest.raises(ValueError, match="Cannot detect format"):
            detect_format(Path("file.xyz"))


class TestCSVReader:
    """Test CSV reader."""

    def test_read_csv(self, sample_csv):
        """Test reading CSV file."""
        from rdkit_cli.io.readers import create_reader

        reader = create_reader(sample_csv)
        records = list(reader)

        assert len(records) == 5
        assert records[0].smiles is not None
        assert records[0].mol is not None

    def test_read_csv_with_name(self, sample_csv):
        """Test reading CSV with name column."""
        from rdkit_cli.io.readers import create_reader

        reader = create_reader(sample_csv, name_column="name")
        records = list(reader)

        assert records[0].name is not None


class TestSMIReader:
    """Test SMI reader."""

    def test_read_smi(self, sample_smi):
        """Test reading SMI file."""
        from rdkit_cli.io.readers import create_reader

        reader = create_reader(sample_smi)
        records = list(reader)

        assert len(records) == 5
        assert records[0].smiles is not None


class TestCSVWriter:
    """Test CSV writer."""

    def test_write_csv(self, output_csv):
        """Test writing CSV file."""
        from rdkit_cli.io.writers import create_writer

        writer = create_writer(output_csv)

        with writer:
            writer.write_row({"smiles": "CCO", "name": "ethanol"})
            writer.write_row({"smiles": "C", "name": "methane"})

        assert output_csv.exists()

        # Read back
        content = output_csv.read_text()
        assert "smiles" in content
        assert "CCO" in content

    def test_write_batch(self, output_csv):
        """Test batch writing."""
        from rdkit_cli.io.writers import create_writer

        writer = create_writer(output_csv)

        with writer:
            writer.write_batch([
                {"smiles": "CCO", "name": "ethanol"},
                {"smiles": "C", "name": "methane"},
            ])

        content = output_csv.read_text()
        assert "CCO" in content
        assert "C" in content


class TestSMIWriter:
    """Test SMI writer."""

    def test_write_smi(self, output_smi):
        """Test writing SMI file."""
        from rdkit_cli.io.writers import create_writer

        writer = create_writer(output_smi)

        with writer:
            writer.write_row({"smiles": "CCO", "name": "ethanol"})

        assert output_smi.exists()

        content = output_smi.read_text()
        assert "CCO" in content


class TestMoleculeRecord:
    """Test MoleculeRecord class."""

    def test_create_record(self):
        """Test creating a molecule record."""
        from rdkit_cli.io.readers import MoleculeRecord
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        record = MoleculeRecord(mol=mol, smiles="CCO", name="ethanol")

        assert record.mol is not None
        assert record.smiles == "CCO"
        assert record.name == "ethanol"

    def test_record_with_metadata(self):
        """Test record with additional metadata."""
        from rdkit_cli.io.readers import MoleculeRecord
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        record = MoleculeRecord(
            mol=mol,
            smiles="CCO",
            name="ethanol",
            metadata={"activity": 1.5},
        )

        assert record.metadata["activity"] == 1.5

    def test_record_with_invalid_smiles(self):
        """Test record with None molecule."""
        from rdkit_cli.io.readers import MoleculeRecord

        record = MoleculeRecord(mol=None, smiles="invalid")

        assert record.mol is None
        assert record.smiles == "invalid"
