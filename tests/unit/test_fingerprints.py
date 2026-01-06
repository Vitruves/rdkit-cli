"""Unit tests for fingerprints module."""

import pytest
from rdkit import Chem


class TestFingerprintCalculator:
    """Test FingerprintCalculator class."""

    def test_compute_morgan_fingerprint(self, sample_molecules):
        """Test computing Morgan fingerprints."""
        from rdkit_cli.core.fingerprints import FingerprintCalculator, FingerprintType
        from rdkit_cli.io.readers import MoleculeRecord

        calc = FingerprintCalculator(fp_type=FingerprintType.MORGAN)

        for name, smi in sample_molecules:
            mol = Chem.MolFromSmiles(smi)
            record = MoleculeRecord(mol=mol, smiles=smi, name=name)
            result = calc.compute(record)

            assert result is not None
            assert "fingerprint" in result
            assert len(result["fingerprint"]) > 0

    def test_compute_maccs_fingerprint(self, sample_molecules):
        """Test computing MACCS fingerprints."""
        from rdkit_cli.core.fingerprints import FingerprintCalculator, FingerprintType
        from rdkit_cli.io.readers import MoleculeRecord

        calc = FingerprintCalculator(fp_type=FingerprintType.MACCS)

        name, smi = sample_molecules[0]
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)
        result = calc.compute(record)

        assert result is not None
        assert "fingerprint" in result

    def test_fingerprint_hex_format(self, sample_molecules):
        """Test hex output format."""
        from rdkit_cli.core.fingerprints import FingerprintCalculator, FingerprintType
        from rdkit_cli.io.readers import MoleculeRecord

        calc = FingerprintCalculator(
            fp_type=FingerprintType.MORGAN,
            output_format="hex",
        )

        name, smi = sample_molecules[0]
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)
        result = calc.compute(record)

        assert result is not None
        # Hex or base64 encoded
        assert len(result["fingerprint"]) > 0

    def test_none_molecule(self):
        """Test handling of None molecule."""
        from rdkit_cli.core.fingerprints import FingerprintCalculator, FingerprintType
        from rdkit_cli.io.readers import MoleculeRecord

        calc = FingerprintCalculator(fp_type=FingerprintType.MORGAN)
        record = MoleculeRecord(mol=None, smiles="invalid")
        result = calc.compute(record)
        assert result is None


class TestListFingerprints:
    """Test list_fingerprints function."""

    def test_list_all(self):
        """Test listing all fingerprint types."""
        from rdkit_cli.core.fingerprints import list_fingerprints

        fps = list_fingerprints()
        assert len(fps) >= 6  # At least 6 fingerprint types

        names = [fp.name for fp in fps]
        assert "morgan" in names
        assert "maccs" in names
