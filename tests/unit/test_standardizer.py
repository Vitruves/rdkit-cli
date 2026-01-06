"""Unit tests for standardizer module."""

import pytest
from rdkit import Chem


class TestMoleculeStandardizer:
    """Test MoleculeStandardizer class."""

    def test_canonicalize(self, sample_molecules):
        """Test SMILES canonicalization."""
        from rdkit_cli.core.standardizer import MoleculeStandardizer
        from rdkit_cli.io.readers import MoleculeRecord

        std = MoleculeStandardizer(canonicalize=True)

        # Test with non-canonical SMILES
        smi = "C(C)(C)C"  # isobutane, non-canonical
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="isobutane")
        result = std.standardize(record)

        assert result is not None
        assert result["smiles"] == "CC(C)C"  # Canonical form

    def test_fragment_parent(self):
        """Test keeping largest fragment."""
        from rdkit_cli.core.standardizer import MoleculeStandardizer
        from rdkit_cli.io.readers import MoleculeRecord

        std = MoleculeStandardizer(fragment_parent=True)

        # Salt: molecule with counterion
        smi = "CC(=O)[O-].[Na+]"  # sodium acetate
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="sodium_acetate")
        result = std.standardize(record)

        assert result is not None
        # Should keep only the larger fragment (acetate)
        assert "." not in result["smiles"]

    def test_uncharge(self):
        """Test neutralizing charges."""
        from rdkit_cli.core.standardizer import MoleculeStandardizer
        from rdkit_cli.io.readers import MoleculeRecord

        std = MoleculeStandardizer(uncharge=True)

        # Carboxylate anion
        smi = "CC(=O)[O-]"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="acetate")
        result = std.standardize(record)

        assert result is not None
        # Should be neutralized (protonated)
        assert "-" not in result["smiles"]

    def test_remove_stereo(self):
        """Test removing stereochemistry."""
        from rdkit_cli.core.standardizer import MoleculeStandardizer
        from rdkit_cli.io.readers import MoleculeRecord

        std = MoleculeStandardizer(remove_stereo=True)

        # Molecule with stereochemistry
        smi = "C[C@H](O)CC"  # chiral carbon
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="chiral")
        result = std.standardize(record)

        assert result is not None
        # Should have no stereochemistry
        assert "@" not in result["smiles"]

    def test_include_original(self):
        """Test including original SMILES."""
        from rdkit_cli.core.standardizer import MoleculeStandardizer
        from rdkit_cli.io.readers import MoleculeRecord

        std = MoleculeStandardizer(include_original=True)

        smi = "c1ccccc1"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="benzene")
        result = std.standardize(record)

        assert result is not None
        assert "original_smiles" in result
        assert result["original_smiles"] == smi

    def test_none_molecule(self):
        """Test handling of None molecule."""
        from rdkit_cli.core.standardizer import MoleculeStandardizer
        from rdkit_cli.io.readers import MoleculeRecord

        std = MoleculeStandardizer()
        record = MoleculeRecord(mol=None, smiles="invalid")
        result = std.standardize(record)
        assert result is None


class TestCanonicalizeSmiles:
    """Test canonicalize_smiles function."""

    def test_canonicalize(self):
        """Test basic canonicalization."""
        from rdkit_cli.core.standardizer import canonicalize_smiles

        result = canonicalize_smiles("C(C)(C)C")
        assert result == "CC(C)C"

    def test_canonicalize_aromatic(self):
        """Test aromatic canonicalization."""
        from rdkit_cli.core.standardizer import canonicalize_smiles

        result = canonicalize_smiles("c1ccccc1")
        assert result == "c1ccccc1"  # Benzene is already canonical

    def test_invalid_smiles(self):
        """Test invalid SMILES returns None."""
        from rdkit_cli.core.standardizer import canonicalize_smiles

        result = canonicalize_smiles("not_a_smiles")
        assert result is None
