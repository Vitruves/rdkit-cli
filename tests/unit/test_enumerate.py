"""Unit tests for enumerate module."""

import pytest
from rdkit import Chem


class TestStereoisomerEnumerator:
    """Test StereoisomerEnumerator class."""

    def test_enumerate_chiral_center(self):
        """Test enumeration of chiral centers."""
        from rdkit_cli.core.enumerate import StereoisomerEnumerator
        from rdkit_cli.io.readers import MoleculeRecord

        enumerator = StereoisomerEnumerator(max_isomers=10)

        # Molecule with undefined stereocenter
        smi = "CC(O)C"  # 2-propanol, achiral
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="propanol")
        results = enumerator.enumerate(record)

        # Should return at least one result
        assert len(results) >= 1

    def test_enumerate_double_bond(self):
        """Test enumeration of E/Z isomers."""
        from rdkit_cli.core.enumerate import StereoisomerEnumerator
        from rdkit_cli.io.readers import MoleculeRecord

        enumerator = StereoisomerEnumerator(max_isomers=10)

        # But-2-ene without specified stereochemistry
        smi = "CC=CC"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="butene")
        results = enumerator.enumerate(record)

        assert len(results) >= 1

    def test_max_isomers_limit(self):
        """Test max isomers limiting."""
        from rdkit_cli.core.enumerate import StereoisomerEnumerator
        from rdkit_cli.io.readers import MoleculeRecord

        enumerator = StereoisomerEnumerator(max_isomers=2)

        # Molecule with multiple stereocenters
        smi = "CC(O)C(O)C(O)C"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="triol")
        results = enumerator.enumerate(record)

        assert len(results) <= 2

    def test_none_molecule(self):
        """Test handling of None molecule."""
        from rdkit_cli.core.enumerate import StereoisomerEnumerator
        from rdkit_cli.io.readers import MoleculeRecord

        enumerator = StereoisomerEnumerator()
        record = MoleculeRecord(mol=None, smiles="invalid")
        results = enumerator.enumerate(record)
        assert results == []


class TestTautomerEnumerator:
    """Test TautomerEnumerator class."""

    def test_enumerate_tautomers(self):
        """Test tautomer enumeration."""
        from rdkit_cli.core.enumerate import TautomerEnumerator
        from rdkit_cli.io.readers import MoleculeRecord

        enumerator = TautomerEnumerator(max_tautomers=10)

        # Phenol - can have tautomers
        smi = "Oc1ccccc1"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="phenol")
        results = enumerator.enumerate(record)

        assert len(results) >= 1
        assert all("smiles" in r for r in results)

    def test_keto_enol(self):
        """Test keto-enol tautomerism."""
        from rdkit_cli.core.enumerate import TautomerEnumerator
        from rdkit_cli.io.readers import MoleculeRecord

        enumerator = TautomerEnumerator(max_tautomers=20)

        # Acetone - can have enol form
        smi = "CC(=O)C"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="acetone")
        results = enumerator.enumerate(record)

        assert len(results) >= 1

    def test_none_molecule(self):
        """Test handling of None molecule."""
        from rdkit_cli.core.enumerate import TautomerEnumerator
        from rdkit_cli.io.readers import MoleculeRecord

        enumerator = TautomerEnumerator()
        record = MoleculeRecord(mol=None, smiles="invalid")
        results = enumerator.enumerate(record)
        assert results == []


class TestCanonicalTautomerizer:
    """Test CanonicalTautomerizer class."""

    def test_canonicalize(self):
        """Test canonical tautomer generation."""
        from rdkit_cli.core.enumerate import CanonicalTautomerizer
        from rdkit_cli.io.readers import MoleculeRecord

        canonicalizer = CanonicalTautomerizer()

        smi = "Oc1ccccc1"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="phenol")
        result = canonicalizer.canonicalize(record)

        assert result is not None
        assert "smiles" in result

    def test_include_original(self):
        """Test including original SMILES."""
        from rdkit_cli.core.enumerate import CanonicalTautomerizer
        from rdkit_cli.io.readers import MoleculeRecord

        canonicalizer = CanonicalTautomerizer(include_original=True)

        smi = "CC(=O)C"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="acetone")
        result = canonicalizer.canonicalize(record)

        assert result is not None
        assert "original_smiles" in result

    def test_none_molecule(self):
        """Test handling of None molecule."""
        from rdkit_cli.core.enumerate import CanonicalTautomerizer
        from rdkit_cli.io.readers import MoleculeRecord

        canonicalizer = CanonicalTautomerizer()
        record = MoleculeRecord(mol=None, smiles="invalid")
        result = canonicalizer.canonicalize(record)
        assert result is None
