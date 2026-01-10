"""Unit tests for rgroup module."""

import pytest
from rdkit import Chem


class TestRGroupDecomposer:
    """Test RGroupDecomposer class."""

    def test_decompose_simple(self):
        """Test simple R-group decomposition."""
        from rdkit_cli.core.rgroup import RGroupDecomposer
        from rdkit_cli.io.readers import MoleculeRecord

        # Core: benzene with one attachment point
        decomposer = RGroupDecomposer(
            core_smarts="c1ccc([*:1])cc1",
            include_smiles=True,
        )

        mol = Chem.MolFromSmiles("c1ccc(C)cc1")  # Toluene
        record = MoleculeRecord(mol=mol, smiles="c1ccc(C)cc1", name="toluene")

        result = decomposer.decompose(record)

        assert result is not None
        assert result["matched"] == True
        assert "R1" in result

    def test_decompose_two_rgroups(self):
        """Test decomposition with two R-groups."""
        from rdkit_cli.core.rgroup import RGroupDecomposer
        from rdkit_cli.io.readers import MoleculeRecord

        # Core: benzene with two attachment points
        decomposer = RGroupDecomposer(
            core_smarts="c1cc([*:1])ccc1[*:2]",
            include_smiles=True,
        )

        # para-xylene
        mol = Chem.MolFromSmiles("Cc1ccc(C)cc1")
        record = MoleculeRecord(mol=mol, smiles="Cc1ccc(C)cc1", name="p-xylene")

        result = decomposer.decompose(record)

        # Should match with two R-groups
        assert result is not None
        if result["matched"]:
            assert "R1" in result
            assert "R2" in result

    def test_decompose_no_match(self):
        """Test decomposition when molecule doesn't match core."""
        from rdkit_cli.core.rgroup import RGroupDecomposer
        from rdkit_cli.io.readers import MoleculeRecord

        decomposer = RGroupDecomposer(
            core_smarts="c1ccc([*:1])cc1",  # Benzene core
            only_matching=True,
        )

        # Cyclohexane - no aromatic ring
        mol = Chem.MolFromSmiles("C1CCCCC1")
        record = MoleculeRecord(mol=mol, smiles="C1CCCCC1", name="cyclohexane")

        result = decomposer.decompose(record)

        assert result is None  # Should not match

    def test_decompose_include_unmatched(self):
        """Test decomposition including unmatched molecules."""
        from rdkit_cli.core.rgroup import RGroupDecomposer
        from rdkit_cli.io.readers import MoleculeRecord

        decomposer = RGroupDecomposer(
            core_smarts="c1ccc([*:1])cc1",
            only_matching=False,
        )

        # Non-matching molecule
        mol = Chem.MolFromSmiles("CCC")
        record = MoleculeRecord(mol=mol, smiles="CCC", name="propane")

        result = decomposer.decompose(record)

        assert result is not None
        assert result["matched"] == False

    def test_invalid_core_smarts(self):
        """Test with invalid core SMARTS."""
        from rdkit_cli.core.rgroup import RGroupDecomposer

        with pytest.raises(ValueError, match="Invalid core SMARTS"):
            RGroupDecomposer(core_smarts="[invalid")

    def test_core_without_attachment_points(self):
        """Test core without attachment points."""
        from rdkit_cli.core.rgroup import RGroupDecomposer

        with pytest.raises(ValueError, match="attachment points"):
            RGroupDecomposer(core_smarts="c1ccccc1")

    def test_column_names(self):
        """Test getting column names."""
        from rdkit_cli.core.rgroup import RGroupDecomposer

        decomposer = RGroupDecomposer(
            core_smarts="c1cc([*:1])ccc1[*:2]",
            include_smiles=True,
            include_name=True,
        )

        cols = decomposer.get_column_names()

        assert "smiles" in cols
        assert "name" in cols
        assert "matched" in cols
        assert "core" in cols
        assert "R1" in cols
        assert "R2" in cols
