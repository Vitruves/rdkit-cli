"""Unit tests for MCS module."""

import pytest
from rdkit import Chem


class TestFindMCS:
    """Test find_mcs function."""

    def test_mcs_identical(self):
        """Test MCS of identical molecules."""
        from rdkit_cli.core.mcs import find_mcs

        mols = [
            Chem.MolFromSmiles("c1ccccc1"),
            Chem.MolFromSmiles("c1ccccc1"),
        ]

        result = find_mcs(mols)

        assert result is not None
        assert result["num_atoms"] == 6  # Benzene has 6 atoms

    def test_mcs_similar(self):
        """Test MCS of similar molecules."""
        from rdkit_cli.core.mcs import find_mcs

        mols = [
            Chem.MolFromSmiles("c1ccccc1"),  # benzene
            Chem.MolFromSmiles("Cc1ccccc1"),  # toluene
        ]

        result = find_mcs(mols)

        assert result is not None
        assert result["num_atoms"] == 6  # Common substructure is benzene

    def test_mcs_different(self):
        """Test MCS of different molecules."""
        from rdkit_cli.core.mcs import find_mcs

        mols = [
            Chem.MolFromSmiles("c1ccccc1"),  # benzene
            Chem.MolFromSmiles("CCCCCC"),  # hexane
        ]

        result = find_mcs(mols)

        assert result is not None
        # Should find some common substructure (carbon chain)

    def test_mcs_timeout(self):
        """Test MCS with timeout."""
        from rdkit_cli.core.mcs import find_mcs

        mols = [
            Chem.MolFromSmiles("c1ccccc1"),
            Chem.MolFromSmiles("c1ccccc1"),
        ]

        result = find_mcs(mols, timeout=1)

        assert result is not None

    def test_mcs_threshold(self):
        """Test MCS with threshold."""
        from rdkit_cli.core.mcs import find_mcs

        mols = [
            Chem.MolFromSmiles("c1ccccc1"),
            Chem.MolFromSmiles("Cc1ccccc1"),
            Chem.MolFromSmiles("CCc1ccccc1"),
        ]

        result = find_mcs(mols, threshold=0.5)

        assert result is not None

    def test_mcs_single_molecule(self):
        """Test MCS with single molecule."""
        from rdkit_cli.core.mcs import find_mcs

        mols = [Chem.MolFromSmiles("C")]

        result = find_mcs(mols)

        assert result is None  # Need at least 2 molecules

    def test_mcs_empty(self):
        """Test MCS with empty list."""
        from rdkit_cli.core.mcs import find_mcs

        result = find_mcs([])

        assert result is None

    def test_mcs_atom_compare_any(self):
        """Test MCS with any atom comparison."""
        from rdkit_cli.core.mcs import find_mcs

        mols = [
            Chem.MolFromSmiles("c1ccccc1"),
            Chem.MolFromSmiles("C1CCCCC1"),  # cyclohexane
        ]

        result = find_mcs(mols, atom_compare="any")

        assert result is not None
        # With "any" comparison, should find 6-membered ring

    def test_mcs_ring_options(self):
        """Test MCS with ring matching options."""
        from rdkit_cli.core.mcs import find_mcs

        mols = [
            Chem.MolFromSmiles("c1ccccc1C"),
            Chem.MolFromSmiles("c1ccccc1CC"),
        ]

        result = find_mcs(
            mols,
            ring_matches_ring_only=True,
            complete_rings_only=True,
        )

        assert result is not None


class TestMCSAligner:
    """Test MCSAligner class."""

    def test_find_common(self):
        """Test finding common substructure."""
        from rdkit_cli.core.mcs import MCSAligner

        aligner = MCSAligner(reference_smiles="c1ccccc1")

        mol = Chem.MolFromSmiles("Cc1ccccc1")
        result = aligner.find_common(mol)

        assert result is not None
        assert result["num_atoms"] == 6

    def test_invalid_reference(self):
        """Test invalid reference SMILES."""
        from rdkit_cli.core.mcs import MCSAligner

        with pytest.raises(ValueError, match="Invalid reference"):
            MCSAligner(reference_smiles="invalid_smiles")

    def test_none_query(self):
        """Test None query molecule."""
        from rdkit_cli.core.mcs import MCSAligner

        aligner = MCSAligner(reference_smiles="C")
        result = aligner.find_common(None)

        assert result is None
