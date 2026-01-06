"""Unit tests for depict module."""

import pytest
from rdkit import Chem


class TestMoleculeDepiction:
    """Test MoleculeDepiction class."""

    def test_depict_svg(self):
        """Test SVG depiction."""
        from rdkit_cli.core.depict import MoleculeDepiction

        mol = Chem.MolFromSmiles("c1ccccc1")
        depictor = MoleculeDepiction(image_format="svg")
        result = depictor.depict(mol)

        assert result is not None
        assert "<svg" in result
        assert "</svg>" in result

    def test_depict_png(self):
        """Test PNG depiction."""
        from rdkit_cli.core.depict import MoleculeDepiction

        mol = Chem.MolFromSmiles("c1ccccc1")
        depictor = MoleculeDepiction(image_format="png")
        result = depictor.depict(mol)

        assert result is not None
        # PNG files start with specific bytes
        assert result[:4] == b'\x89PNG'

    def test_depict_with_size(self):
        """Test depiction with custom size."""
        from rdkit_cli.core.depict import MoleculeDepiction

        mol = Chem.MolFromSmiles("c1ccccc1")
        depictor = MoleculeDepiction(width=500, height=500, image_format="svg")
        result = depictor.depict(mol)

        assert result is not None
        assert "500" in result  # Width should be in SVG

    def test_depict_none_molecule(self):
        """Test depiction of None molecule."""
        from rdkit_cli.core.depict import MoleculeDepiction

        depictor = MoleculeDepiction()
        result = depictor.depict(None)

        assert result is None

    def test_depict_record(self):
        """Test depiction of molecule record."""
        from rdkit_cli.core.depict import MoleculeDepiction
        from rdkit_cli.io.readers import MoleculeRecord

        mol = Chem.MolFromSmiles("CCO")
        record = MoleculeRecord(mol=mol, smiles="CCO", name="ethanol")

        depictor = MoleculeDepiction()
        result = depictor.depict_record(record)

        assert result is not None
        assert result["smiles"] == "CCO"
        assert "image" in result


class TestGridDepiction:
    """Test GridDepiction class."""

    def test_grid_svg(self):
        """Test SVG grid depiction."""
        from rdkit_cli.core.depict import GridDepiction

        mols = [
            Chem.MolFromSmiles("C"),
            Chem.MolFromSmiles("CC"),
            Chem.MolFromSmiles("CCC"),
        ]

        grid = GridDepiction(mols_per_row=2, use_svg=True)
        result = grid.depict(mols)

        assert result is not None
        assert "<svg" in result

    def test_grid_with_legends(self):
        """Test grid with legends."""
        from rdkit_cli.core.depict import GridDepiction

        mols = [
            Chem.MolFromSmiles("C"),
            Chem.MolFromSmiles("CC"),
        ]
        legends = ["methane", "ethane"]

        grid = GridDepiction(legends=legends, use_svg=True)
        result = grid.depict(mols)

        assert result is not None

    def test_grid_empty(self):
        """Test empty grid."""
        from rdkit_cli.core.depict import GridDepiction

        grid = GridDepiction()
        result = grid.depict([])

        assert result is None

    def test_grid_with_none(self):
        """Test grid with None molecules."""
        from rdkit_cli.core.depict import GridDepiction

        mols = [
            Chem.MolFromSmiles("C"),
            None,
            Chem.MolFromSmiles("CC"),
        ]

        grid = GridDepiction(use_svg=True)
        result = grid.depict(mols)

        assert result is not None


class TestDepictSmiles:
    """Test depict_smiles function."""

    def test_depict_simple(self):
        """Test simple SMILES depiction."""
        from rdkit_cli.core.depict import depict_smiles

        result = depict_smiles("c1ccccc1")

        assert result is not None
        assert "<svg" in result

    def test_depict_invalid(self):
        """Test invalid SMILES depiction."""
        from rdkit_cli.core.depict import depict_smiles

        result = depict_smiles("invalid_smiles")

        assert result is None

    def test_depict_custom_size(self):
        """Test custom size depiction."""
        from rdkit_cli.core.depict import depict_smiles

        result = depict_smiles("C", width=200, height=200)

        assert result is not None
