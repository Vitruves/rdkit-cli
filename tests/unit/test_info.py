"""Unit tests for info module."""

import pytest


class TestGetMoleculeInfo:
    """Test get_molecule_info function."""

    def test_basic_info(self):
        """Test basic molecule info extraction."""
        from rdkit_cli.core.info import get_molecule_info

        info = get_molecule_info("CCO")  # Ethanol

        assert info is not None
        assert info["canonical_smiles"] == "CCO"
        assert info["formula"] == "C2H6O"
        assert info["mol_weight"] == pytest.approx(46.07, rel=0.01)
        assert info["heavy_atom_count"] == 3
        assert info["hbd"] == 1
        assert info["hba"] == 1

    def test_complex_molecule(self):
        """Test with a more complex molecule."""
        from rdkit_cli.core.info import get_molecule_info

        info = get_molecule_info("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin

        assert info is not None
        assert info["ring_count"] == 1
        assert info["aromatic_ring_count"] == 1
        assert info["rotatable_bond_count"] >= 2
        assert "C" in info["elements"]
        assert "O" in info["elements"]

    def test_invalid_smiles(self):
        """Test with invalid SMILES."""
        from rdkit_cli.core.info import get_molecule_info

        info = get_molecule_info("not_a_smiles")
        assert info is None

    def test_inchi_generation(self):
        """Test InChI and InChIKey generation."""
        from rdkit_cli.core.info import get_molecule_info

        info = get_molecule_info("c1ccccc1")  # Benzene

        assert info is not None
        assert info["inchi"].startswith("InChI=")
        assert len(info["inchikey"]) == 27  # Standard InChIKey length

    def test_stereochemistry(self):
        """Test stereocenter detection."""
        from rdkit_cli.core.info import get_molecule_info

        # Molecule with chiral center
        info = get_molecule_info("C[C@H](O)CC")

        assert info is not None
        assert info["stereocenters"] >= 1

    def test_lipinski_violations(self):
        """Test Lipinski violation counting."""
        from rdkit_cli.core.info import get_molecule_info

        # Small molecule - no violations expected
        info = get_molecule_info("CCO")
        assert info["lipinski_violations"] == 0


class TestFormatInfo:
    """Test formatting functions."""

    def test_format_text(self):
        """Test text formatting."""
        from rdkit_cli.core.info import get_molecule_info, format_info_text

        info = get_molecule_info("CCO")
        text = format_info_text(info)

        assert "SMILES:" in text
        assert "Formula:" in text
        assert "Mol Weight:" in text

    def test_format_json(self):
        """Test JSON formatting."""
        import json
        from rdkit_cli.core.info import get_molecule_info, format_info_json

        info = get_molecule_info("CCO")
        json_str = format_info_json(info)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "canonical_smiles" in parsed
        assert "mol_weight" in parsed
