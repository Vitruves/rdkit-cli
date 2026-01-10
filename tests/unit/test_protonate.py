"""Unit tests for protonate module."""

import pytest
from rdkit import Chem


class TestProtonationSites:
    """Test protonation site detection."""

    def test_detect_carboxylic_acid(self):
        """Test detecting carboxylic acid sites."""
        from rdkit_cli.core.protonate import get_protonation_sites

        mol = Chem.MolFromSmiles("CC(=O)O")  # Acetic acid
        sites = get_protonation_sites(mol)

        assert len(sites) >= 1
        # Should detect carboxylic acid
        pkas = [s["pka"] for s in sites]
        assert any(3 < pka < 6 for pka in pkas)

    def test_detect_amine(self):
        """Test detecting amine sites."""
        from rdkit_cli.core.protonate import get_protonation_sites

        mol = Chem.MolFromSmiles("CCN")  # Ethylamine
        sites = get_protonation_sites(mol)

        assert len(sites) >= 1
        # Should detect amine
        pkas = [s["pka"] for s in sites]
        assert any(pka > 9 for pka in pkas)

    def test_no_protonation_sites(self):
        """Test molecule without protonation sites."""
        from rdkit_cli.core.protonate import get_protonation_sites

        mol = Chem.MolFromSmiles("CCCC")  # Butane
        sites = get_protonation_sites(mol)

        assert len(sites) == 0


class TestEnumerateProtonationStates:
    """Test protonation state enumeration."""

    def test_enumerate_acid(self):
        """Test enumerating protonation states for acid."""
        from rdkit_cli.core.protonate import enumerate_protonation_states

        mol = Chem.MolFromSmiles("CC(=O)O")  # Acetic acid
        states = enumerate_protonation_states(mol, target_ph=7.4)

        assert len(states) >= 1

    def test_enumerate_amine(self):
        """Test enumerating protonation states for amine."""
        from rdkit_cli.core.protonate import enumerate_protonation_states

        mol = Chem.MolFromSmiles("CCN")
        states = enumerate_protonation_states(mol, target_ph=7.4)

        assert len(states) >= 1

    def test_enumerate_none_molecule(self):
        """Test with None molecule."""
        from rdkit_cli.core.protonate import enumerate_protonation_states

        states = enumerate_protonation_states(None)
        assert states == []


class TestProtonateAtPH:
    """Test pH-based protonation."""

    def test_protonate_acidic_ph(self):
        """Test protonation at acidic pH."""
        from rdkit_cli.core.protonate import protonate_at_ph

        mol = Chem.MolFromSmiles("CC(=O)O")
        result = protonate_at_ph(mol, ph=2.0)

        assert result is not None

    def test_protonate_basic_ph(self):
        """Test protonation at basic pH."""
        from rdkit_cli.core.protonate import protonate_at_ph

        mol = Chem.MolFromSmiles("CCN")
        result = protonate_at_ph(mol, ph=12.0)

        assert result is not None


class TestNeutralize:
    """Test neutralization function."""

    def test_neutralize_anion(self):
        """Test neutralizing anionic molecule."""
        from rdkit_cli.core.protonate import neutralize_mol

        mol = Chem.MolFromSmiles("CC(=O)[O-]")  # Acetate
        result = neutralize_mol(mol)

        assert result is not None
        # Should be neutralized
        charge = Chem.GetFormalCharge(result)
        assert charge == 0

    def test_neutralize_cation(self):
        """Test neutralizing cationic molecule."""
        from rdkit_cli.core.protonate import neutralize_mol

        mol = Chem.MolFromSmiles("CC[NH3+]")  # Ethylammonium
        result = neutralize_mol(mol)

        assert result is not None
        charge = Chem.GetFormalCharge(result)
        assert charge == 0

    def test_neutralize_neutral(self):
        """Test neutralizing already neutral molecule."""
        from rdkit_cli.core.protonate import neutralize_mol

        mol = Chem.MolFromSmiles("CCCC")
        result = neutralize_mol(mol)

        assert result is not None
        charge = Chem.GetFormalCharge(result)
        assert charge == 0


class TestProtonationEnumerator:
    """Test ProtonationEnumerator class."""

    def test_enumerate(self, sample_molecules):
        """Test protonation enumeration."""
        from rdkit_cli.core.protonate import ProtonationEnumerator
        from rdkit_cli.io.readers import MoleculeRecord

        enumerator = ProtonationEnumerator(
            ph=7.4,
            enumerate_all=False,
        )

        name, smi = sample_molecules[0]
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)

        results = enumerator.enumerate(record)

        assert isinstance(results, list)
        for r in results:
            assert "protonated_smiles" in r
            assert "ph" in r

    def test_enumerate_invalid(self):
        """Test enumeration with invalid molecule."""
        from rdkit_cli.core.protonate import ProtonationEnumerator
        from rdkit_cli.io.readers import MoleculeRecord

        enumerator = ProtonationEnumerator()
        record = MoleculeRecord(mol=None, smiles="invalid")

        results = enumerator.enumerate(record)
        assert results == []
