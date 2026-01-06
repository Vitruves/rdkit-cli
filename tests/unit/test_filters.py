"""Unit tests for filters module."""

import pytest
from rdkit import Chem


class TestSubstructureFilter:
    """Test SubstructureFilter class."""

    def test_filter_benzene_ring(self, sample_molecules):
        """Test filtering for benzene ring."""
        from rdkit_cli.core.filters import SubstructureFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = SubstructureFilter(smarts="c1ccccc1")

        # Aspirin has benzene ring
        name, smi = sample_molecules[0]
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)
        result = filt.filter(record)
        assert result is not None

        # Ethanol has no benzene ring
        ethanol_smi = "CCO"
        mol = Chem.MolFromSmiles(ethanol_smi)
        record = MoleculeRecord(mol=mol, smiles=ethanol_smi, name="ethanol")
        result = filt.filter(record)
        assert result is None

    def test_filter_exclude(self, sample_molecules):
        """Test exclude mode."""
        from rdkit_cli.core.filters import SubstructureFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = SubstructureFilter(smarts="c1ccccc1", exclude=True)

        # Aspirin has benzene ring - should be excluded
        name, smi = sample_molecules[0]
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)
        result = filt.filter(record)
        assert result is None

        # Ethanol has no benzene ring - should pass
        ethanol_smi = "CCO"
        mol = Chem.MolFromSmiles(ethanol_smi)
        record = MoleculeRecord(mol=mol, smiles=ethanol_smi, name="ethanol")
        result = filt.filter(record)
        assert result is not None

    def test_invalid_smarts(self):
        """Test invalid SMARTS raises error."""
        from rdkit_cli.core.filters import SubstructureFilter

        with pytest.raises(ValueError, match="Invalid SMARTS"):
            SubstructureFilter(smarts="not_valid_smarts((")

    def test_none_molecule(self):
        """Test handling of None molecule."""
        from rdkit_cli.core.filters import SubstructureFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = SubstructureFilter(smarts="C")
        record = MoleculeRecord(mol=None, smiles="invalid")
        result = filt.filter(record)
        assert result is None


class TestDruglikeFilter:
    """Test DruglikeFilter class."""

    def test_lipinski_filter(self, sample_molecules):
        """Test Lipinski filter."""
        from rdkit_cli.core.filters import DruglikeFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = DruglikeFilter(rule_name="lipinski")

        # Small drug-like molecules should pass
        name, smi = sample_molecules[0]  # aspirin
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)
        result = filt.filter(record)
        assert result is not None

    def test_veber_filter(self, sample_molecules):
        """Test Veber filter."""
        from rdkit_cli.core.filters import DruglikeFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = DruglikeFilter(rule_name="veber")

        name, smi = sample_molecules[0]  # aspirin
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)
        result = filt.filter(record)
        assert result is not None

    def test_max_violations(self):
        """Test max violations parameter."""
        from rdkit_cli.core.filters import DruglikeFilter
        from rdkit_cli.io.readers import MoleculeRecord

        # Large molecule that likely violates rules
        big_smi = "CC" * 50  # Long alkane chain
        mol = Chem.MolFromSmiles(big_smi)
        record = MoleculeRecord(mol=mol, smiles=big_smi, name="big")

        # Strict filter should reject
        filt_strict = DruglikeFilter(rule_name="lipinski", max_violations=0)
        assert filt_strict.filter(record) is None

        # Permissive filter might pass
        filt_permissive = DruglikeFilter(rule_name="lipinski", max_violations=4)
        # This may or may not pass depending on violations

    def test_unknown_rule(self):
        """Test unknown rule raises error."""
        from rdkit_cli.core.filters import DruglikeFilter

        with pytest.raises(ValueError, match="Unknown rule"):
            DruglikeFilter(rule_name="not_a_rule")


class TestPropertyFilter:
    """Test PropertyFilter class."""

    def test_mw_range(self, sample_molecules):
        """Test molecular weight range filter."""
        from rdkit_cli.core.filters import PropertyFilter
        from rdkit_cli.io.readers import MoleculeRecord

        # Filter for molecules with MW 100-300
        filt = PropertyFilter(rules={"MolWt": (100, 300)})

        name, smi = sample_molecules[0]  # aspirin ~180 Da
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)
        result = filt.filter(record)
        assert result is not None

        # Benzene ~78 Da should fail
        benzene_smi = "c1ccccc1"
        mol = Chem.MolFromSmiles(benzene_smi)
        record = MoleculeRecord(mol=mol, smiles=benzene_smi, name="benzene")
        result = filt.filter(record)
        assert result is None

    def test_multiple_rules(self, sample_molecules):
        """Test multiple property rules."""
        from rdkit_cli.core.filters import PropertyFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = PropertyFilter(rules={
            "MolWt": (100, 500),
            "NumHDonors": (None, 5),
        })

        name, smi = sample_molecules[0]  # aspirin
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)
        result = filt.filter(record)
        assert result is not None


class TestPAINSFilter:
    """Test PAINSFilter class."""

    def test_clean_molecule_passes(self, sample_molecules):
        """Test that clean molecules pass PAINS filter."""
        from rdkit_cli.core.filters import PAINSFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = PAINSFilter()

        # Most simple drug molecules should pass
        name, smi = sample_molecules[0]  # aspirin
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)
        result = filt.filter(record)
        assert result is not None


class TestCheckDruglikeRules:
    """Test check_druglike_rules function."""

    def test_lipinski_pass(self):
        """Test Lipinski rules with passing molecule."""
        from rdkit_cli.core.filters import check_druglike_rules

        mol = Chem.MolFromSmiles("CCO")  # ethanol
        result = check_druglike_rules(mol, "lipinski")
        assert result.passed is True

    def test_lipinski_fail(self):
        """Test Lipinski rules with failing molecule."""
        from rdkit_cli.core.filters import check_druglike_rules

        # Very large molecule
        mol = Chem.MolFromSmiles("C" * 60)
        result = check_druglike_rules(mol, "lipinski")
        assert result.passed is False

    def test_unknown_rule(self):
        """Test unknown rule raises error."""
        from rdkit_cli.core.filters import check_druglike_rules

        mol = Chem.MolFromSmiles("C")
        with pytest.raises(ValueError, match="Unknown rule"):
            check_druglike_rules(mol, "not_a_rule")
