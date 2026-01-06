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


class TestElementFilter:
    """Test ElementFilter class."""

    def test_allowed_elements_pass(self):
        """Test molecule with only allowed elements passes."""
        from rdkit_cli.core.filters import ElementFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ElementFilter(allowed_elements=["C", "H", "O"])

        # Ethanol only has C, H, O
        smi = "CCO"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="ethanol")
        result = filt.filter(record)
        assert result is not None

    def test_allowed_elements_fail(self):
        """Test molecule with disallowed elements fails."""
        from rdkit_cli.core.filters import ElementFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ElementFilter(allowed_elements=["C", "H", "O"])

        # Molecule with nitrogen should fail
        smi = "CCN"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="ethylamine")
        result = filt.filter(record)
        assert result is None

    def test_required_elements_pass(self):
        """Test molecule with required elements passes."""
        from rdkit_cli.core.filters import ElementFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ElementFilter(required_elements=["N"])

        # Molecule with nitrogen should pass
        smi = "CCN"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="ethylamine")
        result = filt.filter(record)
        assert result is not None

    def test_required_elements_fail(self):
        """Test molecule without required elements fails."""
        from rdkit_cli.core.filters import ElementFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ElementFilter(required_elements=["N"])

        # Ethanol has no nitrogen
        smi = "CCO"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="ethanol")
        result = filt.filter(record)
        assert result is None

    def test_forbidden_elements_pass(self):
        """Test molecule without forbidden elements passes."""
        from rdkit_cli.core.filters import ElementFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ElementFilter(forbidden_elements=["Cl", "Br", "I"])

        # Ethanol has no halogens
        smi = "CCO"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="ethanol")
        result = filt.filter(record)
        assert result is not None

    def test_forbidden_elements_fail(self):
        """Test molecule with forbidden elements fails."""
        from rdkit_cli.core.filters import ElementFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ElementFilter(forbidden_elements=["Cl", "Br", "I"])

        # Chloroethane has chlorine
        smi = "CCCl"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="chloroethane")
        result = filt.filter(record)
        assert result is None

    def test_combined_element_filters(self):
        """Test combined element constraints."""
        from rdkit_cli.core.filters import ElementFilter
        from rdkit_cli.io.readers import MoleculeRecord

        # Only allow C, H, O, N and require N, forbid S
        filt = ElementFilter(
            allowed_elements=["C", "H", "O", "N"],
            required_elements=["N"],
            forbidden_elements=["S"],
        )

        # Ethylamine: C, H, N - should pass
        smi = "CCN"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="ethylamine")
        result = filt.filter(record)
        assert result is not None

        # Methanethiol: C, H, S - should fail (has S, no N)
        smi = "CS"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="methanethiol")
        result = filt.filter(record)
        assert result is None

    def test_none_molecule(self):
        """Test handling of None molecule."""
        from rdkit_cli.core.filters import ElementFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ElementFilter(allowed_elements=["C", "H"])
        record = MoleculeRecord(mol=None, smiles="invalid")
        result = filt.filter(record)
        assert result is None


class TestComplexityFilter:
    """Test ComplexityFilter class."""

    def test_atom_count_pass(self):
        """Test molecule within atom count range passes."""
        from rdkit_cli.core.filters import ComplexityFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ComplexityFilter(min_atoms=2, max_atoms=10)

        smi = "CCO"  # 3 heavy atoms
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="ethanol")
        result = filt.filter(record)
        assert result is not None

    def test_atom_count_too_few(self):
        """Test molecule with too few atoms fails."""
        from rdkit_cli.core.filters import ComplexityFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ComplexityFilter(min_atoms=5, max_atoms=100)

        smi = "CO"  # 2 heavy atoms
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="methanol")
        result = filt.filter(record)
        assert result is None

    def test_atom_count_too_many(self):
        """Test molecule with too many atoms fails."""
        from rdkit_cli.core.filters import ComplexityFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ComplexityFilter(min_atoms=1, max_atoms=5)

        smi = "CCCCCCCCCC"  # 10 heavy atoms
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="decane")
        result = filt.filter(record)
        assert result is None

    def test_ring_count_pass(self):
        """Test molecule within ring count range passes."""
        from rdkit_cli.core.filters import ComplexityFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ComplexityFilter(min_rings=1, max_rings=3)

        smi = "c1ccccc1"  # 1 ring
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="benzene")
        result = filt.filter(record)
        assert result is not None

    def test_ring_count_too_few(self):
        """Test molecule with too few rings fails."""
        from rdkit_cli.core.filters import ComplexityFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ComplexityFilter(min_rings=1, max_rings=10)

        smi = "CCCC"  # 0 rings
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="butane")
        result = filt.filter(record)
        assert result is None

    def test_ring_count_too_many(self):
        """Test molecule with too many rings fails."""
        from rdkit_cli.core.filters import ComplexityFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ComplexityFilter(min_rings=0, max_rings=1)

        # Naphthalene has 2 rings
        smi = "c1ccc2ccccc2c1"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="naphthalene")
        result = filt.filter(record)
        assert result is None

    def test_rotatable_bonds_pass(self):
        """Test molecule within rotatable bond range passes."""
        from rdkit_cli.core.filters import ComplexityFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ComplexityFilter(min_rotatable=0, max_rotatable=5)

        smi = "CCCC"  # 1 rotatable bond
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="butane")
        result = filt.filter(record)
        assert result is not None

    def test_rotatable_bonds_too_many(self):
        """Test molecule with too many rotatable bonds fails."""
        from rdkit_cli.core.filters import ComplexityFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ComplexityFilter(min_rotatable=0, max_rotatable=3)

        # Octane has many rotatable bonds
        smi = "CCCCCCCC"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="octane")
        result = filt.filter(record)
        assert result is None

    def test_combined_complexity_filter(self):
        """Test combined complexity constraints."""
        from rdkit_cli.core.filters import ComplexityFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ComplexityFilter(
            min_atoms=5,
            max_atoms=15,
            min_rings=1,
            max_rings=2,
            min_rotatable=0,
            max_rotatable=3,
        )

        # Toluene: 7 atoms, 1 ring, 0 rotatable
        smi = "Cc1ccccc1"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="toluene")
        result = filt.filter(record)
        assert result is not None

    def test_none_molecule(self):
        """Test handling of None molecule."""
        from rdkit_cli.core.filters import ComplexityFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = ComplexityFilter()
        record = MoleculeRecord(mol=None, smiles="invalid")
        result = filt.filter(record)
        assert result is None


class TestPAINSFilterExcludeMode:
    """Test PAINSFilter with exclude mode toggled."""

    def test_exclude_false_keeps_pains(self):
        """Test that exclude=False keeps PAINS molecules."""
        from rdkit_cli.core.filters import PAINSFilter
        from rdkit_cli.io.readers import MoleculeRecord

        # With exclude=False, clean molecules should be excluded
        filt = PAINSFilter(exclude=False)

        # Simple molecule (no PAINS) should fail when exclude=False
        smi = "CCO"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="ethanol")
        result = filt.filter(record)
        assert result is None  # Excluded because it's NOT a PAINS hit

    def test_exclude_true_removes_pains(self):
        """Test that exclude=True (default) removes PAINS molecules."""
        from rdkit_cli.core.filters import PAINSFilter
        from rdkit_cli.io.readers import MoleculeRecord

        filt = PAINSFilter(exclude=True)

        # Simple molecule (no PAINS) should pass
        smi = "CCO"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="ethanol")
        result = filt.filter(record)
        assert result is not None

    def test_known_pains_pattern(self):
        """Test a known PAINS pattern."""
        from rdkit_cli.core.filters import PAINSFilter
        from rdkit_cli.io.readers import MoleculeRecord

        # Rhodanine is a known PAINS pattern
        rhodanine = "O=C1NC(=S)SC1"
        mol = Chem.MolFromSmiles(rhodanine)
        record = MoleculeRecord(mol=mol, smiles=rhodanine, name="rhodanine")

        # Should be excluded with exclude=True
        filt_exclude = PAINSFilter(exclude=True)
        result = filt_exclude.filter(record)
        assert result is None  # PAINS hit is excluded

        # Should be kept with exclude=False
        filt_keep = PAINSFilter(exclude=False)
        result = filt_keep.filter(record)
        assert result is not None  # PAINS hit is kept
