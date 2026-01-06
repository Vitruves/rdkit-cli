"""Unit tests for descriptors module."""

import pytest
from rdkit import Chem


class TestDescriptorCalculator:
    """Test DescriptorCalculator class."""

    def test_compute_single_descriptor(self, sample_molecules):
        """Test computing a single descriptor."""
        from rdkit_cli.core.descriptors import DescriptorCalculator
        from rdkit_cli.io.readers import MoleculeRecord

        calc = DescriptorCalculator(descriptors=["MolWt"])

        for name, smi in sample_molecules:
            mol = Chem.MolFromSmiles(smi)
            record = MoleculeRecord(mol=mol, smiles=smi, name=name)
            result = calc.compute(record)

            assert result is not None
            assert "MolWt" in result
            assert isinstance(result["MolWt"], float)
            assert result["MolWt"] > 0

    def test_compute_multiple_descriptors(self, sample_molecules):
        """Test computing multiple descriptors."""
        from rdkit_cli.core.descriptors import DescriptorCalculator
        from rdkit_cli.io.readers import MoleculeRecord

        descriptors = ["MolWt", "MolLogP", "TPSA"]
        calc = DescriptorCalculator(descriptors=descriptors)

        name, smi = sample_molecules[0]
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)
        result = calc.compute(record)

        for desc in descriptors:
            assert desc in result

    def test_compute_all_descriptors(self, sample_molecules):
        """Test computing all descriptors."""
        from rdkit_cli.core.descriptors import DescriptorCalculator
        from rdkit_cli.io.readers import MoleculeRecord

        calc = DescriptorCalculator()  # All by default

        name, smi = sample_molecules[0]
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)
        result = calc.compute(record)

        # Should have 100+ descriptors
        assert len(result) > 100

    def test_invalid_descriptor_name(self):
        """Test that invalid descriptor names raise error."""
        from rdkit_cli.core.descriptors import DescriptorCalculator

        with pytest.raises(ValueError, match="Unknown descriptor"):
            DescriptorCalculator(descriptors=["NotADescriptor"])

    def test_none_molecule(self):
        """Test handling of None molecule."""
        from rdkit_cli.core.descriptors import DescriptorCalculator
        from rdkit_cli.io.readers import MoleculeRecord

        calc = DescriptorCalculator(descriptors=["MolWt"])
        record = MoleculeRecord(mol=None, smiles="invalid")
        result = calc.compute(record)
        assert result is None


class TestListDescriptors:
    """Test list_descriptors function."""

    def test_list_all(self):
        """Test listing all descriptors."""
        from rdkit_cli.core.descriptors import list_descriptors

        descriptors = list_descriptors()
        assert len(descriptors) > 100

    def test_list_by_category(self):
        """Test listing descriptors by category."""
        from rdkit_cli.core.descriptors import list_descriptors

        constitutional = list_descriptors(category="constitutional")
        assert len(constitutional) > 0
