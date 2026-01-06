"""Unit tests for fragment module."""

import pytest
from rdkit import Chem


class TestBRICSFragmenter:
    """Test BRICSFragmenter class."""

    def test_fragment_simple(self):
        """Test BRICS fragmentation of simple molecule."""
        from rdkit_cli.core.fragment import BRICSFragmenter
        from rdkit_cli.io.readers import MoleculeRecord

        fragmenter = BRICSFragmenter()

        # Aspirin
        smi = "CC(=O)OC1=CC=CC=C1C(=O)O"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="aspirin")
        results = fragmenter.fragment(record)

        assert len(results) >= 1
        assert all("fragment_smiles" in r for r in results)

    def test_min_fragment_size(self):
        """Test minimum fragment size filter."""
        from rdkit_cli.core.fragment import BRICSFragmenter
        from rdkit_cli.io.readers import MoleculeRecord

        fragmenter = BRICSFragmenter(min_fragment_size=5)

        smi = "CC(=O)OC1=CC=CC=C1C(=O)O"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="aspirin")
        results = fragmenter.fragment(record)

        # All fragments should have at least 5 heavy atoms
        for r in results:
            assert r.get("heavy_atom_count", 0) >= 5

    def test_none_molecule(self):
        """Test handling of None molecule."""
        from rdkit_cli.core.fragment import BRICSFragmenter
        from rdkit_cli.io.readers import MoleculeRecord

        fragmenter = BRICSFragmenter()
        record = MoleculeRecord(mol=None, smiles="invalid")
        results = fragmenter.fragment(record)
        assert results == []


class TestRECAPFragmenter:
    """Test RECAPFragmenter class."""

    def test_fragment_simple(self):
        """Test RECAP fragmentation."""
        from rdkit_cli.core.fragment import RECAPFragmenter
        from rdkit_cli.io.readers import MoleculeRecord

        fragmenter = RECAPFragmenter()

        # Molecule with amide bond
        smi = "CC(=O)NC1=CC=CC=C1"
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="acetanilide")
        results = fragmenter.fragment(record)

        assert len(results) >= 1

    def test_none_molecule(self):
        """Test handling of None molecule."""
        from rdkit_cli.core.fragment import RECAPFragmenter
        from rdkit_cli.io.readers import MoleculeRecord

        fragmenter = RECAPFragmenter()
        record = MoleculeRecord(mol=None, smiles="invalid")
        results = fragmenter.fragment(record)
        assert results == []


class TestFunctionalGroupExtractor:
    """Test FunctionalGroupExtractor class."""

    def test_extract_alcohol(self):
        """Test alcohol detection."""
        from rdkit_cli.core.fragment import FunctionalGroupExtractor
        from rdkit_cli.io.readers import MoleculeRecord

        extractor = FunctionalGroupExtractor()

        smi = "CCO"  # ethanol
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="ethanol")
        result = extractor.extract(record)

        assert result is not None
        assert result.get("n_alcohol", 0) >= 1

    def test_extract_carboxylic_acid(self):
        """Test carboxylic acid detection."""
        from rdkit_cli.core.fragment import FunctionalGroupExtractor
        from rdkit_cli.io.readers import MoleculeRecord

        extractor = FunctionalGroupExtractor()

        smi = "CC(=O)O"  # acetic acid
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="acetic_acid")
        result = extractor.extract(record)

        assert result is not None
        assert result.get("n_carboxylic_acid", 0) >= 1

    def test_extract_aromatic(self):
        """Test aromatic ring detection."""
        from rdkit_cli.core.fragment import FunctionalGroupExtractor
        from rdkit_cli.io.readers import MoleculeRecord

        extractor = FunctionalGroupExtractor()

        smi = "c1ccccc1"  # benzene
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name="benzene")
        result = extractor.extract(record)

        assert result is not None
        assert result.get("n_aromatic_ring", 0) >= 1

    def test_none_molecule(self):
        """Test handling of None molecule."""
        from rdkit_cli.core.fragment import FunctionalGroupExtractor
        from rdkit_cli.io.readers import MoleculeRecord

        extractor = FunctionalGroupExtractor()
        record = MoleculeRecord(mol=None, smiles="invalid")
        result = extractor.extract(record)
        assert result is None


class TestAnalyzeFragments:
    """Test analyze_fragments function."""

    def test_frequency_analysis(self):
        """Test fragment frequency analysis."""
        from rdkit_cli.core.fragment import analyze_fragments

        fragments = ["[*]C", "[*]C", "[*]C", "[*]O", "[*]O"]
        results = analyze_fragments(fragments, top_n=5)

        assert len(results) == 2
        assert results[0][0] == "[*]C"
        assert results[0][1] == 3

    def test_empty_list(self):
        """Test empty fragment list."""
        from rdkit_cli.core.fragment import analyze_fragments

        results = analyze_fragments([], top_n=10)
        assert len(results) == 0
