"""Unit tests for scaffold module."""

import pytest
from rdkit import Chem


class TestScaffoldExtractor:
    """Test ScaffoldExtractor class."""

    def test_extract_murcko_scaffold(self, sample_molecules):
        """Test Murcko scaffold extraction."""
        from rdkit_cli.core.scaffold import ScaffoldExtractor
        from rdkit_cli.io.readers import MoleculeRecord

        extractor = ScaffoldExtractor()

        name, smi = sample_molecules[0]  # aspirin
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)
        result = extractor.extract(record)

        assert result is not None
        assert "scaffold" in result
        assert len(result["scaffold"]) > 0

    def test_extract_generic_scaffold(self, sample_molecules):
        """Test generic scaffold extraction."""
        from rdkit_cli.core.scaffold import ScaffoldExtractor
        from rdkit_cli.io.readers import MoleculeRecord

        extractor = ScaffoldExtractor(generic=True)

        name, smi = sample_molecules[0]  # aspirin
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)
        result = extractor.extract(record)

        assert result is not None
        assert "scaffold" in result
        # Generic scaffold should only have C atoms (all elements -> C)
        scaffold = result["scaffold"]
        assert "c" in scaffold.lower() or "C" in scaffold

    def test_preserve_metadata(self, sample_molecules):
        """Test that SMILES and name are preserved."""
        from rdkit_cli.core.scaffold import ScaffoldExtractor
        from rdkit_cli.io.readers import MoleculeRecord

        extractor = ScaffoldExtractor(include_smiles=True, include_name=True)

        name, smi = sample_molecules[0]
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)
        result = extractor.extract(record)

        assert result is not None
        assert result["smiles"] == smi
        assert result["name"] == name

    def test_none_molecule(self):
        """Test handling of None molecule."""
        from rdkit_cli.core.scaffold import ScaffoldExtractor
        from rdkit_cli.io.readers import MoleculeRecord

        extractor = ScaffoldExtractor()
        record = MoleculeRecord(mol=None, smiles="invalid")
        result = extractor.extract(record)
        assert result is None


class TestScaffoldDecomposer:
    """Test ScaffoldDecomposer class."""

    def test_decompose(self, sample_molecules):
        """Test scaffold decomposition."""
        from rdkit_cli.core.scaffold import ScaffoldDecomposer
        from rdkit_cli.io.readers import MoleculeRecord

        decomposer = ScaffoldDecomposer()

        name, smi = sample_molecules[0]  # aspirin
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)
        result = decomposer.decompose(record)

        assert result is not None
        assert "scaffold" in result
        assert "generic_scaffold" in result

    def test_none_molecule(self):
        """Test handling of None molecule."""
        from rdkit_cli.core.scaffold import ScaffoldDecomposer
        from rdkit_cli.io.readers import MoleculeRecord

        decomposer = ScaffoldDecomposer()
        record = MoleculeRecord(mol=None, smiles="invalid")
        result = decomposer.decompose(record)
        assert result is None


class TestGetMurckoScaffold:
    """Test get_murcko_scaffold function."""

    def test_benzene_scaffold(self):
        """Test benzene returns itself as scaffold."""
        from rdkit_cli.core.scaffold import get_murcko_scaffold

        mol = Chem.MolFromSmiles("c1ccccc1")
        scaffold = get_murcko_scaffold(mol)
        assert scaffold == "c1ccccc1"

    def test_substituted_benzene(self):
        """Test substituted benzene returns benzene core."""
        from rdkit_cli.core.scaffold import get_murcko_scaffold

        mol = Chem.MolFromSmiles("Cc1ccccc1")  # toluene
        scaffold = get_murcko_scaffold(mol)
        # Scaffold should be benzene ring
        assert scaffold == "c1ccccc1"

    def test_generic_scaffold(self):
        """Test generic scaffold generation."""
        from rdkit_cli.core.scaffold import get_murcko_scaffold

        mol = Chem.MolFromSmiles("c1ccc(N)cc1")  # aniline
        scaffold = get_murcko_scaffold(mol, generic=True)

        assert scaffold is not None
        # Generic scaffold replaces all atoms with C
        # Should be a 6-membered ring with all carbon-like atoms


class TestAnalyzeScaffolds:
    """Test analyze_scaffolds function."""

    def test_frequency_analysis(self):
        """Test scaffold frequency analysis."""
        from rdkit_cli.core.scaffold import analyze_scaffolds

        scaffolds = [
            "c1ccccc1",
            "c1ccccc1",
            "c1ccccc1",
            "C1CCCCC1",
            "C1CCCCC1",
        ]

        results = analyze_scaffolds(scaffolds, top_n=5)

        assert len(results) == 2
        # First should be benzene (most frequent)
        assert results[0][0] == "c1ccccc1"
        assert results[0][1] == 3  # count
        assert results[0][2] == 60.0  # percentage

    def test_top_n_limit(self):
        """Test top_n limiting."""
        from rdkit_cli.core.scaffold import analyze_scaffolds

        scaffolds = ["A", "B", "C", "D", "E", "F", "G"]

        results = analyze_scaffolds(scaffolds, top_n=3)
        assert len(results) == 3

    def test_empty_list(self):
        """Test empty scaffold list."""
        from rdkit_cli.core.scaffold import analyze_scaffolds

        results = analyze_scaffolds([], top_n=10)
        assert len(results) == 0
