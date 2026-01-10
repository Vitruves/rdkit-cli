"""Unit tests for rings module."""

import pytest
from rdkit import Chem


class TestRingSystemExtractor:
    """Test RingSystemExtractor class."""

    def test_extract_single_ring(self):
        """Test extracting single ring system."""
        from rdkit_cli.core.rings import RingSystemExtractor
        from rdkit_cli.io.readers import MoleculeRecord

        extractor = RingSystemExtractor()

        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        record = MoleculeRecord(mol=mol, smiles="c1ccccc1", name="benzene")

        results = extractor.extract(record)

        assert len(results) == 1
        assert results[0]["ring_type"] == "simple"
        assert results[0]["is_aromatic"] == True

    def test_extract_fused_rings(self):
        """Test extracting fused ring systems."""
        from rdkit_cli.core.rings import RingSystemExtractor
        from rdkit_cli.io.readers import MoleculeRecord

        extractor = RingSystemExtractor()

        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # Naphthalene
        record = MoleculeRecord(mol=mol, smiles="c1ccc2ccccc2c1", name="naphthalene")

        results = extractor.extract(record)

        assert len(results) >= 1
        # Naphthalene has fused rings
        ring_types = [r["ring_type"] for r in results]
        assert "fused" in ring_types or "simple" in ring_types

    def test_extract_no_rings(self):
        """Test extracting from molecule without rings."""
        from rdkit_cli.core.rings import RingSystemExtractor
        from rdkit_cli.io.readers import MoleculeRecord

        extractor = RingSystemExtractor()

        mol = Chem.MolFromSmiles("CCCC")  # Butane
        record = MoleculeRecord(mol=mol, smiles="CCCC", name="butane")

        results = extractor.extract(record)

        assert len(results) == 0

    def test_extract_filter_by_type(self):
        """Test filtering ring types."""
        from rdkit_cli.core.rings import RingSystemExtractor
        from rdkit_cli.io.readers import MoleculeRecord

        extractor = RingSystemExtractor(
            include_fused=False,
            include_spiro=True,
            include_bridged=True,
        )

        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # Naphthalene (fused)
        record = MoleculeRecord(mol=mol, smiles="c1ccc2ccccc2c1", name="naphthalene")

        results = extractor.extract(record)

        # Should not include fused rings
        ring_types = [r["ring_type"] for r in results]
        assert "fused" not in ring_types


class TestRingInfo:
    """Test RingInfo class."""

    def test_analyze_aromatic(self):
        """Test analyzing aromatic molecule."""
        from rdkit_cli.core.rings import RingInfo
        from rdkit_cli.io.readers import MoleculeRecord

        analyzer = RingInfo()

        mol = Chem.MolFromSmiles("c1ccccc1")
        record = MoleculeRecord(mol=mol, smiles="c1ccccc1")

        result = analyzer.analyze(record)

        assert result is not None
        assert result["num_rings"] == 1
        assert result["num_aromatic_rings"] == 1
        assert result["num_aliphatic_rings"] == 0

    def test_analyze_aliphatic(self):
        """Test analyzing aliphatic ring."""
        from rdkit_cli.core.rings import RingInfo
        from rdkit_cli.io.readers import MoleculeRecord

        analyzer = RingInfo()

        mol = Chem.MolFromSmiles("C1CCCCC1")  # Cyclohexane
        record = MoleculeRecord(mol=mol, smiles="C1CCCCC1")

        result = analyzer.analyze(record)

        assert result is not None
        assert result["num_rings"] == 1
        assert result["num_aromatic_rings"] == 0
        assert result["num_aliphatic_rings"] == 1

    def test_analyze_heterocycle(self):
        """Test analyzing heterocycle."""
        from rdkit_cli.core.rings import RingInfo
        from rdkit_cli.io.readers import MoleculeRecord

        analyzer = RingInfo()

        mol = Chem.MolFromSmiles("c1ccncc1")  # Pyridine
        record = MoleculeRecord(mol=mol, smiles="c1ccncc1")

        result = analyzer.analyze(record)

        assert result is not None
        assert result["num_heterocycles"] == 1


class TestAnalyzeRingSystems:
    """Test analyze_ring_systems function."""

    def test_analyze_frequency(self):
        """Test ring system frequency analysis."""
        from rdkit_cli.core.rings import analyze_ring_systems

        ring_systems = [
            "c1ccccc1",
            "c1ccccc1",
            "c1ccccc1",
            "C1CCCCC1",
            "c1ccncc1",
        ]

        results = analyze_ring_systems(ring_systems, top_n=10)

        assert len(results) > 0
        # Benzene should be most common
        assert results[0][0] == "c1ccccc1"
        assert results[0][1] == 3

    def test_analyze_empty(self):
        """Test with empty list."""
        from rdkit_cli.core.rings import analyze_ring_systems

        results = analyze_ring_systems([])
        assert results == []
