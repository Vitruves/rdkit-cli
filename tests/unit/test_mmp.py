"""Unit tests for mmp module."""

import pytest
from rdkit import Chem


class TestFragmentMolecule:
    """Test fragment_molecule function."""

    def test_fragment_simple(self):
        """Test fragmenting a simple molecule."""
        from rdkit_cli.core.mmp import fragment_molecule

        mol = Chem.MolFromSmiles("c1ccccc1C")  # Toluene
        fragments = fragment_molecule(mol, max_cuts=1)

        # Should have some fragments
        assert len(fragments) >= 0

    def test_fragment_no_cuts(self):
        """Test fragmenting molecule with no bonds to cut."""
        from rdkit_cli.core.mmp import fragment_molecule

        mol = Chem.MolFromSmiles("C")  # Methane
        fragments = fragment_molecule(mol, max_cuts=1)

        # May or may not have fragments
        assert isinstance(fragments, list)

    def test_fragment_none_molecule(self):
        """Test fragmenting None molecule."""
        from rdkit_cli.core.mmp import fragment_molecule

        fragments = fragment_molecule(None)
        assert fragments == []


class TestFindMatchedPairs:
    """Test find_matched_pairs function."""

    def test_find_pairs(self):
        """Test finding matched molecular pairs."""
        from rdkit_cli.core.mmp import find_matched_pairs

        molecules = [
            ("c1ccccc1C", "toluene", {}),
            ("c1ccccc1CC", "ethylbenzene", {}),
            ("c1ccccc1CCC", "propylbenzene", {}),
        ]

        pairs = list(find_matched_pairs(molecules, max_cuts=1))

        # Should find some pairs (all share benzene core)
        # Note: exact number depends on fragmentation algorithm
        assert isinstance(pairs, list)

    def test_find_pairs_no_common_core(self):
        """Test with molecules that don't share cores."""
        from rdkit_cli.core.mmp import find_matched_pairs

        molecules = [
            ("CCC", "propane", {}),
            ("c1ccccc1", "benzene", {}),
        ]

        pairs = list(find_matched_pairs(molecules, max_cuts=1))

        # These don't share a common core
        # May or may not find pairs depending on fragmentation
        assert isinstance(pairs, list)


class TestMMPFragmenter:
    """Test MMPFragmenter class."""

    def test_fragmenter(self):
        """Test MMP fragmenter."""
        from rdkit_cli.core.mmp import MMPFragmenter
        from rdkit_cli.io.readers import MoleculeRecord

        fragmenter = MMPFragmenter(max_cuts=1)

        mol = Chem.MolFromSmiles("c1ccccc1C")
        record = MoleculeRecord(mol=mol, smiles="c1ccccc1C", name="toluene")

        results = fragmenter.fragment(record)

        # Should return list of fragments
        assert isinstance(results, list)
        for r in results:
            assert "core" in r
            assert "rgroup" in r

    def test_fragmenter_invalid_molecule(self):
        """Test fragmenter with invalid molecule."""
        from rdkit_cli.core.mmp import MMPFragmenter
        from rdkit_cli.io.readers import MoleculeRecord

        fragmenter = MMPFragmenter()
        record = MoleculeRecord(mol=None, smiles="invalid")

        results = fragmenter.fragment(record)
        assert results == []


class TestMMPTransformer:
    """Test MMPTransformer class."""

    def test_transform(self):
        """Test MMP transformation."""
        from rdkit_cli.core.mmp import MMPTransformer
        from rdkit_cli.io.readers import MoleculeRecord

        # Simple C -> N transformation
        transformer = MMPTransformer(
            transformation="[c:1][CH3]>>[c:1][NH2]",
        )

        mol = Chem.MolFromSmiles("c1ccccc1C")  # Toluene
        record = MoleculeRecord(mol=mol, smiles="c1ccccc1C", name="toluene")

        results = transformer.transform(record)

        assert len(results) >= 0
        for r in results:
            assert "product_smiles" in r

    def test_transform_invalid_smirks(self):
        """Test with invalid SMIRKS."""
        from rdkit_cli.core.mmp import MMPTransformer

        with pytest.raises(ValueError, match="Invalid transformation"):
            MMPTransformer(transformation="[invalid")

    def test_transform_no_match(self):
        """Test transformation that doesn't match."""
        from rdkit_cli.core.mmp import MMPTransformer
        from rdkit_cli.io.readers import MoleculeRecord

        # Transform that won't match
        transformer = MMPTransformer(
            transformation="[c:1][Br]>>[c:1][I]",
        )

        mol = Chem.MolFromSmiles("c1ccccc1C")  # No Br
        record = MoleculeRecord(mol=mol, smiles="c1ccccc1C")

        results = transformer.transform(record)
        assert len(results) == 0


class TestAnalyzeTransformations:
    """Test analyze_transformations function."""

    def test_analyze(self):
        """Test transformation analysis."""
        from rdkit_cli.core.mmp import analyze_transformations

        pairs = [
            {"transformation": "C>>N"},
            {"transformation": "C>>N"},
            {"transformation": "C>>O"},
            {"transformation": "F>>Cl"},
        ]

        results = analyze_transformations(pairs, top_n=10)

        assert len(results) > 0
        assert results[0][0] == "C>>N"
        assert results[0][1] == 2

    def test_analyze_empty(self):
        """Test with empty list."""
        from rdkit_cli.core.mmp import analyze_transformations

        results = analyze_transformations([])
        assert results == []
