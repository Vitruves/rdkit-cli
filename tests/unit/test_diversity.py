"""Unit tests for diversity module."""

import pytest
from rdkit import Chem


class TestDiversityPicker:
    """Test DiversityPicker class."""

    def test_pick_diverse(self):
        """Test diverse subset selection."""
        from rdkit_cli.core.diversity import DiversityPicker

        mols = [
            Chem.MolFromSmiles("c1ccccc1"),  # benzene
            Chem.MolFromSmiles("Cc1ccccc1"),  # toluene
            Chem.MolFromSmiles("CCc1ccccc1"),  # ethylbenzene
            Chem.MolFromSmiles("CCCCCC"),  # hexane
            Chem.MolFromSmiles("CCCCCCC"),  # heptane
            Chem.MolFromSmiles("CCCCCCCC"),  # octane
        ]

        picker = DiversityPicker(n_picks=3, seed=42)
        selected = picker.pick(mols)

        assert len(selected) == 3
        assert all(0 <= idx < len(mols) for idx in selected)

    def test_pick_more_than_available(self):
        """Test picking more than available molecules."""
        from rdkit_cli.core.diversity import DiversityPicker

        mols = [
            Chem.MolFromSmiles("C"),
            Chem.MolFromSmiles("CC"),
        ]

        picker = DiversityPicker(n_picks=10)
        selected = picker.pick(mols)

        assert len(selected) == 2  # Only 2 available

    def test_pick_with_none_molecules(self):
        """Test handling of None molecules in list."""
        from rdkit_cli.core.diversity import DiversityPicker

        mols = [
            Chem.MolFromSmiles("C"),
            None,
            Chem.MolFromSmiles("CC"),
            None,
            Chem.MolFromSmiles("CCC"),
        ]

        picker = DiversityPicker(n_picks=2)
        selected = picker.pick(mols)

        assert len(selected) == 2
        # Selected indices should be from valid molecules
        assert all(mols[idx] is not None for idx in selected)

    def test_empty_input(self):
        """Test empty molecule list."""
        from rdkit_cli.core.diversity import DiversityPicker

        picker = DiversityPicker(n_picks=5)
        selected = picker.pick([])

        assert selected == []

    def test_leader_method(self):
        """Test leader picking method."""
        from rdkit_cli.core.diversity import DiversityPicker

        mols = [
            Chem.MolFromSmiles("c1ccccc1"),
            Chem.MolFromSmiles("CCCCCC"),
            Chem.MolFromSmiles("c1ccc(O)cc1"),
        ]

        picker = DiversityPicker(n_picks=2, method="leader")
        selected = picker.pick(mols)

        # Leader method may return fewer than requested if threshold not met
        assert len(selected) >= 1
        assert len(selected) <= 2


class TestDiversityAnalyzer:
    """Test DiversityAnalyzer class."""

    def test_analyze_similar_set(self):
        """Test analyzing similar molecules."""
        from rdkit_cli.core.diversity import DiversityAnalyzer

        # Similar molecules (all aromatic)
        mols = [
            Chem.MolFromSmiles("c1ccccc1"),
            Chem.MolFromSmiles("Cc1ccccc1"),
            Chem.MolFromSmiles("CCc1ccccc1"),
        ]

        analyzer = DiversityAnalyzer()
        stats = analyzer.analyze(mols)

        assert "mean_similarity" in stats
        assert "diversity_score" in stats
        assert stats["n_molecules"] == 3
        # Similar molecules should have some similarity
        assert stats["mean_similarity"] > 0.2

    def test_analyze_diverse_set(self):
        """Test analyzing diverse molecules."""
        from rdkit_cli.core.diversity import DiversityAnalyzer

        # Diverse molecules
        mols = [
            Chem.MolFromSmiles("c1ccccc1"),  # aromatic
            Chem.MolFromSmiles("CCCCCCCCCC"),  # aliphatic
            Chem.MolFromSmiles("C1CCCCC1"),  # cyclic
        ]

        analyzer = DiversityAnalyzer()
        stats = analyzer.analyze(mols)

        assert "diversity_score" in stats
        # Diverse molecules should have lower similarity
        assert stats["diversity_score"] > 0.3

    def test_analyze_with_none(self):
        """Test handling None molecules."""
        from rdkit_cli.core.diversity import DiversityAnalyzer

        mols = [
            Chem.MolFromSmiles("C"),
            None,
            Chem.MolFromSmiles("CC"),
        ]

        analyzer = DiversityAnalyzer()
        stats = analyzer.analyze(mols)

        assert stats["n_molecules"] == 2  # Only valid molecules

    def test_analyze_single_molecule(self):
        """Test analyzing single molecule."""
        from rdkit_cli.core.diversity import DiversityAnalyzer

        mols = [Chem.MolFromSmiles("C")]

        analyzer = DiversityAnalyzer()
        stats = analyzer.analyze(mols)

        assert "error" in stats  # Need at least 2 molecules
