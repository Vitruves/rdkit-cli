"""Unit tests for sascorer module."""

import pytest
from rdkit import Chem


class TestSAScore:
    """Test synthetic accessibility score calculation."""

    def test_calculate_sa_score_simple(self):
        """Test SA score for simple molecule."""
        from rdkit_cli.core.sascorer import calculate_sa_score

        mol = Chem.MolFromSmiles("CCO")  # Ethanol - should be easy to synthesize
        score = calculate_sa_score(mol)

        assert score is not None
        assert 1 <= score <= 10

    def test_calculate_sa_score_complex(self):
        """Test SA score for complex molecule."""
        from rdkit_cli.core.sascorer import calculate_sa_score

        # Erythromycin - complex macrolide
        mol = Chem.MolFromSmiles("CCC1C(C(C(C(=O)C(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)O)(C)O")
        score = calculate_sa_score(mol)

        assert score is not None
        assert 1 <= score <= 10

    def test_calculate_sa_score_none(self):
        """Test SA score with None molecule."""
        from rdkit_cli.core.sascorer import calculate_sa_score

        score = calculate_sa_score(None)
        assert score is None


class TestQEDScore:
    """Test QED score calculation."""

    def test_calculate_qed_score(self):
        """Test QED score for drug-like molecule."""
        from rdkit_cli.core.sascorer import calculate_qed_score

        mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
        score = calculate_qed_score(mol)

        assert score is not None
        assert 0 <= score <= 1

    def test_calculate_qed_score_none(self):
        """Test QED score with None molecule."""
        from rdkit_cli.core.sascorer import calculate_qed_score

        score = calculate_qed_score(None)
        assert score is None


class TestSAScoreCalculator:
    """Test SAScoreCalculator class."""

    def test_compute_all_scores(self, sample_molecules):
        """Test computing all scores."""
        from rdkit_cli.core.sascorer import SAScoreCalculator
        from rdkit_cli.io.readers import MoleculeRecord

        calc = SAScoreCalculator(
            include_sa=True,
            include_npc=True,
            include_qed=True,
        )

        name, smi = sample_molecules[0]
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)

        result = calc.compute(record)

        assert result is not None
        assert "sa_score" in result
        assert "qed_score" in result

    def test_compute_sa_only(self, sample_molecules):
        """Test computing only SA score."""
        from rdkit_cli.core.sascorer import SAScoreCalculator
        from rdkit_cli.io.readers import MoleculeRecord

        calc = SAScoreCalculator(
            include_sa=True,
            include_npc=False,
            include_qed=False,
        )

        name, smi = sample_molecules[0]
        mol = Chem.MolFromSmiles(smi)
        record = MoleculeRecord(mol=mol, smiles=smi, name=name)

        result = calc.compute(record)

        assert result is not None
        assert "sa_score" in result
        assert "npc_score" not in result
        assert "qed_score" not in result

    def test_compute_invalid_molecule(self):
        """Test computing scores for invalid molecule."""
        from rdkit_cli.core.sascorer import SAScoreCalculator
        from rdkit_cli.io.readers import MoleculeRecord

        calc = SAScoreCalculator()
        record = MoleculeRecord(mol=None, smiles="invalid")

        result = calc.compute(record)
        assert result is None

    def test_column_names(self):
        """Test getting column names."""
        from rdkit_cli.core.sascorer import SAScoreCalculator

        calc = SAScoreCalculator(
            include_sa=True,
            include_npc=True,
            include_qed=True,
            include_smiles=True,
            include_name=True,
        )

        cols = calc.get_column_names()

        assert "smiles" in cols
        assert "name" in cols
        assert "sa_score" in cols
        assert "npc_score" in cols
        assert "qed_score" in cols
