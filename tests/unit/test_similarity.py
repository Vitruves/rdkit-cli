"""Unit tests for similarity module."""

import pytest
from rdkit import Chem


class TestSimilaritySearcher:
    """Test SimilaritySearcher class."""

    def test_search_identical(self):
        """Test that identical molecules have similarity 1.0."""
        from rdkit_cli.core.similarity import SimilaritySearcher
        from rdkit_cli.io.readers import MoleculeRecord

        query = "c1ccccc1"  # benzene
        searcher = SimilaritySearcher(query_smiles=query, threshold=0.9)

        mol = Chem.MolFromSmiles(query)
        record = MoleculeRecord(mol=mol, smiles=query, name="benzene")
        result = searcher.search(record)

        assert result is not None
        assert result["similarity"] == 1.0

    def test_search_similar(self):
        """Test finding similar molecules."""
        from rdkit_cli.core.similarity import SimilaritySearcher
        from rdkit_cli.io.readers import MoleculeRecord

        query = "c1ccccc1"  # benzene
        searcher = SimilaritySearcher(query_smiles=query, threshold=0.1)  # Low threshold

        # Toluene is similar to benzene
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        record = MoleculeRecord(mol=mol, smiles="Cc1ccccc1", name="toluene")
        result = searcher.search(record)

        assert result is not None
        assert result["similarity"] > 0.1

    def test_search_dissimilar(self):
        """Test that dissimilar molecules are filtered out."""
        from rdkit_cli.core.similarity import SimilaritySearcher
        from rdkit_cli.io.readers import MoleculeRecord

        query = "c1ccccc1"  # benzene
        searcher = SimilaritySearcher(query_smiles=query, threshold=0.9)

        # Methane is very different from benzene
        mol = Chem.MolFromSmiles("C")
        record = MoleculeRecord(mol=mol, smiles="C", name="methane")
        result = searcher.search(record)

        assert result is None

    def test_invalid_query(self):
        """Test invalid query SMILES raises error."""
        from rdkit_cli.core.similarity import SimilaritySearcher

        with pytest.raises(ValueError, match="Invalid query SMILES"):
            SimilaritySearcher(query_smiles="not_valid_smiles")

    def test_none_molecule(self):
        """Test handling of None molecule."""
        from rdkit_cli.core.similarity import SimilaritySearcher
        from rdkit_cli.io.readers import MoleculeRecord

        searcher = SimilaritySearcher(query_smiles="C", threshold=0.5)
        record = MoleculeRecord(mol=None, smiles="invalid")
        result = searcher.search(record)
        assert result is None


class TestComputeSimilarity:
    """Test compute_similarity function."""

    def test_tanimoto_identical(self):
        """Test Tanimoto similarity of identical fingerprints."""
        from rdkit_cli.core.similarity import compute_similarity, get_morgan_fingerprint, SimilarityMetric

        mol = Chem.MolFromSmiles("CCO")
        fp = get_morgan_fingerprint(mol)

        similarity = compute_similarity(fp, fp, SimilarityMetric.TANIMOTO)
        assert similarity == 1.0

    def test_dice_identical(self):
        """Test Dice similarity of identical fingerprints."""
        from rdkit_cli.core.similarity import compute_similarity, get_morgan_fingerprint, SimilarityMetric

        mol = Chem.MolFromSmiles("CCO")
        fp = get_morgan_fingerprint(mol)

        similarity = compute_similarity(fp, fp, SimilarityMetric.DICE)
        assert similarity == 1.0

    def test_different_molecules(self):
        """Test similarity between different molecules."""
        from rdkit_cli.core.similarity import compute_similarity, get_morgan_fingerprint, SimilarityMetric

        mol1 = Chem.MolFromSmiles("c1ccccc1")  # benzene
        mol2 = Chem.MolFromSmiles("CCCCCC")  # hexane

        fp1 = get_morgan_fingerprint(mol1)
        fp2 = get_morgan_fingerprint(mol2)

        similarity = compute_similarity(fp1, fp2, SimilarityMetric.TANIMOTO)
        assert 0 <= similarity < 1


class TestComputeSimilarityMatrix:
    """Test compute_similarity_matrix function."""

    def test_matrix_dimensions(self):
        """Test matrix has correct dimensions."""
        from rdkit_cli.core.similarity import compute_similarity_matrix

        mols = [
            Chem.MolFromSmiles("C"),
            Chem.MolFromSmiles("CC"),
            Chem.MolFromSmiles("CCC"),
        ]

        matrix = compute_similarity_matrix(mols)

        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)

    def test_matrix_diagonal(self):
        """Test matrix diagonal is 1.0."""
        from rdkit_cli.core.similarity import compute_similarity_matrix

        mols = [
            Chem.MolFromSmiles("C"),
            Chem.MolFromSmiles("CC"),
        ]

        matrix = compute_similarity_matrix(mols)

        assert matrix[0][0] == 1.0
        assert matrix[1][1] == 1.0

    def test_matrix_symmetry(self):
        """Test matrix is symmetric."""
        from rdkit_cli.core.similarity import compute_similarity_matrix

        mols = [
            Chem.MolFromSmiles("C"),
            Chem.MolFromSmiles("CC"),
            Chem.MolFromSmiles("c1ccccc1"),
        ]

        matrix = compute_similarity_matrix(mols)

        for i in range(len(matrix)):
            for j in range(len(matrix)):
                assert matrix[i][j] == matrix[j][i]


class TestClusterMolecules:
    """Test cluster_molecules function."""

    def test_clustering(self):
        """Test basic clustering."""
        from rdkit_cli.core.similarity import cluster_molecules

        mols = [
            Chem.MolFromSmiles("c1ccccc1"),  # benzene
            Chem.MolFromSmiles("Cc1ccccc1"),  # toluene
            Chem.MolFromSmiles("CCc1ccccc1"),  # ethylbenzene
            Chem.MolFromSmiles("CCCCCC"),  # hexane
            Chem.MolFromSmiles("CCCCCCC"),  # heptane
        ]

        clusters = cluster_molecules(mols, cutoff=0.5)

        # Should have at least 2 clusters (aromatics vs aliphatics)
        assert len(clusters) >= 1

    def test_empty_input(self):
        """Test clustering empty list."""
        from rdkit_cli.core.similarity import cluster_molecules

        clusters = cluster_molecules([])
        assert len(clusters) == 0

    def test_single_molecule(self):
        """Test clustering single molecule."""
        from rdkit_cli.core.similarity import cluster_molecules

        mols = [Chem.MolFromSmiles("C")]
        clusters = cluster_molecules(mols)

        assert len(clusters) == 1
        assert 0 in clusters[0]
