"""Molecular similarity computation engine."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.ML.Cluster import Butina

from rdkit_cli.io.readers import MoleculeRecord


class SimilarityMetric(Enum):
    """Supported similarity metrics."""

    TANIMOTO = "tanimoto"
    DICE = "dice"
    COSINE = "cosine"
    SOKAL = "sokal"
    RUSSEL = "russel"


def get_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    """Get Morgan fingerprint for a molecule."""
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_similarity(
    fp1,
    fp2,
    metric: SimilarityMetric = SimilarityMetric.TANIMOTO,
) -> float:
    """
    Compute similarity between two fingerprints.

    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        metric: Similarity metric to use

    Returns:
        Similarity score (0-1)
    """
    if metric == SimilarityMetric.TANIMOTO:
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    elif metric == SimilarityMetric.DICE:
        return DataStructs.DiceSimilarity(fp1, fp2)
    elif metric == SimilarityMetric.COSINE:
        return DataStructs.CosineSimilarity(fp1, fp2)
    elif metric == SimilarityMetric.SOKAL:
        return DataStructs.SokalSimilarity(fp1, fp2)
    elif metric == SimilarityMetric.RUSSEL:
        return DataStructs.RusselSimilarity(fp1, fp2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def bulk_tanimoto_similarity(query_fp, fps: list) -> list[float]:
    """Compute Tanimoto similarity of query against multiple fingerprints."""
    return list(DataStructs.BulkTanimotoSimilarity(query_fp, fps))


class SimilaritySearcher:
    """Search for similar molecules."""

    def __init__(
        self,
        query_smiles: str,
        threshold: float = 0.7,
        metric: SimilarityMetric = SimilarityMetric.TANIMOTO,
        radius: int = 2,
        n_bits: int = 2048,
    ):
        """
        Initialize similarity searcher.

        Args:
            query_smiles: Query molecule SMILES
            threshold: Minimum similarity threshold
            metric: Similarity metric
            radius: Morgan fingerprint radius
            n_bits: Fingerprint bit size
        """
        self.threshold = threshold
        self.metric = metric
        self.radius = radius
        self.n_bits = n_bits

        # Generate query fingerprint
        query_mol = Chem.MolFromSmiles(query_smiles)
        if query_mol is None:
            raise ValueError(f"Invalid query SMILES: {query_smiles}")

        self.query_fp = get_morgan_fingerprint(query_mol, radius, n_bits)

    def search(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """
        Check if molecule is similar to query.

        Args:
            record: MoleculeRecord to check

        Returns:
            Dictionary with similarity score if above threshold, None otherwise
        """
        if record.mol is None:
            return None

        fp = get_morgan_fingerprint(record.mol, self.radius, self.n_bits)
        similarity = compute_similarity(self.query_fp, fp, self.metric)

        if similarity < self.threshold:
            return None

        result: dict[str, Any] = {
            "smiles": record.smiles,
            "similarity": round(similarity, 4),
        }

        if record.name:
            result["name"] = record.name

        return result


def compute_similarity_matrix(
    mols: list[Chem.Mol],
    metric: SimilarityMetric = SimilarityMetric.TANIMOTO,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[list[float]]:
    """
    Compute pairwise similarity matrix.

    Args:
        mols: List of molecules
        metric: Similarity metric
        radius: Morgan fingerprint radius
        n_bits: Fingerprint bit size

    Returns:
        Symmetric similarity matrix
    """
    # Generate fingerprints
    fps = [get_morgan_fingerprint(mol, radius, n_bits) for mol in mols if mol is not None]
    n = len(fps)

    # Compute pairwise similarities
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            sim = compute_similarity(fps[i], fps[j], metric)
            matrix[i][j] = sim
            matrix[j][i] = sim

    return matrix


def cluster_molecules(
    mols: list[Chem.Mol],
    cutoff: float = 0.3,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[list[int]]:
    """
    Cluster molecules using Butina algorithm.

    Args:
        mols: List of molecules
        cutoff: Distance cutoff (1 - similarity)
        radius: Morgan fingerprint radius
        n_bits: Fingerprint bit size

    Returns:
        List of clusters (each cluster is a list of molecule indices)
    """
    # Generate fingerprints
    fps = []
    valid_indices = []
    for i, mol in enumerate(mols):
        if mol is not None:
            fps.append(get_morgan_fingerprint(mol, radius, n_bits))
            valid_indices.append(i)

    n = len(fps)
    if n == 0:
        return []

    # Compute distance matrix (lower triangle)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    # Cluster using Butina
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

    # Map back to original indices
    result = []
    for cluster in clusters:
        result.append([valid_indices[i] for i in cluster])

    return result
