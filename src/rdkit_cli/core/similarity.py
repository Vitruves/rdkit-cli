"""Molecular similarity computation engine."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Cluster import Butina

from rdkit_cli.io.readers import MoleculeRecord


class SimilarityMetric(Enum):
    """Supported similarity metrics."""

    TANIMOTO = "tanimoto"
    DICE = "dice"
    COSINE = "cosine"
    SOKAL = "sokal"
    RUSSEL = "russel"
    ALLBIT = "allbit"
    ASYMMETRIC = "asymmetric"
    BRAUNBLANQUET = "braunblanquet"
    KULCZYNSKI = "kulczynski"
    MCCONNAUGHEY = "mcconnaughey"
    ONBIT = "onbit"
    ROGOTGOLDBERG = "rogotgoldberg"
    TVERSKY = "tversky"


def get_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    """Get Morgan fingerprint for a molecule."""
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprint(mol)


def compute_similarity(
    fp1,
    fp2,
    metric: SimilarityMetric = SimilarityMetric.TANIMOTO,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
) -> float:
    """
    Compute similarity between two fingerprints.

    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        metric: Similarity metric to use
        tversky_alpha: Alpha parameter for Tversky index
        tversky_beta: Beta parameter for Tversky index

    Returns:
        Similarity score (0-1)
    """
    metric_funcs = {
        SimilarityMetric.TANIMOTO: DataStructs.TanimotoSimilarity,
        SimilarityMetric.DICE: DataStructs.DiceSimilarity,
        SimilarityMetric.COSINE: DataStructs.CosineSimilarity,
        SimilarityMetric.SOKAL: DataStructs.SokalSimilarity,
        SimilarityMetric.RUSSEL: DataStructs.RusselSimilarity,
        SimilarityMetric.ALLBIT: DataStructs.AllBitSimilarity,
        SimilarityMetric.ASYMMETRIC: DataStructs.AsymmetricSimilarity,
        SimilarityMetric.BRAUNBLANQUET: DataStructs.BraunBlanquetSimilarity,
        SimilarityMetric.KULCZYNSKI: DataStructs.KulczynskiSimilarity,
        SimilarityMetric.MCCONNAUGHEY: DataStructs.McConnaugheySimilarity,
        SimilarityMetric.ONBIT: DataStructs.OnBitSimilarity,
        SimilarityMetric.ROGOTGOLDBERG: DataStructs.RogotGoldbergSimilarity,
    }

    if metric == SimilarityMetric.TVERSKY:
        return DataStructs.TverskySimilarity(fp1, fp2, tversky_alpha, tversky_beta)

    func = metric_funcs.get(metric)
    if func is None:
        raise ValueError(f"Unknown metric: {metric}")
    return func(fp1, fp2)


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
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5,
    ):
        """
        Initialize similarity searcher.

        Args:
            query_smiles: Query molecule SMILES
            threshold: Minimum similarity threshold
            metric: Similarity metric
            radius: Morgan fingerprint radius
            n_bits: Fingerprint bit size
            tversky_alpha: Alpha parameter for Tversky index
            tversky_beta: Beta parameter for Tversky index
        """
        self.threshold = threshold
        self.metric = metric
        self.radius = radius
        self.n_bits = n_bits
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

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
        similarity = compute_similarity(
            self.query_fp, fp, self.metric,
            tversky_alpha=self.tversky_alpha,
            tversky_beta=self.tversky_beta,
        )

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
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
) -> list[list[float]]:
    """
    Compute pairwise similarity matrix.

    Args:
        mols: List of molecules
        metric: Similarity metric
        radius: Morgan fingerprint radius
        n_bits: Fingerprint bit size
        tversky_alpha: Alpha parameter for Tversky index
        tversky_beta: Beta parameter for Tversky index

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
            sim = compute_similarity(
                fps[i], fps[j], metric,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
            )
            matrix[i][j] = sim
            matrix[j][i] = sim

    return matrix


class ShapeSimilaritySearcher:
    """Search for molecules with similar 3D shape."""

    def __init__(
        self,
        reference_file: str,
        threshold: float = 0.5,
        metric: str = "tanimoto",
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5,
    ):
        from rdkit.Chem import rdShapeHelpers, rdmolfiles

        self.threshold = threshold
        self.metric = metric
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self._rdShapeHelpers = rdShapeHelpers

        # Load reference molecule with 3D coords
        try:
            ref_mol = None
            ext = reference_file.rsplit(".", 1)[-1].lower()
            if ext in ("sdf", "mol"):
                suppl = rdmolfiles.SDMolSupplier(
                    reference_file, removeHs=False,
                )
                ref_mol = next(iter(suppl), None)
            elif ext == "pdb":
                ref_mol = rdmolfiles.MolFromPDBFile(
                    reference_file, removeHs=False,
                )
        except OSError:
            ref_mol = None

        if ref_mol is None or ref_mol.GetNumConformers() == 0:
            raise ValueError(
                f"Cannot load 3D reference from {reference_file}"
            )
        self.ref_mol = ref_mol

    def _compute_shape_sim(self, mol: Chem.Mol) -> float:
        sh = self._rdShapeHelpers
        if self.metric == "protrude":
            return 1.0 - sh.ShapeProtrudeDist(self.ref_mol, mol)
        elif self.metric == "tversky":
            return sh.ShapeTverskyIndex(
                self.ref_mol, mol,
                self.tversky_alpha, self.tversky_beta,
            )
        else:
            return 1.0 - sh.ShapeTanimotoDist(self.ref_mol, mol)

    def search(self, record: MoleculeRecord):
        if record.mol is None:
            return None

        mol = record.mol
        # Generate 3D if needed
        if mol.GetNumConformers() == 0:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)

        try:
            sim = self._compute_shape_sim(mol)
        except Exception:
            return None

        if sim < self.threshold:
            return None

        result = {
            "smiles": record.smiles,
            "shape_similarity": round(sim, 4),
        }
        if record.name:
            result["name"] = record.name
        return result


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
