"""Molecular diversity analysis engine."""

from typing import Optional, Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.SimDivFilters import rdSimDivPickers


def get_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    """Get Morgan fingerprint for a molecule."""
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


class DiversityPicker:
    """Select diverse subset of molecules using MaxMin algorithm."""

    def __init__(
        self,
        n_picks: int = 100,
        seed: Optional[int] = None,
        radius: int = 2,
        n_bits: int = 2048,
        method: str = "maxmin",
    ):
        """
        Initialize diversity picker.

        Args:
            n_picks: Number of molecules to pick
            seed: Random seed for reproducibility
            radius: Morgan fingerprint radius
            n_bits: Fingerprint bit size
            method: Picking method ('maxmin' or 'leader')
        """
        self.n_picks = n_picks
        self.seed = seed
        self.radius = radius
        self.n_bits = n_bits
        self.method = method

    def pick(
        self,
        mols: list[Chem.Mol],
        first_picks: Optional[list[int]] = None,
    ) -> list[int]:
        """
        Pick diverse subset of molecules.

        Args:
            mols: List of molecules
            first_picks: Indices of molecules that must be included

        Returns:
            List of selected indices
        """
        # Filter None molecules and track indices
        valid_mols = []
        valid_indices = []
        for i, mol in enumerate(mols):
            if mol is not None:
                valid_mols.append(mol)
                valid_indices.append(i)

        if len(valid_mols) == 0:
            return []

        # Generate fingerprints
        fps = [get_morgan_fingerprint(mol, self.radius, self.n_bits) for mol in valid_mols]

        # Adjust n_picks if larger than available
        n_to_pick = min(self.n_picks, len(fps))

        # Create picker
        if self.method == "maxmin":
            picker = rdSimDivPickers.MaxMinPicker()
        else:
            picker = rdSimDivPickers.LeaderPicker()

        # Define distance function
        def dist_func(i, j):
            return 1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])

        # Pick diverse molecules
        if first_picks:
            # Map first_picks to valid indices
            mapped_first = [valid_indices.index(i) for i in first_picks if i in valid_indices]
            picks = list(picker.LazyBitVectorPick(fps, len(fps), n_to_pick, firstPicks=mapped_first))
        else:
            if self.seed is not None:
                picks = list(picker.LazyBitVectorPick(fps, len(fps), n_to_pick, seed=self.seed))
            else:
                picks = list(picker.LazyBitVectorPick(fps, len(fps), n_to_pick))

        # Map back to original indices
        return [valid_indices[i] for i in picks]


class DiversityAnalyzer:
    """Analyze diversity of a molecule set."""

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        sample_size: int = 1000,
    ):
        """
        Initialize diversity analyzer.

        Args:
            radius: Morgan fingerprint radius
            n_bits: Fingerprint bit size
            sample_size: Max molecules to sample for analysis
        """
        self.radius = radius
        self.n_bits = n_bits
        self.sample_size = sample_size

    def analyze(self, mols: list[Chem.Mol]) -> dict[str, Any]:
        """
        Analyze diversity of molecule set.

        Args:
            mols: List of molecules

        Returns:
            Dictionary with diversity statistics
        """
        import random

        # Filter None molecules
        valid_mols = [mol for mol in mols if mol is not None]

        if len(valid_mols) < 2:
            return {"error": "Need at least 2 valid molecules"}

        # Sample if too large
        if len(valid_mols) > self.sample_size:
            valid_mols = random.sample(valid_mols, self.sample_size)

        # Generate fingerprints
        fps = [get_morgan_fingerprint(mol, self.radius, self.n_bits) for mol in valid_mols]

        # Compute pairwise similarities
        similarities = []
        n = len(fps)
        for i in range(n):
            for j in range(i + 1, n):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)

        if not similarities:
            return {"error": "Could not compute similarities"}

        # Calculate statistics
        import statistics

        mean_sim = statistics.mean(similarities)
        median_sim = statistics.median(similarities)
        min_sim = min(similarities)
        max_sim = max(similarities)
        stdev_sim = statistics.stdev(similarities) if len(similarities) > 1 else 0

        return {
            "n_molecules": len(valid_mols),
            "n_pairs": len(similarities),
            "mean_similarity": round(mean_sim, 4),
            "median_similarity": round(median_sim, 4),
            "min_similarity": round(min_sim, 4),
            "max_similarity": round(max_sim, 4),
            "stdev_similarity": round(stdev_sim, 4),
            "diversity_score": round(1 - mean_sim, 4),
        }
