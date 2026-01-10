"""RMSD calculation module."""

from typing import Optional


def calculate_rmsd(
    mol1,
    mol2,
    align: bool = True,
    symmetry: bool = True,
    heavy_atoms_only: bool = False,
) -> Optional[float]:
    """
    Calculate RMSD between two molecules.

    Args:
        mol1: First molecule with 3D coordinates
        mol2: Second molecule with 3D coordinates
        align: Align molecules before calculating RMSD
        symmetry: Consider molecular symmetry
        heavy_atoms_only: Only use heavy atoms

    Returns:
        RMSD value or None if calculation fails
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign

    if mol1 is None or mol2 is None:
        return None

    if mol1.GetNumConformers() == 0 or mol2.GetNumConformers() == 0:
        return None

    try:
        # Handle heavy atoms only
        if heavy_atoms_only:
            mol1 = Chem.RemoveHs(mol1)
            mol2 = Chem.RemoveHs(mol2)

        if align and symmetry:
            # Best RMSD with alignment and symmetry
            rmsd = AllChem.GetBestRMS(mol1, mol2)
        elif align:
            # Align then calculate RMSD
            rmsd = rdMolAlign.AlignMol(mol1, mol2)
        else:
            # Calculate RMSD without alignment
            rmsd = rdMolAlign.CalcRMS(mol1, mol2)

        return rmsd

    except Exception:
        return None


def calculate_conformer_rmsd_matrix(mol, symmetry: bool = True) -> list[list[float]]:
    """
    Calculate pairwise RMSD matrix between all conformers of a molecule.

    Args:
        mol: Molecule with multiple conformers
        symmetry: Consider molecular symmetry

    Returns:
        2D list with RMSD values
    """
    from rdkit.Chem import AllChem

    n_conf = mol.GetNumConformers()
    if n_conf == 0:
        return []

    matrix = [[0.0] * n_conf for _ in range(n_conf)]

    for i in range(n_conf):
        for j in range(i + 1, n_conf):
            if symmetry:
                rmsd = AllChem.GetConformerRMS(mol, i, j, prealigned=False)
            else:
                rmsd = AllChem.GetConformerRMS(mol, i, j, prealigned=True)
            matrix[i][j] = rmsd
            matrix[j][i] = rmsd

    return matrix


def cluster_conformers_by_rmsd(
    mol,
    threshold: float = 1.0,
) -> list[list[int]]:
    """
    Cluster conformers by RMSD.

    Args:
        mol: Molecule with multiple conformers
        threshold: RMSD threshold for clustering

    Returns:
        List of clusters (each cluster is a list of conformer indices)
    """
    from rdkit.Chem import AllChem

    n_conf = mol.GetNumConformers()
    if n_conf == 0:
        return []

    # Get all RMSDs
    rmsds = AllChem.GetConformerRMSMatrix(mol, prealigned=False)

    # Simple greedy clustering
    clusters = []
    assigned = set()

    conf_indices = list(range(n_conf))

    for i in conf_indices:
        if i in assigned:
            continue

        cluster = [i]
        assigned.add(i)

        for j in conf_indices:
            if j in assigned:
                continue

            # Calculate index in condensed matrix
            if i < j:
                idx = i * n_conf - i * (i + 1) // 2 + j - i - 1
            else:
                idx = j * n_conf - j * (j + 1) // 2 + i - j - 1

            if idx < len(rmsds) and rmsds[idx] < threshold:
                cluster.append(j)
                assigned.add(j)

        clusters.append(cluster)

    return clusters


class RMSDCalculator:
    """Calculate RMSD between molecules and a reference."""

    def __init__(
        self,
        reference_mol,
        align: bool = True,
        symmetry: bool = True,
        heavy_atoms_only: bool = False,
    ):
        """
        Initialize RMSD calculator.

        Args:
            reference_mol: Reference molecule with 3D coordinates
            align: Align molecules before calculating RMSD
            symmetry: Consider molecular symmetry
            heavy_atoms_only: Only use heavy atoms
        """
        from rdkit import Chem

        if reference_mol is None:
            raise ValueError("Reference molecule is None")
        if reference_mol.GetNumConformers() == 0:
            raise ValueError("Reference molecule has no 3D coordinates")

        self.reference_mol = reference_mol
        self.align = align
        self.symmetry = symmetry
        self.heavy_atoms_only = heavy_atoms_only

        if heavy_atoms_only:
            self.reference_mol = Chem.RemoveHs(reference_mol)

    def calculate(self, mol) -> Optional[float]:
        """Calculate RMSD to reference."""
        return calculate_rmsd(
            mol,
            self.reference_mol,
            align=self.align,
            symmetry=self.symmetry,
            heavy_atoms_only=self.heavy_atoms_only,
        )


class ConformerRMSDAnalyzer:
    """Analyze RMSD between conformers of a molecule."""

    def __init__(
        self,
        symmetry: bool = True,
        heavy_atoms_only: bool = False,
    ):
        self.symmetry = symmetry
        self.heavy_atoms_only = heavy_atoms_only

    def analyze(self, mol) -> Optional[dict]:
        """
        Analyze conformer RMSDs for a molecule.

        Returns dictionary with statistics.
        """
        from rdkit import Chem

        if mol is None:
            return None

        n_conf = mol.GetNumConformers()
        if n_conf < 2:
            return {
                "num_conformers": n_conf,
                "min_rmsd": 0.0,
                "max_rmsd": 0.0,
                "mean_rmsd": 0.0,
            }

        if self.heavy_atoms_only:
            mol = Chem.RemoveHs(mol)

        try:
            matrix = calculate_conformer_rmsd_matrix(mol, symmetry=self.symmetry)

            # Calculate statistics (upper triangle only)
            rmsds = []
            for i in range(len(matrix)):
                for j in range(i + 1, len(matrix)):
                    rmsds.append(matrix[i][j])

            if not rmsds:
                return {
                    "num_conformers": n_conf,
                    "min_rmsd": 0.0,
                    "max_rmsd": 0.0,
                    "mean_rmsd": 0.0,
                }

            return {
                "num_conformers": n_conf,
                "min_rmsd": round(min(rmsds), 4),
                "max_rmsd": round(max(rmsds), 4),
                "mean_rmsd": round(sum(rmsds) / len(rmsds), 4),
            }

        except Exception:
            return None
