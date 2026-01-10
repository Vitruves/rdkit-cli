"""Synthetic Accessibility Score module.

Based on the work by Ertl & Schuffenhauer:
"Estimation of synthetic accessibility score of drug-like molecules based on
molecular complexity and fragment contributions"
Journal of Cheminformatics 2009, 1:8
"""

from typing import Optional
import math
import pickle
import gzip
from pathlib import Path

from rdkit_cli.io.readers import MoleculeRecord


# Cache for fragment scores (loaded lazily)
_fscores: Optional[dict] = None


def _load_fragment_scores() -> dict:
    """Load fragment contribution scores from RDKit Contrib."""
    global _fscores
    if _fscores is not None:
        return _fscores

    from rdkit import RDConfig

    # Try to load from RDKit Contrib
    fscores_path = Path(RDConfig.RDContribDir) / "SA_Score" / "fpscores.pkl.gz"

    if fscores_path.exists():
        with gzip.open(fscores_path, "rb") as f:
            _fscores = pickle.load(f)
    else:
        # Generate scores from SMILES if file not available
        _fscores = _generate_default_scores()

    return _fscores


def _generate_default_scores() -> dict:
    """Generate default fragment scores (simplified version)."""
    # This is a simplified fallback - the full implementation requires
    # the fpscores.pkl.gz file from RDKit Contrib
    return {}


def calculate_sa_score(mol) -> Optional[float]:
    """
    Calculate Synthetic Accessibility Score for a molecule.

    The SA Score ranges from 1 (easy to synthesize) to 10 (difficult).

    Args:
        mol: RDKit molecule object

    Returns:
        SA Score (1-10) or None if calculation fails
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    if mol is None:
        return None

    try:
        # Load fragment scores
        fscores = _load_fragment_scores()

        # Calculate Morgan fingerprint fragments
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()

        # Fragment score
        score1 = 0.0
        nf = 0
        for bit_id, count in fps.items():
            nf += count
            if bit_id in fscores:
                score1 += fscores[bit_id] * count
            else:
                score1 += -4  # Penalty for unknown fragments

        score1 /= nf if nf > 0 else 1

        # Size penalty
        num_atoms = mol.GetNumAtoms()
        sizePenalty = num_atoms ** 1.005 - num_atoms

        # Stereo penalty
        stereo_info = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        stereo_penalty = math.log10(len(stereo_info) + 1)

        # Spiro penalty
        spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        spiro_penalty = math.log10(spiro + 1)

        # Bridge penalty
        bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        bridge_penalty = math.log10(bridge + 1)

        # Macrocycle penalty
        ring_info = mol.GetRingInfo()
        macrocycles = sum(1 for r in ring_info.AtomRings() if len(r) > 8)
        macrocycle_penalty = math.log10(macrocycles + 1)

        # Combine all components
        sa_score = (
            -score1
            + sizePenalty
            + stereo_penalty
            + spiro_penalty
            + bridge_penalty
            + macrocycle_penalty
        )

        # Normalize to 1-10 range
        sa_score = 11.0 - (sa_score + 5.0) / 1.5
        sa_score = max(1.0, min(10.0, sa_score))

        return round(sa_score, 4)

    except Exception:
        return None


def calculate_npc_score(mol) -> Optional[float]:
    """
    Calculate Natural Product-likeness Score.

    Based on the natural product-likeness score from RDKit.

    Args:
        mol: RDKit molecule object

    Returns:
        NP Score or None if calculation fails
    """
    if mol is None:
        return None

    try:
        from rdkit.Chem import NaturalProductScorer
        score = NaturalProductScorer.GetNaturalProductScore(mol)
        return round(score, 4)
    except ImportError:
        # NaturalProductScorer may not be available in all RDKit versions
        return None
    except Exception:
        return None


def calculate_qed_score(mol) -> Optional[float]:
    """
    Calculate QED (Quantitative Estimate of Drug-likeness) score.

    Args:
        mol: RDKit molecule object

    Returns:
        QED Score (0-1) or None if calculation fails
    """
    if mol is None:
        return None

    try:
        from rdkit.Chem.QED import qed
        score = qed(mol)
        return round(score, 4)
    except Exception:
        return None


class SAScoreCalculator:
    """Calculator for synthetic accessibility and related scores."""

    def __init__(
        self,
        include_sa: bool = True,
        include_npc: bool = False,
        include_qed: bool = False,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize calculator.

        Args:
            include_sa: Include SA Score
            include_npc: Include NP-likeness Score
            include_qed: Include QED Score
            include_smiles: Include SMILES in output
            include_name: Include name in output
        """
        self.include_sa = include_sa
        self.include_npc = include_npc
        self.include_qed = include_qed
        self.include_smiles = include_smiles
        self.include_name = include_name

    def compute(self, record: MoleculeRecord) -> Optional[dict]:
        """Compute scores for a molecule record."""
        if record.mol is None:
            return None

        result = {}

        if self.include_smiles:
            result["smiles"] = record.smiles
        if self.include_name and record.name:
            result["name"] = record.name

        if self.include_sa:
            result["sa_score"] = calculate_sa_score(record.mol)

        if self.include_npc:
            result["npc_score"] = calculate_npc_score(record.mol)

        if self.include_qed:
            result["qed_score"] = calculate_qed_score(record.mol)

        return result

    def get_column_names(self) -> list[str]:
        """Get output column names."""
        cols = []
        if self.include_smiles:
            cols.append("smiles")
        if self.include_name:
            cols.append("name")
        if self.include_sa:
            cols.append("sa_score")
        if self.include_npc:
            cols.append("npc_score")
        if self.include_qed:
            cols.append("qed_score")
        return cols
