"""Maximum Common Substructure engine."""

from typing import Optional, Any

from rdkit import Chem
from rdkit.Chem import rdFMCS


def find_mcs(
    mols: list[Chem.Mol],
    timeout: int = 60,
    threshold: float = 1.0,
    maximize: str = "atoms",
    ring_matches_ring_only: bool = True,
    complete_rings_only: bool = True,
    match_valences: bool = False,
    match_chiral_tag: bool = False,
    atom_compare: str = "elements",
    bond_compare: str = "order",
) -> Optional[dict[str, Any]]:
    """
    Find Maximum Common Substructure of molecules.

    Args:
        mols: List of molecules
        timeout: Maximum time in seconds
        threshold: Fraction of molecules that must contain MCS
        maximize: What to maximize ('atoms' or 'bonds')
        ring_matches_ring_only: Ring atoms only match ring atoms
        complete_rings_only: Only return complete rings
        match_valences: Match atom valences
        match_chiral_tag: Match chirality
        atom_compare: Atom comparison ('any', 'elements', 'isotopes')
        bond_compare: Bond comparison ('any', 'order', 'orderexact')

    Returns:
        Dictionary with MCS results or None
    """
    # Filter None molecules
    valid_mols = [mol for mol in mols if mol is not None]

    if len(valid_mols) < 2:
        return None

    # Set up atom comparison
    atom_compare_map = {
        "any": rdFMCS.AtomCompare.CompareAny,
        "elements": rdFMCS.AtomCompare.CompareElements,
        "isotopes": rdFMCS.AtomCompare.CompareIsotopes,
    }

    # Set up bond comparison
    bond_compare_map = {
        "any": rdFMCS.BondCompare.CompareAny,
        "order": rdFMCS.BondCompare.CompareOrder,
        "orderexact": rdFMCS.BondCompare.CompareOrderExact,
    }

    try:
        result = rdFMCS.FindMCS(
            valid_mols,
            timeout=timeout,
            threshold=threshold,
            maximizeBonds=(maximize == "bonds"),
            ringMatchesRingOnly=ring_matches_ring_only,
            completeRingsOnly=complete_rings_only,
            matchValences=match_valences,
            matchChiralTag=match_chiral_tag,
            atomCompare=atom_compare_map.get(atom_compare, rdFMCS.AtomCompare.CompareElements),
            bondCompare=bond_compare_map.get(bond_compare, rdFMCS.BondCompare.CompareOrder),
        )

        if result.canceled:
            return {"canceled": True, "timeout": timeout}

        if result.numAtoms == 0:
            return {"smarts": "", "num_atoms": 0, "num_bonds": 0}

        return {
            "smarts": result.smartsString,
            "num_atoms": result.numAtoms,
            "num_bonds": result.numBonds,
            "canceled": result.canceled,
        }

    except Exception as e:
        return {"error": str(e)}


class MCSAligner:
    """Align molecules based on MCS."""

    def __init__(
        self,
        reference_smiles: str,
        timeout: int = 30,
    ):
        """
        Initialize MCS aligner.

        Args:
            reference_smiles: Reference molecule SMILES
            timeout: MCS timeout in seconds
        """
        self.reference_mol = Chem.MolFromSmiles(reference_smiles)
        if self.reference_mol is None:
            raise ValueError(f"Invalid reference SMILES: {reference_smiles}")
        self.timeout = timeout

    def find_common(self, mol: Chem.Mol) -> Optional[dict[str, Any]]:
        """
        Find MCS between reference and query molecule.

        Args:
            mol: Query molecule

        Returns:
            Dictionary with MCS info or None
        """
        if mol is None:
            return None

        result = find_mcs(
            [self.reference_mol, mol],
            timeout=self.timeout,
        )

        return result
