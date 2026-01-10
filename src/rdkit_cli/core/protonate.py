"""Protonation state enumeration module.

Note: This is a simplified implementation. For more accurate pH-dependent
protonation, consider using Dimorphite-DL or similar specialized tools.
"""

from typing import Optional


# Common functional group pKa values (approximate)
# Format: (SMARTS, pKa, protonated SMARTS, deprotonated SMARTS)
PKA_RULES = [
    # Carboxylic acids (pKa ~4-5)
    ("[CX3](=O)[OX2H1]", 4.5, "[CX3](=O)[OH]", "[CX3](=O)[O-]"),
    # Phosphates (pKa ~2, 7, 12)
    ("[PX4](=O)([OX2H])([OX2H])[OX2H]", 2.0, None, None),  # Complex handling
    # Sulfonates (pKa ~-1)
    ("[SX4](=O)(=O)[OX2H]", -1.0, "[SX4](=O)(=O)[OH]", "[SX4](=O)(=O)[O-]"),
    # Primary amines (pKa ~10)
    ("[NX3H2;!$(NC=O)]", 10.5, "[NH3+]", "[NH2]"),
    # Secondary amines (pKa ~10-11)
    ("[NX3H1;!$(NC=O)]([#6])([#6])", 10.5, "[NH2+]", "[NH]"),
    # Tertiary amines (pKa ~10-11)
    ("[NX3H0;!$(NC=O)]([#6])([#6])([#6])", 10.5, "[NH+]", "[N]"),
    # Imidazoles (pKa ~6-7)
    ("[nR1]1[cR1][nR1H][cR1][cR1]1", 6.5, None, None),
    # Phenols (pKa ~10)
    ("[OX2H][cR1]", 10.0, "[OH]c", "[O-]c"),
    # Thiols (pKa ~8-10)
    ("[SX2H]", 8.5, "[SH]", "[S-]"),
]


def get_protonation_sites(mol) -> list[dict]:
    """
    Identify potential protonation/deprotonation sites in a molecule.

    Args:
        mol: RDKit molecule

    Returns:
        List of dictionaries with site information
    """
    from rdkit import Chem

    if mol is None:
        return []

    sites = []

    for smarts, pka, prot_smarts, deprot_smarts in PKA_RULES:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue

        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            sites.append({
                "atom_indices": match,
                "pka": pka,
                "smarts": smarts,
            })

    return sites


def enumerate_protonation_states(mol, target_ph: float = 7.4) -> list:
    """
    Enumerate protonation states at a given pH.

    This is a simplified implementation that considers the most likely
    protonation state based on pKa vs pH.

    Args:
        mol: RDKit molecule
        target_ph: Target pH (default: 7.4)

    Returns:
        List of RDKit molecules with different protonation states
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if mol is None:
        return []

    # Get protonation sites
    sites = get_protonation_sites(mol)

    if not sites:
        return [mol]

    # Generate the most likely state based on pH
    # If pH < pKa: protonated
    # If pH > pKa: deprotonated
    # Note: This is a simplification; real enumeration is more complex

    states = [mol]  # Include original

    # Generate dominant state at target pH
    dominant = Chem.RWMol(mol)
    modified = False

    for site in sites:
        pka = site["pka"]
        atoms = site["atom_indices"]

        if not atoms:
            continue

        atom_idx = atoms[0]
        atom = dominant.GetAtomWithIdx(atom_idx)

        if pka > target_ph:
            # Should be protonated
            if atom.GetFormalCharge() < 0:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
                modified = True
            elif atom.GetSymbol() == "N" and atom.GetFormalCharge() == 0:
                # Protonate nitrogen
                atom.SetFormalCharge(1)
                modified = True
        else:
            # Should be deprotonated
            if atom.GetSymbol() in ("O", "S") and atom.GetFormalCharge() == 0:
                # Check if it's an acid
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == "C":
                        # Check for carbonyl
                        for n_neighbor in neighbor.GetNeighbors():
                            if n_neighbor.GetSymbol() == "O" and neighbor.GetBondBetweenAtoms(neighbor.GetIdx(), n_neighbor.GetIdx()):
                                bond = dominant.GetBondBetweenAtoms(neighbor.GetIdx(), n_neighbor.GetIdx())
                                if bond and bond.GetBondTypeAsDouble() == 2:
                                    atom.SetFormalCharge(-1)
                                    atom.SetNumExplicitHs(0)
                                    modified = True
                                    break

    if modified:
        try:
            Chem.SanitizeMol(dominant)
            states.append(dominant.GetMol())
        except Exception:
            pass

    return states


def protonate_at_ph(mol, ph: float = 7.4) -> Optional:
    """
    Get the dominant protonation state at a given pH.

    Args:
        mol: RDKit molecule
        ph: Target pH

    Returns:
        Protonated molecule or None
    """
    from rdkit import Chem

    if mol is None:
        return None

    states = enumerate_protonation_states(mol, target_ph=ph)

    # Return the last state (most modified for target pH)
    return states[-1] if states else mol


class ProtonationEnumerator:
    """Enumerate protonation states for molecules."""

    def __init__(
        self,
        ph: float = 7.4,
        enumerate_all: bool = False,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize enumerator.

        Args:
            ph: Target pH
            enumerate_all: Enumerate all possible states (not just dominant)
            include_smiles: Include original SMILES
            include_name: Include name
        """
        self.ph = ph
        self.enumerate_all = enumerate_all
        self.include_smiles = include_smiles
        self.include_name = include_name

    def enumerate(self, record) -> list[dict]:
        """Enumerate protonation states for a molecule."""
        from rdkit import Chem

        if record.mol is None:
            return []

        if self.enumerate_all:
            states = enumerate_protonation_states(record.mol, target_ph=self.ph)
        else:
            dominant = protonate_at_ph(record.mol, ph=self.ph)
            states = [dominant] if dominant else []

        results = []
        for i, state in enumerate(states):
            result = {}
            if self.include_smiles:
                result["original_smiles"] = record.smiles
            if self.include_name and record.name:
                result["name"] = record.name

            result["protonated_smiles"] = Chem.MolToSmiles(state)
            result["ph"] = self.ph
            result["state_index"] = i
            result["formal_charge"] = Chem.GetFormalCharge(state)

            results.append(result)

        return results


def neutralize_mol(mol):
    """
    Neutralize charges in a molecule.

    Args:
        mol: RDKit molecule

    Returns:
        Neutralized molecule
    """
    from rdkit import Chem

    if mol is None:
        return None

    # Patterns for neutralization
    patts = (
        # Negative charges
        ("[n-]", "[nH]"),
        ("[N-;X2]", "N"),
        ("[O-]", "O"),
        ("[S-]", "S"),
        # Positive charges
        ("[N+;!H0]", "N"),
        ("[NH3+]", "[NH2]"),
        ("[NH2+]", "[NH]"),
    )

    mol = Chem.RWMol(mol)

    for reactant, product in patts:
        patt = Chem.MolFromSmarts(reactant)
        if patt is None:
            continue

        while mol.HasSubstructMatch(patt):
            matches = mol.GetSubstructMatches(patt)
            if not matches:
                break

            for match in matches:
                atom = mol.GetAtomWithIdx(match[0])
                charge = atom.GetFormalCharge()

                if charge < 0:
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() + abs(charge))
                elif charge > 0:
                    atom.SetFormalCharge(0)
                    if atom.GetNumExplicitHs() > 0:
                        atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)

    try:
        Chem.SanitizeMol(mol)
        return mol.GetMol()
    except Exception:
        return None
