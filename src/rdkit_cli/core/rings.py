"""Ring system analysis module."""

from typing import Optional
from collections import Counter

from rdkit_cli.io.readers import MoleculeRecord


class RingSystemExtractor:
    """Extract ring systems from molecules."""

    def __init__(
        self,
        include_smiles: bool = True,
        include_name: bool = True,
        include_fused: bool = True,
        include_spiro: bool = True,
        include_bridged: bool = True,
    ):
        """
        Initialize ring system extractor.

        Args:
            include_smiles: Include original SMILES in output
            include_name: Include name in output
            include_fused: Include fused ring systems
            include_spiro: Include spiro ring systems
            include_bridged: Include bridged ring systems
        """
        self.include_smiles = include_smiles
        self.include_name = include_name
        self.include_fused = include_fused
        self.include_spiro = include_spiro
        self.include_bridged = include_bridged

    def extract(self, record: MoleculeRecord) -> list[dict]:
        """
        Extract ring systems from a molecule.

        Args:
            record: Molecule record

        Returns:
            List of dictionaries, one per ring system
        """
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors

        if record.mol is None:
            return []

        mol = record.mol
        ring_info = mol.GetRingInfo()

        if ring_info.NumRings() == 0:
            return []

        try:
            # Get ring systems (connected rings form a system)
            ring_systems = self._get_ring_systems(mol)

            results = []
            for i, (atom_indices, ring_type) in enumerate(ring_systems):
                # Skip based on type filter
                if ring_type == "fused" and not self.include_fused:
                    continue
                if ring_type == "spiro" and not self.include_spiro:
                    continue
                if ring_type == "bridged" and not self.include_bridged:
                    continue

                # Extract the ring system as a molecule
                ring_smiles = self._extract_ring_smiles(mol, atom_indices)

                if ring_smiles:
                    result = {}
                    if self.include_smiles:
                        result["smiles"] = record.smiles
                    if self.include_name and record.name:
                        result["name"] = record.name

                    result["ring_system"] = ring_smiles
                    result["ring_type"] = ring_type
                    result["ring_size"] = len(atom_indices)
                    result["num_rings"] = self._count_rings_in_system(mol, atom_indices)
                    result["is_aromatic"] = self._is_aromatic_system(mol, atom_indices)

                    results.append(result)

            return results

        except Exception:
            return []

    def _get_ring_systems(self, mol) -> list[tuple[set, str]]:
        """Get ring systems grouped by connectivity."""
        from rdkit.Chem import rdMolDescriptors

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        if not atom_rings:
            return []

        # Group rings that share atoms
        ring_sets = [set(r) for r in atom_rings]

        # Union-find to group connected rings
        systems = []
        used = set()

        for i, ring in enumerate(ring_sets):
            if i in used:
                continue

            system = set(ring)
            used.add(i)

            # Find all connected rings
            changed = True
            while changed:
                changed = False
                for j, other_ring in enumerate(ring_sets):
                    if j not in used and system & other_ring:
                        system |= other_ring
                        used.add(j)
                        changed = True

            # Determine ring system type
            ring_type = self._classify_ring_system(mol, system, ring_sets)
            systems.append((system, ring_type))

        return systems

    def _classify_ring_system(self, mol, atom_indices: set, all_rings: list) -> str:
        """Classify a ring system as fused, spiro, or bridged."""
        from rdkit.Chem import rdMolDescriptors

        # Count rings in this system
        rings_in_system = [r for r in all_rings if set(r) & atom_indices]

        if len(rings_in_system) == 1:
            return "simple"

        # Check for spiro atoms (atom shared by exactly 2 rings with no other connections)
        spiro_count = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        bridge_count = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

        if bridge_count > 0:
            return "bridged"
        elif spiro_count > 0:
            return "spiro"
        else:
            return "fused"

    def _extract_ring_smiles(self, mol, atom_indices: set) -> str:
        """Extract SMILES for a ring system."""
        from rdkit import Chem

        try:
            # Create a copy with only ring atoms
            atom_list = sorted(atom_indices)

            # Use RWMol to extract substructure
            emol = Chem.RWMol(mol)

            # Mark atoms to keep
            atoms_to_remove = []
            for atom in mol.GetAtoms():
                if atom.GetIdx() not in atom_indices:
                    atoms_to_remove.append(atom.GetIdx())

            # Remove atoms in reverse order
            for idx in sorted(atoms_to_remove, reverse=True):
                emol.RemoveAtom(idx)

            if emol.GetNumAtoms() == 0:
                return ""

            # Sanitize and get SMILES
            try:
                Chem.SanitizeMol(emol)
            except Exception:
                pass

            return Chem.MolToSmiles(emol)

        except Exception:
            return ""

    def _count_rings_in_system(self, mol, atom_indices: set) -> int:
        """Count number of individual rings in a system."""
        ring_info = mol.GetRingInfo()
        count = 0
        for ring in ring_info.AtomRings():
            if set(ring) & atom_indices:
                count += 1
        return count

    def _is_aromatic_system(self, mol, atom_indices: set) -> bool:
        """Check if ring system is aromatic."""
        for idx in atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetIsAromatic():
                return True
        return False


def analyze_ring_systems(ring_systems: list[str], top_n: int = 20) -> list[tuple]:
    """
    Analyze frequency of ring systems.

    Args:
        ring_systems: List of ring system SMILES
        top_n: Number of top systems to return

    Returns:
        List of (smiles, count, percentage) tuples
    """
    counter = Counter(ring_systems)
    total = len(ring_systems)

    results = []
    for smiles, count in counter.most_common(top_n):
        pct = round(100 * count / total, 2) if total > 0 else 0
        results.append((smiles, count, pct))

    return results


class RingInfo:
    """Get ring information for a molecule."""

    def __init__(
        self,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        self.include_smiles = include_smiles
        self.include_name = include_name

    def analyze(self, record: MoleculeRecord) -> Optional[dict]:
        """Get ring information for a molecule."""
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors

        if record.mol is None:
            return None

        mol = record.mol
        ring_info = mol.GetRingInfo()

        result = {}
        if self.include_smiles:
            result["smiles"] = record.smiles
        if self.include_name and record.name:
            result["name"] = record.name

        result["num_rings"] = rdMolDescriptors.CalcNumRings(mol)
        result["num_aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
        result["num_aliphatic_rings"] = rdMolDescriptors.CalcNumAliphaticRings(mol)
        result["num_saturated_rings"] = rdMolDescriptors.CalcNumSaturatedRings(mol)
        result["num_heterocycles"] = rdMolDescriptors.CalcNumHeterocycles(mol)
        result["num_aromatic_heterocycles"] = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
        result["num_spiro_atoms"] = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        result["num_bridgehead_atoms"] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

        # Ring sizes
        ring_sizes = [len(r) for r in ring_info.AtomRings()]
        result["ring_sizes"] = ",".join(map(str, sorted(ring_sizes))) if ring_sizes else ""
        result["largest_ring"] = max(ring_sizes) if ring_sizes else 0
        result["smallest_ring"] = min(ring_sizes) if ring_sizes else 0

        return result
