"""Molecular fragmentation engine."""

from typing import Optional, Any
from collections import Counter

from rdkit import Chem
from rdkit.Chem import BRICS, Recap, AllChem, rdMolDescriptors

from rdkit_cli.io.readers import MoleculeRecord


class BRICSFragmenter:
    """Fragment molecules using BRICS algorithm."""

    def __init__(
        self,
        min_fragment_size: int = 1,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize BRICS fragmenter.

        Args:
            min_fragment_size: Minimum fragment heavy atom count
            include_smiles: Include original SMILES in output
            include_name: Include molecule name in output
        """
        self.min_fragment_size = min_fragment_size
        self.include_smiles = include_smiles
        self.include_name = include_name

    def fragment(self, record: MoleculeRecord) -> list[dict[str, Any]]:
        """
        Fragment a molecule using BRICS.

        Args:
            record: MoleculeRecord to process

        Returns:
            List of dictionaries with fragment SMILES
        """
        if record.mol is None:
            return []

        try:
            fragments = BRICS.BRICSDecompose(record.mol)

            results = []
            for i, frag_smi in enumerate(fragments):
                # Parse fragment to check size
                frag_mol = Chem.MolFromSmiles(frag_smi)
                if frag_mol is None:
                    continue

                heavy_atoms = frag_mol.GetNumHeavyAtoms()
                if heavy_atoms < self.min_fragment_size:
                    continue

                result: dict[str, Any] = {"fragment_smiles": frag_smi}

                if self.include_smiles:
                    result["smiles"] = record.smiles
                if self.include_name and record.name:
                    result["name"] = record.name

                result["fragment_idx"] = i
                result["heavy_atom_count"] = heavy_atoms

                results.append(result)

            return results

        except Exception:
            return []


class RECAPFragmenter:
    """Fragment molecules using RECAP algorithm."""

    def __init__(
        self,
        min_fragment_size: int = 1,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize RECAP fragmenter.

        Args:
            min_fragment_size: Minimum fragment heavy atom count
            include_smiles: Include original SMILES in output
            include_name: Include molecule name in output
        """
        self.min_fragment_size = min_fragment_size
        self.include_smiles = include_smiles
        self.include_name = include_name

    def fragment(self, record: MoleculeRecord) -> list[dict[str, Any]]:
        """
        Fragment a molecule using RECAP.

        Args:
            record: MoleculeRecord to process

        Returns:
            List of dictionaries with fragment SMILES
        """
        if record.mol is None:
            return []

        try:
            recap_tree = Recap.RecapDecompose(record.mol)
            leaves = recap_tree.GetLeaves()

            results = []
            for i, (frag_smi, node) in enumerate(leaves.items()):
                # Parse fragment to check size
                frag_mol = node.mol
                if frag_mol is None:
                    continue

                heavy_atoms = frag_mol.GetNumHeavyAtoms()
                if heavy_atoms < self.min_fragment_size:
                    continue

                result: dict[str, Any] = {"fragment_smiles": frag_smi}

                if self.include_smiles:
                    result["smiles"] = record.smiles
                if self.include_name and record.name:
                    result["name"] = record.name

                result["fragment_idx"] = i
                result["heavy_atom_count"] = heavy_atoms

                results.append(result)

            return results

        except Exception:
            return []


class FunctionalGroupExtractor:
    """Extract functional groups from molecules."""

    def __init__(
        self,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize functional group extractor.

        Args:
            include_smiles: Include original SMILES in output
            include_name: Include molecule name in output
        """
        self.include_smiles = include_smiles
        self.include_name = include_name
        # Use RDKit's functional group hierarchy
        self._fgs = rdMolDescriptors.GetMorganFingerprint

    def extract(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """
        Extract functional groups from a molecule.

        Args:
            record: MoleculeRecord to process

        Returns:
            Dictionary with functional group info or None
        """
        if record.mol is None:
            return None

        try:
            # Get functional groups using SMARTS patterns
            fg_patterns = {
                "alcohol": "[OX2H]",
                "aldehyde": "[CX3H1](=O)[#6]",
                "ketone": "[#6][CX3](=O)[#6]",
                "carboxylic_acid": "[CX3](=O)[OX2H1]",
                "ester": "[#6][CX3](=O)[OX2][#6]",
                "ether": "[OD2]([#6])[#6]",
                "amine_primary": "[NX3H2][#6]",
                "amine_secondary": "[NX3H1]([#6])[#6]",
                "amine_tertiary": "[NX3]([#6])([#6])[#6]",
                "amide": "[NX3][CX3](=[OX1])[#6]",
                "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
                "nitrile": "[NX1]#[CX2]",
                "halogen": "[F,Cl,Br,I]",
                "thiol": "[SX2H]",
                "sulfide": "[#16X2]([#6])[#6]",
                "aromatic_ring": "a1aaaaa1",
            }

            result: dict[str, Any] = {}

            if self.include_smiles:
                result["smiles"] = record.smiles
            if self.include_name and record.name:
                result["name"] = record.name

            for name, smarts in fg_patterns.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    matches = record.mol.GetSubstructMatches(pattern)
                    result[f"n_{name}"] = len(matches)

            return result

        except Exception:
            return None


def analyze_fragments(fragments: list[str], top_n: int = 20) -> list[tuple[str, int, float]]:
    """
    Analyze fragment frequency distribution.

    Args:
        fragments: List of fragment SMILES
        top_n: Number of top fragments to return

    Returns:
        List of (fragment, count, percentage) tuples
    """
    total = len(fragments)
    counter = Counter(fragments)

    results = []
    for frag, count in counter.most_common(top_n):
        percentage = (count / total) * 100 if total > 0 else 0
        results.append((frag, count, round(percentage, 2)))

    return results
