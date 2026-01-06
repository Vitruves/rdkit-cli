"""Scaffold analysis engine."""

from typing import Optional, Any
from collections import Counter

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from rdkit_cli.io.readers import MoleculeRecord


def get_murcko_scaffold(mol: Chem.Mol, generic: bool = False) -> Optional[str]:
    """
    Get Murcko scaffold for a molecule.

    Args:
        mol: RDKit molecule
        generic: If True, return generic scaffold (element-agnostic)

    Returns:
        Scaffold SMILES or None if failed
    """
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)

        if generic:
            core = MurckoScaffold.MakeScaffoldGeneric(core)

        return Chem.MolToSmiles(core)
    except Exception:
        return None


def get_side_chains(mol: Chem.Mol) -> list[str]:
    """
    Get side chains (R-groups) for a molecule.

    Args:
        mol: RDKit molecule

    Returns:
        List of side chain SMILES
    """
    try:
        side_chains = MurckoScaffold.MurckoDecompose(mol)
        return [Chem.MolToSmiles(sc) for sc in side_chains if sc is not None]
    except Exception:
        return []


class ScaffoldExtractor:
    """Extract Murcko scaffolds from molecules."""

    def __init__(
        self,
        generic: bool = False,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize scaffold extractor.

        Args:
            generic: Generate generic (element-agnostic) scaffolds
            include_smiles: Include original SMILES in output
            include_name: Include molecule name in output
        """
        self.generic = generic
        self.include_smiles = include_smiles
        self.include_name = include_name

    def extract(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """
        Extract scaffold from a molecule.

        Args:
            record: MoleculeRecord to process

        Returns:
            Dictionary with scaffold info or None if failed
        """
        if record.mol is None:
            return None

        scaffold = get_murcko_scaffold(record.mol, generic=self.generic)

        if scaffold is None:
            return None

        result: dict[str, Any] = {}

        if self.include_smiles:
            result["smiles"] = record.smiles
        if self.include_name and record.name:
            result["name"] = record.name

        result["scaffold"] = scaffold

        return result


class ScaffoldDecomposer:
    """Decompose molecules into scaffold and side chains."""

    def __init__(
        self,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize scaffold decomposer.

        Args:
            include_smiles: Include original SMILES in output
            include_name: Include molecule name in output
        """
        self.include_smiles = include_smiles
        self.include_name = include_name

    def decompose(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """
        Decompose a molecule into scaffold and side chains.

        Args:
            record: MoleculeRecord to process

        Returns:
            Dictionary with decomposition info or None if failed
        """
        if record.mol is None:
            return None

        scaffold = get_murcko_scaffold(record.mol)
        if scaffold is None:
            return None

        generic_scaffold = get_murcko_scaffold(record.mol, generic=True)

        result: dict[str, Any] = {}

        if self.include_smiles:
            result["smiles"] = record.smiles
        if self.include_name and record.name:
            result["name"] = record.name

        result["scaffold"] = scaffold
        result["generic_scaffold"] = generic_scaffold

        return result


def analyze_scaffolds(
    scaffolds: list[str],
    top_n: int = 20,
) -> list[tuple[str, int, float]]:
    """
    Analyze scaffold frequency distribution.

    Args:
        scaffolds: List of scaffold SMILES
        top_n: Number of top scaffolds to return

    Returns:
        List of (scaffold, count, percentage) tuples
    """
    total = len(scaffolds)
    counter = Counter(scaffolds)

    results = []
    for scaffold, count in counter.most_common(top_n):
        percentage = (count / total) * 100 if total > 0 else 0
        results.append((scaffold, count, round(percentage, 2)))

    return results
