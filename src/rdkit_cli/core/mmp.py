"""Matched Molecular Pairs (MMP) module."""

from typing import Optional, Iterator
from collections import defaultdict


def fragment_molecule(mol, max_cuts: int = 1) -> list[tuple[str, str]]:
    """
    Fragment a molecule for MMP analysis.

    Uses single-cut fragmentation to identify core and R-group pairs.

    Args:
        mol: RDKit molecule
        max_cuts: Maximum number of cuts (1 or 2)

    Returns:
        List of (core_smiles, rgroup_smiles) tuples
    """
    from rdkit import Chem
    from rdkit.Chem import BRICS, AllChem
    from rdkit.Chem.rdMMPA import FragmentMol

    if mol is None:
        return []

    try:
        # Use RDKit's MMPA fragmentation
        fragments = FragmentMol(mol, maxCuts=max_cuts, resultsAsMols=False)

        result = []
        for core, rgroup in fragments:
            if core and rgroup:
                # Normalize core (remove attachment point labels for matching)
                result.append((core, rgroup))

        return result

    except Exception:
        return []


def find_matched_pairs(
    molecules: list[tuple],
    max_cuts: int = 1,
    min_core_size: int = 3,
) -> Iterator[dict]:
    """
    Find matched molecular pairs in a dataset.

    Args:
        molecules: List of (smiles, name, properties) tuples
        max_cuts: Maximum number of cuts
        min_core_size: Minimum core size in heavy atoms

    Yields:
        Dictionaries with pair information
    """
    from rdkit import Chem

    # Fragment all molecules and group by core
    core_groups = defaultdict(list)

    for smiles, name, props in molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        fragments = fragment_molecule(mol, max_cuts=max_cuts)

        for core, rgroup in fragments:
            # Check core size
            core_mol = Chem.MolFromSmiles(core.replace("[*:1]", "[H]").replace("[*:2]", "[H]"))
            if core_mol and core_mol.GetNumHeavyAtoms() >= min_core_size:
                core_groups[core].append({
                    "smiles": smiles,
                    "name": name,
                    "rgroup": rgroup,
                    "properties": props,
                })

    # Generate pairs from each core group
    for core, members in core_groups.items():
        if len(members) < 2:
            continue

        # Generate all pairs within the group
        for i, mol1 in enumerate(members):
            for mol2 in members[i + 1:]:
                yield {
                    "core": core,
                    "smiles_1": mol1["smiles"],
                    "smiles_2": mol2["smiles"],
                    "name_1": mol1["name"],
                    "name_2": mol2["name"],
                    "rgroup_1": mol1["rgroup"],
                    "rgroup_2": mol2["rgroup"],
                    "transformation": f"{mol1['rgroup']}>>{mol2['rgroup']}",
                }


class MMPFragmenter:
    """Fragment molecules for MMP analysis."""

    def __init__(
        self,
        max_cuts: int = 1,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        self.max_cuts = max_cuts
        self.include_smiles = include_smiles
        self.include_name = include_name

    def fragment(self, record) -> list[dict]:
        """Fragment a molecule and return core/R-group pairs."""
        if record.mol is None:
            return []

        fragments = fragment_molecule(record.mol, max_cuts=self.max_cuts)

        results = []
        for core, rgroup in fragments:
            result = {}
            if self.include_smiles:
                result["smiles"] = record.smiles
            if self.include_name and record.name:
                result["name"] = record.name
            result["core"] = core
            result["rgroup"] = rgroup
            results.append(result)

        return results


class MMPTransformer:
    """Apply MMP transformations to molecules."""

    def __init__(
        self,
        transformation: str,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize transformer.

        Args:
            transformation: SMIRKS-like transformation (e.g., "[*:1]C>>[*:1]N")
            include_smiles: Include original SMILES
            include_name: Include name
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        self.transformation = transformation
        self.include_smiles = include_smiles
        self.include_name = include_name

        # Parse transformation as reaction
        try:
            self.rxn = AllChem.ReactionFromSmarts(transformation)
            if self.rxn is None:
                raise ValueError(f"Invalid transformation: {transformation}")
        except Exception as e:
            raise ValueError(f"Invalid transformation: {transformation}") from e

    def transform(self, record) -> list[dict]:
        """Apply transformation to a molecule."""
        from rdkit import Chem

        if record.mol is None:
            return []

        try:
            products = self.rxn.RunReactants((record.mol,))

            results = []
            seen = set()

            for product_set in products:
                for product in product_set:
                    try:
                        Chem.SanitizeMol(product)
                        product_smiles = Chem.MolToSmiles(product)

                        if product_smiles not in seen:
                            seen.add(product_smiles)
                            result = {}
                            if self.include_smiles:
                                result["original_smiles"] = record.smiles
                            if self.include_name and record.name:
                                result["name"] = record.name
                            result["product_smiles"] = product_smiles
                            result["transformation"] = self.transformation
                            results.append(result)
                    except Exception:
                        continue

            return results

        except Exception:
            return []


def analyze_transformations(pairs: list[dict], top_n: int = 20) -> list[tuple]:
    """
    Analyze frequency of transformations.

    Args:
        pairs: List of pair dictionaries with 'transformation' key
        top_n: Number of top transformations to return

    Returns:
        List of (transformation, count, percentage) tuples
    """
    from collections import Counter

    transformations = [p.get("transformation", "") for p in pairs if p.get("transformation")]
    counter = Counter(transformations)
    total = len(transformations)

    results = []
    for trans, count in counter.most_common(top_n):
        pct = round(100 * count / total, 2) if total > 0 else 0
        results.append((trans, count, pct))

    return results
