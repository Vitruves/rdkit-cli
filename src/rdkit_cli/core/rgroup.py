"""R-Group Decomposition module."""

from typing import Optional

from rdkit_cli.io.readers import MoleculeRecord


class RGroupDecomposer:
    """Decompose molecules into core and R-groups."""

    def __init__(
        self,
        core_smarts: str,
        include_smiles: bool = True,
        include_name: bool = True,
        only_matching: bool = True,
    ):
        """
        Initialize R-group decomposer.

        Args:
            core_smarts: SMARTS pattern for core with labeled attachment points [*:1], [*:2], etc.
            include_smiles: Include original SMILES in output
            include_name: Include name in output
            only_matching: Only output molecules that match the core
        """
        from rdkit import Chem

        self.core_smarts = core_smarts
        self.include_smiles = include_smiles
        self.include_name = include_name
        self.only_matching = only_matching

        # Parse core
        self.core_mol = Chem.MolFromSmarts(core_smarts)
        if self.core_mol is None:
            raise ValueError(f"Invalid core SMARTS: {core_smarts}")

        # Find R-group labels in core
        self.r_labels = []
        for atom in self.core_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # Dummy atom
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    self.r_labels.append(f"R{map_num}")

        if not self.r_labels:
            raise ValueError("Core SMARTS must have labeled attachment points like [*:1], [*:2], etc.")

        self._decomposer = None

    def _get_decomposer(self):
        """Get or create RGroupDecomposition object."""
        if self._decomposer is None:
            from rdkit.Chem.rdRGroupDecomposition import RGroupDecomposition, RGroupDecompositionParameters

            params = RGroupDecompositionParameters()
            params.removeAllHydrogenRGroups = True
            params.removeHydrogensPostMatch = True

            self._decomposer = RGroupDecomposition(self.core_mol, params)

        return self._decomposer

    def decompose(self, record: MoleculeRecord) -> Optional[dict]:
        """
        Decompose a molecule into core and R-groups.

        Args:
            record: Molecule record

        Returns:
            Dictionary with core and R-group SMILES, or None if no match
        """
        from rdkit import Chem
        from rdkit.Chem.rdRGroupDecomposition import RGroupDecomposition, RGroupDecompositionParameters

        if record.mol is None:
            return None

        try:
            # Create a fresh decomposer for each molecule (stateless)
            params = RGroupDecompositionParameters()
            params.removeAllHydrogenRGroups = True
            params.removeHydrogensPostMatch = True

            decomp = RGroupDecomposition(self.core_mol, params)
            match_idx = decomp.Add(record.mol)

            if match_idx < 0:
                if self.only_matching:
                    return None
                else:
                    result = {}
                    if self.include_smiles:
                        result["smiles"] = record.smiles
                    if self.include_name and record.name:
                        result["name"] = record.name
                    result["matched"] = False
                    result["core"] = ""
                    for label in self.r_labels:
                        result[label] = ""
                    return result

            decomp.Process()
            rgroups = decomp.GetRGroupsAsColumns()

            result = {}
            if self.include_smiles:
                result["smiles"] = record.smiles
            if self.include_name and record.name:
                result["name"] = record.name

            result["matched"] = True

            # Get core
            if "Core" in rgroups and len(rgroups["Core"]) > 0:
                core_mol = rgroups["Core"][0]
                result["core"] = Chem.MolToSmiles(core_mol) if core_mol else ""
            else:
                result["core"] = ""

            # Get R-groups
            for label in self.r_labels:
                if label in rgroups and len(rgroups[label]) > 0:
                    rg_mol = rgroups[label][0]
                    result[label] = Chem.MolToSmiles(rg_mol) if rg_mol else "[H]"
                else:
                    result[label] = ""

            return result

        except Exception:
            return None

    def get_column_names(self) -> list[str]:
        """Get output column names."""
        cols = []
        if self.include_smiles:
            cols.append("smiles")
        if self.include_name:
            cols.append("name")
        cols.append("matched")
        cols.append("core")
        cols.extend(self.r_labels)
        return cols


def decompose_batch(
    mols: list,
    core_smarts: str,
) -> list[dict]:
    """
    Decompose multiple molecules in batch (more efficient).

    Args:
        mols: List of RDKit molecule objects
        core_smarts: SMARTS pattern for core

    Returns:
        List of dictionaries with decomposition results
    """
    from rdkit import Chem
    from rdkit.Chem.rdRGroupDecomposition import RGroupDecomposition, RGroupDecompositionParameters

    core_mol = Chem.MolFromSmarts(core_smarts)
    if core_mol is None:
        raise ValueError(f"Invalid core SMARTS: {core_smarts}")

    params = RGroupDecompositionParameters()
    params.removeAllHydrogenRGroups = True
    params.removeHydrogensPostMatch = True

    decomp = RGroupDecomposition(core_mol, params)

    # Add all molecules
    match_indices = []
    for mol in mols:
        if mol is not None:
            idx = decomp.Add(mol)
            match_indices.append(idx)
        else:
            match_indices.append(-1)

    # Process
    decomp.Process()
    rgroups = decomp.GetRGroupsAsColumns()

    # Build results
    results = []
    matched_count = 0
    for i, mol in enumerate(mols):
        if match_indices[i] >= 0:
            result = {
                "smiles": Chem.MolToSmiles(mol) if mol else "",
                "matched": True,
            }

            # Get core
            if "Core" in rgroups and matched_count < len(rgroups["Core"]):
                core_mol = rgroups["Core"][matched_count]
                result["core"] = Chem.MolToSmiles(core_mol) if core_mol else ""
            else:
                result["core"] = ""

            # Get R-groups dynamically
            for key in rgroups:
                if key.startswith("R") and matched_count < len(rgroups[key]):
                    rg_mol = rgroups[key][matched_count]
                    result[key] = Chem.MolToSmiles(rg_mol) if rg_mol else "[H]"

            matched_count += 1
            results.append(result)
        else:
            results.append({
                "smiles": Chem.MolToSmiles(mol) if mol else "",
                "matched": False,
            })

    return results
