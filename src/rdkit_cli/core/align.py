"""3D Molecular alignment module."""

from typing import Optional

from rdkit_cli.io.readers import MoleculeRecord


class MoleculeAligner:
    """Align molecules to a reference structure."""

    def __init__(
        self,
        reference_mol,
        method: str = "mcs",
        use_crippen: bool = False,
    ):
        """
        Initialize molecule aligner.

        Args:
            reference_mol: Reference RDKit molecule with 3D coordinates
            method: Alignment method ('mcs' for MCS-based, 'o3a' for Open3DAlign)
            use_crippen: Use Crippen contributions for O3A alignment
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        self.reference_mol = reference_mol
        self.method = method
        self.use_crippen = use_crippen

        # Validate reference has 3D coordinates
        if reference_mol.GetNumConformers() == 0:
            raise ValueError("Reference molecule must have 3D coordinates")

        # Precompute reference fingerprint for MCS
        if method == "mcs":
            self.ref_pattern = None  # Will compute MCS per molecule

    def align(self, record: MoleculeRecord) -> Optional[dict]:
        """
        Align a molecule to the reference.

        Args:
            record: Molecule record with 3D coordinates

        Returns:
            Dictionary with aligned molecule info and RMSD
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolAlign

        if record.mol is None:
            return None

        mol = record.mol

        # Check for 3D coordinates
        if mol.GetNumConformers() == 0:
            # Generate 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)

        if mol.GetNumConformers() == 0:
            return None

        try:
            if self.method == "o3a":
                rmsd = self._align_o3a(mol)
            else:  # mcs
                rmsd = self._align_mcs(mol)

            if rmsd is None:
                return None

            return {
                "smiles": Chem.MolToSmiles(mol),
                "name": record.name if record.name else "",
                "rmsd": round(rmsd, 4),
                "mol": mol,  # Keep mol object for SDF output
            }

        except Exception:
            return None

    def _align_mcs(self, mol) -> Optional[float]:
        """Align using Maximum Common Substructure."""
        from rdkit.Chem import rdFMCS, AllChem, rdMolAlign

        try:
            # Find MCS
            mcs = rdFMCS.FindMCS(
                [self.reference_mol, mol],
                timeout=10,
                matchValences=False,
                ringMatchesRingOnly=True,
                completeRingsOnly=True,
            )

            if mcs.numAtoms < 3:
                return None

            # Get matching atoms
            mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
            if mcs_mol is None:
                return None

            ref_match = self.reference_mol.GetSubstructMatch(mcs_mol)
            mol_match = mol.GetSubstructMatch(mcs_mol)

            if not ref_match or not mol_match:
                return None

            # Create atom map
            atom_map = list(zip(mol_match, ref_match))

            # Align
            rmsd = AllChem.AlignMol(mol, self.reference_mol, atomMap=atom_map)
            return rmsd

        except Exception:
            return None

    def _align_o3a(self, mol) -> Optional[float]:
        """Align using Open3DAlign."""
        from rdkit.Chem import AllChem, rdMolAlign

        try:
            if self.use_crippen:
                # Crippen-based O3A
                pyO3A = rdMolAlign.GetCrippenO3A(mol, self.reference_mol)
            else:
                # MMFF-based O3A
                pyO3A = rdMolAlign.GetO3A(mol, self.reference_mol)

            pyO3A.Align()
            rmsd = pyO3A.Score()

            return rmsd

        except Exception:
            return None


def load_reference_molecule(path: str):
    """Load a reference molecule from file."""
    from pathlib import Path
    from rdkit import Chem

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".mol", ".sdf", ".mol2"):
        if suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(path))
        else:
            supplier = Chem.SDMolSupplier(str(path))
            mol = next(iter(supplier), None)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path))
    else:
        # Try as SMILES (generate 3D)
        with open(path) as f:
            smiles = f.read().strip().split()[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            from rdkit.Chem import AllChem
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)

    return mol


from rdkit import Chem  # Need this for type annotation


def calculate_rmsd(mol1: Chem.Mol, mol2: Chem.Mol, align: bool = True) -> Optional[float]:
    """
    Calculate RMSD between two molecules.

    Args:
        mol1: First molecule
        mol2: Second molecule
        align: Whether to align before calculating RMSD

    Returns:
        RMSD value or None if calculation fails
    """
    from rdkit.Chem import AllChem, rdMolAlign

    if mol1 is None or mol2 is None:
        return None

    if mol1.GetNumConformers() == 0 or mol2.GetNumConformers() == 0:
        return None

    try:
        if align:
            rmsd = AllChem.GetBestRMS(mol1, mol2)
        else:
            rmsd = rdMolAlign.CalcRMS(mol1, mol2)
        return rmsd
    except Exception:
        return None


def calculate_rmsd_symmetry(mol1: Chem.Mol, mol2: Chem.Mol) -> Optional[float]:
    """
    Calculate RMSD considering molecular symmetry.

    Args:
        mol1: First molecule
        mol2: Second molecule

    Returns:
        Best RMSD value considering symmetry
    """
    from rdkit.Chem import AllChem

    if mol1 is None or mol2 is None:
        return None

    if mol1.GetNumConformers() == 0 or mol2.GetNumConformers() == 0:
        return None

    try:
        # GetBestRMS considers symmetry
        rmsd = AllChem.GetBestRMS(mol1, mol2)
        return rmsd
    except Exception:
        return None
