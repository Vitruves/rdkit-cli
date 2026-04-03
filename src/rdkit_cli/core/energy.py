"""Force field energy calculation engine."""

from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from rdkit_cli.io.readers import MoleculeRecord


def _ensure_3d(mol: Chem.Mol) -> Chem.Mol:
    """Add 3D coords if missing."""
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, rdDistGeom.ETKDGv3())
    return mol


def compute_energy(
    mol: Chem.Mol,
    force_field: str = "mmff",
) -> float | None:
    """Compute single-point energy."""
    mol = _ensure_3d(mol)
    if mol.GetNumConformers() == 0:
        return None

    if force_field == "mmff":
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            return None
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    else:
        ff = AllChem.UFFGetMoleculeForceField(mol)

    if ff is None:
        return None
    return ff.CalcEnergy()


class EnergyCalculator:
    """Compute force field energies for molecules."""

    def __init__(self, force_field: str = "mmff"):
        self.force_field = force_field.lower()

    def compute(self, record: MoleculeRecord):
        if record.mol is None:
            return None

        try:
            mol = Chem.RWMol(record.mol)
            mol = _ensure_3d(mol)
            energy = compute_energy(mol, self.force_field)

            if energy is None:
                return None

            result = {
                "smiles": record.smiles,
                "energy": round(energy, 4),
                "force_field": self.force_field.upper(),
            }
            if record.name:
                result["name"] = record.name
            return result
        except Exception:
            return None


class EnergyMinimizer:
    """Minimize molecules and report energy."""

    def __init__(
        self,
        force_field: str = "mmff",
        max_iterations: int = 500,
    ):
        self.force_field = force_field.lower()
        self.max_iterations = max_iterations

    def minimize(self, record: MoleculeRecord):
        if record.mol is None:
            return None

        try:
            mol = Chem.AddHs(record.mol)
            mol = _ensure_3d(mol)
            if mol.GetNumConformers() == 0:
                return None

            # Energy before
            e_before = compute_energy(mol, self.force_field)

            # Minimize
            if self.force_field == "mmff":
                converged = AllChem.MMFFOptimizeMolecule(
                    mol, maxIters=self.max_iterations,
                )
            else:
                converged = AllChem.UFFOptimizeMolecule(
                    mol, maxIters=self.max_iterations,
                )

            e_after = compute_energy(mol, self.force_field)

            result = {
                "smiles": record.smiles,
                "energy_before": round(e_before, 4) if e_before else None,
                "energy_after": round(e_after, 4) if e_after else None,
                "converged": converged == 0,
                "mol": mol,
            }
            if record.name:
                result["name"] = record.name
            return result
        except Exception:
            return None
