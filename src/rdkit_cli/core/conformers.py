"""Conformer generation engine."""

from typing import Optional, Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from rdkit_cli.io.readers import MoleculeRecord


class ConformerGenerator:
    """Generate 3D conformers for molecules."""

    def __init__(
        self,
        num_conformers: int = 10,
        method: str = "etkdgv3",
        optimize: bool = True,
        force_field: str = "mmff",
        max_iterations: int = 200,
        random_seed: int = 42,
    ):
        """
        Initialize conformer generator.

        Args:
            num_conformers: Number of conformers to generate
            method: Embedding method (etkdgv3, etkdgv2, etdg)
            optimize: Whether to optimize conformers
            force_field: Force field for optimization (mmff, uff)
            max_iterations: Maximum optimization iterations
            random_seed: Random seed for reproducibility
        """
        self.num_conformers = num_conformers
        self.method = method.lower()
        self.optimize = optimize
        self.force_field = force_field.lower()
        self.max_iterations = max_iterations
        self.random_seed = random_seed

        # Set up embedding parameters
        if self.method == "etkdgv3":
            self.params = rdDistGeom.ETKDGv3()
        elif self.method == "etkdgv2":
            self.params = rdDistGeom.ETKDGv2()
        elif self.method == "etdg":
            self.params = rdDistGeom.ETDG()
        else:
            raise ValueError(f"Unknown method: {method}")

        self.params.randomSeed = random_seed
        self.params.numThreads = 0  # Use all available threads

    def generate(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """
        Generate conformers for a molecule.

        Args:
            record: MoleculeRecord to process

        Returns:
            Dictionary with molecule and conformer info, or None if failed
        """
        if record.mol is None:
            return None

        try:
            # Add hydrogens
            mol = Chem.AddHs(record.mol)

            # Embed conformers
            conf_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=self.num_conformers,
                params=self.params,
            )

            if len(conf_ids) == 0:
                return None

            # Optimize if requested
            energies = []
            if self.optimize:
                if self.force_field == "mmff":
                    results = AllChem.MMFFOptimizeMoleculeConfs(
                        mol,
                        maxIters=self.max_iterations,
                        numThreads=0,
                    )
                    energies = [r[1] for r in results]
                elif self.force_field == "uff":
                    results = AllChem.UFFOptimizeMoleculeConfs(
                        mol,
                        maxIters=self.max_iterations,
                        numThreads=0,
                    )
                    energies = [r[1] for r in results]

            # Get lowest energy conformer
            if energies:
                best_conf = min(range(len(energies)), key=lambda i: energies[i])
                best_energy = energies[best_conf]
            else:
                best_conf = 0
                best_energy = None

            result: dict[str, Any] = {
                "smiles": record.smiles,
                "mol": mol,
                "num_conformers": len(conf_ids),
                "best_conformer": best_conf,
            }

            if best_energy is not None:
                result["energy"] = round(best_energy, 2)

            if record.name:
                result["name"] = record.name

            return result

        except Exception:
            return None


class ConformerOptimizer:
    """Optimize existing 3D structures."""

    def __init__(
        self,
        force_field: str = "mmff",
        max_iterations: int = 200,
    ):
        """
        Initialize conformer optimizer.

        Args:
            force_field: Force field (mmff, uff)
            max_iterations: Maximum iterations
        """
        self.force_field = force_field.lower()
        self.max_iterations = max_iterations

    def optimize(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """
        Optimize a molecule's 3D structure.

        Args:
            record: MoleculeRecord with 3D coordinates

        Returns:
            Dictionary with optimized molecule, or None if failed
        """
        if record.mol is None:
            return None

        try:
            mol = Chem.Mol(record.mol)

            # Check if molecule has 3D coordinates
            if mol.GetNumConformers() == 0:
                # Try to generate 3D structure
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, rdDistGeom.ETKDGv3())

            if mol.GetNumConformers() == 0:
                return None

            # Optimize
            if self.force_field == "mmff":
                result = AllChem.MMFFOptimizeMolecule(mol, maxIters=self.max_iterations)
                props = AllChem.MMFFGetMoleculeProperties(mol)
                if props:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
                    energy = ff.CalcEnergy() if ff else None
                else:
                    energy = None
            else:
                result = AllChem.UFFOptimizeMolecule(mol, maxIters=self.max_iterations)
                ff = AllChem.UFFGetMoleculeForceField(mol)
                energy = ff.CalcEnergy() if ff else None

            output: dict[str, Any] = {
                "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol)),
                "mol": mol,
            }

            if energy is not None:
                output["energy"] = round(energy, 2)

            if record.name:
                output["name"] = record.name

            return output

        except Exception:
            return None
