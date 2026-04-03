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


class TorsionScanner:
    """Scan torsion angles and compute energy profiles."""

    def __init__(
        self,
        atom_indices: tuple[int, int, int, int],
        start_angle: float = -180.0,
        end_angle: float = 180.0,
        step: float = 10.0,
        force_field: str = "mmff",
    ):
        from rdkit.Chem import rdMolTransforms
        self.atom_indices = atom_indices
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.step = step
        self.force_field = force_field.lower()
        self._rdMolTransforms = rdMolTransforms

    def scan(self, record: MoleculeRecord):
        if record.mol is None:
            return None

        try:
            mol = Chem.AddHs(record.mol)

            # Generate 3D if needed
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol, rdDistGeom.ETKDGv3())
                AllChem.MMFFOptimizeMolecule(mol)

            if mol.GetNumConformers() == 0:
                return None

            i, j, k, l = self.atom_indices
            conf = mol.GetConformer()

            angles = []
            energies = []
            angle = self.start_angle
            while angle <= self.end_angle:
                self._rdMolTransforms.SetDihedralDeg(
                    conf, i, j, k, l, angle,
                )
                # Compute energy at this angle
                if self.force_field == "mmff":
                    props = AllChem.MMFFGetMoleculeProperties(mol)
                    if props is None:
                        return None
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
                else:
                    ff = AllChem.UFFGetMoleculeForceField(mol)

                if ff is None:
                    return None

                energy = ff.CalcEnergy()
                angles.append(round(angle, 1))
                energies.append(round(energy, 4))
                angle += self.step

            min_energy = min(energies)
            min_angle = angles[energies.index(min_energy)]

            result = {
                "smiles": record.smiles,
                "angles": str(angles),
                "energies": str(energies),
                "min_angle": min_angle,
                "min_energy": min_energy,
                "barrier": round(max(energies) - min_energy, 4),
            }
            if record.name:
                result["name"] = record.name
            return result
        except Exception:
            return None


class ConstrainedEmbedder:
    """Embed molecules constrained to a reference template."""

    def __init__(
        self,
        reference_file: str,
        force_field: str = "mmff",
        random_seed: int = 42,
    ):
        from rdkit.Chem import rdmolfiles

        self.force_field = force_field.lower()
        self.random_seed = random_seed

        # Load reference molecule
        try:
            ext = reference_file.rsplit(".", 1)[-1].lower()
            ref_mol = None
            if ext in ("sdf", "mol"):
                suppl = rdmolfiles.SDMolSupplier(
                    reference_file, removeHs=True,
                )
                ref_mol = next(iter(suppl), None)
            elif ext == "pdb":
                ref_mol = rdmolfiles.MolFromPDBFile(
                    reference_file, removeHs=True,
                )
        except OSError:
            ref_mol = None

        if ref_mol is None or ref_mol.GetNumConformers() == 0:
            raise ValueError(
                f"Cannot load 3D reference from {reference_file}"
            )
        self.ref_mol = ref_mol

    def embed(self, record: MoleculeRecord):
        if record.mol is None:
            return None

        try:
            mol = Chem.AddHs(record.mol)
            AllChem.ConstrainedEmbed(
                mol, self.ref_mol,
                randomseed=self.random_seed,
            )

            if mol.GetNumConformers() == 0:
                return None

            if self.force_field == "mmff":
                AllChem.MMFFOptimizeMolecule(mol)
            else:
                AllChem.UFFOptimizeMolecule(mol)

            result = {
                "smiles": record.smiles,
                "mol": mol,
            }
            if record.name:
                result["name"] = record.name
            return result

        except Exception:
            return None
