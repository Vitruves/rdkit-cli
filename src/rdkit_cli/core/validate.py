"""Molecular structure validation engine."""

from typing import Optional, Any

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


class ValidationResult:
    """Result of validating a single molecule."""

    __slots__ = ("is_valid", "errors", "warnings", "info")

    def __init__(self):
        self.is_valid = True
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: dict[str, Any] = {}

    def add_error(self, message: str) -> None:
        """Add an error (makes molecule invalid)."""
        self.is_valid = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Add a warning (molecule still valid)."""
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": ";".join(self.errors) if self.errors else "",
            "warnings": ";".join(self.warnings) if self.warnings else "",
            **self.info,
        }


class MoleculeValidator:
    """Validate molecular structures."""

    def __init__(
        self,
        check_valence: bool = True,
        check_kekulize: bool = True,
        check_stereo: bool = False,
        check_atoms: bool = True,
        max_atoms: Optional[int] = None,
        max_rings: Optional[int] = None,
        allowed_elements: Optional[set[str]] = None,
    ):
        """
        Initialize validator.

        Args:
            check_valence: Check for valence errors
            check_kekulize: Check if molecule can be kekulized
            check_stereo: Check for undefined stereocenters
            check_atoms: Check for unusual atom states
            max_atoms: Maximum allowed atoms (None = no limit)
            max_rings: Maximum allowed rings (None = no limit)
            allowed_elements: Set of allowed element symbols (None = all allowed)
        """
        self.check_valence = check_valence
        self.check_kekulize = check_kekulize
        self.check_stereo = check_stereo
        self.check_atoms = check_atoms
        self.max_atoms = max_atoms
        self.max_rings = max_rings
        self.allowed_elements = allowed_elements

    def validate(self, mol: Optional[Chem.Mol], smiles: str = "") -> ValidationResult:
        """
        Validate a molecule.

        Args:
            mol: RDKit molecule object (or None if parsing failed)
            smiles: Original SMILES string for error reporting

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()

        # Check if molecule was parsed
        if mol is None:
            result.add_error("Failed to parse SMILES")
            return result

        # Store basic info
        result.info["num_atoms"] = mol.GetNumAtoms()
        result.info["num_heavy_atoms"] = mol.GetNumHeavyAtoms()
        result.info["num_rings"] = rdMolDescriptors.CalcNumRings(mol)

        # Check atom count
        if self.max_atoms is not None and mol.GetNumAtoms() > self.max_atoms:
            result.add_error(f"Too many atoms: {mol.GetNumAtoms()} > {self.max_atoms}")

        # Check ring count
        if self.max_rings is not None:
            n_rings = result.info["num_rings"]
            if n_rings > self.max_rings:
                result.add_error(f"Too many rings: {n_rings} > {self.max_rings}")

        # Check valence
        if self.check_valence:
            self._check_valence(mol, result)

        # Check kekulization
        if self.check_kekulize:
            self._check_kekulize(mol, result)

        # Check stereo
        if self.check_stereo:
            self._check_stereo(mol, result)

        # Check atoms
        if self.check_atoms:
            self._check_atoms(mol, result)

        # Check allowed elements
        if self.allowed_elements is not None:
            self._check_elements(mol, result)

        return result

    def _check_valence(self, mol: Chem.Mol, result: ValidationResult) -> None:
        """Check for valence errors."""
        try:
            problems = Chem.DetectChemistryProblems(mol)
            for problem in problems:
                if "valence" in problem.GetType().lower():
                    result.add_error(f"Valence error: {problem.Message()}")
        except Exception as e:
            result.add_warning(f"Valence check failed: {e}")

    def _check_kekulize(self, mol: Chem.Mol, result: ValidationResult) -> None:
        """Check if molecule can be kekulized."""
        try:
            # Try to kekulize a copy
            mol_copy = Chem.RWMol(mol)
            Chem.Kekulize(mol_copy)
        except Exception as e:
            result.add_error(f"Kekulization failed: {e}")

    def _check_stereo(self, mol: Chem.Mol, result: ValidationResult) -> None:
        """Check for undefined stereocenters."""
        # Check chiral centers
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        undefined_chiral = [c for c in chiral_centers if c[1] == "?"]
        if undefined_chiral:
            atoms = [str(c[0]) for c in undefined_chiral]
            result.add_warning(f"Undefined chiral centers at atoms: {', '.join(atoms)}")
            result.info["undefined_stereocenters"] = len(undefined_chiral)

        # Check double bond stereo
        undefined_db = 0
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                stereo = bond.GetStereo()
                if stereo == Chem.BondStereo.STEREONONE:
                    # Check if it could have stereo
                    begin = bond.GetBeginAtom()
                    end = bond.GetEndAtom()
                    if begin.GetDegree() > 1 and end.GetDegree() > 1:
                        undefined_db += 1

        if undefined_db > 0:
            result.add_warning(f"Potentially undefined double bond stereochemistry: {undefined_db}")
            result.info["undefined_double_bonds"] = undefined_db

    def _check_atoms(self, mol: Chem.Mol, result: ValidationResult) -> None:
        """Check for unusual atom states."""
        for atom in mol.GetAtoms():
            # Check for unusual charges
            charge = atom.GetFormalCharge()
            if abs(charge) > 2:
                result.add_warning(
                    f"Unusual charge {charge} on {atom.GetSymbol()} at index {atom.GetIdx()}"
                )

            # Check for unusual valence (already covered by DetectChemistryProblems but
            # we can add more specific checks here)

            # Check for radicals
            if atom.GetNumRadicalElectrons() > 0:
                result.add_warning(
                    f"Radical electron on {atom.GetSymbol()} at index {atom.GetIdx()}"
                )

    def _check_elements(self, mol: Chem.Mol, result: ValidationResult) -> None:
        """Check for disallowed elements."""
        present_elements = set()
        for atom in mol.GetAtoms():
            present_elements.add(atom.GetSymbol())

        disallowed = present_elements - self.allowed_elements
        if disallowed:
            result.add_error(f"Disallowed elements: {', '.join(sorted(disallowed))}")
            result.info["disallowed_elements"] = sorted(disallowed)


def validate_smiles(smiles: str) -> tuple[bool, Optional[Chem.Mol], str]:
    """
    Quick validation of a SMILES string.

    Args:
        smiles: SMILES string to validate

    Returns:
        Tuple of (is_valid, mol_or_none, error_message)
    """
    if not smiles or not smiles.strip():
        return False, None, "Empty SMILES"

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None, "Failed to parse SMILES"
        return True, mol, ""
    except Exception as e:
        return False, None, str(e)
