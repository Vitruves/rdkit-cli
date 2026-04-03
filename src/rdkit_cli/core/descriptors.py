"""Molecular descriptor computation engine."""

from dataclasses import dataclass
from typing import Any, Optional

from rdkit import Chem
from rdkit.Chem import QED, AllChem, Descriptors, rdMolDescriptors

from rdkit_cli.io.readers import MoleculeRecord

# Descriptor categories
DESCRIPTOR_CATEGORIES = [
    "constitutional",
    "topological",
    "electronic",
    "geometric",
    "molecular",
    "mqn",
    "3d",
]


@dataclass
class DescriptorInfo:
    """Information about a descriptor."""

    name: str
    description: str
    category: str


# Build descriptor registry from RDKit
def _build_descriptor_registry() -> dict[str, tuple[callable, str, str]]:
    """Build registry of all available descriptors."""
    registry = {}

    # Get all descriptors from Descriptors module
    for name, func in Descriptors.descList:
        # Categorize based on name patterns
        category = "molecular"
        lower_name = name.lower()

        if any(x in lower_name for x in ["chi", "kappa", "hall", "balaban", "bertz"]):
            category = "topological"
        elif any(x in lower_name for x in ["tpsa", "labute", "peoe", "gasteiger"]):
            category = "electronic"
        elif any(x in lower_name for x in ["num", "count", "heavy", "ring", "rotatable"]):
            category = "constitutional"
        elif any(x in lower_name for x in ["mol", "exact", "weight", "logp", "mr"]):
            category = "molecular"

        registry[name] = (func, f"RDKit descriptor: {name}", category)

    return registry


DESCRIPTOR_REGISTRY = _build_descriptor_registry()

# Add QED (not in Descriptors.descList)
DESCRIPTOR_REGISTRY["QED"] = (QED.qed, "Quantitative Estimate of Drug-likeness", "molecular")


# --- MQN descriptors (42 Molecular Quantum Numbers) ---
_MQN_NAMES = [
    "MQN1", "MQN2", "MQN3", "MQN4", "MQN5", "MQN6", "MQN7", "MQN8",
    "MQN9", "MQN10", "MQN11", "MQN12", "MQN13", "MQN14", "MQN15", "MQN16",
    "MQN17", "MQN18", "MQN19", "MQN20", "MQN21", "MQN22", "MQN23", "MQN24",
    "MQN25", "MQN26", "MQN27", "MQN28", "MQN29", "MQN30", "MQN31", "MQN32",
    "MQN33", "MQN34", "MQN35", "MQN36", "MQN37", "MQN38", "MQN39", "MQN40",
    "MQN41", "MQN42",
]


def _make_mqn_func(index: int):
    """Create a function that returns a specific MQN index."""
    def mqn_func(mol: Chem.Mol) -> float:
        return float(rdMolDescriptors.MQNs_(mol)[index])
    return mqn_func


for _i, _name in enumerate(_MQN_NAMES):
    DESCRIPTOR_REGISTRY[_name] = (
        _make_mqn_func(_i),
        f"Molecular Quantum Number {_i + 1}",
        "mqn",
    )


# --- 3D descriptors (require 3D coordinates) ---
_3D_DESCRIPTORS: list[tuple[str, callable, str]] = [
    ("PMI1", rdMolDescriptors.CalcPMI1, "First principal moment of inertia"),
    ("PMI2", rdMolDescriptors.CalcPMI2, "Second principal moment of inertia"),
    ("PMI3", rdMolDescriptors.CalcPMI3, "Third principal moment of inertia"),
    ("NPR1", rdMolDescriptors.CalcNPR1, "Normalized principal moments ratio 1"),
    ("NPR2", rdMolDescriptors.CalcNPR2, "Normalized principal moments ratio 2"),
    ("Asphericity", rdMolDescriptors.CalcAsphericity, "Asphericity"),
    ("Eccentricity", rdMolDescriptors.CalcEccentricity, "Eccentricity"),
    ("InertialShapeFactor", rdMolDescriptors.CalcInertialShapeFactor, "Inertial shape factor"),
    ("SpherocityIndex", rdMolDescriptors.CalcSpherocityIndex, "Spherocity index"),
    ("PBF", rdMolDescriptors.CalcPBF, "Plane of best fit"),
]

for _name, _func, _desc in _3D_DESCRIPTORS:
    DESCRIPTOR_REGISTRY[_name] = (_func, _desc, "3d")


def _needs_3d(descriptor_names: list[str]) -> bool:
    """Check if any requested descriptors require 3D coordinates."""
    three_d_names = {name for name, _, _ in _3D_DESCRIPTORS}
    return bool(set(descriptor_names) & three_d_names)


def _ensure_3d(mol: Chem.Mol) -> Chem.Mol:
    """Generate 3D coordinates if molecule lacks them."""
    if mol.GetNumConformers() == 0:
        mol = Chem.RWMol(mol)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
    return mol


def compute_lipinski_violations(mol: Chem.Mol) -> int:
    """
    Count Lipinski Rule of 5 violations.

    Args:
        mol: RDKit molecule

    Returns:
        Number of violations (0-4)
    """
    violations = 0

    if Descriptors.MolWt(mol) > 500:
        violations += 1
    if Descriptors.MolLogP(mol) > 5:
        violations += 1
    if Descriptors.NumHDonors(mol) > 5:
        violations += 1
    if Descriptors.NumHAcceptors(mol) > 10:
        violations += 1

    return violations


def list_descriptors(
    category: Optional[str] = None,
    verbose: bool = False,
) -> list[DescriptorInfo]:
    """
    List available descriptors.

    Args:
        category: Filter by category
        verbose: Include descriptions

    Returns:
        List of DescriptorInfo objects
    """
    result = []

    for name, (func, desc, cat) in sorted(DESCRIPTOR_REGISTRY.items()):
        if category is None or cat == category:
            result.append(DescriptorInfo(name=name, description=desc, category=cat))

    return result


def compute_descriptor(mol: Chem.Mol, name: str) -> Optional[float]:
    """
    Compute a single descriptor for a molecule.

    Args:
        mol: RDKit molecule
        name: Descriptor name

    Returns:
        Descriptor value or None if computation failed
    """
    if name not in DESCRIPTOR_REGISTRY:
        raise ValueError(f"Unknown descriptor: {name}")

    func = DESCRIPTOR_REGISTRY[name][0]

    try:
        value = func(mol)
        # Handle NaN and inf
        if value is None or (isinstance(value, float) and (value != value or abs(value) == float("inf"))):
            return None
        return float(value)
    except Exception:
        return None


class DescriptorCalculator:
    """Calculator for molecular descriptors."""

    def __init__(
        self,
        descriptors: Optional[list[str]] = None,
        include_smiles: bool = True,
        include_name: bool = True,
        precision: int = 4,
        error_value: str = "NaN",
        generate_conformers: bool = False,
    ):
        """
        Initialize descriptor calculator.

        Args:
            descriptors: List of descriptor names (None for all)
            include_smiles: Include SMILES in output
            include_name: Include molecule name in output
            precision: Decimal precision for float values
            error_value: Value to use for failed calculations
            generate_conformers: Auto-generate 3D coords for 3D descriptors
        """
        if descriptors is None:
            self.descriptors = list(DESCRIPTOR_REGISTRY.keys())
        else:
            # Validate descriptor names
            unknown = set(descriptors) - set(DESCRIPTOR_REGISTRY.keys())
            if unknown:
                raise ValueError(f"Unknown descriptors: {', '.join(unknown)}")
            self.descriptors = descriptors

        self.include_smiles = include_smiles
        self.include_name = include_name
        self.precision = precision
        self.error_value = error_value
        self.generate_conformers = generate_conformers
        self._has_3d = _needs_3d(self.descriptors)

    def _format_value(self, value: Optional[float]) -> Any:
        """Format a descriptor value with precision and error handling."""
        if value is None:
            return self.error_value
        if isinstance(value, float):
            return round(value, self.precision)
        return value

    def compute(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """
        Compute descriptors for a molecule record.

        Args:
            record: MoleculeRecord to process

        Returns:
            Dictionary with descriptor values or None if molecule is invalid
        """
        if record.mol is None:
            return None

        mol = record.mol
        if self._has_3d and self.generate_conformers:
            try:
                mol = _ensure_3d(mol)
            except Exception:
                pass  # Will get NaN for 3D descriptors

        result: dict[str, Any] = {}

        if self.include_smiles:
            result["smiles"] = record.smiles
        if self.include_name and record.name:
            result["name"] = record.name

        for desc_name in self.descriptors:
            value = compute_descriptor(mol, desc_name)
            result[desc_name] = self._format_value(value)

        return result

    def get_column_names(self) -> list[str]:
        """Get output column names in order."""
        cols = []
        if self.include_smiles:
            cols.append("smiles")
        if self.include_name:
            cols.append("name")
        cols.extend(self.descriptors)
        return cols


# Common descriptor sets
COMMON_DESCRIPTORS = [
    "MolWt",
    "ExactMolWt",
    "HeavyAtomCount",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "NumHeteroatoms",
    "NumAromaticRings",
    "RingCount",
    "TPSA",
    "MolLogP",
    "MolMR",
    "FractionCSP3",
]

LIPINSKI_DESCRIPTORS = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
]

DRUGLIKE_DESCRIPTORS = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "TPSA",
    "NumRotatableBonds",
    "RingCount",
    "HeavyAtomCount",
]

MQN_DESCRIPTORS = _MQN_NAMES

THREE_D_DESCRIPTORS = [name for name, _, _ in _3D_DESCRIPTORS]
