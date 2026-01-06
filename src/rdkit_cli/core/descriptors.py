"""Molecular descriptor computation engine."""

from dataclasses import dataclass
from typing import Optional, Any

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED

from rdkit_cli.io.readers import MoleculeRecord


# Descriptor categories
DESCRIPTOR_CATEGORIES = [
    "constitutional",
    "topological",
    "electronic",
    "geometric",
    "molecular",
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
    ):
        """
        Initialize descriptor calculator.

        Args:
            descriptors: List of descriptor names (None for all)
            include_smiles: Include SMILES in output
            include_name: Include molecule name in output
            precision: Decimal precision for float values
            error_value: Value to use for failed calculations
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

        result: dict[str, Any] = {}

        if self.include_smiles:
            result["smiles"] = record.smiles
        if self.include_name and record.name:
            result["name"] = record.name

        for desc_name in self.descriptors:
            value = compute_descriptor(record.mol, desc_name)
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
