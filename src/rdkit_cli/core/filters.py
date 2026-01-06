"""Molecular filtering engine."""

from dataclasses import dataclass
from typing import Optional, Any, Callable

from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, rdfiltercatalog

from rdkit_cli.io.readers import MoleculeRecord


@dataclass
class FilterResult:
    """Result of a filter check."""

    passed: bool
    reason: Optional[str] = None


# Drug-likeness rules
DRUGLIKE_RULES = {
    "lipinski": {
        "MolWt": (None, 500),
        "MolLogP": (None, 5),
        "NumHDonors": (None, 5),
        "NumHAcceptors": (None, 10),
    },
    "veber": {
        "NumRotatableBonds": (None, 10),
        "TPSA": (None, 140),
    },
    "ghose": {
        "MolWt": (160, 480),
        "MolLogP": (-0.4, 5.6),
        "NumAtoms": (20, 70),
        "MolMR": (40, 130),
    },
    "egan": {
        "MolLogP": (None, 5.88),
        "TPSA": (None, 131.6),
    },
    "muegge": {
        "MolWt": (200, 600),
        "MolLogP": (-2, 5),
        "TPSA": (None, 150),
        "RingCount": (None, 7),
        "NumHDonors": (None, 5),
        "NumHAcceptors": (None, 10),
        "NumRotatableBonds": (None, 15),
    },
}


def check_property_range(
    mol: Chem.Mol,
    property_name: str,
    min_val: Optional[float],
    max_val: Optional[float],
) -> bool:
    """Check if a property is within range."""
    # Get property function
    if property_name == "NumAtoms":
        value = mol.GetNumAtoms()
    elif hasattr(Descriptors, property_name):
        func = getattr(Descriptors, property_name)
        value = func(mol)
    else:
        return True  # Unknown property, pass

    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False

    return True


def check_druglike_rules(mol: Chem.Mol, rule_name: str) -> FilterResult:
    """
    Check drug-likeness rules.

    Args:
        mol: RDKit molecule
        rule_name: Name of rule set (lipinski, veber, etc.)

    Returns:
        FilterResult with pass/fail status
    """
    if rule_name not in DRUGLIKE_RULES:
        raise ValueError(f"Unknown rule: {rule_name}")

    rules = DRUGLIKE_RULES[rule_name]
    violations = []

    for prop, (min_val, max_val) in rules.items():
        if not check_property_range(mol, prop, min_val, max_val):
            violations.append(prop)

    if violations:
        return FilterResult(passed=False, reason=f"Failed: {', '.join(violations)}")

    return FilterResult(passed=True)


class SubstructureFilter:
    """Filter molecules by substructure."""

    def __init__(
        self,
        smarts: str,
        exclude: bool = False,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize substructure filter.

        Args:
            smarts: SMARTS pattern to match
            exclude: If True, exclude matching molecules
            include_smiles: Include SMILES in output
            include_name: Include molecule name in output
        """
        self.pattern = Chem.MolFromSmarts(smarts)
        if self.pattern is None:
            raise ValueError(f"Invalid SMARTS pattern: {smarts}")

        self.exclude = exclude
        self.include_smiles = include_smiles
        self.include_name = include_name

    def filter(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """
        Filter a molecule record.

        Args:
            record: MoleculeRecord to check

        Returns:
            Dictionary if molecule passes filter, None otherwise
        """
        if record.mol is None:
            return None

        has_match = record.mol.HasSubstructMatch(self.pattern)

        # If exclude=True, we want molecules WITHOUT the match
        # If exclude=False, we want molecules WITH the match
        passes = (self.exclude and not has_match) or (not self.exclude and has_match)

        if not passes:
            return None

        result: dict[str, Any] = {}
        if self.include_smiles:
            result["smiles"] = record.smiles
        if self.include_name and record.name:
            result["name"] = record.name

        # Copy other metadata
        for key, value in record.metadata.items():
            if key not in result:
                result[key] = value

        return result


class PropertyFilter:
    """Filter molecules by property values."""

    def __init__(
        self,
        rules: dict[str, tuple[Optional[float], Optional[float]]],
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize property filter.

        Args:
            rules: Dictionary of property_name -> (min_val, max_val)
            include_smiles: Include SMILES in output
            include_name: Include molecule name in output
        """
        self.rules = rules
        self.include_smiles = include_smiles
        self.include_name = include_name

    def filter(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """Filter a molecule record."""
        if record.mol is None:
            return None

        for prop, (min_val, max_val) in self.rules.items():
            if not check_property_range(record.mol, prop, min_val, max_val):
                return None

        result: dict[str, Any] = {}
        if self.include_smiles:
            result["smiles"] = record.smiles
        if self.include_name and record.name:
            result["name"] = record.name

        for key, value in record.metadata.items():
            if key not in result:
                result[key] = value

        return result


class DruglikeFilter:
    """Filter molecules by drug-likeness rules."""

    def __init__(
        self,
        rule_name: str = "lipinski",
        max_violations: int = 0,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize drug-likeness filter.

        Args:
            rule_name: Rule set to use
            max_violations: Maximum allowed violations
            include_smiles: Include SMILES in output
            include_name: Include molecule name in output
        """
        if rule_name not in DRUGLIKE_RULES:
            raise ValueError(f"Unknown rule: {rule_name}. Available: {', '.join(DRUGLIKE_RULES.keys())}")

        self.rule_name = rule_name
        self.max_violations = max_violations
        self.include_smiles = include_smiles
        self.include_name = include_name

    def filter(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """Filter a molecule record."""
        if record.mol is None:
            return None

        rules = DRUGLIKE_RULES[self.rule_name]
        violations = 0

        for prop, (min_val, max_val) in rules.items():
            if not check_property_range(record.mol, prop, min_val, max_val):
                violations += 1

        if violations > self.max_violations:
            return None

        result: dict[str, Any] = {}
        if self.include_smiles:
            result["smiles"] = record.smiles
        if self.include_name and record.name:
            result["name"] = record.name

        for key, value in record.metadata.items():
            if key not in result:
                result[key] = value

        return result


class PAINSFilter:
    """Filter molecules for PAINS (Pan-Assay Interference Compounds)."""

    def __init__(
        self,
        exclude: bool = True,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """Initialize PAINS filter."""
        self.exclude = exclude
        self.include_smiles = include_smiles
        self.include_name = include_name

        # Initialize PAINS catalog
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        self.catalog = FilterCatalog.FilterCatalog(params)

    def filter(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """Filter a molecule record (returns None if PAINS hit and exclude=True)."""
        if record.mol is None:
            return None

        # Check for PAINS
        entry = self.catalog.GetFirstMatch(record.mol)
        is_pains = entry is not None

        # If exclude=True (default), filter out PAINS hits
        # If exclude=False, keep only PAINS hits
        if self.exclude and is_pains:
            return None
        if not self.exclude and not is_pains:
            return None

        result: dict[str, Any] = {}
        if self.include_smiles:
            result["smiles"] = record.smiles
        if self.include_name and record.name:
            result["name"] = record.name

        for key, value in record.metadata.items():
            if key not in result:
                result[key] = value

        return result


class ElementFilter:
    """Filter molecules by allowed/required/forbidden elements."""

    def __init__(
        self,
        allowed_elements: Optional[list[str]] = None,
        required_elements: Optional[list[str]] = None,
        forbidden_elements: Optional[list[str]] = None,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize element filter.

        Args:
            allowed_elements: Only these elements are allowed
            required_elements: Molecule must contain all of these
            forbidden_elements: Molecule must not contain any of these
        """
        self.allowed = set(allowed_elements) if allowed_elements else None
        self.required = set(required_elements) if required_elements else None
        self.forbidden = set(forbidden_elements) if forbidden_elements else None
        self.include_smiles = include_smiles
        self.include_name = include_name

    def filter(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """Filter a molecule record by elements."""
        if record.mol is None:
            return None

        # Get elements in molecule
        elements = set()
        for atom in record.mol.GetAtoms():
            elements.add(atom.GetSymbol())

        # Check allowed
        if self.allowed is not None:
            if not elements.issubset(self.allowed):
                return None

        # Check required
        if self.required is not None:
            if not self.required.issubset(elements):
                return None

        # Check forbidden
        if self.forbidden is not None:
            if elements.intersection(self.forbidden):
                return None

        result: dict[str, Any] = {}
        if self.include_smiles:
            result["smiles"] = record.smiles
        if self.include_name and record.name:
            result["name"] = record.name

        for key, value in record.metadata.items():
            if key not in result:
                result[key] = value

        return result


class ComplexityFilter:
    """Filter molecules by complexity measures."""

    def __init__(
        self,
        min_atoms: int = 1,
        max_atoms: int = 100,
        min_rings: int = 0,
        max_rings: int = 10,
        min_rotatable: int = 0,
        max_rotatable: int = 20,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize complexity filter.

        Args:
            min_atoms: Minimum heavy atom count
            max_atoms: Maximum heavy atom count
            min_rings: Minimum ring count
            max_rings: Maximum ring count
            min_rotatable: Minimum rotatable bonds
            max_rotatable: Maximum rotatable bonds
        """
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.min_rings = min_rings
        self.max_rings = max_rings
        self.min_rotatable = min_rotatable
        self.max_rotatable = max_rotatable
        self.include_smiles = include_smiles
        self.include_name = include_name

    def filter(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """Filter a molecule record by complexity."""
        if record.mol is None:
            return None

        mol = record.mol

        # Check heavy atom count
        heavy_atoms = mol.GetNumHeavyAtoms()
        if heavy_atoms < self.min_atoms or heavy_atoms > self.max_atoms:
            return None

        # Check ring count
        ring_count = Descriptors.RingCount(mol)
        if ring_count < self.min_rings or ring_count > self.max_rings:
            return None

        # Check rotatable bonds
        rotatable = Descriptors.NumRotatableBonds(mol)
        if rotatable < self.min_rotatable or rotatable > self.max_rotatable:
            return None

        result: dict[str, Any] = {}
        if self.include_smiles:
            result["smiles"] = record.smiles
        if self.include_name and record.name:
            result["name"] = record.name

        for key, value in record.metadata.items():
            if key not in result:
                result[key] = value

        return result
