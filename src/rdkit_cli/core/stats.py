"""Molecular dataset statistics engine."""

from typing import Any, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


class DatasetStatistics:
    """Calculate statistics over a molecular dataset."""

    # Core properties to calculate
    PROPERTY_FUNCS = {
        "MolWt": Descriptors.MolWt,
        "LogP": Descriptors.MolLogP,
        "TPSA": Descriptors.TPSA,
        "NumHDonors": Descriptors.NumHDonors,
        "NumHAcceptors": Descriptors.NumHAcceptors,
        "NumRotatableBonds": Descriptors.NumRotatableBonds,
        "NumHeavyAtoms": Descriptors.HeavyAtomCount,
        "NumRings": rdMolDescriptors.CalcNumRings,
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    }

    def __init__(self, properties: Optional[list[str]] = None):
        """
        Initialize statistics calculator.

        Args:
            properties: List of properties to calculate. If None, uses default set.
        """
        if properties is None:
            self.properties = ["MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors"]
        else:
            self.properties = properties

    def calculate(self, mols: list[Optional[Chem.Mol]]) -> dict[str, Any]:
        """
        Calculate dataset statistics.

        Args:
            mols: List of molecules (may contain None)

        Returns:
            Dictionary with statistics
        """
        import statistics

        # Count valid/invalid
        valid_mols = [m for m in mols if m is not None]
        n_total = len(mols)
        n_valid = len(valid_mols)
        n_invalid = n_total - n_valid

        result = {
            "total_molecules": n_total,
            "valid_molecules": n_valid,
            "invalid_molecules": n_invalid,
            "validity_rate": round(n_valid / n_total, 4) if n_total > 0 else 0.0,
        }

        if n_valid == 0:
            return result

        # Calculate property statistics
        for prop_name in self.properties:
            if prop_name not in self.PROPERTY_FUNCS:
                continue

            func = self.PROPERTY_FUNCS[prop_name]
            values = []
            for mol in valid_mols:
                try:
                    val = func(mol)
                    if val is not None:
                        values.append(val)
                except Exception:
                    pass

            if values:
                result[f"{prop_name}_min"] = round(min(values), 2)
                result[f"{prop_name}_max"] = round(max(values), 2)
                result[f"{prop_name}_mean"] = round(statistics.mean(values), 2)
                result[f"{prop_name}_median"] = round(statistics.median(values), 2)
                if len(values) > 1:
                    result[f"{prop_name}_stdev"] = round(statistics.stdev(values), 2)
                else:
                    result[f"{prop_name}_stdev"] = 0.0

        return result

    @classmethod
    def available_properties(cls) -> list[str]:
        """Return list of available properties."""
        return list(cls.PROPERTY_FUNCS.keys())
