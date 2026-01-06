"""Molecular enumeration engine."""

from typing import Optional, Any

from rdkit import Chem
from rdkit.Chem import AllChem, EnumerateStereoisomers
from rdkit.Chem.MolStandardize import rdMolStandardize

from rdkit_cli.io.readers import MoleculeRecord


class StereoisomerEnumerator:
    """Enumerate stereoisomers of molecules."""

    def __init__(
        self,
        max_isomers: int = 32,
        include_given: bool = True,
        only_unassigned: bool = True,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize stereoisomer enumerator.

        Args:
            max_isomers: Maximum stereoisomers to generate
            include_given: Include the input stereoisomer
            only_unassigned: Only enumerate unassigned stereocenters
            include_smiles: Include original SMILES in output
            include_name: Include molecule name in output
        """
        self.max_isomers = max_isomers
        self.include_given = include_given
        self.only_unassigned = only_unassigned
        self.include_smiles = include_smiles
        self.include_name = include_name

    def enumerate(self, record: MoleculeRecord) -> list[dict[str, Any]]:
        """
        Enumerate stereoisomers.

        Args:
            record: MoleculeRecord to process

        Returns:
            List of dictionaries with stereoisomer SMILES
        """
        if record.mol is None:
            return []

        try:
            opts = EnumerateStereoisomers.StereoEnumerationOptions()
            opts.maxIsomers = self.max_isomers
            opts.onlyUnassigned = self.only_unassigned

            isomers = list(EnumerateStereoisomers.EnumerateStereoisomers(record.mol, opts))

            results = []
            for i, iso in enumerate(isomers[:self.max_isomers]):
                smi = Chem.MolToSmiles(iso, isomericSmiles=True)
                result: dict[str, Any] = {"smiles": smi}

                if self.include_name and record.name:
                    result["name"] = f"{record.name}_iso{i}"
                elif record.name:
                    result["name"] = record.name

                result["stereoisomer_idx"] = i
                result["original_smiles"] = record.smiles

                results.append(result)

            return results

        except Exception:
            return []


class TautomerEnumerator:
    """Enumerate tautomers of molecules."""

    def __init__(
        self,
        max_tautomers: int = 50,
        max_transforms: int = 1000,
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize tautomer enumerator.

        Args:
            max_tautomers: Maximum tautomers to generate
            max_transforms: Maximum transforms to apply
            include_smiles: Include original SMILES in output
            include_name: Include molecule name in output
        """
        self.max_tautomers = max_tautomers
        self.max_transforms = max_transforms
        self.include_smiles = include_smiles
        self.include_name = include_name

        # Create enumerator
        self._enumerator = rdMolStandardize.TautomerEnumerator()
        self._enumerator.SetMaxTautomers(max_tautomers)
        self._enumerator.SetMaxTransforms(max_transforms)

    def enumerate(self, record: MoleculeRecord) -> list[dict[str, Any]]:
        """
        Enumerate tautomers.

        Args:
            record: MoleculeRecord to process

        Returns:
            List of dictionaries with tautomer SMILES
        """
        if record.mol is None:
            return []

        try:
            tautomers = list(self._enumerator.Enumerate(record.mol))

            results = []
            for i, taut in enumerate(tautomers[:self.max_tautomers]):
                smi = Chem.MolToSmiles(taut, isomericSmiles=True)
                result: dict[str, Any] = {"smiles": smi}

                if self.include_name and record.name:
                    result["name"] = f"{record.name}_taut{i}"
                elif record.name:
                    result["name"] = record.name

                result["tautomer_idx"] = i
                result["original_smiles"] = record.smiles

                results.append(result)

            return results

        except Exception:
            return []


class CanonicalTautomerizer:
    """Get canonical tautomer of molecules."""

    def __init__(
        self,
        include_original: bool = False,
    ):
        """
        Initialize canonical tautomerizer.

        Args:
            include_original: Include original SMILES in output
        """
        self.include_original = include_original
        self._canonicalizer = rdMolStandardize.TautomerEnumerator()

    def canonicalize(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """
        Get canonical tautomer.

        Args:
            record: MoleculeRecord to process

        Returns:
            Dictionary with canonical tautomer or None
        """
        if record.mol is None:
            return None

        try:
            canonical = self._canonicalizer.Canonicalize(record.mol)
            smi = Chem.MolToSmiles(canonical, isomericSmiles=True)

            result: dict[str, Any] = {"smiles": smi}

            if record.name:
                result["name"] = record.name

            if self.include_original:
                result["original_smiles"] = record.smiles

            return result

        except Exception:
            return None
