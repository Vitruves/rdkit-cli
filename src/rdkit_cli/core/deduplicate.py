"""Molecular deduplication engine."""

from typing import Optional, Callable, Iterator

from rdkit import Chem

from rdkit_cli.io.readers import MoleculeRecord


def canonical_smiles_key(mol: Chem.Mol) -> str:
    """Get canonical SMILES as deduplication key."""
    return Chem.MolToSmiles(mol, canonical=True)


def inchi_key(mol: Chem.Mol) -> str:
    """Get InChI as deduplication key."""
    from rdkit.Chem.inchi import MolToInchi
    return MolToInchi(mol) or ""


def inchikey_key(mol: Chem.Mol) -> str:
    """Get InChIKey as deduplication key."""
    from rdkit.Chem.inchi import MolToInchiKey
    return MolToInchiKey(mol) or ""


def murcko_scaffold_key(mol: Chem.Mol) -> str:
    """Get Murcko scaffold SMILES as deduplication key."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True)


# Registry of key functions
KEY_FUNCTIONS = {
    "smiles": canonical_smiles_key,
    "inchi": inchi_key,
    "inchikey": inchikey_key,
    "scaffold": murcko_scaffold_key,
}


class Deduplicator:
    """Remove duplicate molecules from a dataset."""

    def __init__(
        self,
        key_type: str = "smiles",
        keep: str = "first",
    ):
        """
        Initialize deduplicator.

        Args:
            key_type: Type of key to use for comparison:
                - 'smiles': Canonical SMILES (default)
                - 'inchi': InChI string
                - 'inchikey': InChIKey
                - 'scaffold': Murcko scaffold SMILES
            keep: Which duplicate to keep:
                - 'first': Keep first occurrence (default)
                - 'last': Keep last occurrence
        """
        if key_type not in KEY_FUNCTIONS:
            raise ValueError(
                f"Unknown key_type: {key_type}. "
                f"Available: {list(KEY_FUNCTIONS.keys())}"
            )
        if keep not in ("first", "last"):
            raise ValueError(f"keep must be 'first' or 'last', got: {keep}")

        self.key_type = key_type
        self.key_func: Callable[[Chem.Mol], str] = KEY_FUNCTIONS[key_type]
        self.keep = keep

    def deduplicate(
        self,
        records: list[MoleculeRecord],
    ) -> tuple[list[MoleculeRecord], int]:
        """
        Remove duplicates from a list of records.

        Args:
            records: List of molecule records

        Returns:
            Tuple of (deduplicated records, number of duplicates removed)
        """
        if self.keep == "first":
            return self._deduplicate_keep_first(records)
        else:
            return self._deduplicate_keep_last(records)

    def _deduplicate_keep_first(
        self,
        records: list[MoleculeRecord],
    ) -> tuple[list[MoleculeRecord], int]:
        """Keep first occurrence of each unique molecule."""
        seen: set[str] = set()
        unique: list[MoleculeRecord] = []
        duplicates = 0

        for record in records:
            if record.mol is None:
                # Keep invalid molecules as-is
                unique.append(record)
                continue

            try:
                key = self.key_func(record.mol)
            except Exception:
                # Keep if we can't compute key
                unique.append(record)
                continue

            if key not in seen:
                seen.add(key)
                unique.append(record)
            else:
                duplicates += 1

        return unique, duplicates

    def _deduplicate_keep_last(
        self,
        records: list[MoleculeRecord],
    ) -> tuple[list[MoleculeRecord], int]:
        """Keep last occurrence of each unique molecule."""
        # Process in reverse, then reverse result
        seen: set[str] = set()
        unique: list[MoleculeRecord] = []
        duplicates = 0

        for record in reversed(records):
            if record.mol is None:
                unique.append(record)
                continue

            try:
                key = self.key_func(record.mol)
            except Exception:
                unique.append(record)
                continue

            if key not in seen:
                seen.add(key)
                unique.append(record)
            else:
                duplicates += 1

        unique.reverse()
        return unique, duplicates

    def deduplicate_stream(
        self,
        records: Iterator[MoleculeRecord],
    ) -> Iterator[MoleculeRecord]:
        """
        Deduplicate records as a stream (only supports keep='first').

        Args:
            records: Iterator of molecule records

        Yields:
            Unique records
        """
        if self.keep != "first":
            raise ValueError("Stream deduplication only supports keep='first'")

        seen: set[str] = set()

        for record in records:
            if record.mol is None:
                yield record
                continue

            try:
                key = self.key_func(record.mol)
            except Exception:
                yield record
                continue

            if key not in seen:
                seen.add(key)
                yield record

    @staticmethod
    def available_key_types() -> list[str]:
        """Return list of available key types."""
        return list(KEY_FUNCTIONS.keys())
