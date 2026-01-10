"""Merge multiple molecule files."""

from pathlib import Path
from typing import Iterator, Optional

from rdkit_cli.io.readers import MoleculeRecord


class MoleculeMerger:
    """Merge molecules from multiple files."""

    def __init__(
        self,
        deduplicate: bool = False,
        dedupe_key: str = "smiles",
        add_source: bool = False,
    ):
        """
        Initialize merger.

        Args:
            deduplicate: Whether to remove duplicates
            dedupe_key: Key to use for deduplication (smiles, inchi, inchikey)
            add_source: Whether to add source file column
        """
        self.deduplicate = deduplicate
        self.dedupe_key = dedupe_key
        self.add_source = add_source
        self._seen: set[str] = set()

    def _get_dedupe_key(self, record: MoleculeRecord) -> Optional[str]:
        """Get deduplication key for a record."""
        if record.mol is None:
            return None

        if self.dedupe_key == "smiles":
            from rdkit import Chem
            return Chem.MolToSmiles(record.mol, canonical=True)
        elif self.dedupe_key == "inchi":
            from rdkit.Chem import inchi
            try:
                return inchi.MolToInchi(record.mol)
            except Exception:
                return None
        elif self.dedupe_key == "inchikey":
            from rdkit.Chem import inchi
            try:
                return inchi.MolToInchiKey(record.mol)
            except Exception:
                return None
        else:
            return record.smiles

    def merge_files(
        self,
        input_paths: list[Path],
        smiles_column: str = "smiles",
        name_column: Optional[str] = None,
        has_header: bool = True,
    ) -> Iterator[dict]:
        """
        Merge molecules from multiple files.

        Args:
            input_paths: List of input file paths
            smiles_column: Name of SMILES column
            name_column: Name of name column
            has_header: Whether files have headers

        Yields:
            Merged molecule records as dicts
        """
        from rdkit_cli.io import create_reader

        for input_path in input_paths:
            reader = create_reader(
                input_path,
                smiles_column=smiles_column,
                name_column=name_column,
                has_header=has_header,
            )

            source_name = input_path.name

            with reader:
                for record in reader:
                    if record.mol is None:
                        continue

                    # Check for duplicates
                    if self.deduplicate:
                        key = self._get_dedupe_key(record)
                        if key is None or key in self._seen:
                            continue
                        self._seen.add(key)

                    result = {
                        "smiles": record.smiles,
                    }

                    if record.name:
                        result["name"] = record.name

                    if self.add_source:
                        result["source_file"] = source_name

                    # Copy metadata (excluding smiles and name)
                    for key, value in record.metadata.items():
                        if key.lower() not in ("smiles", "name", smiles_column.lower()):
                            if name_column is None or key.lower() != name_column.lower():
                                result[key] = value

                    yield result

    def get_stats(self) -> dict:
        """Get merge statistics."""
        return {
            "unique_molecules": len(self._seen) if self.deduplicate else 0,
        }
