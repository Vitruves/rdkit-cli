"""File readers for various molecular file formats."""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Any

import pandas as pd
from rdkit import Chem

from rdkit_cli.io.formats import FileFormat, FormatConfig, detect_format


class MoleculeRecord:
    """A molecule with its associated metadata."""

    __slots__ = ("mol", "smiles", "name", "metadata", "row_idx")

    def __init__(
        self,
        mol: Optional[Chem.Mol],
        smiles: str = "",
        name: str = "",
        metadata: Optional[dict[str, Any]] = None,
        row_idx: int = -1,
    ):
        self.mol = mol
        self.smiles = smiles
        self.name = name
        self.metadata = metadata or {}
        self.row_idx = row_idx

    @property
    def is_valid(self) -> bool:
        """Check if molecule was parsed successfully."""
        return self.mol is not None


class MoleculeReader(ABC):
    """Abstract base class for molecule file readers."""

    @abstractmethod
    def __iter__(self) -> Iterator[MoleculeRecord]:
        """Yield MoleculeRecord objects."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of molecules (for progress)."""
        pass

    @abstractmethod
    def close(self):
        """Close any open resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CSVReader(MoleculeReader):
    """Read molecules from CSV/TSV files."""

    def __init__(
        self,
        path: Path | str,
        smiles_column: str = "smiles",
        name_column: Optional[str] = None,
        delimiter: str = ",",
        has_header: bool = True,
    ):
        self.path = Path(path)
        self.smiles_column = smiles_column
        self.name_column = name_column
        self.delimiter = delimiter
        self.has_header = has_header
        self._count: Optional[int] = None
        self._df: Optional[pd.DataFrame] = None

    def __len__(self) -> int:
        if self._count is None:
            # Count lines efficiently
            with open(self.path, "rb") as f:
                self._count = sum(1 for _ in f) - (1 if self.has_header else 0)
        return self._count

    def __iter__(self) -> Iterator[MoleculeRecord]:
        header = 0 if self.has_header else None

        # Read in chunks for memory efficiency
        for chunk in pd.read_csv(
            self.path,
            delimiter=self.delimiter,
            header=header,
            chunksize=10000,
            dtype=str,
            na_filter=False,
        ):
            # Handle no-header case
            if not self.has_header:
                # Assume first column is SMILES
                chunk.columns = [self.smiles_column] + [f"col_{i}" for i in range(1, len(chunk.columns))]

            for idx, row in chunk.iterrows():
                smiles = str(row.get(self.smiles_column, ""))
                name = str(row.get(self.name_column, "")) if self.name_column else ""

                mol = None
                if smiles:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                    except Exception:
                        pass

                if mol is None and smiles:
                    print(f"Warning: Failed to parse SMILES at row {idx}: {smiles[:50]}", file=sys.stderr)

                yield MoleculeRecord(
                    mol=mol,
                    smiles=smiles,
                    name=name,
                    metadata=row.to_dict(),
                    row_idx=int(idx),
                )

    def close(self):
        pass


class SMIReader(MoleculeReader):
    """Read molecules from SMILES files."""

    def __init__(
        self,
        path: Path | str,
        has_header: bool = False,
        delimiter: str = " ",
    ):
        self.path = Path(path)
        self.has_header = has_header
        self.delimiter = delimiter
        self._count: Optional[int] = None

    def __len__(self) -> int:
        if self._count is None:
            with open(self.path, "r") as f:
                count = sum(1 for _ in f)
            self._count = count - (1 if self.has_header else 0)
        return self._count

    def __iter__(self) -> Iterator[MoleculeRecord]:
        with open(self.path, "r") as f:
            if self.has_header:
                next(f)

            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(self.delimiter, 1)
                smiles = parts[0] if parts else ""
                name = parts[1].strip() if len(parts) > 1 else ""

                mol = None
                if smiles:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                    except Exception:
                        pass

                if mol is None and smiles:
                    print(f"Warning: Failed to parse SMILES at line {idx + 1}: {smiles[:50]}", file=sys.stderr)

                yield MoleculeRecord(
                    mol=mol,
                    smiles=smiles,
                    name=name,
                    metadata={"smiles": smiles, "name": name},
                    row_idx=idx,
                )

    def close(self):
        pass


class SDFReader(MoleculeReader):
    """Read molecules from SDF files."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._count: Optional[int] = None

    def __len__(self) -> int:
        if self._count is None:
            # Count $$$$ delimiters
            with open(self.path, "rb") as f:
                self._count = sum(1 for line in f if line.strip() == b"$$$$")
        return self._count

    def __iter__(self) -> Iterator[MoleculeRecord]:
        supplier = Chem.SDMolSupplier(str(self.path))

        for idx, mol in enumerate(supplier):
            metadata = {}
            smiles = ""
            name = ""

            if mol is not None:
                # Extract properties
                for prop in mol.GetPropsAsDict():
                    metadata[prop] = mol.GetProp(prop)

                name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                smiles = Chem.MolToSmiles(mol)
            else:
                print(f"Warning: Failed to parse molecule at index {idx}", file=sys.stderr)

            yield MoleculeRecord(
                mol=mol,
                smiles=smiles,
                name=name,
                metadata=metadata,
                row_idx=idx,
            )

    def close(self):
        pass


class ParquetReader(MoleculeReader):
    """Read molecules from Parquet files."""

    def __init__(
        self,
        path: Path | str,
        smiles_column: str = "smiles",
        name_column: Optional[str] = None,
    ):
        self.path = Path(path)
        self.smiles_column = smiles_column
        self.name_column = name_column
        self._count: Optional[int] = None

    def __len__(self) -> int:
        if self._count is None:
            import pyarrow.parquet as pq
            self._count = pq.read_metadata(self.path).num_rows
        return self._count

    def __iter__(self) -> Iterator[MoleculeRecord]:
        import pyarrow.parquet as pq

        # Read in batches for memory efficiency
        parquet_file = pq.ParquetFile(self.path)

        row_idx = 0
        for batch in parquet_file.iter_batches(batch_size=10000):
            df = batch.to_pandas()

            for _, row in df.iterrows():
                smiles = str(row.get(self.smiles_column, ""))
                name = str(row.get(self.name_column, "")) if self.name_column else ""

                mol = None
                if smiles:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                    except Exception:
                        pass

                if mol is None and smiles:
                    print(f"Warning: Failed to parse SMILES at row {row_idx}: {smiles[:50]}", file=sys.stderr)

                yield MoleculeRecord(
                    mol=mol,
                    smiles=smiles,
                    name=name,
                    metadata=row.to_dict(),
                    row_idx=row_idx,
                )
                row_idx += 1

    def close(self):
        pass


def create_reader(
    path: str | Path,
    format_config: Optional[FormatConfig] = None,
    smiles_column: str = "smiles",
    name_column: Optional[str] = None,
    has_header: Optional[bool] = None,
) -> MoleculeReader:
    """
    Factory function to create appropriate reader.

    Args:
        path: Path to input file
        format_config: Optional format configuration
        smiles_column: Name of SMILES column (for CSV/Parquet)
        name_column: Name of name column
        has_header: Override header detection

    Returns:
        Appropriate MoleculeReader instance
    """
    path = Path(path)

    if format_config is None:
        file_format = detect_format(path)
    else:
        file_format = format_config.format
        smiles_column = format_config.smiles_column
        name_column = format_config.name_column
        if has_header is None:
            has_header = format_config.has_header

    if file_format == FileFormat.CSV:
        return CSVReader(
            path,
            smiles_column=smiles_column,
            name_column=name_column,
            delimiter=",",
            has_header=has_header if has_header is not None else True,
        )
    elif file_format == FileFormat.TSV:
        return CSVReader(
            path,
            smiles_column=smiles_column,
            name_column=name_column,
            delimiter="\t",
            has_header=has_header if has_header is not None else True,
        )
    elif file_format == FileFormat.SMI:
        return SMIReader(
            path,
            has_header=has_header if has_header is not None else False,
        )
    elif file_format == FileFormat.SDF:
        return SDFReader(path)
    elif file_format == FileFormat.PARQUET:
        return ParquetReader(
            path,
            smiles_column=smiles_column,
            name_column=name_column,
        )
    else:
        raise ValueError(f"Unsupported format: {file_format}")
