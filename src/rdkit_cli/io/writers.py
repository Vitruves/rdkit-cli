"""File writers for various molecular file formats."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from rdkit import Chem

from rdkit_cli.io.formats import FileFormat, detect_format


class MoleculeWriter(ABC):
    """Abstract base class for molecule file writers."""

    @abstractmethod
    def write_row(self, data: dict[str, Any]):
        """Write a single row of data."""
        pass

    @abstractmethod
    def write_batch(self, data: list[dict[str, Any]]):
        """Write a batch of results."""
        pass

    @abstractmethod
    def close(self):
        """Finalize and close the file."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CSVWriter(MoleculeWriter):
    """Write results to CSV/TSV files."""

    def __init__(
        self,
        path: Path | str,
        delimiter: str = ",",
        columns: Optional[list[str]] = None,
    ):
        self.path = Path(path)
        self.delimiter = delimiter
        self.columns = columns
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._header_written = False
        self._column_order: Optional[list[str]] = None

    def write_row(self, data: dict[str, Any]):
        """Write a single row."""
        self.write_batch([data])

    def write_batch(self, data: list[dict[str, Any]]):
        """Write a batch of results."""
        if not data:
            return

        # Determine column order from first row if not set
        if self._column_order is None:
            if self.columns:
                self._column_order = self.columns
            else:
                self._column_order = list(data[0].keys())

        # Write header if not done yet
        if not self._header_written:
            self._file.write(self.delimiter.join(self._column_order) + "\n")
            self._header_written = True

        # Write data rows
        for row in data:
            values = []
            for col in self._column_order:
                val = row.get(col, "")
                # Handle special types
                if val is None:
                    val = ""
                elif isinstance(val, float):
                    if pd.isna(val):
                        val = ""
                    else:
                        val = str(val)
                else:
                    val = str(val)
                # Escape delimiter and quotes
                if self.delimiter in val or '"' in val or "\n" in val:
                    val = '"' + val.replace('"', '""') + '"'
                values.append(val)
            self._file.write(self.delimiter.join(values) + "\n")

    def close(self):
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None


class SMIWriter(MoleculeWriter):
    """Write molecules to SMILES files."""

    def __init__(
        self,
        path: Path | str,
        smiles_column: str = "smiles",
        name_column: Optional[str] = "name",
    ):
        self.path = Path(path)
        self.smiles_column = smiles_column
        self.name_column = name_column
        self._file = open(path, "w", encoding="utf-8")

    def write_row(self, data: dict[str, Any]):
        """Write a single row."""
        smiles = data.get(self.smiles_column, "")
        name = data.get(self.name_column, "") if self.name_column else ""

        if smiles:
            if name:
                self._file.write(f"{smiles} {name}\n")
            else:
                self._file.write(f"{smiles}\n")

    def write_batch(self, data: list[dict[str, Any]]):
        """Write a batch of results."""
        for row in data:
            self.write_row(row)

    def close(self):
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None


class SDFWriter(MoleculeWriter):
    """Write molecules to SDF files."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._writer = Chem.SDWriter(str(path))

    def write_row(self, data: dict[str, Any]):
        """Write a single row."""
        mol = data.get("mol")

        if mol is None:
            # Try to create from SMILES
            smiles = data.get("smiles", "")
            if smiles:
                mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            # Set properties from data
            for key, value in data.items():
                if key not in ("mol", "smiles") and value is not None:
                    try:
                        mol.SetProp(str(key), str(value))
                    except Exception:
                        pass
            self._writer.write(mol)

    def write_batch(self, data: list[dict[str, Any]]):
        """Write a batch of results."""
        for row in data:
            self.write_row(row)

    def close(self):
        """Close the writer."""
        if self._writer:
            self._writer.close()
            self._writer = None


class ParquetWriter(MoleculeWriter):
    """Write results to Parquet files."""

    def __init__(
        self,
        path: Path | str,
        columns: Optional[list[str]] = None,
    ):
        self.path = Path(path)
        self.columns = columns
        self._batches: list[dict[str, Any]] = []
        self._batch_size = 100000  # Write in batches of 100k

    def write_row(self, data: dict[str, Any]):
        """Write a single row."""
        # Remove mol objects (not serializable)
        clean_data = {k: v for k, v in data.items() if k != "mol"}
        self._batches.append(clean_data)

        if len(self._batches) >= self._batch_size:
            self._flush()

    def write_batch(self, data: list[dict[str, Any]]):
        """Write a batch of results."""
        for row in data:
            clean_data = {k: v for k, v in row.items() if k != "mol"}
            self._batches.append(clean_data)

        if len(self._batches) >= self._batch_size:
            self._flush()

    def _flush(self):
        """Write accumulated batches to file."""
        if not self._batches:
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        df = pd.DataFrame(self._batches)

        # Reorder columns if specified
        if self.columns:
            cols = [c for c in self.columns if c in df.columns]
            extra = [c for c in df.columns if c not in self.columns]
            df = df[cols + extra]

        table = pa.Table.from_pandas(df, preserve_index=False)

        if self.path.exists():
            # Append to existing file
            existing = pq.read_table(self.path)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, self.path)
        self._batches = []

    def close(self):
        """Finalize and close the file."""
        self._flush()


def create_writer(
    path: str | Path,
    format_override: Optional[FileFormat] = None,
    columns: Optional[list[str]] = None,
    smiles_column: str = "smiles",
    name_column: Optional[str] = "name",
) -> MoleculeWriter:
    """
    Factory function to create appropriate writer.

    Args:
        path: Output file path
        format_override: Override auto-detected format
        columns: Column order for output
        smiles_column: Name of SMILES column (for SMI files)
        name_column: Name of name column (for SMI files)

    Returns:
        Appropriate MoleculeWriter instance
    """
    path = Path(path)
    file_format = format_override or detect_format(path)

    if file_format == FileFormat.CSV:
        return CSVWriter(path, delimiter=",", columns=columns)
    elif file_format == FileFormat.TSV:
        return CSVWriter(path, delimiter="\t", columns=columns)
    elif file_format == FileFormat.SMI:
        return SMIWriter(path, smiles_column=smiles_column, name_column=name_column)
    elif file_format == FileFormat.SDF:
        return SDFWriter(path)
    elif file_format == FileFormat.PARQUET:
        return ParquetWriter(path, columns=columns)
    else:
        raise ValueError(f"Unsupported format: {file_format}")
