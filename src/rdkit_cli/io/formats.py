"""File format detection and configuration."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class FileFormat(Enum):
    """Supported file formats."""

    CSV = "csv"
    TSV = "tsv"
    SMI = "smi"
    SDF = "sdf"
    PARQUET = "parquet"


@dataclass
class FormatConfig:
    """Configuration for file format handling."""

    format: FileFormat
    has_header: bool = True
    smiles_column: str = "smiles"
    name_column: Optional[str] = None
    delimiter: str = ","
    extra_columns: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Set format-specific defaults."""
        if self.format == FileFormat.TSV:
            self.delimiter = "\t"
        elif self.format == FileFormat.SMI:
            self.has_header = False
            self.delimiter = " "


# File extension to format mapping
EXTENSION_MAP: dict[str, FileFormat] = {
    ".csv": FileFormat.CSV,
    ".tsv": FileFormat.TSV,
    ".smi": FileFormat.SMI,
    ".smiles": FileFormat.SMI,
    ".sdf": FileFormat.SDF,
    ".mol": FileFormat.SDF,
    ".parquet": FileFormat.PARQUET,
    ".pq": FileFormat.PARQUET,
}


def detect_format(path: str | Path) -> FileFormat:
    """
    Detect file format from file extension.

    Args:
        path: Path to the file

    Returns:
        Detected FileFormat

    Raises:
        ValueError: If format cannot be detected
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in EXTENSION_MAP:
        return EXTENSION_MAP[suffix]

    raise ValueError(
        f"Cannot detect format for '{path}'. "
        f"Supported extensions: {', '.join(EXTENSION_MAP.keys())}"
    )


def create_format_config(
    path: str | Path,
    format_override: Optional[FileFormat] = None,
    has_header: Optional[bool] = None,
    smiles_column: str = "smiles",
    name_column: Optional[str] = None,
) -> FormatConfig:
    """
    Create a FormatConfig for a file.

    Args:
        path: Path to the file
        format_override: Override auto-detected format
        has_header: Override default header setting
        smiles_column: Name of the SMILES column
        name_column: Name of the molecule name column

    Returns:
        Configured FormatConfig
    """
    file_format = format_override or detect_format(path)

    config = FormatConfig(
        format=file_format,
        smiles_column=smiles_column,
        name_column=name_column,
    )

    # Override header if explicitly specified
    if has_header is not None:
        config.has_header = has_header

    return config
