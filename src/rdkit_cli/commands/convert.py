"""Convert command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options

# Define formats here to avoid loading io module at startup
FILE_FORMATS = ["csv", "tsv", "smi", "sdf", "parquet"]


def register_parser(subparsers):
    """Register the convert command."""
    parser = subparsers.add_parser(
        "convert",
        help="Convert between molecular file formats",
        description="Convert molecules between different file formats and representations.",
        formatter_class=RdkitHelpFormatter,
    )

    add_common_io_options(parser)
    add_common_processing_options(parser)

    parser.add_argument(
        "--in-format",
        choices=FILE_FORMATS,
        help="Input format (auto-detected from extension if not specified)",
    )
    parser.add_argument(
        "--out-format",
        choices=FILE_FORMATS,
        help="Output format (auto-detected from extension if not specified)",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        default=True,
        help="Canonicalize SMILES output (default: True)",
    )
    parser.add_argument(
        "--no-canonical",
        action="store_false",
        dest="canonical",
        help="Don't canonicalize SMILES output",
    )
    parser.add_argument(
        "--add-inchi",
        action="store_true",
        help="Add InChI column to output",
    )
    parser.add_argument(
        "--add-inchikey",
        action="store_true",
        help="Add InChIKey column to output",
    )

    parser.set_defaults(func=run_convert)


def run_convert(args) -> int:
    """Run the convert command."""
    # Lazy imports
    from typing import Optional, Any
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, MolToInchiKey
    from rdkit_cli.io import create_reader, create_writer, FileFormat, detect_format
    from rdkit_cli.io.readers import MoleculeRecord
    from rdkit_cli.parallel.batch import process_molecules

    class FormatConverter:
        """Convert molecules between formats."""

        def __init__(
            self,
            canonical: bool = True,
            add_inchi: bool = False,
            add_inchikey: bool = False,
        ):
            self.canonical = canonical
            self.add_inchi = add_inchi
            self.add_inchikey = add_inchikey

        def convert(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
            """Convert a molecule record."""
            if record.mol is None:
                return None

            result: dict[str, Any] = {}

            # Generate canonical SMILES
            result["smiles"] = Chem.MolToSmiles(record.mol, canonical=self.canonical)

            if record.name:
                result["name"] = record.name

            # Add InChI if requested
            if self.add_inchi:
                try:
                    result["inchi"] = MolToInchi(record.mol)
                except Exception:
                    result["inchi"] = ""

            # Add InChIKey if requested
            if self.add_inchikey:
                try:
                    result["inchikey"] = MolToInchiKey(record.mol)
                except Exception:
                    result["inchikey"] = ""

            # Copy other metadata
            for key, value in record.metadata.items():
                if key not in result and key != "smiles":
                    result[key] = value

            return result

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Detect formats
    in_format = FileFormat(args.in_format) if args.in_format else detect_format(input_path)
    output_path = Path(args.output)
    out_format = FileFormat(args.out_format) if args.out_format else detect_format(output_path)

    # Create converter
    converter = FormatConverter(
        canonical=args.canonical,
        add_inchi=args.add_inchi,
        add_inchikey=args.add_inchikey,
    )

    # Create reader
    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        has_header=not args.no_header,
    )

    # Create writer
    writer = create_writer(output_path)

    # Process
    with reader, writer:
        result = process_molecules(
            reader=reader,
            writer=writer,
            processor=converter.convert,
            n_workers=args.ncpu,
            quiet=args.quiet,
        )

    if not args.quiet:
        print(
            f"Converted {result.successful}/{result.total_processed} molecules "
            f"({result.failed} failed) in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )

    return 0 if result.failed == 0 else 1
