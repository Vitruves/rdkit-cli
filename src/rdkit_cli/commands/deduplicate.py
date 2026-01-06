"""Deduplicate command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options


def register_parser(subparsers):
    """Register the deduplicate command."""
    parser = subparsers.add_parser(
        "deduplicate",
        help="Remove duplicate molecules",
        description="Remove duplicate molecules from a dataset based on various molecular identifiers.",
        formatter_class=RdkitHelpFormatter,
    )

    add_common_io_options(parser)
    add_common_processing_options(parser)

    parser.add_argument(
        "-b", "--by",
        choices=["smiles", "inchi", "inchikey", "scaffold"],
        default="smiles",
        help="Deduplication key type (default: smiles)",
    )
    parser.add_argument(
        "--keep",
        choices=["first", "last"],
        default="first",
        help="Which duplicate to keep (default: first)",
    )
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="List available key types and exit",
    )

    parser.set_defaults(func=run_deduplicate)


def run_deduplicate(args) -> int:
    """Run the deduplicate command."""
    from rdkit_cli.core.deduplicate import Deduplicator
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.progress.ninja import NinjaProgress

    # Handle --list-keys
    if args.list_keys:
        print("Available deduplication keys:")
        print("  smiles    - Canonical SMILES (default)")
        print("  inchi     - InChI string")
        print("  inchikey  - InChIKey (27 character hash)")
        print("  scaffold  - Murcko scaffold SMILES")
        return 0

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Create reader
    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        has_header=not args.no_header,
    )

    # Read all records with progress
    if not args.quiet:
        print("Reading molecules...", file=sys.stderr)

    records = []
    with reader:
        total = len(reader)
        progress = NinjaProgress(total=total, quiet=args.quiet)
        progress.start()

        for record in reader:
            records.append(record)
            progress.update(1)

        progress.finish()

    if not records:
        print("Error: No molecules found in input file", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Deduplicating {len(records)} molecules by {args.by}...", file=sys.stderr)

    # Create deduplicator
    deduplicator = Deduplicator(
        key_type=args.by,
        keep=args.keep,
    )

    # Deduplicate
    unique_records, n_duplicates = deduplicator.deduplicate(records)

    # Write output
    output_path = Path(args.output)
    writer = create_writer(output_path)

    with writer:
        for record in unique_records:
            row = {"smiles": record.smiles}
            if record.name:
                row["name"] = record.name
            for key, value in record.metadata.items():
                if key not in row and key != "smiles":
                    row[key] = value
            writer.write_row(row)

    if not args.quiet:
        print(
            f"Removed {n_duplicates} duplicates. "
            f"Wrote {len(unique_records)} unique molecules to {output_path}",
            file=sys.stderr,
        )

    return 0
