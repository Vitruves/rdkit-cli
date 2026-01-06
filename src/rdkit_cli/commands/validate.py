"""Validate command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options


def register_parser(subparsers):
    """Register the validate command."""
    parser = subparsers.add_parser(
        "validate",
        help="Validate molecular structures",
        description="Validate molecular structures and check for common issues.",
        formatter_class=RdkitHelpFormatter,
    )

    add_common_io_options(parser)
    add_common_processing_options(parser)

    parser.add_argument(
        "--check-valence",
        action="store_true",
        default=True,
        help="Check for valence errors (default: True)",
    )
    parser.add_argument(
        "--no-check-valence",
        action="store_false",
        dest="check_valence",
        help="Skip valence checking",
    )
    parser.add_argument(
        "--check-kekulize",
        action="store_true",
        default=True,
        help="Check if molecules can be kekulized (default: True)",
    )
    parser.add_argument(
        "--no-check-kekulize",
        action="store_false",
        dest="check_kekulize",
        help="Skip kekulization checking",
    )
    parser.add_argument(
        "--check-stereo",
        action="store_true",
        help="Check for undefined stereocenters",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        metavar="N",
        help="Maximum allowed atoms per molecule",
    )
    parser.add_argument(
        "--max-rings",
        type=int,
        metavar="N",
        help="Maximum allowed rings per molecule",
    )
    parser.add_argument(
        "--allowed-elements",
        metavar="ELEMS",
        help="Comma-separated list of allowed elements (e.g., C,H,N,O,S)",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Only output valid molecules (no error columns)",
    )
    parser.add_argument(
        "--invalid-only",
        action="store_true",
        help="Only output invalid molecules",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics to stderr",
    )

    parser.set_defaults(func=run_validate)


def run_validate(args) -> int:
    """Run the validate command."""
    from rdkit_cli.core.validate import MoleculeValidator
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.progress.ninja import NinjaProgress

    if args.valid_only and args.invalid_only:
        print("Error: Cannot use both --valid-only and --invalid-only", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Parse allowed elements
    allowed_elements = None
    if args.allowed_elements:
        allowed_elements = set(e.strip() for e in args.allowed_elements.split(","))

    # Create validator
    validator = MoleculeValidator(
        check_valence=args.check_valence,
        check_kekulize=args.check_kekulize,
        check_stereo=args.check_stereo,
        max_atoms=args.max_atoms,
        max_rings=args.max_rings,
        allowed_elements=allowed_elements,
    )

    # Create reader
    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        has_header=not args.no_header,
    )

    # Create writer
    output_path = Path(args.output)
    writer = create_writer(output_path)

    # Process with progress
    if not args.quiet:
        print("Validating molecules...", file=sys.stderr)

    n_valid = 0
    n_invalid = 0
    n_written = 0

    with reader, writer:
        total = len(reader)
        progress = NinjaProgress(total=total, quiet=args.quiet)
        progress.start()

        for record in reader:
            result = validator.validate(record.mol, record.smiles)

            if result.is_valid:
                n_valid += 1
            else:
                n_invalid += 1

            # Decide whether to output this record
            should_write = True
            if args.valid_only and not result.is_valid:
                should_write = False
            if args.invalid_only and result.is_valid:
                should_write = False

            if should_write:
                row = {"smiles": record.smiles}
                if record.name:
                    row["name"] = record.name

                # Add validation columns unless --valid-only
                if not args.valid_only:
                    row["is_valid"] = result.is_valid
                    row["errors"] = result.to_dict()["errors"]
                    row["warnings"] = result.to_dict()["warnings"]

                # Copy metadata
                for key, value in record.metadata.items():
                    if key not in row and key != "smiles":
                        row[key] = value

                writer.write_row(row)
                n_written += 1

            progress.update(1)

        progress.finish()

    # Print summary
    if not args.quiet or args.summary:
        total = n_valid + n_invalid
        valid_pct = (n_valid / total * 100) if total > 0 else 0
        print(f"\nValidation Summary:", file=sys.stderr)
        print(f"  Total molecules: {total}", file=sys.stderr)
        print(f"  Valid: {n_valid} ({valid_pct:.1f}%)", file=sys.stderr)
        print(f"  Invalid: {n_invalid} ({100-valid_pct:.1f}%)", file=sys.stderr)
        print(f"  Written: {n_written} molecules to {output_path}", file=sys.stderr)

    return 0
