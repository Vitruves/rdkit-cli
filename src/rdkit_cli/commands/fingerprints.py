"""Fingerprints command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options

# Fingerprint types defined here to avoid importing core at startup
FINGERPRINT_TYPES = ["morgan", "maccs", "rdkit", "atompair", "torsion", "pattern"]


def register_parser(subparsers):
    """Register the fingerprints command and subcommands."""
    parser = subparsers.add_parser(
        "fingerprints",
        help="Compute molecular fingerprints",
        description="Generate various molecular fingerprint types.",
        formatter_class=RdkitHelpFormatter,
    )

    fp_subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # fingerprints list
    list_parser = fp_subparsers.add_parser(
        "list",
        help="List available fingerprint types",
        formatter_class=RdkitHelpFormatter,
    )
    list_parser.set_defaults(func=run_list)

    # fingerprints compute
    compute_parser = fp_subparsers.add_parser(
        "compute",
        help="Compute fingerprints for molecules",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(compute_parser)
    add_common_processing_options(compute_parser)

    compute_parser.add_argument(
        "-t", "--type",
        choices=FINGERPRINT_TYPES,
        default="morgan",
        help="Fingerprint type (default: morgan)",
    )
    compute_parser.add_argument(
        "-r", "--radius",
        type=int,
        default=2,
        metavar="N",
        help="Radius for Morgan fingerprints (default: 2, equivalent to ECFP4)",
    )
    compute_parser.add_argument(
        "-b", "--bits",
        type=int,
        default=2048,
        metavar="N",
        help="Number of bits (default: 2048)",
    )
    compute_parser.add_argument(
        "--counts",
        action="store_true",
        help="Output count fingerprints instead of binary (Morgan only)",
    )
    compute_parser.add_argument(
        "-f", "--format",
        choices=["hex", "bitstring", "bits", "numpy"],
        default="hex",
        dest="output_format",
        help="Output format (default: hex)",
    )
    compute_parser.add_argument(
        "--use-chirality",
        action="store_true",
        help="Include chirality in fingerprint (Morgan only)",
    )
    compute_parser.add_argument(
        "--use-features",
        action="store_true",
        help="Use pharmacophoric features instead of atom invariants (Morgan only)",
    )
    compute_parser.add_argument(
        "--use-bond-types",
        action="store_true",
        default=True,
        help="Include bond types in fingerprint (Morgan, default: True)",
    )
    compute_parser.add_argument(
        "--no-bond-types",
        action="store_true",
        help="Exclude bond types from fingerprint (Morgan)",
    )

    compute_parser.set_defaults(func=run_compute)

    # Set default for main parser
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_list(args) -> int:
    """Run the list subcommand."""
    # Lazy import
    from rdkit_cli.core.fingerprints import list_fingerprints

    fps = list_fingerprints()

    print("Available fingerprint types:\n")
    for fp in fps:
        radius_info = " (radius configurable)" if fp.has_radius else ""
        print(f"  {fp.name:<12} - {fp.description}")
        print(f"                 Default bits: {fp.default_bits}{radius_info}")
        print()

    return 0


def run_compute(args) -> int:
    """Run the compute subcommand."""
    # Lazy imports
    from rdkit_cli.core.fingerprints import FingerprintCalculator, FingerprintType
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    # Parse fingerprint type
    fp_type = FingerprintType(args.type)

    # Create calculator
    calculator = FingerprintCalculator(
        fp_type=fp_type,
        n_bits=args.bits,
        radius=args.radius,
        use_counts=args.counts,
        output_format=args.output_format,
        include_smiles=True,
        include_name=True,
    )

    # Create reader
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        has_header=not args.no_header,
    )

    # Create writer
    output_path = Path(args.output)
    writer = create_writer(
        output_path,
        columns=calculator.get_column_names(),
    )

    # Process
    with reader, writer:
        result = process_molecules(
            reader=reader,
            writer=writer,
            processor=calculator.compute,
            n_workers=args.ncpu,
            quiet=args.quiet,
        )

    if not args.quiet:
        print(
            f"Processed {result.successful}/{result.total_processed} molecules "
            f"({result.failed} failed) in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )

    return 0 if result.failed == 0 else 1
