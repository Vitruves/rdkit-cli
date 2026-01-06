"""Descriptors command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options

# Lazy imports - these are only imported when command runs
# from rdkit_cli.core.descriptors import ...
# from rdkit_cli.io import ...
# from rdkit_cli.parallel.batch import ...

# Categories defined here to avoid importing core.descriptors for help
DESCRIPTOR_CATEGORIES = [
    "constitutional",
    "topological",
    "electronic",
    "geometric",
    "molecular",
]


def register_parser(subparsers):
    """Register the descriptors command and subcommands."""
    parser = subparsers.add_parser(
        "descriptors",
        help="Compute molecular descriptors",
        description="Calculate RDKit molecular descriptors.",
        formatter_class=RdkitHelpFormatter,
    )

    desc_subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # descriptors list
    list_parser = desc_subparsers.add_parser(
        "list",
        help="List available descriptors",
        formatter_class=RdkitHelpFormatter,
    )
    list_parser.add_argument(
        "--all",
        action="store_true",
        dest="show_all",
        help="Show all descriptors with descriptions",
    )
    list_parser.add_argument(
        "--category",
        choices=DESCRIPTOR_CATEGORIES,
        help="Filter by category",
    )
    list_parser.set_defaults(func=run_list)

    # descriptors compute
    compute_parser = desc_subparsers.add_parser(
        "compute",
        help="Compute descriptors for molecules",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(compute_parser)
    add_common_processing_options(compute_parser)

    desc_group = compute_parser.add_mutually_exclusive_group()
    desc_group.add_argument(
        "-d", "--descriptors",
        metavar="DESC",
        help="Comma-separated list of descriptors to compute",
    )
    desc_group.add_argument(
        "--all",
        action="store_true",
        dest="compute_all",
        help="Compute all available descriptors",
    )
    desc_group.add_argument(
        "--common",
        action="store_true",
        help="Compute common descriptors (default)",
    )
    desc_group.add_argument(
        "--lipinski",
        action="store_true",
        help="Compute Lipinski rule-of-5 descriptors",
    )
    desc_group.add_argument(
        "--druglike",
        action="store_true",
        help="Compute drug-likeness descriptors",
    )
    desc_group.add_argument(
        "--category",
        choices=DESCRIPTOR_CATEGORIES,
        dest="compute_category",
        help="Compute all descriptors in category",
    )

    # Additional options
    compute_parser.add_argument(
        "--exclude",
        metavar="DESC",
        help="Comma-separated list of descriptors to exclude",
    )
    compute_parser.add_argument(
        "--precision",
        type=int,
        default=4,
        metavar="N",
        help="Decimal precision for float values (default: 4)",
    )
    compute_parser.add_argument(
        "--error-value",
        default="NaN",
        metavar="VAL",
        help="Value to use for failed calculations (default: NaN)",
    )
    compute_parser.add_argument(
        "--3d",
        action="store_true",
        dest="compute_3d",
        help="Include 3D descriptors (requires 3D coordinates)",
    )
    compute_parser.add_argument(
        "--no-smiles",
        action="store_true",
        help="Don't include SMILES in output",
    )
    compute_parser.add_argument(
        "--no-name",
        action="store_true",
        help="Don't include name in output",
    )
    compute_parser.add_argument(
        "--add-inchi",
        action="store_true",
        help="Add InChI column to output",
    )
    compute_parser.add_argument(
        "--add-inchikey",
        action="store_true",
        help="Add InChIKey column to output",
    )
    compute_parser.add_argument(
        "--add-formula",
        action="store_true",
        help="Add molecular formula column to output",
    )
    compute_parser.add_argument(
        "--add-canonical",
        action="store_true",
        help="Add canonical SMILES column (if input differs)",
    )
    compute_parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Stop processing on first error instead of continuing",
    )
    compute_parser.add_argument(
        "--skip-invalid",
        action="store_true",
        default=True,
        help="Skip molecules that fail parsing (default: True)",
    )
    compute_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="Batch size for parallel processing (default: 100)",
    )

    compute_parser.set_defaults(func=run_compute)

    # Set default for main parser
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_list(args) -> int:
    """Run the list subcommand."""
    # Lazy import
    from rdkit_cli.core.descriptors import list_descriptors

    descriptors = list_descriptors(
        category=getattr(args, "category", None),
        verbose=getattr(args, "show_all", False),
    )

    if args.show_all:
        # Print with descriptions
        max_name_len = max(len(d.name) for d in descriptors) if descriptors else 0
        for desc in descriptors:
            print(f"{desc.name:<{max_name_len}}  [{desc.category}]")
    else:
        for desc in descriptors:
            print(desc.name)

    print(f"\nTotal: {len(descriptors)} descriptors", file=sys.stderr)
    return 0


def run_compute(args) -> int:
    """Run the compute subcommand."""
    # Lazy imports
    from rdkit_cli.core.descriptors import (
        DescriptorCalculator,
        list_descriptors,
        COMMON_DESCRIPTORS,
        LIPINSKI_DESCRIPTORS,
        DRUGLIKE_DESCRIPTORS,
    )
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    # Determine which descriptors to compute
    descriptor_names = None

    if args.descriptors:
        descriptor_names = [d.strip() for d in args.descriptors.split(",")]
    elif args.compute_all:
        descriptor_names = None  # All
    elif args.lipinski:
        descriptor_names = LIPINSKI_DESCRIPTORS
    elif args.druglike:
        descriptor_names = DRUGLIKE_DESCRIPTORS
    elif args.compute_category:
        descs = list_descriptors(category=args.compute_category)
        descriptor_names = [d.name for d in descs]
    else:
        # Default to common descriptors
        descriptor_names = COMMON_DESCRIPTORS

    # Handle exclusions
    if args.exclude and descriptor_names:
        exclude_set = {d.strip() for d in args.exclude.split(",")}
        descriptor_names = [d for d in descriptor_names if d not in exclude_set]

    # Create calculator
    try:
        calculator = DescriptorCalculator(
            descriptors=descriptor_names,
            include_smiles=not args.no_smiles,
            include_name=not args.no_name,
            precision=args.precision,
            error_value=args.error_value,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

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
