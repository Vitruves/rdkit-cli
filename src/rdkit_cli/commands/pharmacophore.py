"""Pharmacophore command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import (
    RdkitHelpFormatter,
    add_common_io_options,
    add_common_processing_options,
)


def register_parser(subparsers):
    """Register the pharmacophore command and subcommands."""
    parser = subparsers.add_parser(
        "pharmacophore",
        help="Pharmacophore feature analysis",
        description="Perceive pharmacophoric features and search "
        "by pharmacophore similarity.",
        formatter_class=RdkitHelpFormatter,
    )

    pharm_sub = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # pharmacophore perceive
    perceive_parser = pharm_sub.add_parser(
        "perceive",
        help="Identify pharmacophore features in molecules",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(perceive_parser)
    add_common_processing_options(perceive_parser)
    perceive_parser.set_defaults(func=run_perceive)

    # pharmacophore search
    search_parser = pharm_sub.add_parser(
        "search",
        help="Search by 2D pharmacophore fingerprint similarity",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(search_parser)
    add_common_processing_options(search_parser)
    search_parser.add_argument(
        "--query",
        required=True,
        metavar="SMILES",
        help="Query molecule SMILES",
    )
    search_parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        metavar="T",
        help="Minimum similarity threshold (default: 0.5)",
    )
    search_parser.set_defaults(func=run_search)

    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_perceive(args) -> int:
    """Perceive pharmacophoric features."""
    from rdkit_cli.core.pharmacophore import PharmacophorePerceiver
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    perceiver = PharmacophorePerceiver()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=getattr(args, "name_column", None),
        has_header=not args.no_header,
    )
    output_path = Path(args.output)
    writer = create_writer(output_path)

    with reader, writer:
        result = process_molecules(
            reader=reader,
            writer=writer,
            processor=perceiver.perceive,
            n_workers=1,  # Gobbi factory not picklable
            quiet=args.quiet,
        )

    if not args.quiet:
        print(
            f"Perceived features for "
            f"{result.successful}/{result.total_processed} molecules "
            f"in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )
    return 0


def run_search(args) -> int:
    """Search by pharmacophore similarity."""
    from rdkit_cli.core.pharmacophore import PharmacophoreSearcher
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    try:
        searcher = PharmacophoreSearcher(
            query_smiles=args.query,
            threshold=args.threshold,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=getattr(args, "name_column", None),
        has_header=not args.no_header,
    )
    output_path = Path(args.output)
    writer = create_writer(output_path)

    with reader, writer:
        result = process_molecules(
            reader=reader,
            writer=writer,
            processor=searcher.search,
            n_workers=1,  # Gobbi factory not picklable
            quiet=args.quiet,
        )

    if not args.quiet:
        print(
            f"Found {result.successful}/{result.total_processed} "
            f"molecules above threshold "
            f"in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )
    return 0
