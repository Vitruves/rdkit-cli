"""Standardize command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options


def register_parser(subparsers):
    """Register the standardize command and subcommands."""
    parser = subparsers.add_parser(
        "standardize",
        help="Standardize and canonicalize molecules",
        description="Apply standardization transforms to molecular structures.",
        formatter_class=RdkitHelpFormatter,
    )

    add_common_io_options(parser)
    add_common_processing_options(parser)

    # Standardization options
    parser.add_argument(
        "--no-canonicalize",
        action="store_true",
        help="Don't canonicalize output SMILES",
    )
    parser.add_argument(
        "--remove-stereo",
        action="store_true",
        help="Remove stereochemistry information",
    )
    parser.add_argument(
        "--disconnect-metals",
        action="store_true",
        help="Disconnect metal atoms from molecules",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply normalization transforms",
    )
    parser.add_argument(
        "--reionize",
        action="store_true",
        help="Standardize ionization states",
    )
    parser.add_argument(
        "--uncharge",
        action="store_true",
        help="Neutralize charges",
    )
    parser.add_argument(
        "--fragment-parent",
        action="store_true",
        help="Keep only the largest fragment",
    )
    parser.add_argument(
        "--tautomer-parent",
        action="store_true",
        help="Canonicalize tautomer form",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Apply standard cleanup (normalize + uncharge + fragment-parent)",
    )
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="Include original SMILES in output",
    )

    parser.set_defaults(func=run_standardize)


def run_standardize(args) -> int:
    """Run the standardize command."""
    # Lazy imports
    from rdkit_cli.core.standardizer import MoleculeStandardizer
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    # Handle --cleanup shortcut
    normalize = args.normalize or args.cleanup
    uncharge = args.uncharge or args.cleanup
    fragment_parent = args.fragment_parent or args.cleanup

    # Create standardizer
    standardizer = MoleculeStandardizer(
        canonicalize=not args.no_canonicalize,
        remove_stereo=args.remove_stereo,
        disconnect_metals=args.disconnect_metals,
        normalize=normalize,
        reionize=args.reionize,
        uncharge=uncharge,
        fragment_parent=fragment_parent,
        tautomer_parent=args.tautomer_parent,
        include_original=args.include_original,
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
        columns=standardizer.get_column_names(),
    )

    # Process
    with reader, writer:
        result = process_molecules(
            reader=reader,
            writer=writer,
            processor=standardizer.standardize,
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
