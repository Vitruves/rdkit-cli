"""Enumerate command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options


def register_parser(subparsers):
    """Register the enumerate command and subcommands."""
    parser = subparsers.add_parser(
        "enumerate",
        help="Enumerate molecular variants",
        description="Enumerate stereoisomers, tautomers, and other molecular variants.",
        formatter_class=RdkitHelpFormatter,
    )

    enum_subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # enumerate stereoisomers
    stereo_parser = enum_subparsers.add_parser(
        "stereoisomers",
        help="Enumerate stereoisomers",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(stereo_parser)
    add_common_processing_options(stereo_parser)
    stereo_parser.add_argument(
        "--max-isomers",
        type=int,
        default=32,
        metavar="N",
        help="Maximum stereoisomers per molecule (default: 32)",
    )
    stereo_parser.add_argument(
        "--only-unassigned",
        action="store_true",
        default=True,
        help="Only enumerate unassigned stereocenters (default: True)",
    )
    stereo_parser.add_argument(
        "--all-centers",
        action="store_true",
        help="Enumerate all stereocenters, not just unassigned",
    )
    stereo_parser.set_defaults(func=run_stereoisomers)

    # enumerate tautomers
    taut_parser = enum_subparsers.add_parser(
        "tautomers",
        help="Enumerate tautomers",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(taut_parser)
    add_common_processing_options(taut_parser)
    taut_parser.add_argument(
        "--max-tautomers",
        type=int,
        default=50,
        metavar="N",
        help="Maximum tautomers per molecule (default: 50)",
    )
    taut_parser.add_argument(
        "--max-transforms",
        type=int,
        default=1000,
        metavar="N",
        help="Maximum transforms to apply (default: 1000)",
    )
    taut_parser.set_defaults(func=run_tautomers)

    # enumerate canonical-tautomer
    canon_parser = enum_subparsers.add_parser(
        "canonical-tautomer",
        help="Get canonical tautomer",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(canon_parser)
    add_common_processing_options(canon_parser)
    canon_parser.add_argument(
        "--include-original",
        action="store_true",
        help="Include original SMILES in output",
    )
    canon_parser.set_defaults(func=run_canonical_tautomer)

    # Set default for main parser
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_stereoisomers(args) -> int:
    """Run stereoisomer enumeration."""
    from rdkit_cli.core.enumerate import StereoisomerEnumerator
    from rdkit_cli.io import create_reader, create_writer

    enumerator = StereoisomerEnumerator(
        max_isomers=args.max_isomers,
        only_unassigned=not args.all_centers,
    )

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

    output_path = Path(args.output)
    writer = create_writer(output_path)

    total_input = 0
    total_output = 0

    with reader, writer:
        for record in reader:
            total_input += 1
            results = enumerator.enumerate(record)
            for result in results:
                writer.write_row(result)
                total_output += 1

    if not args.quiet:
        print(
            f"Enumerated {total_output} stereoisomers from {total_input} molecules",
            file=sys.stderr,
        )

    return 0


def run_tautomers(args) -> int:
    """Run tautomer enumeration."""
    from rdkit_cli.core.enumerate import TautomerEnumerator
    from rdkit_cli.io import create_reader, create_writer

    enumerator = TautomerEnumerator(
        max_tautomers=args.max_tautomers,
        max_transforms=args.max_transforms,
    )

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

    output_path = Path(args.output)
    writer = create_writer(output_path)

    total_input = 0
    total_output = 0

    with reader, writer:
        for record in reader:
            total_input += 1
            results = enumerator.enumerate(record)
            for result in results:
                writer.write_row(result)
                total_output += 1

    if not args.quiet:
        print(
            f"Enumerated {total_output} tautomers from {total_input} molecules",
            file=sys.stderr,
        )

    return 0


def run_canonical_tautomer(args) -> int:
    """Run canonical tautomer extraction."""
    from rdkit_cli.core.enumerate import CanonicalTautomerizer
    from rdkit_cli.io import create_reader, create_writer

    canonicalizer = CanonicalTautomerizer(
        include_original=args.include_original,
    )

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

    output_path = Path(args.output)
    writer = create_writer(output_path)

    # Note: Running single-threaded because RDKit TautomerEnumerator
    # objects can't be pickled for multiprocessing
    total = 0
    successful = 0

    with reader, writer:
        for record in reader:
            total += 1
            result = canonicalizer.canonicalize(record)
            if result is not None:
                writer.write_row(result)
                successful += 1

    if not args.quiet:
        print(
            f"Canonicalized {successful}/{total} molecules "
            f"({total - successful} failed)",
            file=sys.stderr,
        )

    return 0
