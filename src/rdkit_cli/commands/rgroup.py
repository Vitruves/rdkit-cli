"""R-Group decomposition command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options


def register_parser(subparsers):
    """Register the rgroup command."""
    parser = subparsers.add_parser(
        "rgroup",
        help="R-group decomposition",
        description="Decompose molecules into core and R-groups based on a core SMARTS pattern.",
        formatter_class=RdkitHelpFormatter,
    )

    add_common_io_options(parser)
    add_common_processing_options(parser)

    parser.add_argument(
        "-c", "--core",
        required=True,
        metavar="SMARTS",
        help="Core SMARTS with labeled attachment points (e.g., 'c1ccc([*:1])cc1[*:2]')",
    )
    parser.add_argument(
        "--include-unmatched",
        action="store_true",
        help="Include molecules that don't match the core (with empty R-groups)",
    )
    parser.add_argument(
        "--no-smiles",
        action="store_true",
        help="Don't include original SMILES in output",
    )
    parser.add_argument(
        "--no-name",
        action="store_true",
        help="Don't include name in output",
    )

    parser.set_defaults(func=run_rgroup)


def run_rgroup(args) -> int:
    """Run the rgroup command."""
    from rdkit_cli.core.rgroup import RGroupDecomposer
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        decomposer = RGroupDecomposer(
            core_smarts=args.core,
            include_smiles=not args.no_smiles,
            include_name=not args.no_name,
            only_matching=not args.include_unmatched,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        has_header=not args.no_header,
    )

    output_path = Path(args.output)
    writer = create_writer(
        output_path,
        columns=decomposer.get_column_names(),
    )

    # Note: Running single-threaded because RGroupDecomposition
    # objects can be stateful and tricky to parallelize per-molecule
    with reader, writer:
        result = process_molecules(
            reader=reader,
            writer=writer,
            processor=decomposer.decompose,
            n_workers=1,  # Single-threaded for RGroupDecomposition
            quiet=args.quiet,
        )

    if not args.quiet:
        matched = result.successful
        total = result.total_processed
        unmatched = total - matched - result.failed
        print(
            f"Decomposed {matched}/{total} molecules "
            f"({unmatched} unmatched, {result.failed} failed) in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )

    return 0
