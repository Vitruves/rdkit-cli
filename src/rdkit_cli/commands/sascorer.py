"""SA Score command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options


def register_parser(subparsers):
    """Register the sascorer command."""
    parser = subparsers.add_parser(
        "sascorer",
        help="Calculate synthetic accessibility score",
        description="Calculate Synthetic Accessibility (SA) Score and related metrics.",
        formatter_class=RdkitHelpFormatter,
    )

    add_common_io_options(parser)
    add_common_processing_options(parser)

    parser.add_argument(
        "--npc",
        action="store_true",
        help="Include Natural Product-likeness Score",
    )
    parser.add_argument(
        "--qed",
        action="store_true",
        help="Include QED (drug-likeness) Score",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_scores",
        help="Include all scores (SA, NPC, QED)",
    )
    parser.add_argument(
        "--no-smiles",
        action="store_true",
        help="Don't include SMILES in output",
    )
    parser.add_argument(
        "--no-name",
        action="store_true",
        help="Don't include name in output",
    )

    parser.set_defaults(func=run_sascorer)


def run_sascorer(args) -> int:
    """Run the sascorer command."""
    from rdkit_cli.core.sascorer import SAScoreCalculator
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Determine which scores to include
    include_npc = args.npc or args.all_scores
    include_qed = args.qed or args.all_scores

    calculator = SAScoreCalculator(
        include_sa=True,
        include_npc=include_npc,
        include_qed=include_qed,
        include_smiles=not args.no_smiles,
        include_name=not args.no_name,
    )

    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        has_header=not args.no_header,
    )

    output_path = Path(args.output)
    writer = create_writer(
        output_path,
        columns=calculator.get_column_names(),
    )

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
            f"Calculated scores for {result.successful}/{result.total_processed} molecules "
            f"({result.failed} failed) in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )

    return 0 if result.failed == 0 else 1
