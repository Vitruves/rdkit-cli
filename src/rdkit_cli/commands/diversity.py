"""Diversity command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_processing_options


def register_parser(subparsers):
    """Register the diversity command and subcommands."""
    parser = subparsers.add_parser(
        "diversity",
        help="Analyze and select diverse molecules",
        description="Analyze molecular diversity and select diverse subsets.",
        formatter_class=RdkitHelpFormatter,
    )

    div_subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # diversity pick
    pick_parser = div_subparsers.add_parser(
        "pick",
        help="Select diverse subset using MaxMin algorithm",
        formatter_class=RdkitHelpFormatter,
    )
    pick_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file",
    )
    pick_parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output file",
    )
    add_common_processing_options(pick_parser)
    pick_parser.add_argument(
        "-k", "--num-picks",
        type=int,
        default=100,
        metavar="N",
        help="Number of molecules to pick (default: 100)",
    )
    pick_parser.add_argument(
        "-m", "--method",
        choices=["maxmin", "leader"],
        default="maxmin",
        help="Picking method (default: maxmin)",
    )
    pick_parser.add_argument(
        "-r", "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (default: 2)",
    )
    pick_parser.add_argument(
        "-b", "--bits",
        type=int,
        default=2048,
        help="Fingerprint bit size (default: 2048)",
    )
    pick_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    pick_parser.set_defaults(func=run_pick)

    # diversity analyze
    analyze_parser = div_subparsers.add_parser(
        "analyze",
        help="Analyze diversity of a molecule set",
        formatter_class=RdkitHelpFormatter,
    )
    analyze_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file",
    )
    analyze_parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Output file (optional, prints to stdout if not specified)",
    )
    add_common_processing_options(analyze_parser)
    analyze_parser.add_argument(
        "-r", "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (default: 2)",
    )
    analyze_parser.add_argument(
        "-b", "--bits",
        type=int,
        default=2048,
        help="Fingerprint bit size (default: 2048)",
    )
    analyze_parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Max molecules to sample for analysis (default: 1000)",
    )
    analyze_parser.set_defaults(func=run_analyze)

    # Set default for main parser
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_pick(args) -> int:
    """Run diversity picking."""
    from rdkit_cli.core.diversity import DiversityPicker
    from rdkit_cli.io import create_reader, create_writer

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

    if not args.quiet:
        print("Reading molecules...", file=sys.stderr)

    # Read all records
    records = list(reader)
    mols = [r.mol for r in records]

    if not args.quiet:
        print(f"Picking {args.num_picks} diverse molecules from {len(mols)}...", file=sys.stderr)

    # Create picker
    picker = DiversityPicker(
        n_picks=args.num_picks,
        seed=args.seed,
        radius=args.radius,
        n_bits=args.bits,
        method=args.method,
    )

    # Pick diverse subset
    selected_indices = picker.pick(mols)

    # Write output
    output_path = Path(args.output)
    writer = create_writer(output_path)

    with writer:
        for idx in selected_indices:
            record = records[idx]
            result = {
                "smiles": record.smiles,
                "diversity_rank": selected_indices.index(idx),
            }
            if record.name:
                result["name"] = record.name
            writer.write_row(result)

    if not args.quiet:
        print(
            f"Selected {len(selected_indices)} diverse molecules. Wrote to {output_path}",
            file=sys.stderr,
        )

    return 0


def run_analyze(args) -> int:
    """Run diversity analysis."""
    from rdkit_cli.core.diversity import DiversityAnalyzer
    from rdkit_cli.io import create_reader

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

    if not args.quiet:
        print("Reading molecules...", file=sys.stderr)

    # Read all molecules
    mols = [r.mol for r in reader]

    if not args.quiet:
        print(f"Analyzing diversity of {len(mols)} molecules...", file=sys.stderr)

    # Analyze
    analyzer = DiversityAnalyzer(
        radius=args.radius,
        n_bits=args.bits,
        sample_size=args.sample_size,
    )

    stats = analyzer.analyze(mols)

    # Output results
    if args.output:
        output_path = Path(args.output)
        from rdkit_cli.io import create_writer
        writer = create_writer(output_path)
        with writer:
            writer.write_row(stats)
        if not args.quiet:
            print(f"Wrote diversity analysis to {output_path}", file=sys.stderr)
    else:
        print("\nDiversity Analysis Results")
        print("=" * 40)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("=" * 40)

    return 0
