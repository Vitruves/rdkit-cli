"""MMP (Matched Molecular Pairs) command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options


def register_parser(subparsers):
    """Register the mmp command and subcommands."""
    parser = subparsers.add_parser(
        "mmp",
        help="Matched Molecular Pairs analysis",
        description="Find and analyze matched molecular pairs in datasets.",
        formatter_class=RdkitHelpFormatter,
    )

    mmp_subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # mmp fragment
    frag_parser = mmp_subparsers.add_parser(
        "fragment",
        help="Fragment molecules for MMP analysis",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(frag_parser)
    add_common_processing_options(frag_parser)
    frag_parser.add_argument(
        "--max-cuts",
        type=int,
        default=1,
        choices=[1, 2],
        help="Maximum number of cuts (default: 1)",
    )
    frag_parser.set_defaults(func=run_fragment)

    # mmp find
    find_parser = mmp_subparsers.add_parser(
        "find",
        help="Find matched molecular pairs in a dataset",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(find_parser)
    add_common_processing_options(find_parser)
    find_parser.add_argument(
        "--max-cuts",
        type=int,
        default=1,
        choices=[1, 2],
        help="Maximum number of cuts (default: 1)",
    )
    find_parser.add_argument(
        "--min-core-size",
        type=int,
        default=3,
        help="Minimum core size in heavy atoms (default: 3)",
    )
    find_parser.set_defaults(func=run_find)

    # mmp transform
    trans_parser = mmp_subparsers.add_parser(
        "transform",
        help="Apply MMP transformation to molecules",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(trans_parser)
    add_common_processing_options(trans_parser)
    trans_parser.add_argument(
        "-t", "--transformation",
        required=True,
        metavar="SMIRKS",
        help="Transformation SMIRKS (e.g., '[*:1]C>>[*:1]N')",
    )
    trans_parser.set_defaults(func=run_transform)

    # mmp analyze
    analyze_parser = mmp_subparsers.add_parser(
        "analyze",
        help="Analyze transformation frequency",
        formatter_class=RdkitHelpFormatter,
    )
    analyze_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file with transformation column",
    )
    analyze_parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Output file (optional, prints to stdout if not specified)",
    )
    analyze_parser.add_argument(
        "--transformation-column",
        default="transformation",
        help="Name of transformation column (default: transformation)",
    )
    analyze_parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top transformations to show (default: 20)",
    )
    analyze_parser.add_argument(
        "--no-header",
        action="store_true",
        help="Input file has no header row",
    )
    analyze_parser.set_defaults(func=run_analyze)

    # Set default for main parser
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_fragment(args) -> int:
    """Run MMP fragmentation."""
    from rdkit_cli.core.mmp import MMPFragmenter
    from rdkit_cli.io import create_reader, create_writer

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    fragmenter = MMPFragmenter(
        max_cuts=args.max_cuts,
        include_smiles=True,
        include_name=True,
    )

    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        has_header=not args.no_header,
    )

    output_path = Path(args.output)
    writer = create_writer(output_path)

    total_input = 0
    total_fragments = 0

    with reader, writer:
        for record in reader:
            total_input += 1
            results = fragmenter.fragment(record)
            for result in results:
                writer.write_row(result)
                total_fragments += 1

    if not args.quiet:
        print(
            f"Generated {total_fragments} fragments from {total_input} molecules",
            file=sys.stderr,
        )

    return 0


def run_find(args) -> int:
    """Run MMP pair finding."""
    from rdkit_cli.core.mmp import find_matched_pairs
    from rdkit_cli.io import create_reader, create_writer

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("Reading molecules...", file=sys.stderr)

    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        has_header=not args.no_header,
    )

    # Read all molecules
    molecules = []
    with reader:
        for record in reader:
            if record.mol is not None:
                molecules.append((record.smiles, record.name or "", {}))

    if not args.quiet:
        print(f"Finding matched pairs in {len(molecules)} molecules...", file=sys.stderr)

    output_path = Path(args.output)
    writer = create_writer(output_path)

    pair_count = 0
    with writer:
        for pair in find_matched_pairs(
            molecules,
            max_cuts=args.max_cuts,
            min_core_size=args.min_core_size,
        ):
            writer.write_row(pair)
            pair_count += 1

    if not args.quiet:
        print(
            f"Found {pair_count} matched pairs. Wrote to {output_path}",
            file=sys.stderr,
        )

    return 0


def run_transform(args) -> int:
    """Run MMP transformation."""
    from rdkit_cli.core.mmp import MMPTransformer
    from rdkit_cli.io import create_reader, create_writer

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        transformer = MMPTransformer(
            transformation=args.transformation,
            include_smiles=True,
            include_name=True,
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
    writer = create_writer(output_path)

    total_input = 0
    total_products = 0

    with reader, writer:
        for record in reader:
            total_input += 1
            results = transformer.transform(record)
            for result in results:
                writer.write_row(result)
                total_products += 1

    if not args.quiet:
        print(
            f"Generated {total_products} products from {total_input} molecules",
            file=sys.stderr,
        )

    return 0


def run_analyze(args) -> int:
    """Run transformation frequency analysis."""
    import pandas as pd
    from rdkit_cli.core.mmp import analyze_transformations

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    header = 0 if not args.no_header else None
    df = pd.read_csv(input_path, header=header)

    if args.no_header:
        trans_col = df.columns[0]
    else:
        trans_col = args.transformation_column

    if trans_col not in df.columns:
        print(f"Error: Transformation column '{trans_col}' not found", file=sys.stderr)
        return 1

    # Convert to list of dicts
    pairs = [{"transformation": t} for t in df[trans_col].dropna().tolist()]
    results = analyze_transformations(pairs, top_n=args.top)

    # Output
    output_lines = ["transformation,count,percentage"]
    for trans, count, pct in results:
        trans_escaped = trans.replace('"', '""')
        output_lines.append(f'"{trans_escaped}",{count},{pct}')

    output_text = "\n".join(output_lines)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            f.write(output_text + "\n")
        print(f"Wrote transformation analysis to {output_path}", file=sys.stderr)
    else:
        print(output_text)

    print(f"\nTotal transformations: {len(pairs)}, Unique: {len(set(p['transformation'] for p in pairs))}", file=sys.stderr)

    return 0
