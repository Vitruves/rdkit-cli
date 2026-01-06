"""Fragment command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options


def register_parser(subparsers):
    """Register the fragment command and subcommands."""
    parser = subparsers.add_parser(
        "fragment",
        help="Fragment molecules",
        description="Fragment molecules using BRICS, RECAP, or functional group analysis.",
        formatter_class=RdkitHelpFormatter,
    )

    frag_subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # fragment brics
    brics_parser = frag_subparsers.add_parser(
        "brics",
        help="Fragment using BRICS algorithm",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(brics_parser)
    add_common_processing_options(brics_parser)
    brics_parser.add_argument(
        "--min-size",
        type=int,
        default=1,
        metavar="N",
        help="Minimum fragment heavy atom count (default: 1)",
    )
    brics_parser.set_defaults(func=run_brics)

    # fragment recap
    recap_parser = frag_subparsers.add_parser(
        "recap",
        help="Fragment using RECAP algorithm",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(recap_parser)
    add_common_processing_options(recap_parser)
    recap_parser.add_argument(
        "--min-size",
        type=int,
        default=1,
        metavar="N",
        help="Minimum fragment heavy atom count (default: 1)",
    )
    recap_parser.set_defaults(func=run_recap)

    # fragment functional-groups
    fg_parser = frag_subparsers.add_parser(
        "functional-groups",
        help="Extract functional group counts",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(fg_parser)
    add_common_processing_options(fg_parser)
    fg_parser.set_defaults(func=run_functional_groups)

    # fragment analyze
    analyze_parser = frag_subparsers.add_parser(
        "analyze",
        help="Analyze fragment frequency distribution",
        formatter_class=RdkitHelpFormatter,
    )
    analyze_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file with fragment_smiles column",
    )
    analyze_parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Output file (optional, prints to stdout if not specified)",
    )
    analyze_parser.add_argument(
        "--fragment-column",
        default="fragment_smiles",
        help="Name of fragment column (default: fragment_smiles)",
    )
    analyze_parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top fragments to show (default: 20)",
    )
    analyze_parser.add_argument(
        "--no-header",
        action="store_true",
        help="Input file has no header row",
    )
    analyze_parser.set_defaults(func=run_analyze)

    # Set default for main parser
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_brics(args) -> int:
    """Run BRICS fragmentation."""
    from rdkit_cli.core.fragment import BRICSFragmenter
    from rdkit_cli.io import create_reader, create_writer

    fragmenter = BRICSFragmenter(
        min_fragment_size=args.min_size,
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
            f"Generated {total_fragments} BRICS fragments from {total_input} molecules",
            file=sys.stderr,
        )

    return 0


def run_recap(args) -> int:
    """Run RECAP fragmentation."""
    from rdkit_cli.core.fragment import RECAPFragmenter
    from rdkit_cli.io import create_reader, create_writer

    fragmenter = RECAPFragmenter(
        min_fragment_size=args.min_size,
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
            f"Generated {total_fragments} RECAP fragments from {total_input} molecules",
            file=sys.stderr,
        )

    return 0


def run_functional_groups(args) -> int:
    """Run functional group extraction."""
    from rdkit_cli.core.fragment import FunctionalGroupExtractor
    from rdkit_cli.io import create_reader, create_writer

    extractor = FunctionalGroupExtractor()

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

    # Note: Running single-threaded because RDKit FragmentCatalog
    # objects can't be pickled for multiprocessing
    total = 0
    successful = 0

    with reader, writer:
        for record in reader:
            total += 1
            result = extractor.extract(record)
            if result is not None:
                writer.write_row(result)
                successful += 1

    if not args.quiet:
        print(
            f"Extracted functional groups for {successful}/{total} molecules "
            f"({total - successful} failed)",
            file=sys.stderr,
        )

    return 0


def run_analyze(args) -> int:
    """Run fragment frequency analysis."""
    import pandas as pd
    from rdkit_cli.core.fragment import analyze_fragments

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Read fragment data
    header = 0 if not args.no_header else None
    df = pd.read_csv(input_path, header=header)

    if args.no_header:
        fragment_col = df.columns[0]
    else:
        fragment_col = args.fragment_column

    if fragment_col not in df.columns:
        print(f"Error: Fragment column '{fragment_col}' not found", file=sys.stderr)
        return 1

    fragments = df[fragment_col].dropna().tolist()
    results = analyze_fragments(fragments, top_n=args.top)

    # Output
    output_lines = ["fragment,count,percentage"]
    for fragment, count, pct in results:
        fragment_escaped = fragment.replace('"', '""')
        output_lines.append(f'"{fragment_escaped}",{count},{pct}')

    output_text = "\n".join(output_lines)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            f.write(output_text + "\n")
        print(f"Wrote fragment analysis to {output_path}", file=sys.stderr)
    else:
        print(output_text)

    print(f"\nTotal fragments: {len(fragments)}, Unique: {len(set(fragments))}", file=sys.stderr)

    return 0
