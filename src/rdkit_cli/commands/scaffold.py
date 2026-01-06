"""Scaffold command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options


def register_parser(subparsers):
    """Register the scaffold command and subcommands."""
    parser = subparsers.add_parser(
        "scaffold",
        help="Analyze molecular scaffolds",
        description="Extract and analyze Murcko scaffolds.",
        formatter_class=RdkitHelpFormatter,
    )

    scaf_subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # scaffold murcko
    murcko_parser = scaf_subparsers.add_parser(
        "murcko",
        help="Extract Murcko scaffolds",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(murcko_parser)
    add_common_processing_options(murcko_parser)
    murcko_parser.add_argument(
        "--generic",
        action="store_true",
        help="Generate generic (element-agnostic) scaffolds",
    )
    murcko_parser.add_argument(
        "--include-sidechains",
        action="store_true",
        help="Include side chains in output",
    )
    murcko_parser.add_argument(
        "--rings-only",
        action="store_true",
        help="Extract only ring systems (no linkers)",
    )
    murcko_parser.add_argument(
        "--include-original",
        action="store_true",
        help="Include original SMILES in output",
    )
    murcko_parser.set_defaults(func=run_murcko)

    # scaffold decompose
    decompose_parser = scaf_subparsers.add_parser(
        "decompose",
        help="Decompose molecules into scaffold components",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(decompose_parser)
    add_common_processing_options(decompose_parser)
    decompose_parser.set_defaults(func=run_decompose)

    # scaffold analyze
    analyze_parser = scaf_subparsers.add_parser(
        "analyze",
        help="Analyze scaffold frequency distribution",
        formatter_class=RdkitHelpFormatter,
    )
    analyze_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file with scaffold column",
    )
    analyze_parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Output file (optional, prints to stdout if not specified)",
    )
    analyze_parser.add_argument(
        "--scaffold-column",
        default="scaffold",
        help="Name of scaffold column (default: scaffold)",
    )
    analyze_parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top scaffolds to show (default: 20)",
    )
    analyze_parser.add_argument(
        "--no-header",
        action="store_true",
        help="Input file has no header row",
    )
    analyze_parser.set_defaults(func=run_analyze)

    # Set default for main parser
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_murcko(args) -> int:
    """Run Murcko scaffold extraction."""
    # Lazy imports
    from rdkit_cli.core.scaffold import ScaffoldExtractor
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    extractor = ScaffoldExtractor(
        generic=args.generic,
        include_smiles=True,
        include_name=True,
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

    with reader, writer:
        result = process_molecules(
            reader=reader,
            writer=writer,
            processor=extractor.extract,
            n_workers=args.ncpu,
            quiet=args.quiet,
        )

    if not args.quiet:
        print(
            f"Extracted scaffolds for {result.successful}/{result.total_processed} molecules "
            f"({result.failed} failed) in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )

    return 0 if result.failed == 0 else 1


def run_decompose(args) -> int:
    """Run scaffold decomposition."""
    # Lazy imports
    from rdkit_cli.core.scaffold import ScaffoldDecomposer
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    decomposer = ScaffoldDecomposer(
        include_smiles=True,
        include_name=True,
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

    with reader, writer:
        result = process_molecules(
            reader=reader,
            writer=writer,
            processor=decomposer.decompose,
            n_workers=args.ncpu,
            quiet=args.quiet,
        )

    if not args.quiet:
        print(
            f"Decomposed {result.successful}/{result.total_processed} molecules "
            f"({result.failed} failed) in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )

    return 0 if result.failed == 0 else 1


def run_analyze(args) -> int:
    """Run scaffold frequency analysis."""
    # Lazy imports
    import pandas as pd
    from rdkit_cli.core.scaffold import analyze_scaffolds

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Read scaffold data
    header = 0 if not args.no_header else None
    df = pd.read_csv(input_path, header=header)

    if args.no_header:
        # Assume first column is scaffold
        scaffold_col = df.columns[0]
    else:
        scaffold_col = args.scaffold_column

    if scaffold_col not in df.columns:
        print(f"Error: Scaffold column '{scaffold_col}' not found", file=sys.stderr)
        return 1

    scaffolds = df[scaffold_col].dropna().tolist()
    results = analyze_scaffolds(scaffolds, top_n=args.top)

    # Output
    output_lines = ["scaffold,count,percentage"]
    for scaffold, count, pct in results:
        # Escape quotes in scaffold SMILES
        scaffold_escaped = scaffold.replace('"', '""')
        output_lines.append(f'"{scaffold_escaped}",{count},{pct}')

    output_text = "\n".join(output_lines)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            f.write(output_text + "\n")
        print(f"Wrote scaffold analysis to {output_path}", file=sys.stderr)
    else:
        print(output_text)

    print(f"\nTotal scaffolds: {len(scaffolds)}, Unique: {len(set(scaffolds))}", file=sys.stderr)

    return 0
