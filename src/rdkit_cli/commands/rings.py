"""Rings command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options


def register_parser(subparsers):
    """Register the rings command and subcommands."""
    parser = subparsers.add_parser(
        "rings",
        help="Analyze ring systems",
        description="Extract and analyze ring systems from molecules.",
        formatter_class=RdkitHelpFormatter,
    )

    rings_subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # rings extract
    extract_parser = rings_subparsers.add_parser(
        "extract",
        help="Extract ring systems from molecules",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(extract_parser)
    add_common_processing_options(extract_parser)
    extract_parser.add_argument(
        "--no-fused",
        action="store_true",
        help="Exclude fused ring systems",
    )
    extract_parser.add_argument(
        "--no-spiro",
        action="store_true",
        help="Exclude spiro ring systems",
    )
    extract_parser.add_argument(
        "--no-bridged",
        action="store_true",
        help="Exclude bridged ring systems",
    )
    extract_parser.set_defaults(func=run_extract)

    # rings info
    info_parser = rings_subparsers.add_parser(
        "info",
        help="Get ring information for molecules",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(info_parser)
    add_common_processing_options(info_parser)
    info_parser.set_defaults(func=run_info)

    # rings analyze
    analyze_parser = rings_subparsers.add_parser(
        "analyze",
        help="Analyze ring system frequency distribution",
        formatter_class=RdkitHelpFormatter,
    )
    analyze_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file with ring_system column",
    )
    analyze_parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Output file (optional, prints to stdout if not specified)",
    )
    analyze_parser.add_argument(
        "--ring-column",
        default="ring_system",
        help="Name of ring system column (default: ring_system)",
    )
    analyze_parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top ring systems to show (default: 20)",
    )
    analyze_parser.add_argument(
        "--no-header",
        action="store_true",
        help="Input file has no header row",
    )
    analyze_parser.set_defaults(func=run_analyze)

    # Set default for main parser
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_extract(args) -> int:
    """Run ring system extraction."""
    from rdkit_cli.core.rings import RingSystemExtractor
    from rdkit_cli.io import create_reader, create_writer

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    extractor = RingSystemExtractor(
        include_smiles=True,
        include_name=True,
        include_fused=not args.no_fused,
        include_spiro=not args.no_spiro,
        include_bridged=not args.no_bridged,
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
    total_rings = 0

    with reader, writer:
        for record in reader:
            total_input += 1
            results = extractor.extract(record)
            for result in results:
                writer.write_row(result)
                total_rings += 1

    if not args.quiet:
        print(
            f"Extracted {total_rings} ring systems from {total_input} molecules",
            file=sys.stderr,
        )

    return 0


def run_info(args) -> int:
    """Run ring info calculation."""
    from rdkit_cli.core.rings import RingInfo
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    analyzer = RingInfo(
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

    with reader, writer:
        result = process_molecules(
            reader=reader,
            writer=writer,
            processor=analyzer.analyze,
            n_workers=args.ncpu,
            quiet=args.quiet,
        )

    if not args.quiet:
        print(
            f"Analyzed {result.successful}/{result.total_processed} molecules "
            f"({result.failed} failed) in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )

    return 0


def run_analyze(args) -> int:
    """Run ring system frequency analysis."""
    import pandas as pd
    from rdkit_cli.core.rings import analyze_ring_systems

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Read data
    header = 0 if not args.no_header else None
    df = pd.read_csv(input_path, header=header)

    if args.no_header:
        ring_col = df.columns[0]
    else:
        ring_col = args.ring_column

    if ring_col not in df.columns:
        print(f"Error: Ring column '{ring_col}' not found", file=sys.stderr)
        return 1

    ring_systems = df[ring_col].dropna().tolist()
    results = analyze_ring_systems(ring_systems, top_n=args.top)

    # Output
    output_lines = ["ring_system,count,percentage"]
    for ring, count, pct in results:
        ring_escaped = ring.replace('"', '""')
        output_lines.append(f'"{ring_escaped}",{count},{pct}')

    output_text = "\n".join(output_lines)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            f.write(output_text + "\n")
        print(f"Wrote ring analysis to {output_path}", file=sys.stderr)
    else:
        print(output_text)

    print(f"\nTotal ring systems: {len(ring_systems)}, Unique: {len(set(ring_systems))}", file=sys.stderr)

    return 0
