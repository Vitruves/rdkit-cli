"""Merge command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_processing_options


def register_parser(subparsers):
    """Register the merge command."""
    parser = subparsers.add_parser(
        "merge",
        help="Merge multiple molecule files",
        description="Combine molecules from multiple input files into one output file.",
        formatter_class=RdkitHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        metavar="FILE",
        dest="input_files",
        help="Input files to merge (multiple allowed)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output file",
    )
    add_common_processing_options(parser)
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Remove duplicate molecules",
    )
    parser.add_argument(
        "--dedupe-by",
        choices=["smiles", "inchi", "inchikey"],
        default="smiles",
        help="Key for deduplication (default: smiles)",
    )
    parser.add_argument(
        "--add-source",
        action="store_true",
        help="Add source_file column with original filename",
    )

    parser.set_defaults(func=run_merge)


def run_merge(args) -> int:
    """Run the merge command."""
    from rdkit_cli.core.merge import MoleculeMerger
    from rdkit_cli.io import create_writer

    # Validate input files
    input_paths = []
    for f in args.input_files:
        path = Path(f)
        if not path.exists():
            print(f"Error: Input file not found: {path}", file=sys.stderr)
            return 1
        input_paths.append(path)

    if not args.quiet:
        print(f"Merging {len(input_paths)} files...", file=sys.stderr)

    merger = MoleculeMerger(
        deduplicate=args.dedupe,
        dedupe_key=args.dedupe_by,
        add_source=args.add_source,
    )

    output_path = Path(args.output)
    writer = create_writer(output_path)

    total = 0
    with writer:
        for record in merger.merge_files(
            input_paths,
            smiles_column=args.smiles_column,
            name_column=args.name_column,
            has_header=not args.no_header,
        ):
            writer.write_row(record)
            total += 1

    if not args.quiet:
        stats = merger.get_stats()
        if args.dedupe:
            print(
                f"Merged {total} molecules ({stats['unique_molecules']} unique). "
                f"Wrote to {output_path}",
                file=sys.stderr,
            )
        else:
            print(f"Merged {total} molecules. Wrote to {output_path}", file=sys.stderr)

    return 0
