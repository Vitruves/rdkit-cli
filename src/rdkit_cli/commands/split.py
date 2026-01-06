"""Split command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_processing_options


def register_parser(subparsers):
    """Register the split command."""
    parser = subparsers.add_parser(
        "split",
        help="Split files into smaller chunks",
        description="Split a molecular dataset into smaller files.",
        formatter_class=RdkitHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file (CSV, TSV, SMI, SDF, or Parquet)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        metavar="DIR",
        help="Output directory for split files",
    )
    add_common_processing_options(parser)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-c", "--num-chunks",
        type=int,
        metavar="N",
        help="Number of output files to create",
    )
    group.add_argument(
        "-s", "--chunk-size",
        type=int,
        metavar="N",
        help="Number of molecules per output file",
    )

    parser.add_argument(
        "--prefix",
        metavar="NAME",
        help="Prefix for output filenames (default: input filename stem)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "tsv", "smi", "sdf"],
        help="Output format (default: same as input)",
    )

    parser.set_defaults(func=run_split)


def run_split(args) -> int:
    """Run the split command."""
    from rdkit_cli.core.split import FileSplitter
    from rdkit_cli.io import create_reader, create_writer, detect_format
    from rdkit_cli.progress.ninja import NinjaProgress

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect formats
    in_format = detect_format(input_path)
    out_format = args.format or in_format.value

    # Create reader
    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        has_header=not args.no_header,
    )

    # Read all records with progress
    if not args.quiet:
        print("Reading molecules...", file=sys.stderr)

    records = []
    with reader:
        total = len(reader)
        progress = NinjaProgress(total=total, quiet=args.quiet)
        progress.start()

        for record in reader:
            records.append(record)
            progress.update(1)

        progress.finish()

    if not records:
        print("Error: No molecules found in input file", file=sys.stderr)
        return 1

    # Create splitter
    splitter = FileSplitter(
        n_chunks=args.num_chunks,
        chunk_size=args.chunk_size,
    )

    # Get output prefix
    prefix = args.prefix or input_path.stem

    # Calculate number of chunks for filename padding
    assignments = splitter.calculate_chunk_assignments(len(records))
    n_chunks = len(assignments)

    if not args.quiet:
        print(f"Splitting {len(records)} molecules into {n_chunks} files...", file=sys.stderr)

    # Split and write
    files_written = 0
    for chunk_idx, chunk_records in splitter.split_records(records):
        output_path = FileSplitter.generate_output_path(
            output_dir=output_dir,
            base_name=prefix,
            chunk_idx=chunk_idx,
            extension=out_format,
            total_chunks=n_chunks,
        )

        writer = create_writer(output_path)
        with writer:
            for record in chunk_records:
                row = {"smiles": record.smiles}
                if record.name:
                    row["name"] = record.name
                for key, value in record.metadata.items():
                    if key not in row and key != "smiles":
                        row[key] = value
                writer.write_row(row)

        files_written += 1

    if not args.quiet:
        print(
            f"Created {files_written} files in {output_dir}",
            file=sys.stderr,
        )

    return 0
