"""Stats command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_processing_options


def register_parser(subparsers):
    """Register the stats command."""
    parser = subparsers.add_parser(
        "stats",
        help="Calculate dataset statistics",
        description="Calculate statistics over molecular properties in a dataset.",
        formatter_class=RdkitHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file (CSV, TSV, SMI, SDF, or Parquet)",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Output file (optional, prints to stdout if not specified)",
    )
    add_common_processing_options(parser)
    parser.add_argument(
        "-p", "--properties",
        metavar="PROPS",
        help="Comma-separated list of properties to calculate (default: MolWt,LogP,TPSA,NumHDonors,NumHAcceptors)",
    )
    parser.add_argument(
        "--list-properties",
        action="store_true",
        help="List available properties and exit",
    )
    parser.add_argument(
        "--format",
        choices=["text", "csv", "json"],
        default="text",
        help="Output format (default: text)",
    )

    parser.set_defaults(func=run_stats)


def run_stats(args) -> int:
    """Run the stats command."""
    from rdkit_cli.core.stats import DatasetStatistics
    from rdkit_cli.io import create_reader
    from rdkit_cli.progress.ninja import NinjaProgress

    # Handle --list-properties
    if args.list_properties:
        print("Available properties:")
        for prop in DatasetStatistics.available_properties():
            print(f"  {prop}")
        return 0

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Parse properties
    properties = None
    if args.properties:
        properties = [p.strip() for p in args.properties.split(",")]

    # Create reader
    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        has_header=not args.no_header,
    )

    # Read all molecules with progress
    if not args.quiet:
        print("Reading molecules...", file=sys.stderr)

    mols = []
    with reader:
        total = len(reader)
        progress = NinjaProgress(total=total, quiet=args.quiet)
        progress.start()

        for record in reader:
            mols.append(record.mol)
            progress.update(1)

        progress.finish()

    if not args.quiet:
        print(f"Calculating statistics for {len(mols)} molecules...", file=sys.stderr)

    # Calculate statistics
    stats_calc = DatasetStatistics(properties=properties)
    stats = stats_calc.calculate(mols)

    # Output results
    if args.output:
        output_path = Path(args.output)
        _write_stats(stats, output_path, args.format)
        if not args.quiet:
            print(f"Wrote statistics to {output_path}", file=sys.stderr)
    else:
        _print_stats(stats, args.format)

    return 0


def _print_stats(stats: dict, format: str) -> None:
    """Print statistics to stdout."""
    if format == "json":
        import json
        print(json.dumps(stats, indent=2))
    elif format == "csv":
        print(",".join(stats.keys()))
        print(",".join(str(v) for v in stats.values()))
    else:
        # Text format
        print("\nDataset Statistics")
        print("=" * 50)

        # Group stats
        general = {k: v for k, v in stats.items() if not any(
            k.endswith(s) for s in ("_min", "_max", "_mean", "_median", "_stdev")
        )}
        properties = {}
        for k, v in stats.items():
            for suffix in ("_min", "_max", "_mean", "_median", "_stdev"):
                if k.endswith(suffix):
                    prop = k[:-len(suffix)]
                    if prop not in properties:
                        properties[prop] = {}
                    properties[prop][suffix[1:]] = v

        # Print general stats
        for key, value in general.items():
            print(f"{key}: {value}")

        # Print property stats
        if properties:
            print("\nProperty Statistics:")
            print("-" * 50)
            for prop, values in properties.items():
                print(f"\n{prop}:")
                for stat, val in values.items():
                    print(f"  {stat}: {val}")

        print("=" * 50)


def _write_stats(stats: dict, path: Path, format: str) -> None:
    """Write statistics to file."""
    if format == "json":
        import json
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
    elif format == "csv":
        with open(path, "w") as f:
            f.write(",".join(stats.keys()) + "\n")
            f.write(",".join(str(v) for v in stats.values()) + "\n")
    else:
        # Text format
        with open(path, "w") as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
