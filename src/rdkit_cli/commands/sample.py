"""Sample command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options


def register_parser(subparsers):
    """Register the sample command."""
    parser = subparsers.add_parser(
        "sample",
        help="Randomly sample molecules",
        description="Randomly sample molecules from a dataset.",
        formatter_class=RdkitHelpFormatter,
    )

    add_common_io_options(parser)
    add_common_processing_options(parser)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-k", "--num-samples",
        type=int,
        metavar="N",
        help="Number of molecules to sample",
    )
    group.add_argument(
        "-f", "--fraction",
        type=float,
        metavar="F",
        help="Fraction of molecules to sample (0.0-1.0)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Maintain valid/invalid molecule ratio in sample",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use reservoir sampling (memory efficient for large files)",
    )

    parser.set_defaults(func=run_sample)


def run_sample(args) -> int:
    """Run the sample command."""
    from rdkit_cli.core.sample import MoleculeSampler, ReservoirSampler
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.progress.ninja import NinjaProgress

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Create reader
    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        has_header=not args.no_header,
    )

    output_path = Path(args.output)
    writer = create_writer(output_path)

    # Use reservoir sampling for streaming mode
    if args.stream:
        if args.fraction is not None:
            print("Error: --stream mode requires --num-samples, not --fraction", file=sys.stderr)
            return 1

        if not args.quiet:
            print(f"Sampling {args.num_samples} molecules using reservoir sampling...", file=sys.stderr)

        sampler = ReservoirSampler(n=args.num_samples, seed=args.seed)

        with reader:
            total = len(reader)
            progress = NinjaProgress(total=total, quiet=args.quiet)
            progress.start()

            for record in reader:
                sampler.add(record)
                progress.update(1)

            progress.finish()

        sampled = sampler.get_sample()

    else:
        # Load all records, then sample
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

        # Create sampler
        sampler = MoleculeSampler(
            n=args.num_samples,
            fraction=args.fraction,
            seed=args.seed,
            stratify_valid=args.stratify,
        )

        if not args.quiet:
            if args.num_samples:
                print(f"Sampling {args.num_samples} molecules from {len(records)}...", file=sys.stderr)
            else:
                print(f"Sampling {args.fraction*100:.1f}% of {len(records)} molecules...", file=sys.stderr)

        sampled = sampler.sample(records)

    # Write output
    with writer:
        for record in sampled:
            row = {"smiles": record.smiles}
            if record.name:
                row["name"] = record.name
            for key, value in record.metadata.items():
                if key not in row and key != "smiles":
                    row[key] = value
            writer.write_row(row)

    if not args.quiet:
        print(f"Sampled {len(sampled)} molecules. Wrote to {output_path}", file=sys.stderr)

    return 0
