"""Energy command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import (
    RdkitHelpFormatter,
    add_common_io_options,
    add_common_processing_options,
)


def register_parser(subparsers):
    """Register the energy command and subcommands."""
    parser = subparsers.add_parser(
        "energy",
        help="Force field energy calculations",
        description="Compute MMFF/UFF energies and minimize structures.",
        formatter_class=RdkitHelpFormatter,
    )

    energy_sub = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # energy compute
    compute_parser = energy_sub.add_parser(
        "compute",
        help="Compute single-point energy",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(compute_parser)
    add_common_processing_options(compute_parser)
    compute_parser.add_argument(
        "-f", "--force-field",
        choices=["mmff", "uff"],
        default="mmff",
        help="Force field (default: mmff)",
    )
    compute_parser.set_defaults(func=run_compute)

    # energy minimize
    minimize_parser = energy_sub.add_parser(
        "minimize",
        help="Minimize and report energy",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(minimize_parser)
    add_common_processing_options(minimize_parser)
    minimize_parser.add_argument(
        "-f", "--force-field",
        choices=["mmff", "uff"],
        default="mmff",
        help="Force field (default: mmff)",
    )
    minimize_parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum iterations (default: 500)",
    )
    minimize_parser.set_defaults(func=run_minimize)

    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_compute(args) -> int:
    """Compute single-point energies."""
    from rdkit_cli.core.energy import EnergyCalculator
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    calculator = EnergyCalculator(force_field=args.force_field)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=getattr(args, "name_column", None),
        has_header=not args.no_header,
    )
    output_path = Path(args.output)
    writer = create_writer(output_path)

    with reader, writer:
        result = process_molecules(
            reader=reader,
            writer=writer,
            processor=calculator.compute,
            n_workers=1,  # 3D generation not picklable
            quiet=args.quiet,
        )

    if not args.quiet:
        print(
            f"Computed energy for "
            f"{result.successful}/{result.total_processed} molecules "
            f"in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )
    return 0


def run_minimize(args) -> int:
    """Minimize structures."""
    from rdkit_cli.core.energy import EnergyMinimizer
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    minimizer = EnergyMinimizer(
        force_field=args.force_field,
        max_iterations=args.max_iter,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=getattr(args, "name_column", None),
        has_header=not args.no_header,
    )
    output_path = Path(args.output)
    writer = create_writer(output_path)

    with reader, writer:
        result = process_molecules(
            reader=reader,
            writer=writer,
            processor=minimizer.minimize,
            n_workers=1,
            quiet=args.quiet,
        )

    if not args.quiet:
        print(
            f"Minimized "
            f"{result.successful}/{result.total_processed} molecules "
            f"in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )
    return 0
