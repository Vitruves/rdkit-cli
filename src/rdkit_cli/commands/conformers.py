"""Conformers command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options


def register_parser(subparsers):
    """Register the conformers command and subcommands."""
    parser = subparsers.add_parser(
        "conformers",
        help="Generate and optimize 3D conformers",
        description="Generate and optimize 3D molecular conformers.",
        formatter_class=RdkitHelpFormatter,
    )

    conf_subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # conformers generate
    gen_parser = conf_subparsers.add_parser(
        "generate",
        help="Generate 3D conformers",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(gen_parser)
    add_common_processing_options(gen_parser)
    gen_parser.add_argument(
        "--num",
        type=int,
        default=10,
        metavar="N",
        help="Number of conformers to generate (default: 10)",
    )
    gen_parser.add_argument(
        "-m", "--method",
        choices=["etkdgv3", "etkdgv2", "etdg"],
        default="etkdgv3",
        help="Embedding method (default: etkdgv3)",
    )
    gen_parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip force field optimization",
    )
    gen_parser.add_argument(
        "-f", "--force-field",
        choices=["mmff", "uff"],
        default="mmff",
        help="Force field for optimization (default: mmff)",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    gen_parser.add_argument(
        "--prune-rms",
        type=float,
        default=0.5,
        metavar="THRESH",
        help="RMSD threshold for pruning similar conformers (default: 0.5)",
    )
    gen_parser.add_argument(
        "--energy-window",
        type=float,
        default=None,
        metavar="KCAL",
        help="Keep only conformers within N kcal/mol of lowest energy",
    )
    gen_parser.add_argument(
        "--add-hydrogens",
        action="store_true",
        default=True,
        help="Add hydrogens before embedding (default: True)",
    )
    gen_parser.add_argument(
        "--no-hydrogens",
        action="store_true",
        help="Don't add hydrogens",
    )
    gen_parser.add_argument(
        "--use-basic-knowledge",
        action="store_true",
        help="Use basic knowledge about conformer preferences",
    )
    gen_parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        metavar="N",
        help="Maximum embedding attempts per conformer (0 = auto)",
    )
    gen_parser.set_defaults(func=run_generate)

    # conformers optimize
    opt_parser = conf_subparsers.add_parser(
        "optimize",
        help="Optimize existing 3D structures",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(opt_parser)
    add_common_processing_options(opt_parser)
    opt_parser.add_argument(
        "-f", "--force-field",
        choices=["mmff", "uff"],
        default="mmff",
        help="Force field for optimization (default: mmff)",
    )
    opt_parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum optimization iterations (default: 200)",
    )
    opt_parser.set_defaults(func=run_optimize)

    # Set default for main parser
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_generate(args) -> int:
    """Run conformer generation."""
    # Lazy imports
    from rdkit_cli.core.conformers import ConformerGenerator
    from rdkit_cli.io import create_reader, create_writer, FileFormat
    from rdkit_cli.parallel.batch import process_molecules

    generator = ConformerGenerator(
        num_conformers=args.num,
        method=args.method,
        optimize=not args.no_optimize,
        force_field=args.force_field,
        random_seed=args.seed,
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

    # Force SDF output for 3D structures
    output_path = Path(args.output)
    writer = create_writer(output_path, format_override=FileFormat.SDF)

    with reader, writer:
        result = process_molecules(
            reader=reader,
            writer=writer,
            processor=generator.generate,
            n_workers=args.ncpu,
            quiet=args.quiet,
        )

    if not args.quiet:
        print(
            f"Generated conformers for {result.successful}/{result.total_processed} molecules "
            f"({result.failed} failed) in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )

    return 0 if result.failed == 0 else 1


def run_optimize(args) -> int:
    """Run conformer optimization."""
    # Lazy imports
    from rdkit_cli.core.conformers import ConformerOptimizer
    from rdkit_cli.io import create_reader, create_writer, FileFormat
    from rdkit_cli.parallel.batch import process_molecules

    optimizer = ConformerOptimizer(
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
        name_column=args.name_column,
        has_header=not args.no_header,
    )

    output_path = Path(args.output)
    writer = create_writer(output_path, format_override=FileFormat.SDF)

    with reader, writer:
        result = process_molecules(
            reader=reader,
            writer=writer,
            processor=optimizer.optimize,
            n_workers=args.ncpu,
            quiet=args.quiet,
        )

    if not args.quiet:
        print(
            f"Optimized {result.successful}/{result.total_processed} molecules "
            f"({result.failed} failed) in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )

    return 0 if result.failed == 0 else 1
