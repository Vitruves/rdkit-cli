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

    # conformers constrained
    const_parser = conf_subparsers.add_parser(
        "constrained",
        help="Embed molecules constrained to a reference template",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(const_parser)
    add_common_processing_options(const_parser)
    const_parser.add_argument(
        "-r", "--reference",
        required=True,
        metavar="FILE",
        help="Reference molecule file with 3D coords (SDF, MOL, PDB)",
    )
    const_parser.add_argument(
        "-f", "--force-field",
        choices=["mmff", "uff"],
        default="mmff",
        help="Force field for optimization (default: mmff)",
    )
    const_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    const_parser.set_defaults(func=run_constrained)

    # conformers torsion
    torsion_parser = conf_subparsers.add_parser(
        "torsion",
        help="Scan torsion angles and compute energy profile",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(torsion_parser)
    add_common_processing_options(torsion_parser)
    torsion_parser.add_argument(
        "--atoms",
        required=True,
        metavar="I,J,K,L",
        help="Comma-separated atom indices for the dihedral",
    )
    torsion_parser.add_argument(
        "--start",
        type=float,
        default=-180.0,
        help="Start angle in degrees (default: -180)",
    )
    torsion_parser.add_argument(
        "--end",
        type=float,
        default=180.0,
        help="End angle in degrees (default: 180)",
    )
    torsion_parser.add_argument(
        "--step",
        type=float,
        default=10.0,
        help="Step size in degrees (default: 10)",
    )
    torsion_parser.add_argument(
        "-f", "--force-field",
        choices=["mmff", "uff"],
        default="mmff",
        help="Force field (default: mmff)",
    )
    torsion_parser.set_defaults(func=run_torsion)

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


def run_constrained(args) -> int:
    """Run constrained embedding."""
    from rdkit_cli.core.conformers import ConstrainedEmbedder
    from rdkit_cli.io import create_reader, create_writer, FileFormat

    try:
        embedder = ConstrainedEmbedder(
            reference_file=args.reference,
            force_field=args.force_field,
            random_seed=args.seed,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

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

    # Single-threaded: ConstrainedEmbed not picklable
    records = list(reader)
    succeeded = 0
    failed = 0
    with writer:
        for record in records:
            result = embedder.embed(record)
            if result is not None:
                writer.write_row(result)
                succeeded += 1
            else:
                failed += 1

    if not args.quiet:
        total = succeeded + failed
        print(
            f"Embedded {succeeded}/{total} molecules "
            f"({failed} failed)",
            file=sys.stderr,
        )

    return 0


def run_torsion(args) -> int:
    """Run torsion angle scan."""
    from rdkit_cli.core.conformers import TorsionScanner
    from rdkit_cli.io import create_reader, create_writer

    try:
        indices = tuple(int(x) for x in args.atoms.split(","))
        if len(indices) != 4:
            raise ValueError("Need exactly 4 atom indices")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    scanner = TorsionScanner(
        atom_indices=indices,
        start_angle=args.start,
        end_angle=args.end,
        step=args.step,
        force_field=args.force_field,
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

    count = 0
    with writer:
        for record in reader:
            result = scanner.scan(record)
            if result is not None:
                writer.write_row(result)
                count += 1

    if not args.quiet:
        print(
            f"Scanned torsion for {count} molecules",
            file=sys.stderr,
        )

    return 0
