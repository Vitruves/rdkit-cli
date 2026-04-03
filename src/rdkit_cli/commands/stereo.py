"""Stereo command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import (
    RdkitHelpFormatter,
    add_common_io_options,
    add_common_processing_options,
)


def register_parser(subparsers):
    """Register the stereo command and subcommands."""
    parser = subparsers.add_parser(
        "stereo",
        help="Analyze and manipulate stereochemistry",
        description="Assign CIP labels, perceive stereocenters, "
        "and analyze enhanced stereo groups.",
        formatter_class=RdkitHelpFormatter,
    )

    stereo_sub = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # stereo assign
    assign_parser = stereo_sub.add_parser(
        "assign",
        help="Assign CIP (R/S, E/Z) labels",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(assign_parser)
    add_common_processing_options(assign_parser)
    assign_parser.set_defaults(func=run_assign)

    # stereo perceive
    perceive_parser = stereo_sub.add_parser(
        "perceive",
        help="Find potential stereocenters",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(perceive_parser)
    add_common_processing_options(perceive_parser)
    perceive_parser.set_defaults(func=run_perceive)

    # stereo enhanced
    enhanced_parser = stereo_sub.add_parser(
        "enhanced",
        help="Show enhanced stereo group info",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(enhanced_parser)
    add_common_processing_options(enhanced_parser)
    enhanced_parser.set_defaults(func=run_enhanced)

    # stereo clean
    clean_parser = stereo_sub.add_parser(
        "clean",
        help="Clean and canonicalize stereo groups",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(clean_parser)
    add_common_processing_options(clean_parser)
    clean_parser.set_defaults(func=run_clean)

    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_assign(args) -> int:
    """Assign CIP labels."""
    from rdkit_cli.core.stereo import StereoAssigner
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    assigner = StereoAssigner()

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
            processor=assigner.assign,
            n_workers=args.ncpu,
            quiet=args.quiet,
        )

    if not args.quiet:
        print(
            f"Assigned CIP labels for "
            f"{result.successful}/{result.total_processed} molecules "
            f"in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )
    return 0


def run_perceive(args) -> int:
    """Perceive potential stereocenters."""
    from rdkit_cli.core.stereo import StereoPerceiver
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    perceiver = StereoPerceiver()

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
            processor=perceiver.perceive,
            n_workers=args.ncpu,
            quiet=args.quiet,
        )

    if not args.quiet:
        print(
            f"Perceived stereo for "
            f"{result.successful}/{result.total_processed} molecules "
            f"in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )
    return 0


def run_enhanced(args) -> int:
    """Show enhanced stereo groups."""
    from rdkit import Chem
    from rdkit_cli.core.stereo import get_enhanced_stereo
    from rdkit_cli.io import create_reader, create_writer

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
            if record.mol is None:
                continue
            groups = get_enhanced_stereo(record.mol)
            row = {
                "smiles": record.smiles,
                "num_stereo_groups": len(groups),
                "stereo_groups": str(groups) if groups else "",
            }
            if record.name:
                row["name"] = record.name
            writer.write_row(row)
            count += 1

    if not args.quiet:
        print(
            f"Analyzed enhanced stereo for {count} molecules",
            file=sys.stderr,
        )
    return 0


def run_clean(args) -> int:
    """Clean and canonicalize stereo groups."""
    from rdkit import Chem
    from rdkit_cli.io import create_reader, create_writer

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
            if record.mol is None:
                continue
            mol = Chem.RWMol(record.mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            cleaned = Chem.MolToSmiles(mol, isomericSmiles=True)
            row = {
                "smiles": cleaned,
                "original_smiles": record.smiles,
            }
            if record.name:
                row["name"] = record.name
            writer.write_row(row)
            count += 1

    if not args.quiet:
        print(f"Cleaned stereo for {count} molecules", file=sys.stderr)
    return 0
