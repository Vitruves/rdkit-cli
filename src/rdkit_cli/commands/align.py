"""Align command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options


def register_parser(subparsers):
    """Register the align command."""
    parser = subparsers.add_parser(
        "align",
        help="Align 3D molecules to a reference",
        description="Align 3D molecular structures to a reference molecule.",
        formatter_class=RdkitHelpFormatter,
    )

    add_common_io_options(parser)
    add_common_processing_options(parser)

    parser.add_argument(
        "-r", "--reference",
        required=True,
        metavar="FILE",
        help="Reference molecule file (SDF, MOL, PDB)",
    )
    parser.add_argument(
        "-m", "--method",
        choices=["mcs", "o3a"],
        default="mcs",
        help="Alignment method (default: mcs)",
    )
    parser.add_argument(
        "--crippen",
        action="store_true",
        help="Use Crippen contributions for O3A (default: MMFF)",
    )
    parser.add_argument(
        "--add-rmsd",
        action="store_true",
        default=True,
        help="Add RMSD column to output (default: True)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="RMSD",
        help="Only output molecules with RMSD below threshold",
    )

    parser.set_defaults(func=run_align)


def run_align(args) -> int:
    """Run the align command."""
    from rdkit_cli.core.align import MoleculeAligner, load_reference_molecule
    from rdkit_cli.io import create_reader, create_writer, FileFormat

    # Load reference molecule
    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"Error: Reference file not found: {ref_path}", file=sys.stderr)
        return 1

    reference_mol = load_reference_molecule(str(ref_path))
    if reference_mol is None:
        print(f"Error: Could not load reference molecule from: {ref_path}", file=sys.stderr)
        return 1

    if reference_mol.GetNumConformers() == 0:
        print("Error: Reference molecule has no 3D coordinates", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        aligner = MoleculeAligner(
            reference_mol=reference_mol,
            method=args.method,
            use_crippen=args.crippen,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        has_header=not args.no_header,
    )

    # Force SDF output for 3D coordinates
    output_path = Path(args.output)
    writer = create_writer(output_path, format_override=FileFormat.SDF)

    aligned_count = 0
    failed_count = 0
    filtered_count = 0

    with reader, writer:
        for record in reader:
            result = aligner.align(record)

            if result is None:
                failed_count += 1
                continue

            # Apply threshold filter
            if args.threshold is not None and result["rmsd"] > args.threshold:
                filtered_count += 1
                continue

            # Write aligned molecule
            aligned_mol = result.get("mol")
            if aligned_mol:
                # Set RMSD as property
                if args.add_rmsd:
                    aligned_mol.SetDoubleProp("RMSD", result["rmsd"])

                # Set name
                if result.get("name"):
                    aligned_mol.SetProp("_Name", result["name"])

                writer.write_row({"mol": aligned_mol, "smiles": result.get("smiles", ""), "name": result.get("name", "")})
                aligned_count += 1

    if not args.quiet:
        total = aligned_count + failed_count + filtered_count
        msg = f"Aligned {aligned_count}/{total} molecules ({failed_count} failed"
        if args.threshold:
            msg += f", {filtered_count} filtered by RMSD > {args.threshold}"
        msg += f"). Wrote to {output_path}"
        print(msg, file=sys.stderr)

    return 0
