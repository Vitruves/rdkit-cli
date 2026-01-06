"""Depict command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_processing_options


def register_parser(subparsers):
    """Register the depict command and subcommands."""
    parser = subparsers.add_parser(
        "depict",
        help="Generate molecular depictions",
        description="Generate 2D images of molecules (SVG or PNG).",
        formatter_class=RdkitHelpFormatter,
    )

    depict_subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # depict single
    single_parser = depict_subparsers.add_parser(
        "single",
        help="Depict a single SMILES",
        formatter_class=RdkitHelpFormatter,
    )
    single_parser.add_argument(
        "-s", "--smiles",
        required=True,
        metavar="SMILES",
        help="SMILES string to depict",
    )
    single_parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output file (SVG or PNG)",
    )
    single_parser.add_argument(
        "-W", "--width",
        type=int,
        default=400,
        help="Image width (default: 400)",
    )
    single_parser.add_argument(
        "-H", "--height",
        type=int,
        default=400,
        help="Image height (default: 400)",
    )
    single_parser.add_argument(
        "--atom-indices",
        action="store_true",
        help="Show atom indices",
    )
    single_parser.add_argument(
        "--stereo-annotations",
        action="store_true",
        help="Show stereo annotations",
    )
    single_parser.add_argument(
        "-f", "--format",
        choices=["svg", "png"],
        help="Output format (default: from file extension)",
    )
    single_parser.set_defaults(func=run_single)

    # depict batch
    batch_parser = depict_subparsers.add_parser(
        "batch",
        help="Depict molecules from file to individual images",
        formatter_class=RdkitHelpFormatter,
    )
    batch_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file with molecules",
    )
    batch_parser.add_argument(
        "-o", "--output-dir",
        required=True,
        metavar="DIR",
        help="Output directory for images",
    )
    add_common_processing_options(batch_parser)
    batch_parser.add_argument(
        "-f", "--format",
        choices=["svg", "png"],
        default="svg",
        help="Output format (default: svg)",
    )
    batch_parser.add_argument(
        "-W", "--width",
        type=int,
        default=300,
        help="Image width (default: 300)",
    )
    batch_parser.add_argument(
        "-H", "--height",
        type=int,
        default=300,
        help="Image height (default: 300)",
    )
    batch_parser.set_defaults(func=run_batch)

    # depict grid
    grid_parser = depict_subparsers.add_parser(
        "grid",
        help="Depict molecules as a grid image",
        formatter_class=RdkitHelpFormatter,
    )
    grid_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file with molecules",
    )
    grid_parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output file (SVG or PNG)",
    )
    add_common_processing_options(grid_parser)
    grid_parser.add_argument(
        "--mols-per-row",
        type=int,
        default=4,
        help="Molecules per row (default: 4)",
    )
    grid_parser.add_argument(
        "-W", "--mol-width",
        type=int,
        default=200,
        help="Width per molecule (default: 200)",
    )
    grid_parser.add_argument(
        "-H", "--mol-height",
        type=int,
        default=200,
        help="Height per molecule (default: 200)",
    )
    grid_parser.add_argument(
        "--max-mols",
        type=int,
        default=100,
        help="Maximum molecules to include (default: 100)",
    )
    grid_parser.set_defaults(func=run_grid)

    # Set default for main parser
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_single(args) -> int:
    """Run single molecule depiction."""
    from rdkit_cli.core.depict import depict_smiles

    output_path = Path(args.output)

    # Use explicit format if provided, otherwise infer from extension
    if args.format:
        image_format = args.format
    else:
        image_format = output_path.suffix.lower().lstrip(".")

    if image_format not in ("svg", "png"):
        print(f"Error: Unsupported format '{image_format}'. Use .svg or .png", file=sys.stderr)
        return 1

    image_data = depict_smiles(
        args.smiles,
        width=args.width,
        height=args.height,
        image_format=image_format,
    )

    if image_data is None:
        print(f"Error: Failed to depict SMILES: {args.smiles}", file=sys.stderr)
        return 1

    # Write output
    mode = "w" if image_format == "svg" else "wb"
    with open(output_path, mode) as f:
        f.write(image_data)

    print(f"Wrote depiction to {output_path}", file=sys.stderr)
    return 0


def run_batch(args) -> int:
    """Run batch depiction."""
    from rdkit_cli.core.depict import MoleculeDepiction
    from rdkit_cli.io import create_reader

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        has_header=not args.no_header,
    )

    depictor = MoleculeDepiction(
        width=args.width,
        height=args.height,
        image_format=args.format,
    )

    count = 0
    failed = 0

    for i, record in enumerate(reader):
        if record.mol is None:
            failed += 1
            continue

        image_data = depictor.depict(record.mol)
        if image_data is None:
            failed += 1
            continue

        # Generate filename
        name = record.name or f"mol_{i}"
        # Sanitize filename
        name = "".join(c for c in name if c.isalnum() or c in "-_")
        filename = f"{name}.{args.format}"

        output_path = output_dir / filename

        mode = "w" if args.format == "svg" else "wb"
        with open(output_path, mode) as f:
            f.write(image_data)

        count += 1

    if not args.quiet:
        print(f"Generated {count} images ({failed} failed) in {output_dir}", file=sys.stderr)

    return 0


def run_grid(args) -> int:
    """Run grid depiction."""
    from rdkit_cli.core.depict import GridDepiction
    from rdkit_cli.io import create_reader

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    image_format = output_path.suffix.lower().lstrip(".")

    if image_format not in ("svg", "png"):
        print(f"Error: Unsupported format '{image_format}'. Use .svg or .png", file=sys.stderr)
        return 1

    reader = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        has_header=not args.no_header,
    )

    if not args.quiet:
        print("Reading molecules...", file=sys.stderr)

    # Read molecules
    records = list(reader)[:args.max_mols]
    mols = [r.mol for r in records]
    legends = [r.name or "" for r in records]

    if not args.quiet:
        print(f"Generating grid for {len(mols)} molecules...", file=sys.stderr)

    grid_depictor = GridDepiction(
        mols_per_row=args.mols_per_row,
        mol_width=args.mol_width,
        mol_height=args.mol_height,
        legends=legends,
        use_svg=(image_format == "svg"),
    )

    image_data = grid_depictor.depict(mols)

    if image_data is None:
        print("Error: Failed to generate grid image", file=sys.stderr)
        return 1

    # Write output
    mode = "w" if image_format == "svg" else "wb"
    with open(output_path, mode) as f:
        f.write(image_data)

    if not args.quiet:
        print(f"Wrote grid image to {output_path}", file=sys.stderr)

    return 0
