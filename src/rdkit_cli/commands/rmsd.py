"""RMSD command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_processing_options


def register_parser(subparsers):
    """Register the rmsd command and subcommands."""
    parser = subparsers.add_parser(
        "rmsd",
        help="Calculate RMSD between 3D structures",
        description="Calculate Root Mean Square Deviation between 3D molecular structures.",
        formatter_class=RdkitHelpFormatter,
    )

    rmsd_subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # rmsd compare
    compare_parser = rmsd_subparsers.add_parser(
        "compare",
        help="Calculate RMSD between molecules and a reference",
        formatter_class=RdkitHelpFormatter,
    )
    compare_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file with 3D coordinates (SDF)",
    )
    compare_parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output file",
    )
    compare_parser.add_argument(
        "-r", "--reference",
        required=True,
        metavar="FILE",
        help="Reference molecule file (SDF, MOL, PDB)",
    )
    add_common_processing_options(compare_parser)
    compare_parser.add_argument(
        "--no-align",
        action="store_true",
        help="Don't align molecules before calculating RMSD",
    )
    compare_parser.add_argument(
        "--no-symmetry",
        action="store_true",
        help="Don't consider molecular symmetry",
    )
    compare_parser.add_argument(
        "--heavy-atoms-only",
        action="store_true",
        help="Only use heavy atoms for RMSD calculation",
    )
    compare_parser.set_defaults(func=run_compare)

    # rmsd matrix
    matrix_parser = rmsd_subparsers.add_parser(
        "matrix",
        help="Calculate pairwise RMSD matrix",
        formatter_class=RdkitHelpFormatter,
    )
    matrix_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file with 3D coordinates (SDF)",
    )
    matrix_parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output file (CSV)",
    )
    add_common_processing_options(matrix_parser)
    matrix_parser.add_argument(
        "--no-symmetry",
        action="store_true",
        help="Don't consider molecular symmetry",
    )
    matrix_parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal precision (default: 4)",
    )
    matrix_parser.set_defaults(func=run_matrix)

    # rmsd conformers
    conf_parser = rmsd_subparsers.add_parser(
        "conformers",
        help="Analyze RMSD between conformers of each molecule",
        formatter_class=RdkitHelpFormatter,
    )
    conf_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file with 3D coordinates (SDF with multiple conformers)",
    )
    conf_parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output file",
    )
    add_common_processing_options(conf_parser)
    conf_parser.add_argument(
        "--no-symmetry",
        action="store_true",
        help="Don't consider molecular symmetry",
    )
    conf_parser.add_argument(
        "--heavy-atoms-only",
        action="store_true",
        help="Only use heavy atoms",
    )
    conf_parser.set_defaults(func=run_conformers)

    # Set default for main parser
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_compare(args) -> int:
    """Run RMSD comparison."""
    from rdkit import Chem
    from rdkit_cli.core.rmsd import RMSDCalculator
    from rdkit_cli.core.align import load_reference_molecule
    from rdkit_cli.io import create_reader, create_writer

    # Load reference
    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"Error: Reference file not found: {ref_path}", file=sys.stderr)
        return 1

    reference_mol = load_reference_molecule(str(ref_path))
    if reference_mol is None:
        print(f"Error: Could not load reference molecule", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        calculator = RMSDCalculator(
            reference_mol=reference_mol,
            align=not args.no_align,
            symmetry=not args.no_symmetry,
            heavy_atoms_only=args.heavy_atoms_only,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    reader = create_reader(input_path)
    output_path = Path(args.output)
    writer = create_writer(output_path)

    results = []
    total = 0
    failed = 0

    with reader:
        for record in reader:
            total += 1
            if record.mol is None or record.mol.GetNumConformers() == 0:
                failed += 1
                continue

            rmsd = calculator.calculate(record.mol)
            if rmsd is None:
                failed += 1
                continue

            results.append({
                "smiles": record.smiles,
                "name": record.name if record.name else "",
                "rmsd": round(rmsd, 4),
            })

    with writer:
        for r in results:
            writer.write_row(r)

    if not args.quiet:
        print(
            f"Calculated RMSD for {len(results)}/{total} molecules "
            f"({failed} failed). Wrote to {output_path}",
            file=sys.stderr,
        )

    return 0


def run_matrix(args) -> int:
    """Run pairwise RMSD matrix calculation."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("Reading molecules...", file=sys.stderr)

    # Read all molecules
    supplier = Chem.SDMolSupplier(str(input_path))
    mols = []
    names = []

    for mol in supplier:
        if mol is not None and mol.GetNumConformers() > 0:
            mols.append(mol)
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{len(mols)}"
            names.append(name)

    if len(mols) < 2:
        print("Error: Need at least 2 molecules for matrix", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Calculating {len(mols)}x{len(mols)} RMSD matrix...", file=sys.stderr)

    # Calculate pairwise RMSD
    n = len(mols)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            try:
                if args.no_symmetry:
                    from rdkit.Chem import rdMolAlign
                    rmsd = rdMolAlign.AlignMol(mols[i], mols[j])
                else:
                    rmsd = AllChem.GetBestRMS(mols[i], mols[j])
                matrix[i][j] = rmsd
                matrix[j][i] = rmsd
            except Exception:
                matrix[i][j] = float('nan')
                matrix[j][i] = float('nan')

    # Write output
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        # Header
        f.write("," + ",".join(names) + "\n")
        # Data
        prec = args.precision
        for i, row in enumerate(matrix):
            f.write(names[i] + "," + ",".join(f"{v:.{prec}f}" for v in row) + "\n")

    if not args.quiet:
        print(f"Wrote RMSD matrix to {output_path}", file=sys.stderr)

    return 0


def run_conformers(args) -> int:
    """Run conformer RMSD analysis."""
    from rdkit import Chem
    from rdkit_cli.core.rmsd import ConformerRMSDAnalyzer
    from rdkit_cli.io import create_writer

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    analyzer = ConformerRMSDAnalyzer(
        symmetry=not args.no_symmetry,
        heavy_atoms_only=args.heavy_atoms_only,
    )

    # Read molecules (group conformers by name)
    supplier = Chem.SDMolSupplier(str(input_path))

    mol_dict = {}
    for mol in supplier:
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unnamed"
        if name not in mol_dict:
            mol_dict[name] = mol
        else:
            # Add conformer to existing molecule
            mol_dict[name].AddConformer(mol.GetConformer(), assignId=True)

    output_path = Path(args.output)
    writer = create_writer(output_path)

    results = []
    for name, mol in mol_dict.items():
        stats = analyzer.analyze(mol)
        if stats:
            stats["name"] = name
            stats["smiles"] = Chem.MolToSmiles(mol)
            results.append(stats)

    with writer:
        for r in results:
            writer.write_row(r)

    if not args.quiet:
        print(
            f"Analyzed conformer RMSD for {len(results)} molecules. "
            f"Wrote to {output_path}",
            file=sys.stderr,
        )

    return 0
