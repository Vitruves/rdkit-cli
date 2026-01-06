"""MCS (Maximum Common Substructure) command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_processing_options


def register_parser(subparsers):
    """Register the mcs command."""
    parser = subparsers.add_parser(
        "mcs",
        help="Find Maximum Common Substructure",
        description="Find the Maximum Common Substructure (MCS) of molecules.",
        formatter_class=RdkitHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file with molecules",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Output file (optional, prints to stdout if not specified)",
    )
    add_common_processing_options(parser)

    # MCS options
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        metavar="SEC",
        help="Maximum time in seconds (default: 60)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        metavar="T",
        help="Fraction of molecules that must contain MCS (default: 1.0)",
    )
    parser.add_argument(
        "--maximize",
        choices=["atoms", "bonds"],
        default="atoms",
        help="What to maximize (default: atoms)",
    )
    parser.add_argument(
        "--no-ring-matches-ring",
        action="store_true",
        help="Allow ring atoms to match non-ring atoms",
    )
    parser.add_argument(
        "--no-complete-rings",
        action="store_true",
        help="Allow partial ring matches",
    )
    parser.add_argument(
        "--match-valences",
        action="store_true",
        help="Match atom valences",
    )
    parser.add_argument(
        "--match-chirality",
        action="store_true",
        help="Match chirality",
    )
    parser.add_argument(
        "--atom-compare",
        choices=["any", "elements", "isotopes"],
        default="elements",
        help="Atom comparison method (default: elements)",
    )
    parser.add_argument(
        "--bond-compare",
        choices=["any", "order", "orderexact"],
        default="order",
        help="Bond comparison method (default: order)",
    )

    parser.set_defaults(func=run_mcs)


def run_mcs(args) -> int:
    """Run MCS finding."""
    from rdkit_cli.core.mcs import find_mcs
    from rdkit_cli.io import create_reader

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

    if not args.quiet:
        print("Reading molecules...", file=sys.stderr)

    # Read all molecules
    records = list(reader)
    mols = [r.mol for r in records if r.mol is not None]

    if len(mols) < 2:
        print("Error: Need at least 2 valid molecules for MCS", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Finding MCS for {len(mols)} molecules...", file=sys.stderr)

    # Find MCS
    result = find_mcs(
        mols,
        timeout=args.timeout,
        threshold=args.threshold,
        maximize=args.maximize,
        ring_matches_ring_only=not args.no_ring_matches_ring,
        complete_rings_only=not args.no_complete_rings,
        match_valences=args.match_valences,
        match_chiral_tag=args.match_chirality,
        atom_compare=args.atom_compare,
        bond_compare=args.bond_compare,
    )

    if result is None:
        print("Error: MCS computation failed", file=sys.stderr)
        return 1

    if result.get("error"):
        print(f"Error: {result['error']}", file=sys.stderr)
        return 1

    # Output results
    if args.output:
        from rdkit_cli.io import create_writer
        output_path = Path(args.output)
        writer = create_writer(output_path)
        with writer:
            writer.write_row(result)
        if not args.quiet:
            print(f"Wrote MCS result to {output_path}", file=sys.stderr)
    else:
        print("\nMCS Results")
        print("=" * 50)

        if result.get("canceled"):
            print(f"WARNING: Search timed out after {args.timeout}s")

        print(f"SMARTS: {result.get('smarts', 'N/A')}")
        print(f"Atoms:  {result.get('num_atoms', 0)}")
        print(f"Bonds:  {result.get('num_bonds', 0)}")
        print("=" * 50)

    return 0
