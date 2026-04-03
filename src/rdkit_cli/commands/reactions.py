"""Reactions command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options


def register_parser(subparsers):
    """Register the reactions command and subcommands."""
    parser = subparsers.add_parser(
        "reactions",
        help="Apply chemical reactions and transformations",
        description="Apply SMIRKS transformations and enumerate reaction products.",
        formatter_class=RdkitHelpFormatter,
    )

    rxn_subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # reactions transform
    transform_parser = rxn_subparsers.add_parser(
        "transform",
        help="Apply SMIRKS transformation",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(transform_parser)
    add_common_processing_options(transform_parser)
    transform_parser.add_argument(
        "-s", "--smirks",
        required=True,
        metavar="SMIRKS",
        help="SMIRKS transformation pattern",
    )
    transform_parser.add_argument(
        "--max-products",
        type=int,
        default=100,
        help="Maximum products per molecule (default: 100)",
    )
    transform_parser.set_defaults(func=run_transform)

    # reactions enumerate
    enum_parser = rxn_subparsers.add_parser(
        "enumerate",
        help="Enumerate reaction products",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(enum_parser)
    add_common_processing_options(enum_parser)
    enum_parser.add_argument(
        "-t", "--template",
        required=True,
        metavar="SMARTS",
        help="Reaction SMARTS template",
    )
    enum_parser.add_argument(
        "--reactant2",
        metavar="FILE",
        help="Second reactant file (if reaction has 2 reactants)",
    )
    enum_parser.add_argument(
        "--max-products",
        type=int,
        default=1000,
        help="Maximum total products (default: 1000)",
    )
    enum_parser.set_defaults(func=run_enumerate)

    # reactions map
    map_parser = rxn_subparsers.add_parser(
        "map",
        help="Show atom-atom mapping in a reaction SMARTS",
        formatter_class=RdkitHelpFormatter,
    )
    map_parser.add_argument(
        "-s", "--smarts",
        required=True,
        metavar="SMARTS",
        help="Reaction SMARTS (with atom mapping numbers)",
    )
    map_parser.add_argument(
        "-f", "--format",
        choices=["text", "json"],
        default="text",
        dest="output_format",
        help="Output format (default: text)",
    )
    map_parser.set_defaults(func=run_map)

    # reactions fingerprint
    fp_parser = rxn_subparsers.add_parser(
        "fingerprint",
        help="Compute reaction fingerprints",
        formatter_class=RdkitHelpFormatter,
    )
    fp_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file with reaction SMARTS column",
    )
    fp_parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output file",
    )
    fp_parser.add_argument(
        "--reaction-column",
        default="reaction",
        metavar="COL",
        help="Name of reaction SMARTS column (default: reaction)",
    )
    fp_parser.add_argument(
        "-t", "--type",
        choices=["difference", "structural"],
        default="difference",
        dest="fp_type",
        help="Fingerprint type (default: difference)",
    )
    fp_parser.add_argument(
        "--no-header",
        action="store_true",
        help="Input file has no header row",
    )
    fp_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    fp_parser.set_defaults(func=run_fingerprint)

    # Set default for main parser
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_transform(args) -> int:
    """Run SMIRKS transformation."""
    # Lazy imports
    from rdkit_cli.core.reactions import ReactionTransformer
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    try:
        transformer = ReactionTransformer(
            smirks=args.smirks,
            max_products=args.max_products,
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
    writer = create_writer(output_path)

    with reader, writer:
        result = process_molecules(
            reader=reader,
            writer=writer,
            processor=transformer.transform,
            n_workers=args.ncpu,
            quiet=args.quiet,
        )

    if not args.quiet:
        print(
            f"Transformed {result.successful}/{result.total_processed} molecules "
            f"({result.total_processed - result.successful - result.failed} no reaction, "
            f"{result.failed} failed) in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )

    return 0


def run_enumerate(args) -> int:
    """Run reaction enumeration."""
    # Lazy imports
    from rdkit_cli.core.reactions import ReactionEnumerator
    from rdkit_cli.io import create_reader, create_writer

    try:
        enumerator = ReactionEnumerator(
            reaction_smarts=args.template,
            max_products=args.max_products,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Read reactants
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("Reading reactants...", file=sys.stderr)

    reader1 = create_reader(
        input_path,
        smiles_column=args.smiles_column,
        has_header=not args.no_header,
    )
    mols1 = [r.mol for r in reader1 if r.mol is not None]

    reactant_lists = [mols1]

    # Read second reactant file if provided
    if args.reactant2:
        reactant2_path = Path(args.reactant2)
        if not reactant2_path.exists():
            print(f"Error: Reactant2 file not found: {reactant2_path}", file=sys.stderr)
            return 1

        reader2 = create_reader(reactant2_path, smiles_column=args.smiles_column)
        mols2 = [r.mol for r in reader2 if r.mol is not None]
        reactant_lists.append(mols2)

    if not args.quiet:
        print(f"Enumerating products from {len(mols1)} reactant(s)...", file=sys.stderr)

    try:
        products = enumerator.enumerate(reactant_lists)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Write output
    output_path = Path(args.output)
    writer = create_writer(output_path)

    with writer:
        writer.write_batch(products)

    if not args.quiet:
        print(f"Generated {len(products)} products. Wrote to {output_path}", file=sys.stderr)

    return 0


def run_fingerprint(args) -> int:
    """Compute reaction fingerprints."""
    import pandas as pd
    from rdkit_cli.core.reactions import compute_reaction_fingerprint
    from rdkit_cli.io import create_writer

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    header = 0 if not args.no_header else None
    df = pd.read_csv(input_path, header=header)

    if args.no_header:
        rxn_col = df.columns[0]
    else:
        rxn_col = args.reaction_column

    if rxn_col not in df.columns:
        print(
            f"Error: Column '{rxn_col}' not found",
            file=sys.stderr,
        )
        return 1

    output_path = Path(args.output)
    writer = create_writer(output_path)

    count = 0
    with writer:
        for rxn_smarts in df[rxn_col].dropna():
            try:
                fp = compute_reaction_fingerprint(
                    rxn_smarts, fp_type=args.fp_type,
                )
                if hasattr(fp, "ToBitString"):
                    fp_str = fp.ToBase64()
                else:
                    fp_str = str(dict(fp.GetNonzeroElements()))
                writer.write_row({
                    "reaction": rxn_smarts,
                    "fingerprint": fp_str,
                    "type": args.fp_type,
                })
                count += 1
            except (ValueError, Exception):
                continue

    if not args.quiet:
        print(
            f"Computed {args.fp_type} fingerprints for "
            f"{count} reactions",
            file=sys.stderr,
        )
    return 0


def run_map(args) -> int:
    """Show atom-atom mapping in a reaction."""
    import json
    from rdkit_cli.core.reactions import get_atom_mapping

    try:
        mapping = get_atom_mapping(args.smarts)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.output_format == "json":
        print(json.dumps(mapping, indent=2))
    else:
        print(f"Reaction: {mapping['reaction_smarts']}")
        print(f"Has mapping: {mapping['has_mapping']}")
        print(
            f"Reactants: {mapping['num_reactants']}, "
            f"Products: {mapping['num_products']}"
        )
        print()

        for i, rmap in enumerate(mapping["reactant_maps"]):
            print(f"Reactant {i + 1} mapped atoms:")
            for map_num, info in sorted(rmap.items()):
                print(f"  :{map_num} -> {info['symbol']} (idx {info['idx']})")

        print()
        for i, pmap in enumerate(mapping["product_maps"]):
            print(f"Product {i + 1} mapped atoms:")
            for map_num, info in sorted(pmap.items()):
                print(f"  :{map_num} -> {info['symbol']} (idx {info['idx']})")

    return 0
