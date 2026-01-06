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
