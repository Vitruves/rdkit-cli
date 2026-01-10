"""Protonate command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options


def register_parser(subparsers):
    """Register the protonate command."""
    parser = subparsers.add_parser(
        "protonate",
        help="Enumerate protonation states",
        description="Generate protonation states at specified pH.",
        formatter_class=RdkitHelpFormatter,
    )

    add_common_io_options(parser)
    add_common_processing_options(parser)

    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        metavar="PH",
        help="Target pH (default: 7.4)",
    )
    parser.add_argument(
        "--enumerate",
        action="store_true",
        help="Enumerate multiple protonation states (default: dominant state only)",
    )
    parser.add_argument(
        "--neutralize",
        action="store_true",
        help="Neutralize all charges instead of pH-based protonation",
    )
    parser.add_argument(
        "--add-charge",
        action="store_true",
        help="Add formal charge column to output",
    )

    parser.set_defaults(func=run_protonate)


def run_protonate(args) -> int:
    """Run the protonate command."""
    from rdkit import Chem
    from rdkit_cli.core.protonate import ProtonationEnumerator, neutralize_mol
    from rdkit_cli.io import create_reader, create_writer

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

    total_input = 0
    total_output = 0

    if args.neutralize:
        # Neutralization mode
        with reader, writer:
            for record in reader:
                total_input += 1
                if record.mol is None:
                    continue

                neutralized = neutralize_mol(record.mol)
                if neutralized:
                    result = {
                        "smiles": Chem.MolToSmiles(neutralized),
                    }
                    if record.name:
                        result["name"] = record.name
                    if args.add_charge:
                        result["formal_charge"] = Chem.GetFormalCharge(neutralized)
                    writer.write_row(result)
                    total_output += 1
    else:
        # pH-based protonation
        enumerator = ProtonationEnumerator(
            ph=args.ph,
            enumerate_all=args.enumerate,
            include_smiles=True,
            include_name=True,
        )

        with reader, writer:
            for record in reader:
                total_input += 1
                results = enumerator.enumerate(record)
                for result in results:
                    if not args.add_charge and "formal_charge" in result:
                        del result["formal_charge"]
                    writer.write_row(result)
                    total_output += 1

    if not args.quiet:
        if args.neutralize:
            print(
                f"Neutralized {total_output}/{total_input} molecules",
                file=sys.stderr,
            )
        else:
            print(
                f"Generated {total_output} protonation states from {total_input} molecules (pH {args.ph})",
                file=sys.stderr,
            )

    return 0
