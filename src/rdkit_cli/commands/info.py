"""Info command implementation."""

import sys

from rdkit_cli.cli import RdkitHelpFormatter


def register_parser(subparsers):
    """Register the info command."""
    parser = subparsers.add_parser(
        "info",
        help="Display quick molecule information",
        description="Get comprehensive information about a molecule from its SMILES.",
        formatter_class=RdkitHelpFormatter,
    )

    parser.add_argument(
        "smiles",
        metavar="SMILES",
        help="SMILES string to analyze",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["text", "json"],
        default="text",
        dest="output_format",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (shortcut for -f json)",
    )

    parser.set_defaults(func=run_info)


def run_info(args) -> int:
    """Run the info command."""
    from rdkit_cli.core.info import get_molecule_info, format_info_text, format_info_json

    info = get_molecule_info(args.smiles)

    if info is None:
        print(f"Error: Failed to parse SMILES: {args.smiles}", file=sys.stderr)
        return 1

    output_format = "json" if args.json else args.output_format

    if output_format == "json":
        print(format_info_json(info))
    else:
        print(format_info_text(info))

    return 0
