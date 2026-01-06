"""Main CLI entry point for rdkit-cli."""

import argparse
import sys
from typing import Optional

from rich_argparse import RichHelpFormatter

from rdkit_cli import __version__


class RdkitHelpFormatter(RichHelpFormatter):
    """Custom formatter with adjusted styles and command-first ordering."""

    styles = {
        **RichHelpFormatter.styles,
        "argparse.args": "cyan",
        "argparse.groups": "bold yellow",
        "argparse.metavar": "green",
        "argparse.prog": "bold magenta",
    }


def add_common_io_options(parser: argparse.ArgumentParser):
    """Add common I/O options to a parser."""
    parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file (CSV, TSV, SMI, SDF, or Parquet)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output file",
    )


def add_common_processing_options(parser: argparse.ArgumentParser):
    """Add common processing options to a parser."""
    parser.add_argument(
        "-n", "--ncpu",
        type=int,
        default=-1,
        metavar="N",
        help="Number of CPU cores (-1 for all, default: -1)",
    )
    parser.add_argument(
        "--smiles-column",
        default="smiles",
        metavar="COL",
        help="Name of SMILES column (default: smiles)",
    )
    parser.add_argument(
        "--name-column",
        default=None,
        metavar="COL",
        help="Name of molecule name column",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Input file has no header row",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--no-warnings",
        action="store_true",
        help="Suppress RDKit warnings (kekulization errors, etc.)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default=None,
        metavar="LEVEL",
        help="RDKit log level (default: warning, use 'error' to suppress warnings)",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="rdkit-cli",
        description="A comprehensive CLI tool for RDKit cheminformatics operations.",
        epilog="Use 'rdkit-cli <command> --help' for command-specific help.",
        formatter_class=RdkitHelpFormatter,
    )

    # Version
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"rdkit-cli {__version__}",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title="Commands",
        dest="command",
        metavar="<command>",
    )

    # Register all command modules
    _register_commands(subparsers)

    return parser


def _register_commands(subparsers):
    """Register all command subparsers (alphabetical order)."""
    from rdkit_cli.commands import (
        conformers,
        convert,
        deduplicate,
        depict,
        descriptors,
        diversity,
        enumerate,
        filter,
        fingerprints,
        fragment,
        mcs,
        reactions,
        sample,
        scaffold,
        similarity,
        split,
        standardize,
        stats,
        validate,
    )

    # Each module has a register_parser(subparsers) function
    conformers.register_parser(subparsers)
    convert.register_parser(subparsers)
    deduplicate.register_parser(subparsers)
    depict.register_parser(subparsers)
    descriptors.register_parser(subparsers)
    diversity.register_parser(subparsers)
    enumerate.register_parser(subparsers)
    filter.register_parser(subparsers)
    fingerprints.register_parser(subparsers)
    fragment.register_parser(subparsers)
    mcs.register_parser(subparsers)
    reactions.register_parser(subparsers)
    sample.register_parser(subparsers)
    scaffold.register_parser(subparsers)
    similarity.register_parser(subparsers)
    split.register_parser(subparsers)
    standardize.register_parser(subparsers)
    stats.register_parser(subparsers)
    validate.register_parser(subparsers)


def main(args: Optional[list[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 1

    # Configure logging based on --no-warnings or --log-level
    from rdkit_cli.utils import configure_all_warnings, set_rdkit_log_level
    no_warnings = getattr(parsed_args, "no_warnings", False)
    log_level = getattr(parsed_args, "log_level", None)

    if no_warnings:
        # Suppress both RDKit and application warnings
        configure_all_warnings(suppress=True)
    elif log_level is not None:
        # Only control RDKit log level
        set_rdkit_log_level(log_level)

    # Each command has a run(args) function via set_defaults(func=...)
    try:
        return parsed_args.func(parsed_args)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        return 130
    except BrokenPipeError:
        # Handle broken pipe gracefully (e.g., piping to head)
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
