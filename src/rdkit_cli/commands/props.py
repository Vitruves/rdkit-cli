"""Props command implementation for property column operations."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_processing_options


def register_parser(subparsers):
    """Register the props command and subcommands."""
    parser = subparsers.add_parser(
        "props",
        help="Property column operations",
        description="Add, rename, or remove property columns from molecule files.",
        formatter_class=RdkitHelpFormatter,
    )

    props_subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # props add
    add_parser = props_subparsers.add_parser(
        "add",
        help="Add a new property column",
        formatter_class=RdkitHelpFormatter,
    )
    add_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file",
    )
    add_parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output file",
    )
    add_parser.add_argument(
        "-c", "--column",
        required=True,
        metavar="NAME",
        help="New column name",
    )
    add_parser.add_argument(
        "-v", "--value",
        required=True,
        metavar="VALUE",
        help="Value for the new column (same for all rows)",
    )
    add_parser.add_argument(
        "--no-header",
        action="store_true",
        help="Input file has no header row",
    )
    add_parser.set_defaults(func=run_add)

    # props rename
    rename_parser = props_subparsers.add_parser(
        "rename",
        help="Rename property columns",
        formatter_class=RdkitHelpFormatter,
    )
    rename_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file",
    )
    rename_parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output file",
    )
    rename_parser.add_argument(
        "--from",
        required=True,
        metavar="NAME",
        dest="from_col",
        help="Current column name",
    )
    rename_parser.add_argument(
        "--to",
        required=True,
        metavar="NAME",
        dest="to_col",
        help="New column name",
    )
    rename_parser.add_argument(
        "--no-header",
        action="store_true",
        help="Input file has no header row",
    )
    rename_parser.set_defaults(func=run_rename)

    # props drop
    drop_parser = props_subparsers.add_parser(
        "drop",
        help="Remove property columns",
        formatter_class=RdkitHelpFormatter,
    )
    drop_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file",
    )
    drop_parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output file",
    )
    drop_parser.add_argument(
        "-c", "--columns",
        required=True,
        metavar="COLS",
        help="Comma-separated column names to drop",
    )
    drop_parser.add_argument(
        "--no-header",
        action="store_true",
        help="Input file has no header row",
    )
    drop_parser.set_defaults(func=run_drop)

    # props keep
    keep_parser = props_subparsers.add_parser(
        "keep",
        help="Keep only specified columns",
        formatter_class=RdkitHelpFormatter,
    )
    keep_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file",
    )
    keep_parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output file",
    )
    keep_parser.add_argument(
        "-c", "--columns",
        required=True,
        metavar="COLS",
        help="Comma-separated column names to keep",
    )
    keep_parser.add_argument(
        "--no-header",
        action="store_true",
        help="Input file has no header row",
    )
    keep_parser.set_defaults(func=run_keep)

    # props list
    list_parser = props_subparsers.add_parser(
        "list",
        help="List property columns in a file",
        formatter_class=RdkitHelpFormatter,
    )
    list_parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input file",
    )
    list_parser.add_argument(
        "--no-header",
        action="store_true",
        help="Input file has no header row",
    )
    list_parser.set_defaults(func=run_list)

    # Set default for main parser
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_add(args) -> int:
    """Add a new property column."""
    import pandas as pd

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    header = 0 if not args.no_header else None
    df = pd.read_csv(input_path, header=header)

    # Add new column
    df[args.column] = args.value

    output_path = Path(args.output)
    df.to_csv(output_path, index=False, header=not args.no_header)

    print(f"Added column '{args.column}' with value '{args.value}'. Wrote to {output_path}", file=sys.stderr)
    return 0


def run_rename(args) -> int:
    """Rename a property column."""
    import pandas as pd

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    header = 0 if not args.no_header else None
    df = pd.read_csv(input_path, header=header)

    if args.from_col not in df.columns:
        print(f"Error: Column '{args.from_col}' not found", file=sys.stderr)
        return 1

    df = df.rename(columns={args.from_col: args.to_col})

    output_path = Path(args.output)
    df.to_csv(output_path, index=False, header=not args.no_header)

    print(f"Renamed '{args.from_col}' to '{args.to_col}'. Wrote to {output_path}", file=sys.stderr)
    return 0


def run_drop(args) -> int:
    """Drop property columns."""
    import pandas as pd

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    header = 0 if not args.no_header else None
    df = pd.read_csv(input_path, header=header)

    columns_to_drop = [c.strip() for c in args.columns.split(",")]

    # Check which columns exist
    missing = [c for c in columns_to_drop if c not in df.columns]
    if missing:
        print(f"Warning: Columns not found: {', '.join(missing)}", file=sys.stderr)

    existing = [c for c in columns_to_drop if c in df.columns]
    df = df.drop(columns=existing)

    output_path = Path(args.output)
    df.to_csv(output_path, index=False, header=not args.no_header)

    print(f"Dropped {len(existing)} column(s). Wrote to {output_path}", file=sys.stderr)
    return 0


def run_keep(args) -> int:
    """Keep only specified columns."""
    import pandas as pd

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    header = 0 if not args.no_header else None
    df = pd.read_csv(input_path, header=header)

    columns_to_keep = [c.strip() for c in args.columns.split(",")]

    # Check which columns exist
    missing = [c for c in columns_to_keep if c not in df.columns]
    if missing:
        print(f"Error: Columns not found: {', '.join(missing)}", file=sys.stderr)
        return 1

    df = df[columns_to_keep]

    output_path = Path(args.output)
    df.to_csv(output_path, index=False, header=not args.no_header)

    print(f"Kept {len(columns_to_keep)} column(s). Wrote to {output_path}", file=sys.stderr)
    return 0


def run_list(args) -> int:
    """List property columns."""
    import pandas as pd

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    header = 0 if not args.no_header else None

    # Read just the header
    df = pd.read_csv(input_path, header=header, nrows=0)

    print(f"Columns in {input_path}:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. {col}")

    print(f"\nTotal: {len(df.columns)} columns", file=sys.stderr)
    return 0
