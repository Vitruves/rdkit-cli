"""Filter command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options

# Define here to avoid loading core at startup
DRUGLIKE_RULES = ["lipinski", "veber", "ghose", "egan", "muegge"]


def register_parser(subparsers):
    """Register the filter command and subcommands."""
    parser = subparsers.add_parser(
        "filter",
        help="Filter molecules by various criteria",
        description="Filter molecules by substructure, properties, or drug-likeness.",
        formatter_class=RdkitHelpFormatter,
    )

    filter_subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # filter substructure
    sub_parser = filter_subparsers.add_parser(
        "substructure",
        help="Filter by substructure (SMARTS)",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(sub_parser)
    add_common_processing_options(sub_parser)
    sub_parser.add_argument(
        "-s", "--smarts",
        required=True,
        metavar="PATTERN",
        help="SMARTS pattern to match",
    )
    sub_parser.add_argument(
        "--exclude",
        action="store_true",
        help="Exclude molecules matching the pattern (default: include)",
    )
    sub_parser.add_argument(
        "--min-matches",
        type=int,
        default=1,
        metavar="N",
        help="Minimum number of matches required (default: 1)",
    )
    sub_parser.add_argument(
        "--max-matches",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of matches allowed",
    )
    sub_parser.add_argument(
        "--count-unique",
        action="store_true",
        help="Count only unique (non-overlapping) matches",
    )
    sub_parser.add_argument(
        "--add-match-count",
        action="store_true",
        help="Add column with number of matches",
    )
    sub_parser.add_argument(
        "--use-chirality",
        action="store_true",
        help="Consider chirality in matching",
    )
    sub_parser.set_defaults(func=run_substructure)

    # filter property
    prop_parser = filter_subparsers.add_parser(
        "property",
        help="Filter by property values",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(prop_parser)
    add_common_processing_options(prop_parser)
    prop_parser.add_argument(
        "-r", "--rule",
        action="append",
        metavar="RULE",
        help="Property rule in format 'PROP<OP>VALUE' (e.g., 'MolWt<500', 'LogP>-2'). Can be repeated.",
    )
    prop_parser.set_defaults(func=run_property)

    # filter druglike
    drug_parser = filter_subparsers.add_parser(
        "druglike",
        help="Filter by drug-likeness rules",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(drug_parser)
    add_common_processing_options(drug_parser)
    drug_parser.add_argument(
        "-r", "--rule",
        choices=DRUGLIKE_RULES,
        default="lipinski",
        help="Drug-likeness rule set (default: lipinski)",
    )
    drug_parser.add_argument(
        "-v", "--max-violations",
        type=int,
        default=0,
        metavar="N",
        help="Maximum allowed violations (default: 0)",
    )
    drug_parser.add_argument(
        "--add-violations",
        action="store_true",
        help="Add column with violation count",
    )
    drug_parser.add_argument(
        "--add-details",
        action="store_true",
        help="Add columns with individual rule values",
    )
    drug_parser.set_defaults(func=run_druglike)

    # filter pains
    pains_parser = filter_subparsers.add_parser(
        "pains",
        help="Filter out PAINS compounds",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(pains_parser)
    add_common_processing_options(pains_parser)
    pains_parser.add_argument(
        "--keep-pains",
        action="store_true",
        help="Keep PAINS compounds (inverse filter)",
    )
    pains_parser.add_argument(
        "--add-pains-type",
        action="store_true",
        help="Add column with PAINS alert type",
    )
    pains_parser.set_defaults(func=run_pains)

    # filter elements
    elem_parser = filter_subparsers.add_parser(
        "elements",
        help="Filter by allowed elements",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(elem_parser)
    add_common_processing_options(elem_parser)
    elem_parser.add_argument(
        "--allowed",
        metavar="ELEMS",
        default="C,H,N,O,S,F,Cl,Br,I,P",
        help="Comma-separated allowed elements (default: C,H,N,O,S,F,Cl,Br,I,P)",
    )
    elem_parser.add_argument(
        "--required",
        metavar="ELEMS",
        help="Comma-separated required elements (must contain all)",
    )
    elem_parser.add_argument(
        "--forbidden",
        metavar="ELEMS",
        help="Comma-separated forbidden elements",
    )
    elem_parser.set_defaults(func=run_elements)

    # filter complexity
    comp_parser = filter_subparsers.add_parser(
        "complexity",
        help="Filter by molecular complexity",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(comp_parser)
    add_common_processing_options(comp_parser)
    comp_parser.add_argument(
        "--min-atoms",
        type=int,
        default=1,
        metavar="N",
        help="Minimum heavy atom count (default: 1)",
    )
    comp_parser.add_argument(
        "--max-atoms",
        type=int,
        default=100,
        metavar="N",
        help="Maximum heavy atom count (default: 100)",
    )
    comp_parser.add_argument(
        "--min-rings",
        type=int,
        default=0,
        metavar="N",
        help="Minimum ring count (default: 0)",
    )
    comp_parser.add_argument(
        "--max-rings",
        type=int,
        default=10,
        metavar="N",
        help="Maximum ring count (default: 10)",
    )
    comp_parser.add_argument(
        "--min-rotatable",
        type=int,
        default=0,
        metavar="N",
        help="Minimum rotatable bonds (default: 0)",
    )
    comp_parser.add_argument(
        "--max-rotatable",
        type=int,
        default=20,
        metavar="N",
        help="Maximum rotatable bonds (default: 20)",
    )
    comp_parser.set_defaults(func=run_complexity)

    # Set default for main parser
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_substructure(args) -> int:
    """Run the substructure filter."""
    # Lazy imports
    from rdkit_cli.core.filters import SubstructureFilter
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    try:
        filter_obj = SubstructureFilter(
            smarts=args.smarts,
            exclude=args.exclude,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return _run_filter(args, filter_obj.filter)


def run_property(args) -> int:
    """Run the property filter."""
    # Lazy imports
    from rdkit_cli.core.filters import PropertyFilter

    if not args.rule:
        print("Error: At least one --rule is required", file=sys.stderr)
        return 1

    # Parse rules
    rules = {}
    for rule in args.rule:
        try:
            if "<=" in rule:
                prop, val = rule.split("<=")
                rules[prop.strip()] = (None, float(val.strip()))
            elif ">=" in rule:
                prop, val = rule.split(">=")
                rules[prop.strip()] = (float(val.strip()), None)
            elif "<" in rule:
                prop, val = rule.split("<")
                rules[prop.strip()] = (None, float(val.strip()))
            elif ">" in rule:
                prop, val = rule.split(">")
                rules[prop.strip()] = (float(val.strip()), None)
            else:
                print(f"Error: Invalid rule format: {rule}", file=sys.stderr)
                return 1
        except ValueError as e:
            print(f"Error parsing rule '{rule}': {e}", file=sys.stderr)
            return 1

    filter_obj = PropertyFilter(rules=rules)
    return _run_filter(args, filter_obj.filter)


def run_druglike(args) -> int:
    """Run the drug-likeness filter."""
    # Lazy import
    from rdkit_cli.core.filters import DruglikeFilter

    try:
        filter_obj = DruglikeFilter(
            rule_name=args.rule,
            max_violations=args.max_violations,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return _run_filter(args, filter_obj.filter)


def run_pains(args) -> int:
    """Run the PAINS filter."""
    # Lazy import
    from rdkit_cli.core.filters import PAINSFilter

    filter_obj = PAINSFilter(
        exclude=not getattr(args, "keep_pains", False),
    )
    return _run_filter(args, filter_obj.filter)


def run_elements(args) -> int:
    """Run the element filter."""
    # Lazy import
    from rdkit_cli.core.filters import ElementFilter

    allowed = [e.strip() for e in args.allowed.split(",")] if args.allowed else None
    required = [e.strip() for e in args.required.split(",")] if args.required else None
    forbidden = [e.strip() for e in args.forbidden.split(",")] if args.forbidden else None

    filter_obj = ElementFilter(
        allowed_elements=allowed,
        required_elements=required,
        forbidden_elements=forbidden,
    )
    return _run_filter(args, filter_obj.filter)


def run_complexity(args) -> int:
    """Run the complexity filter."""
    # Lazy import
    from rdkit_cli.core.filters import ComplexityFilter

    filter_obj = ComplexityFilter(
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        min_rings=args.min_rings,
        max_rings=args.max_rings,
        min_rotatable=args.min_rotatable,
        max_rotatable=args.max_rotatable,
    )
    return _run_filter(args, filter_obj.filter)


def _run_filter(args, filter_func) -> int:
    """Common filter execution."""
    # Lazy imports
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

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
            processor=filter_func,
            n_workers=args.ncpu,
            quiet=args.quiet,
        )

    if not args.quiet:
        passed = result.successful
        total = result.total_processed
        filtered = total - passed - result.failed
        print(
            f"Passed: {passed}/{total} molecules "
            f"(filtered: {filtered}, failed: {result.failed}) in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )

    return 0
