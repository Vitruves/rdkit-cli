"""Similarity command implementation."""

import sys
from pathlib import Path

from rdkit_cli.cli import RdkitHelpFormatter, add_common_io_options, add_common_processing_options

# Define here to avoid loading core at startup
SIMILARITY_METRICS = ["tanimoto", "dice", "cosine", "sokal", "russel"]


def register_parser(subparsers):
    """Register the similarity command and subcommands."""
    parser = subparsers.add_parser(
        "similarity",
        help="Compute molecular similarity",
        description="Search, compare, and cluster molecules by similarity.",
        formatter_class=RdkitHelpFormatter,
    )

    sim_subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        metavar="<subcommand>",
    )

    # similarity search
    search_parser = sim_subparsers.add_parser(
        "search",
        help="Search for molecules similar to a query",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(search_parser)
    add_common_processing_options(search_parser)
    search_parser.add_argument(
        "--query",
        required=True,
        metavar="SMILES",
        help="Query molecule SMILES",
    )
    search_parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.7,
        metavar="T",
        help="Minimum similarity threshold (default: 0.7)",
    )
    search_parser.add_argument(
        "-m", "--metric",
        choices=SIMILARITY_METRICS,
        default="tanimoto",
        help="Similarity metric (default: tanimoto)",
    )
    search_parser.add_argument(
        "-r", "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (default: 2)",
    )
    search_parser.add_argument(
        "-b", "--bits",
        type=int,
        default=2048,
        help="Fingerprint bit size (default: 2048)",
    )
    search_parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        metavar="N",
        help="Return only top N most similar molecules",
    )
    search_parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort output by similarity (descending)",
    )
    search_parser.add_argument(
        "--fp-type",
        choices=["morgan", "maccs", "rdkit", "atompair", "torsion"],
        default="morgan",
        help="Fingerprint type (default: morgan)",
    )
    search_parser.add_argument(
        "--include-query",
        action="store_true",
        help="Include query molecule in output",
    )
    search_parser.add_argument(
        "--add-rank",
        action="store_true",
        help="Add similarity rank column",
    )
    search_parser.set_defaults(func=run_search)

    # similarity matrix
    matrix_parser = sim_subparsers.add_parser(
        "matrix",
        help="Compute pairwise similarity matrix",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(matrix_parser)
    add_common_processing_options(matrix_parser)
    matrix_parser.add_argument(
        "-m", "--metric",
        choices=SIMILARITY_METRICS,
        default="tanimoto",
        help="Similarity metric (default: tanimoto)",
    )
    matrix_parser.add_argument(
        "--fp-type",
        choices=["morgan", "maccs", "rdkit", "atompair", "torsion"],
        default="morgan",
        help="Fingerprint type (default: morgan)",
    )
    matrix_parser.add_argument(
        "-r", "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (default: 2)",
    )
    matrix_parser.add_argument(
        "-b", "--bits",
        type=int,
        default=2048,
        help="Fingerprint bit size (default: 2048)",
    )
    matrix_parser.add_argument(
        "--distance",
        action="store_true",
        help="Output distance matrix (1 - similarity) instead of similarity",
    )
    matrix_parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal precision (default: 4)",
    )
    matrix_parser.set_defaults(func=run_matrix)

    # similarity cluster
    cluster_parser = sim_subparsers.add_parser(
        "cluster",
        help="Cluster molecules by similarity",
        formatter_class=RdkitHelpFormatter,
    )
    add_common_io_options(cluster_parser)
    add_common_processing_options(cluster_parser)
    cluster_parser.add_argument(
        "-c", "--cutoff",
        type=float,
        default=0.3,
        metavar="C",
        help="Distance cutoff (1-similarity, default: 0.3)",
    )
    cluster_parser.add_argument(
        "-r", "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (default: 2)",
    )
    cluster_parser.add_argument(
        "-b", "--bits",
        type=int,
        default=2048,
        help="Fingerprint bit size (default: 2048)",
    )
    cluster_parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=1,
        help="Minimum cluster size to include (default: 1)",
    )
    cluster_parser.add_argument(
        "--fp-type",
        choices=["morgan", "maccs", "rdkit", "atompair", "torsion"],
        default="morgan",
        help="Fingerprint type (default: morgan)",
    )
    cluster_parser.add_argument(
        "--method",
        choices=["butina", "hierarchical"],
        default="butina",
        help="Clustering method (default: butina)",
    )
    cluster_parser.add_argument(
        "--add-centroid",
        action="store_true",
        help="Mark cluster centroids",
    )
    cluster_parser.set_defaults(func=run_cluster)

    # Set default for main parser
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def run_search(args) -> int:
    """Run similarity search."""
    # Lazy imports
    from rdkit_cli.core.similarity import SimilaritySearcher, SimilarityMetric
    from rdkit_cli.io import create_reader, create_writer
    from rdkit_cli.parallel.batch import process_molecules

    try:
        searcher = SimilaritySearcher(
            query_smiles=args.query,
            threshold=args.threshold,
            metric=SimilarityMetric(args.metric),
            radius=args.radius,
            n_bits=args.bits,
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
            processor=searcher.search,
            n_workers=args.ncpu,
            quiet=args.quiet,
        )

    if not args.quiet:
        found = result.successful
        total = result.total_processed
        print(
            f"Found {found}/{total} molecules above threshold "
            f"({result.failed} failed) in {result.elapsed_time:.1f}s",
            file=sys.stderr,
        )

    return 0


def run_matrix(args) -> int:
    """Compute similarity matrix."""
    # Lazy imports
    from rdkit_cli.core.similarity import compute_similarity_matrix, SimilarityMetric
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

    # Read all molecules
    if not args.quiet:
        print("Reading molecules...", file=sys.stderr)

    records = list(reader)
    mols = [r.mol for r in records]
    names = [r.name or r.smiles[:20] for r in records]

    if not args.quiet:
        print(f"Computing {len(mols)}x{len(mols)} similarity matrix...", file=sys.stderr)

    matrix = compute_similarity_matrix(
        mols,
        metric=SimilarityMetric(args.metric),
    )

    # Write output
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        # Header
        f.write("," + ",".join(names) + "\n")
        # Data
        for i, row in enumerate(matrix):
            f.write(names[i] + "," + ",".join(f"{v:.4f}" for v in row) + "\n")

    if not args.quiet:
        print(f"Wrote similarity matrix to {output_path}", file=sys.stderr)

    return 0


def run_cluster(args) -> int:
    """Cluster molecules."""
    # Lazy imports
    from rdkit_cli.core.similarity import cluster_molecules
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

    # Read all molecules
    if not args.quiet:
        print("Reading molecules...", file=sys.stderr)

    records = list(reader)
    mols = [r.mol for r in records]

    if not args.quiet:
        print(f"Clustering {len(mols)} molecules...", file=sys.stderr)

    clusters = cluster_molecules(
        mols,
        cutoff=args.cutoff,
        radius=args.radius,
        n_bits=args.bits,
    )

    # Filter by minimum cluster size
    min_size = getattr(args, "min_cluster_size", 1)
    clusters = [c for c in clusters if len(c) >= min_size]

    # Write output
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        f.write("smiles,name,cluster,cluster_size\n")
        for cluster_id, cluster in enumerate(clusters):
            cluster_size = len(cluster)
            for idx in cluster:
                r = records[idx]
                smiles = r.smiles.replace('"', '""')
                name = (r.name or "").replace('"', '""')
                f.write(f'"{smiles}","{name}",{cluster_id},{cluster_size}\n')

    if not args.quiet:
        print(
            f"Found {len(clusters)} clusters from {len(mols)} molecules. "
            f"Wrote to {output_path}",
            file=sys.stderr,
        )

    return 0
