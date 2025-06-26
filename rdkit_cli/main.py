# rdkit_cli/main.py
import argparse
import sys
from typing import List, Optional


def create_parser() -> argparse.ArgumentParser:
	try:
		from rich_argparse import RichHelpFormatter
		# Configure rich-argparse with enhanced colors and styling
		RichHelpFormatter.styles["argparse.text"] = "dim white"
		RichHelpFormatter.styles["argparse.prog"] = "bold bright_magenta"
		RichHelpFormatter.styles["argparse.args"] = "bold cyan"
		RichHelpFormatter.styles["argparse.help"] = "white"
		RichHelpFormatter.styles["argparse.groups"] = "bold bright_green"
		RichHelpFormatter.styles["argparse.metavar"] = "bold yellow"
		RichHelpFormatter.styles["argparse.syntax"] = "bold"
		RichHelpFormatter.styles["argparse.default"] = "italic dim yellow"
		
		parser = argparse.ArgumentParser(
			prog="rdkit-cli",
			description="[bold blue]Comprehensive command-line interface for RDKit cheminformatics operations[/bold blue]\n\n"
						"[dim]Process molecular data through file I/O, property calculation, similarity analysis,\n"
						"structure manipulation, 3D operations, reaction processing, visualization, and ML.[/dim]",
			formatter_class=RichHelpFormatter,
		)
		use_rich = True
	except ImportError:
		parser = argparse.ArgumentParser(
			prog="rdkit-cli",
			description="Comprehensive command-line interface for RDKit cheminformatics operations",
		)
		use_rich = False

	parser.add_argument(
		'-v', '--verbose',
		action='store_true',
		help='Enable verbose logging (INFO level)'
	)
	parser.add_argument(
		'--debug',
		action='store_true',
		help='Enable debug logging (DEBUG level with detailed internal steps)'
	)
	parser.add_argument(
		'-j', '--jobs',
		type=int,
		metavar='N',
		help='Number of parallel jobs (default: auto-detect based on CPU cores)'
	)

	# Create command groups for better organization
	if use_rich:
		subparsers = parser.add_subparsers(
			dest='command',
			help='[bold green]Available Commands (organized by category)[/bold green]',
			metavar='COMMAND'
		)
	else:
		subparsers = parser.add_subparsers(
			dest='command',
			help='Available commands',
			metavar='COMMAND'
		)

	from .commands import (
		io_ops, descriptors, fingerprints, substructure, conformers,
		fragments, reactions, optimization, visualization, database,
		ml_support, specialized, utils
	)

	# Add commands organized in logical groups
	# Commands are ordered by typical workflow: I/O -> Analysis -> Modeling -> Output
	
	# === FILE OPERATIONS ===
	io_ops.add_subparser(subparsers)  # convert, standardize, validate, split, merge, deduplicate
	
	# === MOLECULAR PROPERTIES ===  
	descriptors.add_subparser(subparsers)  # descriptors, physicochemical, admet
	
	# === SIMILARITY & FINGERPRINTS ===
	fingerprints.add_subparser(subparsers)  # fingerprints, similarity, similarity-matrix, cluster, diversity-pick
	
	# === STRUCTURE ANALYSIS ===
	substructure.add_subparser(subparsers)  # substructure-search, substructure-filter, scaffold-analysis, murcko-scaffolds, functional-groups
	fragments.add_subparser(subparsers)  # fragment, fragment-similarity, lead-optimization
	
	# === 3D GEOMETRY & CONFORMERS ===
	conformers.add_subparser(subparsers)  # conformers, align-molecules, shape-similarity, pharmacophore-screen
	optimization.add_subparser(subparsers)  # optimize, minimize, dock-prep
	
	# === CHEMICAL REACTIONS ===
	reactions.add_subparser(subparsers)  # reaction-search, reaction-apply, reaction-enumerate, retrosynthesis
	
	# === VISUALIZATION & REPORTS ===
	visualization.add_subparser(subparsers)  # visualize, grid-image, plot-descriptors, report
	
	# === DATABASE OPERATIONS ===
	database.add_subparser(subparsers)  # db-create, db-search, db-filter, db-export
	
	# === MACHINE LEARNING ===
	ml_support.add_subparser(subparsers)  # ml-features, ml-split, ml-validate
	
	# === SPECIALIZED ANALYSIS ===
	specialized.add_subparser(subparsers)  # toxicity-alerts, matched-pairs, sar-analysis, free-wilson, qsar-descriptors
	
	# === UTILITIES ===
	utils.add_subparser(subparsers)  # info, stats, sample, benchmark, config

	return parser


def main(argv: Optional[List[str]] = None) -> int:
	parser = create_parser()
	args = parser.parse_args(argv)

	if not args.command:
		parser.print_help()
		return 1

	from rdkit_cli.core.logging import setup_logging
	from rdkit_cli.core.common import GracefulExit

	logger = setup_logging(verbose=args.verbose, debug=args.debug)
	graceful_exit = GracefulExit()

	try:
		from .commands import (
			io_ops, descriptors, fingerprints, substructure, conformers,
			fragments, reactions, optimization, visualization, database,
			ml_support, specialized, utils
		)

		command_map = {
			'convert': io_ops.convert,
			'standardize': io_ops.standardize,
			'validate': io_ops.validate,
			'split': io_ops.split,
			'merge': io_ops.merge,
			'deduplicate': io_ops.deduplicate,
			'descriptors': descriptors.calculate,
			'physicochemical': descriptors.physicochemical,
			'admet': descriptors.admet,
			'fingerprints': fingerprints.generate,
			'similarity': fingerprints.similarity,
			'similarity-matrix': fingerprints.similarity_matrix,
			'cluster': fingerprints.cluster,
			'diversity-pick': fingerprints.diversity_pick,
			'substructure-search': substructure.search,
			'substructure-filter': substructure.filter_smarts,
			'scaffold-analysis': substructure.scaffold_analysis,
			'murcko-scaffolds': substructure.murcko_scaffolds,
			'functional-groups': substructure.functional_groups,
			'conformers': conformers.generate,
			'align-molecules': conformers.align,
			'shape-similarity': conformers.shape_similarity,
			'pharmacophore-screen': conformers.pharmacophore_screen,
			'fragment': fragments.fragment_molecules,
			'fragment-similarity': fragments.fragment_similarity,
			'lead-optimization': fragments.lead_optimization,
			'reaction-search': reactions.search,
			'reaction-apply': reactions.apply,
			'reaction-enumerate': reactions.enumerate,
			'retrosynthesis': reactions.retrosynthesis,
			'optimize': optimization.optimize,
			'minimize': optimization.minimize,
			'dock-prep': optimization.dock_prep,
			'visualize': visualization.visualize,
			'grid-image': visualization.grid_image,
			'plot-descriptors': visualization.plot_descriptors,
			'report': visualization.report,
			'db-create': database.create,
			'db-search': database.search,
			'db-filter': database.filter_db,
			'db-export': database.export,
			'ml-features': ml_support.features,
			'ml-split': ml_support.split_data,
			'ml-validate': ml_support.validate,
			'toxicity-alerts': specialized.toxicity_alerts,
			'matched-pairs': specialized.matched_pairs,
			'sar-analysis': specialized.sar_analysis,
			'free-wilson': specialized.free_wilson,
			'qsar-descriptors': specialized.qsar_descriptors,
			'info': utils.info,
			'stats': utils.stats,
			'sample': utils.sample,
			'benchmark': utils.benchmark,
			'config': utils.config_cmd,
		}

		if args.command in command_map:
			result = command_map[args.command](args, graceful_exit)
			return result if isinstance(result, int) else 0
		else:
			logger.error(f"Unknown command: {args.command}")
			return 1

	except KeyboardInterrupt:
		logger.info("Operation cancelled by user")
		return 130
	except Exception as e:
		if args.debug:
			logger.exception(f"Command failed: {e}")
		else:
			logger.error(f"Command failed: {e}")
		return 1


if __name__ == "__main__":
	sys.exit(main())


# rdkit_cli/commands/__init__.py
"""Command modules for RDKit CLI."""