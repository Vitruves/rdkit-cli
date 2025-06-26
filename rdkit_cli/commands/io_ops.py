# rdkit_cli/commands/io_ops.py
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolHash
from tqdm import tqdm

from ..core.common import (
	GracefulExit, get_parallel_jobs, read_molecules, validate_input_file,
	validate_output_path, write_molecules
)
from ..core.logging import log_success


def add_subparser(subparsers) -> None:
	parser_convert = subparsers.add_parser(
		'convert',
		help='Convert molecular file formats (SDF, SMILES, CSV, etc.)'
	)
	parser_convert.add_argument(
		'-i', '--input-file',
		help='Input file path (smi, sdf, mol, csv, parquet)'
	)
	parser_convert.add_argument(
		'-S', '--smiles',
		help='Direct SMILES string(s) - comma-separated for multiple'
	)
	parser_convert.add_argument(
		'-c', '--smiles-column',
		help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
	)
	parser_convert.add_argument(
		'-o', '--output',
		required=True,
		help='Output file path'
	)
	parser_convert.add_argument(
		'--format',
		choices=['smiles', 'sdf', 'csv', 'parquet', 'mol'],
		help='Force output format (default: auto-detect from extension)'
	)
	parser_convert.add_argument(
		'--no-header',
		action='store_true',
		help='Skip header in CSV output'
	)
	parser_convert.add_argument(
		'--delimiter',
		default=',',
		help='CSV delimiter (default: comma)'
	)

	parser_standardize = subparsers.add_parser(
		'standardize',
		help='Standardize molecular structures (remove salts, neutralize, normalize)'
	)
	parser_standardize.add_argument(
		'-i', '--input-file',
		help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
	)
	parser_standardize.add_argument(
		'-S', '--smiles',
		help='Direct SMILES string(s) - comma-separated for multiple'
	)
	parser_standardize.add_argument(
		'-c', '--smiles-column',
		help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
	)
	parser_standardize.add_argument(
		'-o', '--output',
		required=True,
		help='Output file for standardized molecules'
	)
	parser_standardize.add_argument(
		'--remove-salts',
		action='store_true',
		help='Remove salt fragments from molecules'
	)
	parser_standardize.add_argument(
		'--neutralize',
		action='store_true',
		help='Neutralize charged molecules'
	)
	parser_standardize.add_argument(
		'--normalize',
		action='store_true',
		help='Apply normalization transforms'
	)

	parser_validate = subparsers.add_parser(
		'validate',
		help='Validate molecular structures and filter invalid molecules'
	)
	parser_validate.add_argument(
		'-i', '--input-file',
		help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
	)
	parser_validate.add_argument(
		'-S', '--smiles',
		help='Direct SMILES string(s) - comma-separated for multiple'
	)
	parser_validate.add_argument(
		'-c', '--smiles-column',
		help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
	)
	parser_validate.add_argument(
		'-o', '--output',
		required=True,
		help='Output file for valid molecules'
	)
	parser_validate.add_argument(
		'--log-errors',
		help='Log validation errors to file'
	)
	parser_validate.add_argument(
		'--strict',
		action='store_true',
		help='Use strict validation (reject molecules with warnings)'
	)

	parser_split = subparsers.add_parser(
		'split',
		help='Split large molecular datasets into smaller chunks'
	)
	parser_split.add_argument(
		'-i', '--input-file',
		required=True,
		help='Input molecular file'
	)
	parser_split.add_argument(
		'--output-dir',
		required=True,
		help='Output directory for split files'
	)
	parser_split.add_argument(
		'--chunk-size',
		type=int,
		default=1000,
		help='Number of molecules per chunk (default: 1000)'
	)
	parser_split.add_argument(
		'--prefix',
		default='chunk',
		help='Prefix for output files (default: chunk)'
	)

	parser_merge = subparsers.add_parser(
		'merge',
		help='Merge multiple molecular files into single file'
	)
	parser_merge.add_argument(
		'-i', '--input-files',
		nargs='+',
		required=True,
		help='Input molecular files to merge'
	)
	parser_merge.add_argument(
		'-o', '--output',
		required=True,
		help='Output merged file'
	)
	parser_merge.add_argument(
		'--validate',
		action='store_true',
		help='Validate molecules during merge'
	)

	parser_deduplicate = subparsers.add_parser(
		'deduplicate',
		help='Remove duplicate molecules from dataset'
	)
	parser_deduplicate.add_argument(
		'-i', '--input-file',
		required=True,
		help='Input molecular file'
	)
	parser_deduplicate.add_argument(
		'-o', '--output',
		required=True,
		help='Output file with unique molecules'
	)
	parser_deduplicate.add_argument(
		'--method',
		choices=['inchi-key', 'canonical-smiles', 'molecular-hash'],
		default='inchi-key',
		help='Deduplication method (default: inchi-key)'
	)
	parser_deduplicate.add_argument(
		'--keep-first',
		action='store_true',
		help='Keep first occurrence of duplicates (default: keep last)'
	)


def convert(args, graceful_exit: GracefulExit) -> int:
	logger = logging.getLogger("rdkit_cli")
	
	try:
		input_path = validate_input_file(args.input_file)
		output_path = validate_output_path(args.output)
		
		logger.info(f"Converting {input_path} to {output_path}")
		
		molecules = read_molecules(input_path)
		logger.info(f"Loaded {len(molecules)} molecules")
		
		if graceful_exit.exit_now:
			return 130
		
		write_molecules(molecules, output_path, format_hint=args.format)
		
		log_success(f"Converted {len(molecules)} molecules to {output_path}")
		return 0
		
	except Exception as e:
		logger.error(f"Conversion failed: {e}")
		return 1


def standardize(args, graceful_exit: GracefulExit) -> int:
	logger = logging.getLogger("rdkit_cli")
	
	try:
		from rdkit.Chem import SaltRemover, rdMolStandardize
		
		input_path = validate_input_file(args.input_file)
		output_path = validate_output_path(args.output)
		
		logger.info(f"Standardizing molecules from {input_path}")
		
		molecules = read_molecules(input_path)
		logger.info(f"Loaded {len(molecules)} molecules")
		
		standardized = []
		
		if args.remove_salts:
			salt_remover = SaltRemover.SaltRemover()
		
		normalizer = rdMolStandardize.Normalizer()
		
		with tqdm(total=len(molecules), desc="Standardizing", ncols=80, colour='blue') as pbar:
			for mol in molecules:
				if graceful_exit.exit_now:
					break
				
				try:
					standardized_mol = mol
					
					if args.remove_salts:
						standardized_mol = salt_remover.StripMol(standardized_mol)
					
					if args.neutralize:
						uncharger = rdMolStandardize.Uncharger()
						standardized_mol = uncharger.uncharge(standardized_mol)
					
					if args.normalize:
						standardized_mol = normalizer.normalize(standardized_mol)
					
					if standardized_mol is not None:
						standardized.append(standardized_mol)
				
				except Exception as e:
					logger.debug(f"Failed to standardize molecule: {e}")
				
				pbar.update(1)
		
		if graceful_exit.exit_now:
			return 130
		
		write_molecules(standardized, output_path)
		
		log_success(f"Standardized {len(standardized)} molecules to {output_path}")
		return 0
		
	except Exception as e:
		logger.error(f"Standardization failed: {e}")
		return 1


def validate(args, graceful_exit: GracefulExit) -> int:
	logger = logging.getLogger("rdkit_cli")
	
	try:
		input_path = validate_input_file(args.input_file)
		output_path = validate_output_path(args.output)
		
		logger.info(f"Validating molecules from {input_path}")
		
		molecules = read_molecules(input_path)
		logger.info(f"Loaded {len(molecules)} molecules")
		
		valid_molecules = []
		errors = []
		
		with tqdm(total=len(molecules), desc="Validating", ncols=80, colour='green') as pbar:
			for i, mol in enumerate(molecules):
				if graceful_exit.exit_now:
					break
				
				try:
					if mol is None:
						errors.append(f"Molecule {i+1}: Invalid structure")
						continue
					
					if mol.GetNumAtoms() == 0:
						errors.append(f"Molecule {i+1}: Empty molecule")
						continue
					
					smiles = Chem.MolToSmiles(mol)
					if not smiles:
						errors.append(f"Molecule {i+1}: Cannot generate SMILES")
						continue
					
					Chem.SanitizeMol(mol)
					valid_molecules.append(mol)
				
				except Exception as e:
					errors.append(f"Molecule {i+1}: {str(e)}")
				
				pbar.update(1)
		
		if graceful_exit.exit_now:
			return 130
		
		write_molecules(valid_molecules, output_path)
		
		if args.log_errors and errors:
			error_path = Path(args.log_errors)
			with open(error_path, 'w') as f:
				for error in errors:
					f.write(f"{error}\n")
			logger.info(f"Logged {len(errors)} validation errors to {error_path}")
		
		log_success(f"Validated {len(valid_molecules)}/{len(molecules)} molecules to {output_path}")
		return 0
		
	except Exception as e:
		logger.error(f"Validation failed: {e}")
		return 1


def split(args, graceful_exit: GracefulExit) -> int:
	logger = logging.getLogger("rdkit_cli")
	
	try:
		input_path = validate_input_file(args.input_file)
		output_dir = Path(args.output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
		
		logger.info(f"Splitting {input_path} into chunks of {args.chunk_size}")
		
		molecules = read_molecules(input_path)
		logger.info(f"Loaded {len(molecules)} molecules")
		
		num_chunks = (len(molecules) + args.chunk_size - 1) // args.chunk_size
		
		with tqdm(total=num_chunks, desc="Writing chunks", ncols=80, colour='cyan') as pbar:
			for i in range(num_chunks):
				if graceful_exit.exit_now:
					break
				
				start_idx = i * args.chunk_size
				end_idx = min((i + 1) * args.chunk_size, len(molecules))
				chunk = molecules[start_idx:end_idx]
				
				chunk_file = output_dir / f"{args.prefix}_{i+1:03d}.sdf"
				write_molecules(chunk, chunk_file)
				
				pbar.update(1)
		
		if graceful_exit.exit_now:
			return 130
		
		log_success(f"Split {len(molecules)} molecules into {num_chunks} chunks in {output_dir}")
		return 0
		
	except Exception as e:
		logger.error(f"Splitting failed: {e}")
		return 1


def merge(args, graceful_exit: GracefulExit) -> int:
	logger = logging.getLogger("rdkit_cli")
	
	try:
		output_path = validate_output_path(args.output)
		
		logger.info(f"Merging {len(args.input_files)} files")
		
		all_molecules = []
		
		with tqdm(total=len(args.input_files), desc="Reading files", ncols=80, colour='magenta') as pbar:
			for file_path in args.input_files:
				if graceful_exit.exit_now:
					break
				
				input_path = validate_input_file(file_path)
				molecules = read_molecules(input_path)
				
				if args.validate:
					valid_molecules = []
					for mol in molecules:
						try:
							if mol is not None:
								Chem.SanitizeMol(mol)
								valid_molecules.append(mol)
						except:
							pass
					all_molecules.extend(valid_molecules)
				else:
					all_molecules.extend(molecules)
				
				pbar.update(1)
		
		if graceful_exit.exit_now:
			return 130
		
		write_molecules(all_molecules, output_path)
		
		log_success(f"Merged {len(all_molecules)} molecules to {output_path}")
		return 0
		
	except Exception as e:
		logger.error(f"Merging failed: {e}")
		return 1


def deduplicate(args, graceful_exit: GracefulExit) -> int:
	logger = logging.getLogger("rdkit_cli")
	
	try:
		input_path = validate_input_file(args.input_file)
		output_path = validate_output_path(args.output)
		
		logger.info(f"Deduplicating molecules from {input_path} using {args.method}")
		
		molecules = read_molecules(input_path)
		logger.info(f"Loaded {len(molecules)} molecules")
		
		seen_hashes: Set[str] = set()
		unique_molecules = []
		
		with tqdm(total=len(molecules), desc="Deduplicating", ncols=80, colour='yellow') as pbar:
			for mol in molecules:
				if graceful_exit.exit_now:
					break
				
				try:
					if mol is None:
						continue
					
					if args.method == 'inchi-key':
						mol_hash = Chem.InchiToInchiKey(Chem.MolToInchi(mol))
					elif args.method == 'canonical-smiles':
						mol_hash = Chem.MolToSmiles(mol, canonical=True)
					elif args.method == 'molecular-hash':
						mol_hash = rdMolHash.MolHash(mol, rdMolHash.HashFunction.HadamardMD5)
					else:
						raise ValueError(f"Unknown deduplication method: {args.method}")
					
					if mol_hash not in seen_hashes:
						seen_hashes.add(mol_hash)
						unique_molecules.append(mol)
					elif not args.keep_first:
						for i, existing_mol in enumerate(unique_molecules):
							existing_hash = None
							if args.method == 'inchi-key':
								existing_hash = Chem.InchiToInchiKey(Chem.MolToInchi(existing_mol))
							elif args.method == 'canonical-smiles':
								existing_hash = Chem.MolToSmiles(existing_mol, canonical=True)
							elif args.method == 'molecular-hash':
								existing_hash = rdMolHash.MolHash(existing_mol, rdMolHash.HashFunction.HadamardMD5)
							
							if existing_hash == mol_hash:
								unique_molecules[i] = mol
								break
				
				except Exception as e:
					logger.debug(f"Failed to process molecule: {e}")
				
				pbar.update(1)
		
		if graceful_exit.exit_now:
			return 130
		
		write_molecules(unique_molecules, output_path)
		
		duplicates_removed = len(molecules) - len(unique_molecules)
		log_success(f"Removed {duplicates_removed} duplicates, {len(unique_molecules)} unique molecules to {output_path}")
		return 0
		
	except Exception as e:
		logger.error(f"Deduplication failed: {e}")
		return 1