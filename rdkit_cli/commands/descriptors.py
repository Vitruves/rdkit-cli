# rdkit_cli/commands/descriptors.py
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, EState, rdMolDescriptors
from rdkit.Chem.EState import EState_VSA
from tqdm import tqdm

from ..core.common import (
	GracefulExit, get_parallel_jobs, read_molecules, validate_input_file,
	validate_output_path, get_molecules_from_args, save_dataframe_with_format_detection
)
from ..core.logging import log_success


def add_subparser(subparsers) -> None:
	parser_descriptors = subparsers.add_parser(
		'descriptors',
		help='Calculate molecular descriptors and properties'
	)
	parser_descriptors.add_argument(
		'-i', '--input-file',
		help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
	)
	parser_descriptors.add_argument(
		'-S', '--smiles',
		help='Direct SMILES string(s) - comma-separated for multiple'
	)
	parser_descriptors.add_argument(
		'-c', '--smiles-column',
		help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
	)
	parser_descriptors.add_argument(
		'-o', '--output',
		required=True,
		help='Output CSV file with calculated descriptors'
	)
	parser_descriptors.add_argument(
		'--descriptor-set',
		choices=['all', 'lipinski', 'druglike', 'basic', 'constitutional', 'topological', 'estate'],
		default='basic',
		help='Predefined descriptor set to calculate (default: basic)'
	)
	parser_descriptors.add_argument(
		'--descriptors',
		help='Comma-separated list of specific descriptors (e.g., "MolWt,LogP,TPSA,NumHDonors")'
	)
	parser_descriptors.add_argument(
		'--include-3d',
		action='store_true',
		help='Include 3D descriptors (requires conformers)'
	)
	parser_descriptors.add_argument(
		'--skip-errors',
		action='store_true',
		help='Skip molecules that fail descriptor calculation'
	)

	parser_physicochemical = subparsers.add_parser(
		'physicochemical',
		help='Calculate physicochemical properties with drug-like filters'
	)
	parser_physicochemical.add_argument(
		'-i', '--input-file',
		help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
	)
	parser_physicochemical.add_argument(
		'-S', '--smiles',
		help='Direct SMILES string(s) - comma-separated for multiple'
	)
	parser_physicochemical.add_argument(
		'-c', '--smiles-column',
		help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
	)
	parser_physicochemical.add_argument(
		'-o', '--output',
		required=True,
		help='Output CSV file with physicochemical properties'
	)
	parser_physicochemical.add_argument(
		'--include-druglike-filters',
		action='store_true',
		help='Include Lipinski, Veber, and other drug-like filters'
	)
	parser_physicochemical.add_argument(
		'--include-qed',
		action='store_true',
		help='Include QED (Quantitative Estimate of Drug-likeness) score'
	)

	parser_admet = subparsers.add_parser(
		'admet',
		help='Calculate ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties'
	)
	parser_admet.add_argument(
		'-i', '--input-file',
		help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
	)
	parser_admet.add_argument(
		'-S', '--smiles',
		help='Direct SMILES string(s) - comma-separated for multiple'
	)
	parser_admet.add_argument(
		'-c', '--smiles-column',
		help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
	)
	parser_admet.add_argument(
		'-o', '--output',
		required=True,
		help='Output CSV file with ADMET predictions'
	)
	parser_admet.add_argument(
		'--models',
		choices=['all', 'basic', 'solubility', 'permeability', 'metabolism'],
		default='basic',
		help='ADMET model set to use (default: basic)'
	)


def get_descriptor_functions() -> Dict[str, callable]:
	"""Get mapping of descriptor names to calculation functions."""
	descriptor_funcs = {
		'MolWt': Descriptors.MolWt,
		'HeavyAtomMolWt': Descriptors.HeavyAtomMolWt,
		'ExactMolWt': Descriptors.ExactMolWt,
		'NumHeavyAtoms': Descriptors.HeavyAtomCount,
		'NumAtoms': lambda mol: mol.GetNumAtoms(),
		'NumHeteroatoms': Descriptors.NumHeteroatoms,
		'NumRotatableBonds': Descriptors.NumRotatableBonds,
		'NumHBD': Descriptors.NumHDonors,
		'NumHBA': Descriptors.NumHAcceptors,
		'NumRings': Descriptors.RingCount,
		'NumAromaticRings': Descriptors.NumAromaticRings,
		'NumSaturatedRings': Descriptors.NumSaturatedRings,
		'NumAliphaticRings': Descriptors.NumAliphaticRings,
		'FractionCsp3': Descriptors.FractionCSP3,
		'TPSA': Descriptors.TPSA,
		'LabuteASA': Descriptors.LabuteASA,
		'PEOE_VSA1': Descriptors.PEOE_VSA1,
		'PEOE_VSA2': Descriptors.PEOE_VSA2,
		'PEOE_VSA3': Descriptors.PEOE_VSA3,
		'PEOE_VSA4': Descriptors.PEOE_VSA4,
		'PEOE_VSA5': Descriptors.PEOE_VSA5,
		'PEOE_VSA6': Descriptors.PEOE_VSA6,
		'SMR_VSA1': Descriptors.SMR_VSA1,
		'SMR_VSA2': Descriptors.SMR_VSA2,
		'SMR_VSA3': Descriptors.SMR_VSA3,
		'SMR_VSA4': Descriptors.SMR_VSA4,
		'SMR_VSA5': Descriptors.SMR_VSA5,
		'SlogP_VSA1': Descriptors.SlogP_VSA1,
		'SlogP_VSA2': Descriptors.SlogP_VSA2,
		'SlogP_VSA3': Descriptors.SlogP_VSA3,
		'SlogP_VSA4': Descriptors.SlogP_VSA4,
		'SlogP_VSA5': Descriptors.SlogP_VSA5,
		'LogP': Crippen.MolLogP,
		'MR': Crippen.MolMR,
		'BalabanJ': Descriptors.BalabanJ,
		'BertzCT': Descriptors.BertzCT,
		'Chi0': Descriptors.Chi0,
		'Chi0n': Descriptors.Chi0n,
		'Chi0v': Descriptors.Chi0v,
		'Chi1': Descriptors.Chi1,
		'Chi1n': Descriptors.Chi1n,
		'Chi1v': Descriptors.Chi1v,
		'Chi2n': Descriptors.Chi2n,
		'Chi2v': Descriptors.Chi2v,
		'Chi3n': Descriptors.Chi3n,
		'Chi3v': Descriptors.Chi3v,
		'Chi4n': Descriptors.Chi4n,
		'Chi4v': Descriptors.Chi4v,
		'HallKierAlpha': Descriptors.HallKierAlpha,
		'Kappa1': Descriptors.Kappa1,
		'Kappa2': Descriptors.Kappa2,
		'Kappa3': Descriptors.Kappa3,
		'AvgIpc': Descriptors.AvgIpc,
	}

	# Dynamically add EState_VSA and VSA_EState descriptors
	for i in range(1, 12):  # EState_VSA descriptors usually go up to EState_VSA11 or more
		try:
			desc_name = f'EState_VSA{i}'
			descriptor_funcs[desc_name] = getattr(EState_VSA, desc_name)
		except AttributeError:
			pass
		try:
			desc_name = f'VSA_EState{i}'
			descriptor_funcs[desc_name] = getattr(EState_VSA, desc_name)
		except AttributeError:
			pass
	return descriptor_funcs


def get_descriptor_sets() -> Dict[str, List[str]]:
	"""Get predefined descriptor sets."""
	estate_descriptors = []
	for i in range(1, 12):
		if f'EState_VSA{i}' in get_descriptor_functions():
			estate_descriptors.append(f'EState_VSA{i}')
		if f'VSA_EState{i}' in get_descriptor_functions():
			estate_descriptors.append(f'VSA_EState{i}')

	all_descriptors = list(get_descriptor_functions().keys())

	return {
		'basic': [
			'MolWt', 'LogP', 'TPSA', 'NumHBD', 'NumHBA', 'NumRotatableBonds',
			'NumHeavyAtoms', 'NumRings', 'NumAromaticRings', 'FractionCsp3'
		],
		'lipinski': [
			'MolWt', 'LogP', 'NumHBD', 'NumHBA', 'NumRotatableBonds'
		],
		'druglike': [
			'MolWt', 'LogP', 'TPSA', 'NumHBD', 'NumHBA', 'NumRotatableBonds',
			'NumHeavyAtoms', 'NumRings', 'NumAromaticRings', 'FractionCsp3',
			'LabuteASA', 'BertzCT'
		],
		'constitutional': [
			'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumHeavyAtoms',
			'NumAtoms', 'NumHeteroatoms', 'FractionCsp3'
		],
		'topological': [
			'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n',
			'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
			'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3'
		],
		'estate': estate_descriptors,
		'all': all_descriptors,
	}


def calculate(args, graceful_exit: GracefulExit) -> int:
	logger = logging.getLogger("rdkit_cli")
	
	try:
		output_path = validate_output_path(args.output)
		
		molecules = get_molecules_from_args(args)
		logger.info(f"Loaded {len(molecules)} molecules")
		
		descriptor_funcs = get_descriptor_functions()
		all_calculated_descriptors = []
		descriptors_to_calculate_names = []

		if args.descriptor_set == 'all':
			from rdkit.Chem import rdMolDescriptors
			for mol in molecules:
				if mol is None: continue
				desc_values = rdMolDescriptors.CalcMolDescriptors(mol)
				all_calculated_descriptors.append(desc_values)
			if all_calculated_descriptors:
				descriptors_to_calculate_names = list(all_calculated_descriptors[0].keys())
		elif args.descriptors:
			descriptors_to_calculate_names = [d.strip() for d in args.descriptors.split(',')]
		else:
			descriptor_sets = get_descriptor_sets()
			if args.descriptor_set in descriptor_sets:
				descriptors_to_calculate_names = descriptor_sets[args.descriptor_set]
			else:
				graceful_exit.exit_error(f"Unknown descriptor set: {args.descriptor_set}")

		logger.info(f"Calculating {len(descriptors_to_calculate_names)} descriptors")
		logger.debug(f"Descriptors: {', '.join(descriptors_to_calculate_names)}")

		data = []

		with tqdm(total=len(molecules), desc="Calculating descriptors", ncols=80, colour='blue') as pbar:
			for i, mol in enumerate(molecules):
				if graceful_exit.exit_now:
					break

				try:
					if mol is None:
						if not args.skip_errors:
							logger.error(f"Invalid molecule at index {i}")
							return 1
						continue

					row = {}

					mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
					row['ID'] = mol_id
					row['SMILES'] = Chem.MolToSmiles(mol)

					if args.descriptor_set == 'all':
						desc_values = rdMolDescriptors.CalcMolDescriptors(mol)
						for desc_name in descriptors_to_calculate_names:
							row[desc_name] = desc_values.get(desc_name, float('nan'))
					else:
						for desc_name in descriptors_to_calculate_names:
							try:
								if desc_name in descriptor_funcs:
									value = descriptor_funcs[desc_name](mol)
									row[desc_name] = value
								else:
									logger.warning(f"Unknown descriptor: {desc_name}")
							except Exception as e:
								if args.skip_errors:
									row[desc_name] = None
									logger.debug(f"Failed to calculate {desc_name} for molecule {mol_id}: {e}")
								else:
									raise

					data.append(row)

				except Exception as e:
					if args.skip_errors:
						logger.debug(f"Skipped molecule {i}: {e}")
						continue
					else:
						logger.error(f"Failed to process molecule {i}: {e}")
						return 1

				pbar.update(1)
		
		if graceful_exit.exit_now:
			return 130
		
		if not data:
			logger.error("No valid molecules processed")
			return 1
		
		df = pd.DataFrame(data)
		save_dataframe_with_format_detection(df, output_path)
		
		log_success(f"Calculated descriptors for {len(data)} molecules, saved to {output_path}")
		return 0
		
	except Exception as e:
		logger.error(f"Descriptor calculation failed: {e}")
		return 1


def physicochemical(args, graceful_exit: GracefulExit) -> int:
	logger = logging.getLogger("rdkit_cli")
	
	try:
		output_path = validate_output_path(args.output)
		
		molecules = get_molecules_from_args(args)
		logger.info(f"Loaded {len(molecules)} molecules")
		
		data = []
		
		with tqdm(total=len(molecules), desc="Calculating properties", ncols=80, colour='green') as pbar:
			for i, mol in enumerate(molecules):
				if graceful_exit.exit_now:
					break
				
				try:
					if mol is None:
						continue
					
					mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
					
					row = {
						'ID': mol_id,
						'SMILES': Chem.MolToSmiles(mol),
						'MolWt': Descriptors.MolWt(mol),
						'LogP': Crippen.MolLogP(mol),
						'TPSA': Descriptors.TPSA(mol),
						'NumHBD': Descriptors.NumHDonors(mol),
						'NumHBA': Descriptors.NumHAcceptors(mol),
						'NumRotBonds': Descriptors.NumRotatableBonds(mol),
						'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),
						'NumRings': Descriptors.RingCount(mol),
						'NumAromaticRings': Descriptors.NumAromaticRings(mol),
						'FractionCsp3': Descriptors.FractionCSP3(mol),
						'LabuteASA': Descriptors.LabuteASA(mol),
					}
					
					if args.include_druglike_filters:
						row['Lipinski_MW'] = row['MolWt'] <= 500
						row['Lipinski_LogP'] = row['LogP'] <= 5
						row['Lipinski_HBD'] = row['NumHBD'] <= 5
						row['Lipinski_HBA'] = row['NumHBA'] <= 10
						row['Lipinski_Pass'] = all([
							row['Lipinski_MW'], row['Lipinski_LogP'],
							row['Lipinski_HBD'], row['Lipinski_HBA']
						])
						
						row['Veber_RotBonds'] = row['NumRotBonds'] <= 10
						row['Veber_TPSA'] = row['TPSA'] <= 140
						row['Veber_Pass'] = row['Veber_RotBonds'] and row['Veber_TPSA']
						
						row['Egan_LogP'] = -1 <= row['LogP'] <= 5.88
						row['Egan_TPSA'] = row['TPSA'] <= 131.6
						row['Egan_Pass'] = row['Egan_LogP'] and row['Egan_TPSA']
					
					if args.include_qed:
						row['QED'] = QED.qed(mol)
					
					data.append(row)
				
				except Exception as e:
					logger.debug(f"Failed to process molecule {i}: {e}")
				
				pbar.update(1)
		
		if graceful_exit.exit_now:
			return 130
		
		if not data:
			logger.error("No valid molecules processed")
			return 1
		
		df = pd.DataFrame(data)
		save_dataframe_with_format_detection(df, output_path)
		
		log_success(f"Calculated physicochemical properties for {len(data)} molecules, saved to {output_path}")
		return 0
		
	except Exception as e:
		logger.error(f"Physicochemical calculation failed: {e}")
		return 1


def admet(args, graceful_exit: GracefulExit) -> int:
	logger = logging.getLogger("rdkit_cli")
	
	try:
		output_path = validate_output_path(args.output)
		
		molecules = get_molecules_from_args(args)
		logger.info(f"Loaded {len(molecules)} molecules")
		
		data = []
		
		with tqdm(total=len(molecules), desc="Calculating ADMET", ncols=80, colour='cyan') as pbar:
			for i, mol in enumerate(molecules):
				if graceful_exit.exit_now:
					break
				
				try:
					if mol is None:
						continue
					
					mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
					
					row = {
						'ID': mol_id,
						'SMILES': Chem.MolToSmiles(mol),
					}
					
					if args.models in ['all', 'basic', 'solubility']:
						row['LogS_Ali'] = _calculate_logs_ali(mol)
						row['LogS_ESOL'] = _calculate_logs_esol(mol)
					
					if args.models in ['all', 'basic', 'permeability']:
						row['BBB_Permeability'] = _calculate_bbb_permeability(mol)
						row['Caco2_Permeability'] = _calculate_caco2_permeability(mol)
					
					if args.models in ['all', 'metabolism']:
						row['CYP2D6_Substrate'] = _calculate_cyp2d6_substrate(mol)
						row['CYP3A4_Substrate'] = _calculate_cyp3a4_substrate(mol)
					
					if args.models == 'all':
						row['hERG_Inhibition'] = _calculate_herg_inhibition(mol)
						row['AMES_Mutagenicity'] = _calculate_ames_mutagenicity(mol)
						row['Hepatotoxicity'] = _calculate_hepatotoxicity(mol)
					
					data.append(row)
				
				except Exception as e:
					logger.debug(f"Failed to process molecule {i}: {e}")
				
				pbar.update(1)
		
		if graceful_exit.exit_now:
			return 130
		
		if not data:
			logger.error("No valid molecules processed")
			return 1
		
		df = pd.DataFrame(data)
		save_dataframe_with_format_detection(df, output_path)
		
		log_success(f"Calculated ADMET properties for {len(data)} molecules, saved to {output_path}")
		return 0
		
	except Exception as e:
		logger.error(f"ADMET calculation failed: {e}")
		return 1


def _calculate_logs_ali(mol) -> float:
	"""Calculate LogS using Ali method (simplified estimation)."""
	logp = Crippen.MolLogP(mol)
	mw = Descriptors.MolWt(mol)
	num_atoms = mol.GetNumAtoms()
	
	logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * num_atoms
	return round(logs, 3)


def _calculate_logs_esol(mol) -> float:
	"""Calculate LogS using ESOL method (simplified estimation)."""
	logp = Crippen.MolLogP(mol)
	mw = Descriptors.MolWt(mol)
	rb = Descriptors.NumRotatableBonds(mol)
	ap = Descriptors.NumAromaticRings(mol)
	
	logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap
	return round(logs, 3)


def _calculate_bbb_permeability(mol) -> float:
	"""Calculate BBB permeability (simplified estimation)."""
	logp = Crippen.MolLogP(mol)
	tpsa = Descriptors.TPSA(mol)
	
	bbb_score = -0.0148 * tpsa + 0.152 * logp + 0.139
	return round(max(0, min(1, bbb_score)), 3)


def _calculate_caco2_permeability(mol) -> float:
	"""Calculate Caco-2 permeability (simplified estimation)."""
	logp = Crippen.MolLogP(mol)
	tpsa = Descriptors.TPSA(mol)
	
	caco2 = 1.35 - 0.01 * tpsa + 0.4 * logp
	return round(caco2, 3)


def _calculate_cyp2d6_substrate(mol) -> float:
	"""Calculate CYP2D6 substrate probability (simplified estimation)."""
	logp = Crippen.MolLogP(mol)
	mw = Descriptors.MolWt(mol)
	
	score = 1 / (1 + pow(2.718, -(0.01 * mw + 0.3 * logp - 5)))
	return round(score, 3)


def _calculate_cyp3a4_substrate(mol) -> float:
	"""Calculate CYP3A4 substrate probability (simplified estimation)."""
	logp = Crippen.MolLogP(mol)
	mw = Descriptors.MolWt(mol)
	
	score = 1 / (1 + pow(2.718, -(0.005 * mw + 0.4 * logp - 4)))
	return round(score, 3)


def _calculate_herg_inhibition(mol) -> float:
	"""Calculate hERG inhibition probability (simplified estimation)."""
	logp = Crippen.MolLogP(mol)
	mw = Descriptors.MolWt(mol)
	
	score = 1 / (1 + pow(2.718, -(0.02 * mw + 0.5 * logp - 8)))
	return round(score, 3)


def _calculate_ames_mutagenicity(mol) -> float:
	"""Calculate AMES mutagenicity probability (simplified estimation)."""
	aromatic_rings = Descriptors.NumAromaticRings(mol)
	nitro_groups = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[N+](=O)[O-]')))
	
	score = min(1.0, 0.1 + 0.15 * aromatic_rings + 0.3 * nitro_groups)
	return round(score, 3)


def _calculate_hepatotoxicity(mol) -> float:
	"""Calculate hepatotoxicity probability (simplified estimation)."""
	logp = Crippen.MolLogP(mol)
	mw = Descriptors.MolWt(mol)
	
	score = 1 / (1 + pow(2.718, -(0.01 * mw + 0.2 * logp - 6)))
	return round(score, 3)