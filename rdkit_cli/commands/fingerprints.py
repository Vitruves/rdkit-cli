# rdkit_cli/commands/fingerprints.py
import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from ..core.common import (
	GracefulExit, get_parallel_jobs, read_molecules, validate_input_file,
	validate_output_path, get_molecules_from_args, save_dataframe_with_format_detection
)
from ..core.logging import log_success


def add_subparser(subparsers) -> None:
	parser_fingerprints = subparsers.add_parser(
		'fingerprints',
		help='Generate molecular fingerprints (Morgan, RDKit, MACCS, etc.)'
	)
	parser_fingerprints.add_argument(
		'-i', '--input-file',
		help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
	)
	parser_fingerprints.add_argument(
		'-S', '--smiles',
		help='Direct SMILES string(s) - comma-separated for multiple'
	)
	parser_fingerprints.add_argument(
		'-c', '--smiles-column',
		help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
	)
	parser_fingerprints.add_argument(
		'-o', '--output',
		required=True,
		help='Output file for fingerprints (pickle format recommended)'
	)
	parser_fingerprints.add_argument(
		'--fp-type',
		choices=['morgan', 'rdkit', 'maccs', 'avalon', 'topological', 'atompair'],
		default='morgan',
		help='Fingerprint type to generate (default: morgan)'
	)
	parser_fingerprints.add_argument(
		'--radius',
		type=int,
		default=2,
		help='Radius for Morgan fingerprints (default: 2)'
	)
	parser_fingerprints.add_argument(
		'--n-bits',
		type=int,
		default=2048,
		help='Number of bits in fingerprint (default: 2048)'
	)
	parser_fingerprints.add_argument(
		'--use-features',
		action='store_true',
		help='Use feature-based fingerprints instead of circular'
	)
	parser_fingerprints.add_argument(
		'--use-chirality',
		action='store_true',
		help='Include chirality information in fingerprints'
	)

	parser_similarity = subparsers.add_parser(
		'similarity',
		help='Calculate molecular similarity using fingerprints'
	)
	parser_similarity.add_argument(
		'-i', '--query',
		help='Query molecule file (single molecule)'
	)
	parser_similarity.add_argument(
		'-S', '--query-smiles',
		help='Direct query SMILES string'
	)
	parser_similarity.add_argument(
		'-c', '--smiles-column',
		help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
	)
	parser_similarity.add_argument(
		'--database',
		required=True,
		help='Database of molecules to search against'
	)
	parser_similarity.add_argument(
		'-o', '--output',
		required=True,
		help='Output CSV file with similarity scores'
	)
	parser_similarity.add_argument(
		'--threshold',
		type=float,
		default=0.7,
		help='Minimum similarity threshold (default: 0.7)'
	)
	parser_similarity.add_argument(
		'--metric',
		choices=['tanimoto', 'dice', 'cosine', 'sokal', 'russel'],
		default='tanimoto',
		help='Similarity metric to use (default: tanimoto)'
	)
	parser_similarity.add_argument(
		'--fp-type',
		choices=['morgan', 'rdkit', 'maccs', 'avalon'],
		default='morgan',
		help='Fingerprint type (default: morgan)'
	)

	parser_similarity_matrix = subparsers.add_parser(
		'similarity-matrix',
		help='Calculate pairwise similarity matrix for molecule set'
	)
	parser_similarity_matrix.add_argument(
		'-i', '--input-file',
		help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
	)
	parser_similarity_matrix.add_argument(
		'-S', '--smiles',
		help='Direct SMILES string(s) - comma-separated for multiple'
	)
	parser_similarity_matrix.add_argument(
		'-c', '--smiles-column',
		help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
	)
	parser_similarity_matrix.add_argument(
		'-o', '--output',
		required=True,
		help='Output CSV file with similarity matrix'
	)
	parser_similarity_matrix.add_argument(
		'--fp-type',
		choices=['morgan', 'rdkit', 'maccs', 'avalon'],
		default='morgan',
		help='Fingerprint type (default: morgan)'
	)
	parser_similarity_matrix.add_argument(
		'--metric',
		choices=['tanimoto', 'dice', 'cosine'],
		default='tanimoto',
		help='Similarity metric (default: tanimoto)'
	)

	parser_cluster = subparsers.add_parser(
		'cluster',
		help='Cluster molecules based on fingerprint similarity'
	)
	parser_cluster.add_argument(
		'-i', '--input-file',
		help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
	)
	parser_cluster.add_argument(
		'-S', '--smiles',
		help='Direct SMILES string(s) - comma-separated for multiple'
	)
	parser_cluster.add_argument(
		'-c', '--smiles-column',
		help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
	)
	parser_cluster.add_argument(
		'-o', '--output',
		required=True,
		help='Output CSV file with cluster assignments'
	)
	parser_cluster.add_argument(
		'--method',
		choices=['butina', 'hierarchical', 'kmeans'],
		default='butina',
		help='Clustering algorithm (default: butina)'
	)
	parser_cluster.add_argument(
		'--threshold',
		type=float,
		default=0.8,
		help='Similarity threshold for clustering (default: 0.8)'
	)
	parser_cluster.add_argument(
		'--fp-type',
		choices=['morgan', 'rdkit', 'maccs'],
		default='morgan',
		help='Fingerprint type (default: morgan)'
	)

	parser_diversity_pick = subparsers.add_parser(
		'diversity-pick',
		help='Pick diverse subset of molecules using fingerprint-based selection'
	)
	parser_diversity_pick.add_argument(
		'-i', '--input-file',
		help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
	)
	parser_diversity_pick.add_argument(
		'-S', '--smiles',
		help='Direct SMILES string(s) - comma-separated for multiple'
	)
	parser_diversity_pick.add_argument(
		'-c', '--smiles-column',
		help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
	)
	parser_diversity_pick.add_argument(
		'-o', '--output',
		required=True,
		help='Output file with diverse molecule subset'
	)
	parser_diversity_pick.add_argument(
		'--method',
		choices=['maxmin', 'sphere-exclusion', 'leader'],
		default='maxmin',
		help='Diversity picking algorithm (default: maxmin)'
	)
	parser_diversity_pick.add_argument(
		'--count',
		type=int,
		default=100,
		help='Number of diverse molecules to select (default: 100)'
	)
	parser_diversity_pick.add_argument(
		'--fp-type',
		choices=['morgan', 'rdkit', 'maccs'],
		default='morgan',
		help='Fingerprint type (default: morgan)'
	)


def generate_fingerprint(mol: Chem.Mol, fp_type: str, radius: int = 2, 
						n_bits: int = 2048, use_features: bool = False,
						use_chirality: bool = False) -> Optional[DataStructs.ExplicitBitVect]:
	"""Generate fingerprint for a molecule."""
	try:
		if fp_type == 'morgan':
			if use_features:
				return AllChem.GetMorganFingerprintAsBitVect(
					mol, radius, nBits=n_bits, useFeatures=True, useChirality=use_chirality
				)
			else:
				return AllChem.GetMorganFingerprintAsBitVect(
					mol, radius, nBits=n_bits, useChirality=use_chirality
				)
		elif fp_type == 'rdkit':
			return Chem.RDKFingerprint(mol, fpSize=n_bits)
		elif fp_type == 'maccs':
			return rdMolDescriptors.GetMACCSKeysFingerprint(mol)
		elif fp_type == 'avalon':
			return rdMolDescriptors.GetAvalonFP(mol, nBits=n_bits)
		elif fp_type == 'topological':
			return Chem.RDKFingerprint(mol, fpSize=n_bits)
		elif fp_type == 'atompair':
			return rdMolDescriptors.GetAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
		else:
			raise ValueError(f"Unsupported fingerprint type: {fp_type}")
	except Exception:
		return None


def calculate_similarity(fp1: DataStructs.ExplicitBitVect, 
						fp2: DataStructs.ExplicitBitVect,
						metric: str = 'tanimoto') -> float:
	"""Calculate similarity between two fingerprints."""
	if metric == 'tanimoto':
		return DataStructs.TanimotoSimilarity(fp1, fp2)
	elif metric == 'dice':
		return DataStructs.DiceSimilarity(fp1, fp2)
	elif metric == 'cosine':
		return DataStructs.CosineSimilarity(fp1, fp2)
	elif metric == 'sokal':
		return DataStructs.SokalSimilarity(fp1, fp2)
	elif metric == 'russel':
		return DataStructs.RusselSimilarity(fp1, fp2)
	else:
		raise ValueError(f"Unsupported similarity metric: {metric}")


def generate(args, graceful_exit: GracefulExit) -> int:
	logger = logging.getLogger("rdkit_cli")
	
	try:
		output_path = validate_output_path(args.output)
		
		molecules = get_molecules_from_args(args)
		logger.info(f"Generating {args.fp_type} fingerprints for {len(molecules)} molecules")
		
		fingerprints = []
		mol_ids = []
		
		with tqdm(total=len(molecules), desc="Generating fingerprints", ncols=80, colour='blue') as pbar:
			for i, mol in enumerate(molecules):
				if graceful_exit.exit_now:
					break
				
				try:
					if mol is None:
						continue
					
					fp = generate_fingerprint(
						mol, args.fp_type, args.radius, args.n_bits,
						args.use_features, args.use_chirality
					)
					
					if fp is not None:
						fingerprints.append(fp)
						mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
						mol_ids.append(mol_id)
				
				except Exception as e:
					logger.debug(f"Failed to generate fingerprint for molecule {i}: {e}")
				
				pbar.update(1)
		
		if graceful_exit.exit_now:
			return 130
		
		if not fingerprints:
			logger.error("No valid fingerprints generated")
			return 1
		
		with open(output_path, 'wb') as f:
			pickle.dump({
				'fingerprints': fingerprints,
				'mol_ids': mol_ids,
				'fp_type': args.fp_type,
				'radius': args.radius,
				'n_bits': args.n_bits,
				'use_features': args.use_features,
				'use_chirality': args.use_chirality
			}, f)
		
		log_success(f"Generated {len(fingerprints)} fingerprints, saved to {output_path}")
		return 0
		
	except Exception as e:
		logger.error(f"Fingerprint generation failed: {e}")
		return 1


def similarity(args, graceful_exit: GracefulExit) -> int:
	logger = logging.getLogger("rdkit_cli")
	
	try:
		query_path = validate_input_file(args.query)
		database_path = validate_input_file(args.database)
		output_path = validate_output_path(args.output)
		
		logger.info(f"Searching for similar molecules to {query_path} in {database_path}")
		
		query_molecules = read_molecules(query_path)
		if not query_molecules:
			logger.error("No valid query molecule found")
			return 1
		
		query_mol = query_molecules[0]
		query_fp = generate_fingerprint(query_mol, args.fp_type)
		
		if query_fp is None:
			logger.error("Failed to generate fingerprint for query molecule")
			return 1
		
		database_molecules = read_molecules(database_path)
		logger.info(f"Loaded {len(database_molecules)} database molecules")
		
		results = []
		
		with tqdm(total=len(database_molecules), desc="Calculating similarity", ncols=80, colour='green') as pbar:
			for i, mol in enumerate(database_molecules):
				if graceful_exit.exit_now:
					break
				
				try:
					if mol is None:
						continue
					
					db_fp = generate_fingerprint(mol, args.fp_type)
					if db_fp is None:
						continue
					
					sim_score = calculate_similarity(query_fp, db_fp, args.metric)
					
					if sim_score >= args.threshold:
						mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
						results.append({
							'ID': mol_id,
							'SMILES': Chem.MolToSmiles(mol),
							'Similarity': round(sim_score, 4)
						})
				
				except Exception as e:
					logger.debug(f"Failed to process molecule {i}: {e}")
				
				pbar.update(1)
		
		if graceful_exit.exit_now:
			return 130
		
		if not results:
			logger.warning(f"No molecules found above similarity threshold {args.threshold}")
			return 0
		
		results.sort(key=lambda x: x['Similarity'], reverse=True)
		
		df = pd.DataFrame(results)
		df.to_csv(output_path, index=False)
		
		log_success(f"Found {len(results)} similar molecules, saved to {output_path}")
		return 0
		
	except Exception as e:
		logger.error(f"Similarity search failed: {e}")
		return 1


def similarity_matrix(args, graceful_exit: GracefulExit) -> int:
	logger = logging.getLogger("rdkit_cli")
	
	try:
		input_path = validate_input_file(args.input_file)
		output_path = validate_output_path(args.output)
		
		logger.info(f"Calculating similarity matrix for {input_path}")
		
		molecules = read_molecules(input_path)
		logger.info(f"Loaded {len(molecules)} molecules")
		
		fingerprints = []
		mol_ids = []
		
		with tqdm(total=len(molecules), desc="Generating fingerprints", ncols=80, colour='cyan') as pbar:
			for i, mol in enumerate(molecules):
				if graceful_exit.exit_now:
					break
				
				try:
					if mol is None:
						continue
					
					fp = generate_fingerprint(mol, args.fp_type)
					if fp is not None:
						fingerprints.append(fp)
						mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
						mol_ids.append(mol_id)
				
				except Exception as e:
					logger.debug(f"Failed to process molecule {i}: {e}")
				
				pbar.update(1)
		
		if graceful_exit.exit_now:
			return 130
		
		if len(fingerprints) < 2:
			logger.error("Need at least 2 valid molecules for similarity matrix")
			return 1
		
		n_mols = len(fingerprints)
		similarity_matrix = np.zeros((n_mols, n_mols))
		
		total_pairs = (n_mols * (n_mols - 1)) // 2
		with tqdm(total=total_pairs, desc="Calculating similarities", ncols=80, colour='magenta') as pbar:
			for i in range(n_mols):
				if graceful_exit.exit_now:
					break
				
				similarity_matrix[i][i] = 1.0
				for j in range(i + 1, n_mols):
					sim = calculate_similarity(fingerprints[i], fingerprints[j], args.metric)
					similarity_matrix[i][j] = sim
					similarity_matrix[j][i] = sim
					pbar.update(1)
		
		if graceful_exit.exit_now:
			return 130
		
		df = pd.DataFrame(similarity_matrix, index=mol_ids, columns=mol_ids)
		df.to_csv(output_path)
		
		log_success(f"Calculated {n_mols}x{n_mols} similarity matrix, saved to {output_path}")
		return 0
		
	except Exception as e:
		logger.error(f"Similarity matrix calculation failed: {e}")
		return 1


def cluster(args, graceful_exit: GracefulExit) -> int:
	logger = logging.getLogger("rdkit_cli")
	
	try:
		input_path = validate_input_file(args.input_file)
		output_path = validate_output_path(args.output)
		
		logger.info(f"Clustering molecules from {input_path} using {args.method}")
		
		molecules = read_molecules(input_path)
		logger.info(f"Loaded {len(molecules)} molecules")
		
		fingerprints = []
		mol_data = []
		
		with tqdm(total=len(molecules), desc="Generating fingerprints", ncols=80, colour='yellow') as pbar:
			for i, mol in enumerate(molecules):
				if graceful_exit.exit_now:
					break
				
				try:
					if mol is None:
						continue
					
					fp = generate_fingerprint(mol, args.fp_type)
					if fp is not None:
						fingerprints.append(fp)
						mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
						mol_data.append({
							'ID': mol_id,
							'SMILES': Chem.MolToSmiles(mol)
						})
				
				except Exception as e:
					logger.debug(f"Failed to process molecule {i}: {e}")
				
				pbar.update(1)
		
		if graceful_exit.exit_now:
			return 130
		
		if len(fingerprints) < 2:
			logger.error("Need at least 2 valid molecules for clustering")
			return 1
		
		if args.method == 'butina':
			clusters = _butina_cluster(fingerprints, args.threshold)
		elif args.method == 'hierarchical':
			clusters = _hierarchical_cluster(fingerprints, args.threshold)
		elif args.method == 'kmeans':
			n_clusters = max(1, len(fingerprints) // 10)
			clusters = _kmeans_cluster(fingerprints, n_clusters)
		else:
			raise ValueError(f"Unknown clustering method: {args.method}")
		
		for i, cluster_id in enumerate(clusters):
			mol_data[i]['Cluster'] = cluster_id
		
		df = pd.DataFrame(mol_data)
		df.to_csv(output_path, index=False)
		
		n_clusters = len(set(clusters))
		log_success(f"Clustered {len(mol_data)} molecules into {n_clusters} clusters, saved to {output_path}")
		return 0
		
	except Exception as e:
		logger.error(f"Clustering failed: {e}")
		return 1


def diversity_pick(args, graceful_exit: GracefulExit) -> int:
	logger = logging.getLogger("rdkit_cli")
	
	try:
		input_path = validate_input_file(args.input_file)
		output_path = validate_output_path(args.output)
		
		logger.info(f"Picking {args.count} diverse molecules from {input_path}")
		
		molecules = read_molecules(input_path)
		logger.info(f"Loaded {len(molecules)} molecules")
		
		if args.count >= len(molecules):
			logger.warning(f"Requested count {args.count} >= dataset size {len(molecules)}, using all molecules")
			selected_molecules = molecules
		else:
			fingerprints = []
			valid_molecules = []
			
			with tqdm(total=len(molecules), desc="Generating fingerprints", ncols=80, colour='red') as pbar:
				for mol in molecules:
					if graceful_exit.exit_now:
						break
					
					try:
						if mol is None:
							continue
						
						fp = generate_fingerprint(mol, args.fp_type)
						if fp is not None:
							fingerprints.append(fp)
							valid_molecules.append(mol)
					
					except Exception as e:
						logger.debug(f"Failed to process molecule: {e}")
					
					pbar.update(1)
			
			if graceful_exit.exit_now:
				return 130
			
			if len(valid_molecules) < args.count:
				logger.warning(f"Only {len(valid_molecules)} valid molecules, selecting all")
				selected_molecules = valid_molecules
			else:
				if args.method == 'maxmin':
					selected_indices = _maxmin_diversity_pick(fingerprints, args.count)
				elif args.method == 'sphere-exclusion':
					selected_indices = _sphere_exclusion_pick(fingerprints, args.count)
				elif args.method == 'leader':
					selected_indices = _leader_pick(fingerprints, args.count)
				else:
					raise ValueError(f"Unknown diversity picking method: {args.method}")
				
				selected_molecules = [valid_molecules[i] for i in selected_indices]
		
		from ..core.common import write_molecules
		write_molecules(selected_molecules, output_path)
		
		log_success(f"Selected {len(selected_molecules)} diverse molecules, saved to {output_path}")
		return 0
		
	except Exception as e:
		logger.error(f"Diversity picking failed: {e}")
		return 1


def _butina_cluster(fingerprints: List, threshold: float) -> List[int]:
	"""Butina clustering algorithm."""
	from rdkit.ML.Cluster import Butina
	
	dists = []
	n_fps = len(fingerprints)
	
	for i in range(n_fps):
		for j in range(i + 1, n_fps):
			dist = 1.0 - DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
			dists.append(dist)
	
	clusters = Butina.ClusterData(dists, n_fps, 1.0 - threshold, isDistData=True)
	
	cluster_assignments = [0] * n_fps
	for cluster_id, cluster in enumerate(clusters):
		for mol_idx in cluster:
			cluster_assignments[mol_idx] = cluster_id
	
	return cluster_assignments


def _hierarchical_cluster(fingerprints: List, threshold: float) -> List[int]:
	"""Hierarchical clustering using sklearn."""
	fp_matrix = np.array([fp.ToBitString() for fp in fingerprints])
	fp_matrix = np.array([[int(bit) for bit in fp_str] for fp_str in fp_matrix])
	
	clustering = AgglomerativeClustering(
		n_clusters=None,
		distance_threshold=1.0 - threshold,
		linkage='average',
		metric='jaccard'
	)
	
	cluster_labels = clustering.fit_predict(fp_matrix)
	return cluster_labels.tolist()


def _kmeans_cluster(fingerprints: List, n_clusters: int) -> List[int]:
	"""K-means clustering using sklearn."""
	from sklearn.cluster import KMeans
	
	fp_matrix = np.array([fp.ToBitString() for fp in fingerprints])
	fp_matrix = np.array([[int(bit) for bit in fp_str] for fp_str in fp_matrix])
	
	kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
	cluster_labels = kmeans.fit_predict(fp_matrix)
	return cluster_labels.tolist()


def _maxmin_diversity_pick(fingerprints: List, count: int) -> List[int]:
	"""MaxMin diversity picking algorithm."""
	from rdkit.SimDivFilters import MaxMinPicker
	
	def distance_fn(i, j, fingerprints=fingerprints):
		return 1.0 - DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
	
	picker = MaxMinPicker()
	selected = picker.LazyPick(distance_fn, len(fingerprints), count)
	return list(selected)


def _sphere_exclusion_pick(fingerprints: List, count: int) -> List[int]:
	"""Sphere exclusion diversity picking."""
	selected = []
	available = list(range(len(fingerprints)))
	threshold = 0.3
	
	selected.append(0)
	available.remove(0)
	
	while len(selected) < count and available:
		best_idx = None
		best_min_sim = -1
		
		for idx in available:
			min_sim = min(
				DataStructs.TanimotoSimilarity(fingerprints[idx], fingerprints[sel_idx])
				for sel_idx in selected
			)
			
			if min_sim > best_min_sim:
				best_min_sim = min_sim
				best_idx = idx
		
		if best_idx is not None:
			selected.append(best_idx)
			available.remove(best_idx)
		else:
			break
	
	return selected


def _leader_pick(fingerprints: List, count: int) -> List[int]:
	"""Leader-based diversity picking."""
	selected = [0]
	threshold = 0.7
	
	for i in range(1, len(fingerprints)):
		if len(selected) >= count:
			break
		
		is_diverse = True
		for sel_idx in selected:
			sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[sel_idx])
			if sim > threshold:
				is_diverse = False
				break
		
		if is_diverse:
			selected.append(i)
	
	while len(selected) < count and len(selected) < len(fingerprints):
		remaining = [i for i in range(len(fingerprints)) if i not in selected]
		if remaining:
			selected.append(remaining[0])
		else:
			break
	
	return selected[:count]