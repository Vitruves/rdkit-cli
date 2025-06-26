# rdkit_cli/commands/substructure.py
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from ..core.common import (
    GracefulExit, get_parallel_jobs, read_molecules, validate_input_file,
    validate_output_path, write_molecules, get_molecules_from_args, save_dataframe_with_format_detection
)
from ..core.logging import log_success


def add_subparser(subparsers) -> None:
    parser_substructure_search = subparsers.add_parser(
        'substructure-search',
        help='Search for molecules containing a specific substructure'
    )
    parser_substructure_search.add_argument(
        '-i', '--input-file',
        help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_substructure_search.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_substructure_search.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_substructure_search.add_argument(
        '--query',
        required=True,
        help='Query substructure (SMARTS or SMILES pattern)'
    )
    parser_substructure_search.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with matching molecules'
    )
    parser_substructure_search.add_argument(
        '--count-matches',
        action='store_true',
        help='Count number of matches per molecule'
    )

    parser_substructure_filter = subparsers.add_parser(
        'substructure-filter',
        help='Filter molecules using SMARTS patterns from file'
    )
    parser_substructure_filter.add_argument(
        '-i', '--input-file',
        help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_substructure_filter.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_substructure_filter.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_substructure_filter.add_argument(
        '--smarts-file',
        required=True,
        help='File containing SMARTS patterns (one per line)'
    )
    parser_substructure_filter.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with filtered molecules'
    )
    parser_substructure_filter.add_argument(
        '--mode',
        choices=['include', 'exclude'],
        default='exclude',
        help='Filter mode: include or exclude matching molecules (default: exclude)'
    )

    parser_scaffold_analysis = subparsers.add_parser(
        'scaffold-analysis',
        help='Analyze molecular scaffolds and their frequency'
    )
    parser_scaffold_analysis.add_argument(
        '-i', '--input-file',
        help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_scaffold_analysis.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_scaffold_analysis.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_scaffold_analysis.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file with scaffold analysis'
    )
    parser_scaffold_analysis.add_argument(
        '--include-counts',
        action='store_true',
        help='Include molecule counts per scaffold'
    )
    parser_scaffold_analysis.add_argument(
        '--min-frequency',
        type=int,
        default=2,
        help='Minimum frequency to include scaffold (default: 2)'
    )

    parser_murcko_scaffolds = subparsers.add_parser(
        'murcko-scaffolds',
        help='Extract Murcko scaffolds from molecules'
    )
    parser_murcko_scaffolds.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input molecular file'
    )
    parser_murcko_scaffolds.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with scaffolds'
    )
    parser_murcko_scaffolds.add_argument(
        '--generic',
        action='store_true',
        help='Generate generic (atom-type independent) scaffolds'
    )
    parser_murcko_scaffolds.add_argument(
        '--unique-only',
        action='store_true',
        help='Output only unique scaffolds'
    )

    parser_functional_groups = subparsers.add_parser(
        'functional-groups',
        help='Identify functional groups in molecules'
    )
    parser_functional_groups.add_argument(
        '-i', '--input-file',
        help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_functional_groups.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_functional_groups.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_functional_groups.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file with functional group analysis'
    )
    parser_functional_groups.add_argument(
        '--hierarchy',
        choices=['ifg', 'brics', 'custom'],
        default='ifg',
        help='Functional group hierarchy to use (default: ifg)'
    )


def search(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        output_path = validate_output_path(args.output)
        
        logger.info(f"Searching for substructure '{args.query}'")
        
        query_mol = Chem.MolFromSmarts(args.query)
        if query_mol is None:
            query_mol = Chem.MolFromSmiles(args.query)
        
        if query_mol is None:
            logger.error(f"Invalid query structure: {args.query}")
            return 1
        
        molecules = get_molecules_from_args(args)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        matching_molecules = []
        match_data = []
        
        with tqdm(total=len(molecules), desc="Searching substructures", ncols=80, colour='blue') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    matches = mol.GetSubstructMatches(query_mol)
                    if matches:
                        matching_molecules.append(mol)
                        
                        if args.count_matches:
                            mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                            match_data.append({
                                'ID': mol_id,
                                'SMILES': Chem.MolToSmiles(mol),
                                'Match_Count': len(matches)
                            })
                
                except Exception as e:
                    logger.debug(f"Failed to process molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        if args.count_matches and match_data:
            df = pd.DataFrame(match_data)
            save_dataframe_with_format_detection(df, output_path)
            log_success(f"Found {len(matching_molecules)} matches, counts saved to {output_path}")
        
        write_molecules(matching_molecules, output_path)
        log_success(f"Found {len(matching_molecules)} molecules with substructure, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Substructure search failed: {e}")
        return 1


def filter_smarts(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        smarts_path = validate_input_file(args.smarts_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Filtering molecules from {input_path} using patterns from {smarts_path}")
        
        with open(smarts_path, 'r') as f:
            smarts_patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        compiled_patterns = []
        for pattern in smarts_patterns:
            mol = Chem.MolFromSmarts(pattern)
            if mol is not None:
                compiled_patterns.append(mol)
            else:
                logger.warning(f"Invalid SMARTS pattern: {pattern}")
        
        if not compiled_patterns:
            logger.error("No valid SMARTS patterns found")
            return 1
        
        logger.info(f"Using {len(compiled_patterns)} SMARTS patterns")
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        filtered_molecules = []
        
        with tqdm(total=len(molecules), desc="Filtering molecules", ncols=80, colour='green') as pbar:
            for mol in molecules:
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    has_match = any(mol.GetSubstructMatches(pattern) for pattern in compiled_patterns)
                    
                    if args.mode == 'include' and has_match:
                        filtered_molecules.append(mol)
                    elif args.mode == 'exclude' and not has_match:
                        filtered_molecules.append(mol)
                
                except Exception as e:
                    logger.debug(f"Failed to process molecule: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        write_molecules(filtered_molecules, output_path)
        log_success(f"Filtered to {len(filtered_molecules)} molecules, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"SMARTS filtering failed: {e}")
        return 1


def scaffold_analysis(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Analyzing scaffolds from {input_path}")
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        scaffold_counts: Dict[str, int] = {}
        scaffold_molecules: Dict[str, List[str]] = {}
        
        with tqdm(total=len(molecules), desc="Extracting scaffolds", ncols=80, colour='cyan') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    if scaffold is not None:
                        scaffold_smiles = Chem.MolToSmiles(scaffold)
                        scaffold_counts[scaffold_smiles] = scaffold_counts.get(scaffold_smiles, 0) + 1
                        
                        if args.include_counts:
                            mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                            if scaffold_smiles not in scaffold_molecules:
                                scaffold_molecules[scaffold_smiles] = []
                            scaffold_molecules[scaffold_smiles].append(mol_id)
                
                except Exception as e:
                    logger.debug(f"Failed to process molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        data = []
        for scaffold_smiles, count in scaffold_counts.items():
            if count >= args.min_frequency:
                row = {
                    'Scaffold_SMILES': scaffold_smiles,
                    'Frequency': count
                }
                
                if args.include_counts:
                    row['Molecule_IDs'] = ';'.join(scaffold_molecules[scaffold_smiles])
                
                data.append(row)
        
        data.sort(key=lambda x: x['Frequency'], reverse=True)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        log_success(f"Analyzed {len(data)} scaffolds (min frequency: {args.min_frequency}), saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Scaffold analysis failed: {e}")
        return 1


def murcko_scaffolds(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Extracting Murcko scaffolds from {input_path}")
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        scaffolds = []
        seen_scaffolds: Set[str] = set()
        
        with tqdm(total=len(molecules), desc="Extracting scaffolds", ncols=80, colour='magenta') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    if args.generic:
                        scaffold = MurckoScaffold.MakeScaffoldGeneric(
                            MurckoScaffold.GetScaffoldForMol(mol)
                        )
                    else:
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    
                    if scaffold is not None:
                        scaffold_smiles = Chem.MolToSmiles(scaffold)
                        
                        if args.unique_only:
                            if scaffold_smiles not in seen_scaffolds:
                                seen_scaffolds.add(scaffold_smiles)
                                scaffold.SetProp("_Name", f"scaffold_{len(scaffolds)+1}")
                                scaffolds.append(scaffold)
                        else:
                            mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                            scaffold.SetProp("_Name", f"scaffold_{mol_id}")
                            scaffolds.append(scaffold)
                
                except Exception as e:
                    logger.debug(f"Failed to process molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        write_molecules(scaffolds, output_path)
        log_success(f"Extracted {len(scaffolds)} scaffolds, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Murcko scaffold extraction failed: {e}")
        return 1


def functional_groups(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Identifying functional groups from {input_path}")
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        functional_group_patterns = _get_functional_group_patterns(args.hierarchy)
        
        data = []
        
        with tqdm(total=len(molecules), desc="Identifying functional groups", ncols=80, colour='yellow') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                    row = {
                        'ID': mol_id,
                        'SMILES': Chem.MolToSmiles(mol)
                    }
                    
                    for fg_name, fg_pattern in functional_group_patterns.items():
                        matches = mol.GetSubstructMatches(fg_pattern)
                        row[fg_name] = len(matches)
                    
                    data.append(row)
                
                except Exception as e:
                    logger.debug(f"Failed to process molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        log_success(f"Analyzed functional groups for {len(data)} molecules, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Functional group analysis failed: {e}")
        return 1


def _get_functional_group_patterns(hierarchy: str) -> Dict[str, Chem.Mol]:
    """Get functional group SMARTS patterns based on hierarchy."""
    
    if hierarchy == 'ifg':
        patterns = {
            'Alcohol': Chem.MolFromSmarts('[CX4][OX2H]'),
            'Phenol': Chem.MolFromSmarts('[OX2H][cX3]:[c]'),
            'Carboxylic_Acid': Chem.MolFromSmarts('[CX3](=O)[OX2H1]'),
            'Ester': Chem.MolFromSmarts('[CX3](=O)[OX2H0]'),
            'Ether': Chem.MolFromSmarts('[OD2]([#6])[#6]'),
            'Aldehyde': Chem.MolFromSmarts('[CX3H1](=O)[#6]'),
            'Ketone': Chem.MolFromSmarts('[#6][CX3](=O)[#6]'),
            'Amine_Primary': Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]'),
            'Amine_Secondary': Chem.MolFromSmarts('[NX3;H1,H0;!$(NC=O)]'),
            'Amine_Tertiary': Chem.MolFromSmarts('[NX3;H0;!$(NC=O)]'),
            'Amide': Chem.MolFromSmarts('[NX3][CX3](=[OX1])[#6]'),
            'Halide': Chem.MolFromSmarts('[#6][F,Cl,Br,I]'),
            'Nitrile': Chem.MolFromSmarts('[NX1]#[CX2]'),
            'Nitro': Chem.MolFromSmarts('[NX3+]([OX1-])=[OX1]'),
            'Aromatic_Ring': Chem.MolFromSmarts('c1ccccc1'),
            'Sulfide': Chem.MolFromSmarts('[#6][SX2][#6]'),
            'Sulfonyl': Chem.MolFromSmarts('[SX4](=[OX1])(=[OX1])([#6])[#6]')
        }
    
    elif hierarchy == 'brics':
        patterns = {
            'C-C': Chem.MolFromSmarts('[#6]!@[#6]'),
            'C-N': Chem.MolFromSmarts('[#6]!@[#7]'),
            'C-O': Chem.MolFromSmarts('[#6]!@[#8]'),
            'C-S': Chem.MolFromSmarts('[#6]!@[#16]'),
            'N-N': Chem.MolFromSmarts('[#7]!@[#7]'),
            'N-O': Chem.MolFromSmarts('[#7]!@[#8]'),
            'O-O': Chem.MolFromSmarts('[#8]!@[#8]'),
            'Ring_C': Chem.MolFromSmarts('[R][#6][R]'),
            'Ring_N': Chem.MolFromSmarts('[R][#7][R]'),
            'Ring_O': Chem.MolFromSmarts('[R][#8][R]')
        }
    
    else:  # custom
        patterns = {
            'Benzene': Chem.MolFromSmarts('c1ccccc1'),
            'Pyridine': Chem.MolFromSmarts('c1ccncc1'),
            'Imidazole': Chem.MolFromSmarts('c1c[nH]cn1'),
            'Furan': Chem.MolFromSmarts('c1ccoc1'),
            'Thiophene': Chem.MolFromSmarts('c1ccsc1'),
            'Piperidine': Chem.MolFromSmarts('C1CCNCC1'),
            'Morpholine': Chem.MolFromSmarts('C1COCCN1'),
            'Cyclohexane': Chem.MolFromSmarts('C1CCCCC1')
        }
    
    return {name: pattern for name, pattern in patterns.items() if pattern is not None}