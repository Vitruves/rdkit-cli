# rdkit_cli/commands/conformers.py
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolDescriptors, rdDistGeom, rdForceFieldHelpers, rdFMCS
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdShapeHelpers import ShapeTanimotoDist
from tqdm import tqdm

from ..core.common import (
    GracefulExit, get_parallel_jobs, read_molecules, validate_input_file,
    validate_output_path, write_molecules, get_molecules_from_args, save_dataframe_with_format_detection
)
from ..core.logging import log_success


def add_subparser(subparsers) -> None:
    parser_conformers = subparsers.add_parser(
        'conformers',
        help='Generate 3D conformers for molecules'
    )
    parser_conformers.add_argument(
        '-i', '--input-file',
        help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_conformers.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_conformers.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_conformers.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with 3D conformers'
    )
    parser_conformers.add_argument(
        '--num-confs',
        type=int,
        default=10,
        help='Number of conformers to generate per molecule (default: 10)'
    )
    parser_conformers.add_argument(
        '--method',
        choices=['etkdg', 'etdg', 'kdg', 'rdkit'],
        default='etkdg',
        help='Conformer generation method (default: etkdg)'
    )
    parser_conformers.add_argument(
        '--optimize',
        action='store_true',
        help='Optimize conformers using force field'
    )
    parser_conformers.add_argument(
        '--ff',
        choices=['mmff94', 'mmff94s', 'uff'],
        default='mmff94',
        help='Force field for optimization (default: mmff94)'
    )
    parser_conformers.add_argument(
        '--max-iters',
        type=int,
        default=200,
        help='Maximum optimization iterations (default: 200)'
    )
    parser_conformers.add_argument(
        '--energy-window',
        type=float,
        default=10.0,
        help='Energy window for conformer filtering (kcal/mol, default: 10.0)'
    )

    parser_align = subparsers.add_parser(
        'align-molecules',
        help='Align molecules to a template structure'
    )
    parser_align.add_argument(
        '-i', '--input-file',
        help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_align.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_align.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_align.add_argument(
        '--template',
        required=True,
        help='Template molecule file for alignment'
    )
    parser_align.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with aligned molecules'
    )
    parser_align.add_argument(
        '--align-mode',
        choices=['mcs', 'shape', 'pharmacophore'],
        default='mcs',
        help='Alignment method (default: mcs)'
    )

    parser_shape_similarity = subparsers.add_parser(
        'shape-similarity',
        help='Calculate 3D shape similarity between molecules'
    )
    parser_shape_similarity.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input molecular file'
    )
    parser_shape_similarity.add_argument(
        '--reference',
        required=True,
        help='Reference molecule file'
    )
    parser_shape_similarity.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file with shape similarity scores'
    )
    parser_shape_similarity.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Minimum similarity threshold (default: 0.5)'
    )

    parser_pharmacophore_screen = subparsers.add_parser(
        'pharmacophore-screen',
        help='Screen molecules against pharmacophore models'
    )
    parser_pharmacophore_screen.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input molecular file'
    )
    parser_pharmacophore_screen.add_argument(
        '--pharmacophore',
        required=True,
        help='Pharmacophore model file (JSON format)'
    )
    parser_pharmacophore_screen.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with matching molecules'
    )
    parser_pharmacophore_screen.add_argument(
        '--tolerance',
        type=float,
        default=1.0,
        help='Distance tolerance for pharmacophore matching (Å, default: 1.0)'
    )


def generate(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Generating {args.num_confs} conformers using {args.method} for {input_path}")
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        output_molecules = []
        
        with tqdm(total=len(molecules), desc="Generating conformers", ncols=80, colour='blue') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    mol_with_h = Chem.AddHs(mol)
                    
                    if args.method == 'etkdg':
                        params = rdDistGeom.ETKDGv3()
                        params.numThreads = 0
                        conf_ids = rdDistGeom.EmbedMultipleConfs(mol_with_h, numConfs=args.num_confs, params=params)
                    elif args.method == 'etdg':
                        params = rdDistGeom.ETDG()
                        conf_ids = rdDistGeom.EmbedMultipleConfs(mol_with_h, numConfs=args.num_confs, params=params)
                    elif args.method == 'kdg':
                        params = rdDistGeom.KDG()
                        conf_ids = rdDistGeom.EmbedMultipleConfs(mol_with_h, numConfs=args.num_confs, params=params)
                    else:  # rdkit
                        conf_ids = AllChem.EmbedMultipleConfs(mol_with_h, numConfs=args.num_confs)
                    
                    if not conf_ids:
                        logger.debug(f"Failed to generate conformers for molecule {i}")
                        continue
                    
                    if args.optimize:
                        energies = []
                        for conf_id in conf_ids:
                            if args.ff == 'mmff94':
                                ff = AllChem.MMFFGetMoleculeForceField(mol_with_h, AllChem.MMFFGetMoleculeProperties(mol_with_h), confId=conf_id)
                            elif args.ff == 'mmff94s':
                                ff = AllChem.MMFFGetMoleculeForceField(mol_with_h, AllChem.MMFFGetMoleculeProperties(mol_with_h, mmffVariant='MMFF94s'), confId=conf_id)
                            else:  # uff
                                ff = AllChem.UFFGetMoleculeForceField(mol_with_h, confId=conf_id)
                            
                            if ff is not None:
                                ff.Minimize(maxIts=args.max_iters)
                                energy = ff.CalcEnergy()
                                energies.append((conf_id, energy))
                        
                        if energies:
                            energies.sort(key=lambda x: x[1])
                            min_energy = energies[0][1]
                            
                            filtered_confs = [
                                conf_id for conf_id, energy in energies 
                                if (energy - min_energy) * 627.509 <= args.energy_window
                            ]
                            
                            for conf_id in conf_ids:
                                if conf_id not in filtered_confs:
                                    mol_with_h.RemoveConformer(conf_id)
                    
                    mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                    
                    for conf_id in mol_with_h.GetConformers():
                        conf_mol = Chem.Mol(mol_with_h)
                        conf_mol.RemoveAllConformers()
                        conf_mol.AddConformer(mol_with_h.GetConformer(conf_id.GetId()), assignId=True)
                        conf_mol.SetProp("_Name", f"{mol_id}_conf_{conf_id.GetId()}")
                        output_molecules.append(conf_mol)
                
                except Exception as e:
                    logger.debug(f"Failed to process molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        write_molecules(output_molecules, output_path)
        log_success(f"Generated conformers for {len(output_molecules)} structures, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Conformer generation failed: {e}")
        return 1


def align(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        # Handle template as either a file path or SMILES string
        if Path(args.template).exists():
            template_molecules = read_molecules(args.template)
            logger.info(f"Aligning molecules from {input_path} to template file {args.template}")
        else:
            # Treat as SMILES string
            template_mol = Chem.MolFromSmiles(args.template)
            if template_mol is None:
                logger.error(f"Invalid template: {args.template}")
                return 1
            template_molecules = [template_mol]
            logger.info(f"Aligning molecules from {input_path} to template SMILES {args.template}")
        
        template_molecules = template_molecules
        if not template_molecules:
            logger.error("No template molecule found")
            return 1
        
        template = template_molecules[0]
        template_with_h = Chem.AddHs(template)
        AllChem.EmbedMolecule(template_with_h)
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        aligned_molecules = []
        
        with tqdm(total=len(molecules), desc="Aligning molecules", ncols=80, colour='green') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    mol_with_h = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol_with_h)
                    
                    if args.align_mode == 'mcs':
                        mcs = rdFMCS.FindMCS([template, mol])
                        if mcs.numAtoms > 3:
                            template_match = template.GetSubstructMatch(Chem.MolFromSmarts(mcs.smartsString))
                            mol_match = mol.GetSubstructMatch(Chem.MolFromSmarts(mcs.smartsString))
                            
                            if template_match and mol_match:
                                rmsd = AlignMol(mol_with_h, template_with_h, atomMap=list(zip(mol_match, template_match)))
                                mol_with_h.SetProp("RMSD", str(rmsd))
                    
                    elif args.align_mode == 'shape':
                        AllChem.AlignMol(mol_with_h, template_with_h)
                    
                    aligned_molecules.append(mol_with_h)
                
                except Exception as e:
                    logger.debug(f"Failed to align molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        write_molecules(aligned_molecules, output_path)
        log_success(f"Aligned {len(aligned_molecules)} molecules, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Molecular alignment failed: {e}")
        return 1


def shape_similarity(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        # Handle reference as either a file path or SMILES string
        if Path(args.reference).exists():
            reference_molecules = read_molecules(args.reference)
            logger.info(f"Calculating shape similarity to reference file {args.reference}")
        else:
            # Treat as SMILES string
            reference_mol = Chem.MolFromSmiles(args.reference)
            if reference_mol is None:
                logger.error(f"Invalid reference: {args.reference}")
                return 1
            reference_molecules = [reference_mol]
            logger.info(f"Calculating shape similarity to reference SMILES {args.reference}")
        
        if not reference_molecules:
            logger.error("No reference molecule found")
            return 1
        
        reference = reference_molecules[0]
        reference_with_h = Chem.AddHs(reference)
        AllChem.EmbedMolecule(reference_with_h)
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        results = []
        
        with tqdm(total=len(molecules), desc="Calculating shape similarity", ncols=80, colour='cyan') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    mol_with_h = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol_with_h)
                    
                    similarity = 1.0 - ShapeTanimotoDist(reference_with_h, mol_with_h)
                    
                    if similarity >= args.threshold:
                        mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                        results.append({
                            'ID': mol_id,
                            'SMILES': Chem.MolToSmiles(mol),
                            'Shape_Similarity': round(similarity, 4)
                        })
                
                except Exception as e:
                    logger.debug(f"Failed to process molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        results.sort(key=lambda x: x['Shape_Similarity'], reverse=True)
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        log_success(f"Calculated shape similarity for {len(results)} molecules, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Shape similarity calculation failed: {e}")
        return 1


def pharmacophore_screen(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        pharmacophore_path = validate_input_file(args.pharmacophore)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Screening molecules against pharmacophore {pharmacophore_path}")
        
        with open(pharmacophore_path, 'r') as f:
            pharmacophore_data = json.load(f)
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        matching_molecules = []
        
        with tqdm(total=len(molecules), desc="Screening pharmacophore", ncols=80, colour='magenta') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    mol_with_h = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol_with_h)
                    
                    if _matches_pharmacophore(mol_with_h, pharmacophore_data, args.tolerance):
                        matching_molecules.append(mol)
                
                except Exception as e:
                    logger.debug(f"Failed to process molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        write_molecules(matching_molecules, output_path)
        log_success(f"Found {len(matching_molecules)} pharmacophore matches, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Pharmacophore screening failed: {e}")
        return 1


def _matches_pharmacophore(mol: Chem.Mol, pharmacophore_data: Dict, tolerance: float) -> bool:
    """Check if molecule matches pharmacophore model."""
    try:
        features = pharmacophore_data.get('features', [])
        
        feature_coords = []
        for feature in features:
            feature_type = feature.get('type')
            coords = feature.get('coordinates')
            
            if feature_type == 'donor':
                pattern = Chem.MolFromSmarts('[N,O;H]')
            elif feature_type == 'acceptor':
                pattern = Chem.MolFromSmarts('[N,O]')
            elif feature_type == 'aromatic':
                pattern = Chem.MolFromSmarts('c1ccccc1')
            elif feature_type == 'hydrophobic':
                pattern = Chem.MolFromSmarts('[C,c]')
            else:
                continue
            
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return False
            
            conf = mol.GetConformer()
            best_distance = float('inf')
            
            for match in matches:
                atom_pos = conf.GetAtomPosition(match[0])
                distance = np.sqrt(
                    (atom_pos.x - coords[0])**2 + 
                    (atom_pos.y - coords[1])**2 + 
                    (atom_pos.z - coords[2])**2
                )
                best_distance = min(best_distance, distance)
            
            if best_distance > tolerance:
                return False
        
        return True
        
    except Exception:
        return False