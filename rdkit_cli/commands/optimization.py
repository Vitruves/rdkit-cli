# rdkit_cli/commands/optimization.py
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers, rdMolDescriptors
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, MMFFOptimizeMolecule
from tqdm import tqdm

from ..core.common import (
    GracefulExit, get_parallel_jobs, read_molecules, validate_input_file,
    validate_output_path, write_molecules, get_molecules_from_args, save_dataframe_with_format_detection
)
from ..core.logging import log_success


def add_subparser(subparsers) -> None:
    parser_optimize = subparsers.add_parser(
        'optimize',
        help='Optimize molecular geometries using force fields'
    )
    parser_optimize.add_argument(
        '-i', '--input-file',
        help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_optimize.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_optimize.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_optimize.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with optimized structures'
    )
    parser_optimize.add_argument(
        '--method',
        choices=['mmff94', 'mmff94s', 'uff'],
        default='mmff94',
        help='Force field method (default: mmff94)'
    )
    parser_optimize.add_argument(
        '--max-iters',
        type=int,
        default=200,
        help='Maximum optimization iterations (default: 200)'
    )
    parser_optimize.add_argument(
        '--energy-threshold',
        type=float,
        default=1e-6,
        help='Energy convergence threshold (default: 1e-6)'
    )

    parser_minimize = subparsers.add_parser(
        'minimize',
        help='Energy minimize molecular structures'
    )
    parser_minimize.add_argument(
        '-i', '--input-file',
        help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_minimize.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_minimize.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_minimize.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with minimized structures'
    )
    parser_minimize.add_argument(
        '--gradient-tolerance',
        type=float,
        default=0.001,
        help='Gradient tolerance for convergence (default: 0.001)'
    )
    parser_minimize.add_argument(
        '--method',
        choices=['mmff94', 'uff', 'gaff'],
        default='mmff94',
        help='Minimization method (default: mmff94)'
    )

    parser_dock_prep = subparsers.add_parser(
        'dock-prep',
        help='Prepare molecules for docking calculations'
    )
    parser_dock_prep.add_argument(
        '-i', '--input-file',
        help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_dock_prep.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_dock_prep.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_dock_prep.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with prepared molecules'
    )
    parser_dock_prep.add_argument(
        '--add-hydrogens',
        action='store_true',
        help='Add hydrogen atoms'
    )
    parser_dock_prep.add_argument(
        '--assign-charges',
        action='store_true',
        help='Assign partial charges'
    )
    parser_dock_prep.add_argument(
        '--generate-3d',
        action='store_true',
        help='Generate 3D coordinates'
    )
    parser_dock_prep.add_argument(
        '--charge-method',
        choices=['gasteiger', 'mmff94', 'am1-bcc'],
        default='gasteiger',
        help='Partial charge calculation method (default: gasteiger)'
    )


def optimize(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Optimizing structures using {args.method} from {input_path}")
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        optimized_molecules = []
        
        with tqdm(total=len(molecules), desc="Optimizing structures", ncols=80, colour='blue') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    mol_with_h = Chem.AddHs(mol)
                    
                    if mol_with_h.GetNumConformers() == 0:
                        AllChem.EmbedMolecule(mol_with_h)
                    
                    initial_energy = None
                    final_energy = None
                    converged = False
                    
                    if args.method == 'mmff94':
                        props = AllChem.MMFFGetMoleculeProperties(mol_with_h)
                        if props is not None:
                            ff = AllChem.MMFFGetMoleculeForceField(mol_with_h, props)
                            if ff is not None:
                                initial_energy = ff.CalcEnergy()
                                result = ff.Minimize(maxIts=args.max_iters, energyTol=args.energy_threshold)
                                final_energy = ff.CalcEnergy()
                                converged = (result == 0)
                    
                    elif args.method == 'mmff94s':
                        props = AllChem.MMFFGetMoleculeProperties(mol_with_h, mmffVariant='MMFF94s')
                        if props is not None:
                            ff = AllChem.MMFFGetMoleculeForceField(mol_with_h, props)
                            if ff is not None:
                                initial_energy = ff.CalcEnergy()
                                result = ff.Minimize(maxIts=args.max_iters, energyTol=args.energy_threshold)
                                final_energy = ff.CalcEnergy()
                                converged = (result == 0)
                    
                    elif args.method == 'uff':
                        ff = AllChem.UFFGetMoleculeForceField(mol_with_h)
                        if ff is not None:
                            initial_energy = ff.CalcEnergy()
                            result = ff.Minimize(maxIts=args.max_iters, energyTol=args.energy_threshold)
                            final_energy = ff.CalcEnergy()
                            converged = (result == 0)
                    
                    if final_energy is not None:
                        mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                        mol_with_h.SetProp("_Name", mol_id)
                        mol_with_h.SetProp("Initial_Energy", str(round(initial_energy, 6)))
                        mol_with_h.SetProp("Final_Energy", str(round(final_energy, 6)))
                        mol_with_h.SetProp("Energy_Change", str(round(final_energy - initial_energy, 6)))
                        mol_with_h.SetProp("Converged", str(converged))
                        optimized_molecules.append(mol_with_h)
                
                except Exception as e:
                    logger.debug(f"Failed to optimize molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        write_molecules(optimized_molecules, output_path)
        log_success(f"Optimized {len(optimized_molecules)} structures, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Structure optimization failed: {e}")
        return 1


def minimize(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Minimizing structures using {args.method} from {input_path}")
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        minimized_molecules = []
        
        with tqdm(total=len(molecules), desc="Minimizing structures", ncols=80, colour='green') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    mol_with_h = Chem.AddHs(mol)
                    
                    if mol_with_h.GetNumConformers() == 0:
                        AllChem.EmbedMolecule(mol_with_h)
                    
                    if args.method == 'mmff94':
                        result = MMFFOptimizeMolecule(mol_with_h)
                    elif args.method == 'uff':
                        result = UFFOptimizeMolecule(mol_with_h)
                    else:  # gaff or other
                        result = UFFOptimizeMolecule(mol_with_h)
                    
                    mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                    mol_with_h.SetProp("_Name", mol_id)
                    mol_with_h.SetProp("Minimization_Result", str(result))
                    mol_with_h.SetProp("Method", args.method)
                    
                    minimized_molecules.append(mol_with_h)
                
                except Exception as e:
                    logger.debug(f"Failed to minimize molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        write_molecules(minimized_molecules, output_path)
        log_success(f"Minimized {len(minimized_molecules)} structures, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Structure minimization failed: {e}")
        return 1


def dock_prep(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Preparing molecules for docking from {input_path}")
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        prepared_molecules = []
        
        with tqdm(total=len(molecules), desc="Preparing for docking", ncols=80, colour='cyan') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    prepared_mol = Chem.Mol(mol)
                    
                    if args.add_hydrogens:
                        prepared_mol = Chem.AddHs(prepared_mol)
                    
                    if args.generate_3d:
                        if prepared_mol.GetNumConformers() == 0:
                            AllChem.EmbedMolecule(prepared_mol)
                            UFFOptimizeMolecule(prepared_mol)
                    
                    if args.assign_charges:
                        _assign_partial_charges(prepared_mol, args.charge_method)
                    
                    mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                    prepared_mol.SetProp("_Name", mol_id)
                    prepared_mol.SetProp("Prepared_For_Docking", "True")
                    prepared_mol.SetProp("Hydrogens_Added", str(args.add_hydrogens))
                    prepared_mol.SetProp("Charges_Assigned", str(args.assign_charges))
                    prepared_mol.SetProp("3D_Generated", str(args.generate_3d))
                    
                    if args.assign_charges:
                        prepared_mol.SetProp("Charge_Method", args.charge_method)
                    
                    prepared_molecules.append(prepared_mol)
                
                except Exception as e:
                    logger.debug(f"Failed to prepare molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        write_molecules(prepared_molecules, output_path)
        log_success(f"Prepared {len(prepared_molecules)} molecules for docking, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Docking preparation failed: {e}")
        return 1


def _assign_partial_charges(mol: Chem.Mol, method: str) -> None:
    """Assign partial charges to molecule."""
    try:
        if method == 'gasteiger':
            AllChem.ComputeGasteigerCharges(mol)
            
            for atom in mol.GetAtoms():
                charge = atom.GetDoubleProp('_GasteigerCharge')
                atom.SetDoubleProp('PartialCharge', charge)
        
        elif method == 'mmff94':
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                for i, atom in enumerate(mol.GetAtoms()):
                    charge = props.GetMMFFPartialCharge(i)
                    atom.SetDoubleProp('PartialCharge', charge)
        
        elif method == 'am1-bcc':
            logger = logging.getLogger("rdkit_cli")
            logger.warning("AM1-BCC charges require external software, using Gasteiger instead")
            AllChem.ComputeGasteigerCharges(mol)
            
            for atom in mol.GetAtoms():
                charge = atom.GetDoubleProp('_GasteigerCharge')
                atom.SetDoubleProp('PartialCharge', charge)
    
    except Exception:
        pass