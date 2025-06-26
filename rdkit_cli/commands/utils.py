# rdkit_cli/commands/utils.py
import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

from ..core.common import (
    GracefulExit, get_parallel_jobs, read_molecules, validate_input_file,
    validate_output_path, write_molecules, get_molecules_from_args, save_dataframe_with_format_detection
)
from ..core.config import config
from ..core.logging import log_success


def add_subparser(subparsers) -> None:
    parser_info = subparsers.add_parser(
        'info',
        help='Display information about molecular file'
    )
    parser_info.add_argument(
        '-i', '--input-file',
        help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_info.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_info.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )

    parser_stats = subparsers.add_parser(
        'stats',
        help='Calculate statistics for molecular dataset'
    )
    parser_stats.add_argument(
        '-i', '--input-file',
        help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_stats.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_stats.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_stats.add_argument(
        '-o', '--output',
        required=True,
        help='Output JSON file with statistics'
    )
    parser_stats.add_argument(
        '--include-descriptors',
        action='store_true',
        help='Include descriptor statistics'
    )

    parser_sample = subparsers.add_parser(
        'sample',
        help='Sample molecules from dataset'
    )
    parser_sample.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input molecular file'
    )
    parser_sample.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with sampled molecules'
    )
    parser_sample.add_argument(
        '--count',
        type=int,
        default=1000,
        help='Number of molecules to sample (default: 1000)'
    )
    parser_sample.add_argument(
        '--method',
        choices=['random', 'systematic', 'diverse'],
        default='random',
        help='Sampling method (default: random)'
    )
    parser_sample.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser_benchmark = subparsers.add_parser(
        'benchmark',
        help='Benchmark performance of operations'
    )
    parser_benchmark.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input molecular file for benchmarking'
    )
    parser_benchmark.add_argument(
        '--operation',
        choices=['descriptors', 'fingerprints', 'similarity', 'substructure'],
        default='descriptors',
        help='Operation to benchmark (default: descriptors)'
    )
    parser_benchmark.add_argument(
        '--jobs',
        type=int,
        help='Number of parallel jobs to test'
    )
    parser_benchmark.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='Number of benchmark iterations (default: 3)'
    )

    parser_config = subparsers.add_parser(
        'config',
        help='Manage configuration settings'
    )
    parser_config.add_argument(
        '--set',
        nargs=2,
        metavar=('KEY', 'VALUE'),
        help='Set configuration value'
    )
    parser_config.add_argument(
        '--get',
        metavar='KEY',
        help='Get configuration value'
    )
    parser_config.add_argument(
        '--list',
        action='store_true',
        help='List all configuration values'
    )
    parser_config.add_argument(
        '--reset',
        action='store_true',
        help='Reset configuration to defaults'
    )



def info(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        
        logger.info(f"Analyzing file information for {input_path}")
        
        molecules = read_molecules(input_path)
        
        valid_molecules = [mol for mol in molecules if mol is not None]
        invalid_count = len(molecules) - len(valid_molecules)
        
        if not valid_molecules:
            logger.warning("No valid molecules found")
            return 0
        
        print(f"\nFile: {input_path}")
        print(f"Format: {input_path.suffix}")
        print(f"Total entries: {len(molecules)}")
        print(f"Valid molecules: {len(valid_molecules)}")
        print(f"Invalid entries: {invalid_count}")
        
        if valid_molecules:
            atom_counts = [mol.GetNumAtoms() for mol in valid_molecules]
            heavy_atom_counts = [mol.GetNumHeavyAtoms() for mol in valid_molecules]
            
            print(f"\nMolecular statistics:")
            print(f"  Atom count range: {min(atom_counts)} - {max(atom_counts)}")
            print(f"  Heavy atom count range: {min(heavy_atom_counts)} - {max(heavy_atom_counts)}")
            print(f"  Average heavy atoms: {sum(heavy_atom_counts) / len(heavy_atom_counts):.1f}")
            
            mw_values = []
            for mol in valid_molecules[:100]:  # Sample first 100 for performance
                try:
                    mw = Descriptors.MolWt(mol)
                    mw_values.append(mw)
                except Exception:
                    continue
            
            if mw_values:
                print(f"  Molecular weight range: {min(mw_values):.1f} - {max(mw_values):.1f}")
                print(f"  Average molecular weight: {sum(mw_values) / len(mw_values):.1f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"File analysis failed: {e}")
        return 1


def stats(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Calculating statistics for {input_path}")
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        valid_molecules = [mol for mol in molecules if mol is not None]
        
        statistics = {
            'file_info': {
                'file_path': str(input_path),
                'file_format': input_path.suffix,
                'total_entries': len(molecules),
                'valid_molecules': len(valid_molecules),
                'invalid_entries': len(molecules) - len(valid_molecules)
            }
        }
        
        if valid_molecules:
            basic_stats = _calculate_basic_statistics(valid_molecules, graceful_exit)
            statistics['basic_statistics'] = basic_stats
            
            if args.include_descriptors:
                descriptor_stats = _calculate_descriptor_statistics(valid_molecules, graceful_exit)
                statistics['descriptor_statistics'] = descriptor_stats
        
        if graceful_exit.exit_now:
            return 130
        
        with open(output_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        log_success(f"Calculated statistics for {len(valid_molecules)} molecules, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Statistics calculation failed: {e}")
        return 1


def sample(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Sampling {args.count} molecules using {args.method} method from {input_path}")
        
        molecules = read_molecules(input_path)
        valid_molecules = [mol for mol in molecules if mol is not None]
        
        logger.info(f"Loaded {len(valid_molecules)} valid molecules")
        
        if args.count >= len(valid_molecules):
            logger.warning(f"Sample size {args.count} >= dataset size {len(valid_molecules)}, using all molecules")
            sampled_molecules = valid_molecules
        else:
            if args.method == 'random':
                random.seed(args.seed)
                sampled_molecules = random.sample(valid_molecules, args.count)
            
            elif args.method == 'systematic':
                step = len(valid_molecules) // args.count
                sampled_molecules = valid_molecules[::step][:args.count]
            
            elif args.method == 'diverse':
                sampled_molecules = _diverse_sample(valid_molecules, args.count, args.seed, graceful_exit)
            
            else:
                logger.error(f"Unknown sampling method: {args.method}")
                return 1
        
        if graceful_exit.exit_now:
            return 130
        
        write_molecules(sampled_molecules, output_path)
        log_success(f"Sampled {len(sampled_molecules)} molecules, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Molecular sampling failed: {e}")
        return 1


def benchmark(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        
        logger.info(f"Benchmarking {args.operation} operation on {input_path}")
        
        molecules = read_molecules(input_path)
        valid_molecules = [mol for mol in molecules if mol is not None][:1000]  # Limit for benchmarking
        
        logger.info(f"Benchmarking with {len(valid_molecules)} molecules")
        
        jobs_to_test = [1, 2, 4, 8] if args.jobs is None else [args.jobs]
        
        results = {}
        
        for n_jobs in jobs_to_test:
            if graceful_exit.exit_now:
                break
            
            times = []
            for iteration in range(args.iterations):
                if graceful_exit.exit_now:
                    break
                
                start_time = time.time()
                
                if args.operation == 'descriptors':
                    _benchmark_descriptors(valid_molecules, n_jobs)
                elif args.operation == 'fingerprints':
                    _benchmark_fingerprints(valid_molecules, n_jobs)
                elif args.operation == 'similarity':
                    _benchmark_similarity(valid_molecules, n_jobs)
                elif args.operation == 'substructure':
                    _benchmark_substructure(valid_molecules, n_jobs)
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                logger.info(f"Jobs: {n_jobs}, Iteration: {iteration + 1}, Time: {times[-1]:.2f}s")
            
            if times:
                results[f'{n_jobs}_jobs'] = {
                    'mean_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'times': times
                }
        
        if graceful_exit.exit_now:
            return 130
        
        print(f"\nBenchmark Results for {args.operation}:")
        print(f"Dataset size: {len(valid_molecules)} molecules")
        print(f"Iterations: {args.iterations}")
        print("-" * 50)
        
        for job_config, metrics in results.items():
            print(f"{job_config}: {metrics['mean_time']:.2f}s ± {(metrics['max_time'] - metrics['min_time'])/2:.2f}s")
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        return 1


def config_cmd(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        if args.set:
            key, value = args.set
            try:
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
            except ValueError:
                pass
            
            config.set(key, value)
            print(f"Set {key} = {value}")
        
        elif args.get:
            value = config.get(args.get)
            print(f"{args.get} = {value}")
        
        elif args.list:
            all_config = config.list_all()
            print("Configuration settings:")
            print("-" * 30)
            for key, value in sorted(all_config.items()):
                print(f"{key}: {value}")
        
        elif args.reset:
            config.reset()
            print("Configuration reset to defaults")
        
        else:
            print("No action specified. Use --set, --get, --list, or --reset")
        
        return 0
        
    except Exception as e:
        logger.error(f"Configuration command failed: {e}")
        return 1




def _calculate_basic_statistics(molecules: List[Chem.Mol], graceful_exit: GracefulExit) -> Dict:
    """Calculate basic molecular statistics."""
    
    atom_counts = []
    heavy_atom_counts = []
    bond_counts = []
    ring_counts = []
    
    for mol in molecules:
        if graceful_exit.exit_now:
            break
        
        try:
            atom_counts.append(mol.GetNumAtoms())
            heavy_atom_counts.append(mol.GetNumHeavyAtoms())
            bond_counts.append(mol.GetNumBonds())
            ring_counts.append(Descriptors.RingCount(mol))
        except Exception:
            continue
    
    def _stats_dict(values):
        if not values:
            return {}
        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'median': sorted(values)[len(values) // 2]
        }
    
    return {
        'atom_counts': _stats_dict(atom_counts),
        'heavy_atom_counts': _stats_dict(heavy_atom_counts),
        'bond_counts': _stats_dict(bond_counts),
        'ring_counts': _stats_dict(ring_counts)
    }


def _calculate_descriptor_statistics(molecules: List[Chem.Mol], graceful_exit: GracefulExit) -> Dict:
    """Calculate descriptor statistics."""
    
    from rdkit.Chem import Crippen
    
    mw_values = []
    logp_values = []
    tpsa_values = []
    hbd_values = []
    hba_values = []
    
    for mol in molecules[:500]:  # Limit for performance
        if graceful_exit.exit_now:
            break
        
        try:
            mw_values.append(Descriptors.MolWt(mol))
            logp_values.append(Crippen.MolLogP(mol))
            tpsa_values.append(Descriptors.TPSA(mol))
            hbd_values.append(Descriptors.NumHDonors(mol))
            hba_values.append(Descriptors.NumHAcceptors(mol))
        except Exception:
            continue
    
    def _stats_dict(values):
        if not values:
            return {}
        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'median': sorted(values)[len(values) // 2]
        }
    
    return {
        'molecular_weight': _stats_dict(mw_values),
        'logp': _stats_dict(logp_values),
        'tpsa': _stats_dict(tpsa_values),
        'hbd': _stats_dict(hbd_values),
        'hba': _stats_dict(hba_values)
    }


def _diverse_sample(molecules: List[Chem.Mol], count: int, seed: int, graceful_exit: GracefulExit) -> List[Chem.Mol]:
    """Select diverse subset using MaxMin algorithm."""
    
    if len(molecules) <= count:
        return molecules
    
    try:
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        
        fingerprints = []
        valid_molecules = []
        
        for mol in molecules:
            if graceful_exit.exit_now:
                break
            
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fingerprints.append(fp)
                valid_molecules.append(mol)
            except Exception:
                continue
        
        if graceful_exit.exit_now or len(valid_molecules) <= count:
            return valid_molecules[:count]
        
        random.seed(seed)
        selected_indices = [random.randint(0, len(valid_molecules) - 1)]
        
        while len(selected_indices) < count and not graceful_exit.exit_now:
            max_min_sim = -1
            best_idx = -1
            
            for i, fp in enumerate(fingerprints):
                if i in selected_indices:
                    continue
                
                min_sim = min(
                    DataStructs.TanimotoSimilarity(fp, fingerprints[j])
                    for j in selected_indices
                )
                
                if min_sim > max_min_sim:
                    max_min_sim = min_sim
                    best_idx = i
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
            else:
                break
        
        return [valid_molecules[i] for i in selected_indices]
    
    except Exception:
        random.seed(seed)
        return random.sample(molecules, count)


def _benchmark_descriptors(molecules: List[Chem.Mol], n_jobs: int) -> None:
    """Benchmark descriptor calculation."""
    for mol in molecules:
        try:
            Descriptors.MolWt(mol)
            Descriptors.TPSA(mol)
            Descriptors.NumHDonors(mol)
        except Exception:
            continue


def _benchmark_fingerprints(molecules: List[Chem.Mol], n_jobs: int) -> None:
    """Benchmark fingerprint generation."""
    from rdkit.Chem import AllChem
    
    for mol in molecules:
        try:
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        except Exception:
            continue


def _benchmark_similarity(molecules: List[Chem.Mol], n_jobs: int) -> None:
    """Benchmark similarity calculation."""
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    
    if len(molecules) < 2:
        return
    
    fps = []
    for mol in molecules[:100]:  # Limit for benchmarking
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
        except Exception:
            continue
    
    for i in range(min(50, len(fps))):
        for j in range(i + 1, min(50, len(fps))):
            try:
                DataStructs.TanimotoSimilarity(fps[i], fps[j])
            except Exception:
                continue


def _benchmark_substructure(molecules: List[Chem.Mol], n_jobs: int) -> None:
    """Benchmark substructure search."""
    pattern = Chem.MolFromSmarts('c1ccccc1')  # Benzene ring
    
    for mol in molecules:
        try:
            mol.GetSubstructMatches(pattern)
        except Exception:
            continue