# rdkit_cli/commands/specialized.py
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..core.common import (
    GracefulExit, get_parallel_jobs, read_molecules, validate_input_file,
    validate_output_path, get_molecules_from_args, save_dataframe_with_format_detection
)
from ..core.logging import log_success


def add_subparser(subparsers) -> None:
    parser_toxicity_alerts = subparsers.add_parser(
        'toxicity-alerts',
        help='Screen molecules for structural toxicity alerts'
    )
    parser_toxicity_alerts.add_argument(
        '-i', '--input-file',
        help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_toxicity_alerts.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_toxicity_alerts.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_toxicity_alerts.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file with toxicity alerts'
    )
    parser_toxicity_alerts.add_argument(
        '--alert-set',
        choices=['brenk', 'pains', 'nih', 'custom'],
        default='brenk',
        help='Toxicity alert set to use (default: brenk)'
    )
    parser_toxicity_alerts.add_argument(
        '--custom-alerts',
        help='Custom alerts file (SMARTS patterns, one per line)'
    )

    parser_matched_pairs = subparsers.add_parser(
        'matched-pairs',
        help='Identify matched molecular pairs for SAR analysis'
    )
    parser_matched_pairs.add_argument(
        '-i', '--input-file',
        help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_matched_pairs.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_matched_pairs.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_matched_pairs.add_argument(
        '--activity-file',
        required=True,
        help='CSV file with molecular activities'
    )
    parser_matched_pairs.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file with matched pairs'
    )
    parser_matched_pairs.add_argument(
        '--max-pairs',
        type=int,
        default=1000,
        help='Maximum number of pairs to identify (default: 1000)'
    )

    parser_sar_analysis = subparsers.add_parser(
        'sar-analysis',
        help='Perform structure-activity relationship analysis'
    )
    parser_sar_analysis.add_argument(
        '-i', '--input-file',
        help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_sar_analysis.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_sar_analysis.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_sar_analysis.add_argument(
        '--activity-file',
        required=True,
        help='CSV file with molecular activities'
    )
    parser_sar_analysis.add_argument(
        '-o', '--output',
        required=True,
        help='Output HTML report file'
    )
    parser_sar_analysis.add_argument(
        '--activity-column',
        default='activity',
        help='Name of activity column (default: activity)'
    )

    parser_free_wilson = subparsers.add_parser(
        'free-wilson',
        help='Perform Free-Wilson analysis'
    )
    parser_free_wilson.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input molecular file'
    )
    parser_free_wilson.add_argument(
        '--activity-file',
        required=True,
        help='CSV file with molecular activities'
    )
    parser_free_wilson.add_argument(
        '-o', '--output',
        required=True,
        help='Output JSON file with Free-Wilson analysis'
    )
    parser_free_wilson.add_argument(
        '--activity-column',
        default='activity',
        help='Name of activity column (default: activity)'
    )

    parser_qsar_descriptors = subparsers.add_parser(
        'qsar-descriptors',
        help='Calculate comprehensive QSAR descriptors'
    )
    parser_qsar_descriptors.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input molecular file'
    )
    parser_qsar_descriptors.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file with QSAR descriptors'
    )
    parser_qsar_descriptors.add_argument(
        '--include-3d',
        action='store_true',
        help='Include 3D descriptors (requires conformers)'
    )
    parser_qsar_descriptors.add_argument(
        '--descriptor-set',
        choices=['all', 'constitutional', 'topological', 'geometric', 'electronic'],
        default='all',
        help='Descriptor set to calculate (default: all)'
    )


def toxicity_alerts(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Screening for {args.alert_set} toxicity alerts from {input_path}")
        
        alert_patterns = _get_toxicity_alerts(args.alert_set, args.custom_alerts)
        if not alert_patterns:
            logger.error("No toxicity alert patterns loaded")
            return 1
        
        logger.info(f"Using {len(alert_patterns)} alert patterns")
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        results = []
        
        with tqdm(total=len(molecules), desc="Screening toxicity alerts", ncols=80, colour='red') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                    smiles = Chem.MolToSmiles(mol)
                    
                    alerts_found = []
                    for alert_name, pattern in alert_patterns.items():
                        if mol.HasSubstructMatch(pattern):
                            alerts_found.append(alert_name)
                    
                    results.append({
                        'ID': mol_id,
                        'SMILES': smiles,
                        'Alert_Count': len(alerts_found),
                        'Alerts': ';'.join(alerts_found) if alerts_found else '',
                        'Clean': len(alerts_found) == 0
                    })
                
                except Exception as e:
                    logger.debug(f"Failed to process molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        clean_molecules = sum(1 for r in results if r['Clean'])
        log_success(f"Screened {len(results)} molecules, {clean_molecules} clean, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Toxicity alert screening failed: {e}")
        return 1


def matched_pairs(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        activity_path = validate_input_file(args.activity_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Identifying matched molecular pairs from {input_path}")
        
        molecules = read_molecules(input_path)
        activity_df = pd.read_csv(activity_path)
        
        logger.info(f"Loaded {len(molecules)} molecules and {len(activity_df)} activity records")
        
        mol_dict = {}
        for mol in molecules:
            if mol is not None:
                mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                if mol_id:
                    mol_dict[mol_id] = mol
        
        activity_dict = {}
        if 'ID' in activity_df.columns:
            for _, row in activity_df.iterrows():
                activity_dict[row['ID']] = row.to_dict()
        
        pairs = []
        mol_items = list(mol_dict.items())
        
        with tqdm(total=min(args.max_pairs, len(mol_items) * (len(mol_items) - 1) // 2), 
                  desc="Finding matched pairs", ncols=80, colour='cyan') as pbar:
            
            pair_count = 0
            for i, (id1, mol1) in enumerate(mol_items):
                if graceful_exit.exit_now or pair_count >= args.max_pairs:
                    break
                
                for j, (id2, mol2) in enumerate(mol_items[i+1:], i+1):
                    if graceful_exit.exit_now or pair_count >= args.max_pairs:
                        break
                    
                    try:
                        similarity = _calculate_structural_similarity(mol1, mol2)
                        
                        if similarity > 0.8:  # High similarity threshold for matched pairs
                            pair_data = {
                                'Mol1_ID': id1,
                                'Mol2_ID': id2,
                                'Mol1_SMILES': Chem.MolToSmiles(mol1),
                                'Mol2_SMILES': Chem.MolToSmiles(mol2),
                                'Similarity': round(similarity, 4)
                            }
                            
                            if id1 in activity_dict:
                                pair_data['Mol1_Activity'] = activity_dict[id1].get('activity', '')
                            if id2 in activity_dict:
                                pair_data['Mol2_Activity'] = activity_dict[id2].get('activity', '')
                            
                            if 'Mol1_Activity' in pair_data and 'Mol2_Activity' in pair_data:
                                try:
                                    act1 = float(pair_data['Mol1_Activity'])
                                    act2 = float(pair_data['Mol2_Activity'])
                                    pair_data['Activity_Difference'] = round(abs(act1 - act2), 4)
                                except (ValueError, TypeError):
                                    pass
                            
                            pairs.append(pair_data)
                            pair_count += 1
                            pbar.update(1)
                    
                    except Exception:
                        continue
        
        if graceful_exit.exit_now:
            return 130
        
        if pairs:
            pairs.sort(key=lambda x: x['Similarity'], reverse=True)
            df = pd.DataFrame(pairs)
            df.to_csv(output_path, index=False)
        
        log_success(f"Identified {len(pairs)} matched pairs, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Matched pairs analysis failed: {e}")
        return 1


def sar_analysis(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        activity_path = validate_input_file(args.activity_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Performing SAR analysis from {input_path}")
        
        molecules = read_molecules(input_path)
        activity_df = pd.read_csv(activity_path)
        
        logger.info(f"Loaded {len(molecules)} molecules and {len(activity_df)} activity records")
        
        html_report = _generate_sar_report(molecules, activity_df, args.activity_column, graceful_exit)
        
        if graceful_exit.exit_now:
            return 130
        
        with open(output_path, 'w') as f:
            f.write(html_report)
        
        log_success(f"Generated SAR analysis report, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"SAR analysis failed: {e}")
        return 1


def free_wilson(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        activity_path = validate_input_file(args.activity_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Performing Free-Wilson analysis from {input_path}")
        
        molecules = read_molecules(input_path)
        activity_df = pd.read_csv(activity_path)
        
        logger.info(f"Loaded {len(molecules)} molecules and {len(activity_df)} activity records")
        
        fw_results = _perform_free_wilson_analysis(molecules, activity_df, args.activity_column, graceful_exit)
        
        if graceful_exit.exit_now:
            return 130
        
        with open(output_path, 'w') as f:
            json.dump(fw_results, f, indent=2)
        
        log_success(f"Completed Free-Wilson analysis, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Free-Wilson analysis failed: {e}")
        return 1


def qsar_descriptors(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Calculating QSAR descriptors from {input_path}")
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        descriptor_functions = _get_qsar_descriptors(args.descriptor_set, args.include_3d)
        
        data = []
        
        with tqdm(total=len(molecules), desc="Calculating QSAR descriptors", ncols=80, colour='blue') as pbar:
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
                    
                    if args.include_3d and mol.GetNumConformers() == 0:
                        try:
                            AllChem.EmbedMolecule(mol)
                            AllChem.UFFOptimizeMol(mol)
                        except Exception:
                            pass
                    
                    for desc_name, desc_func in descriptor_functions.items():
                        try:
                            value = desc_func(mol)
                            row[desc_name] = value
                        except Exception:
                            row[desc_name] = None
                    
                    data.append(row)
                
                except Exception as e:
                    logger.debug(f"Failed to process molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        n_descriptors = len(df.columns) - 2  # subtract ID and SMILES
        log_success(f"Calculated {n_descriptors} QSAR descriptors for {len(data)} molecules, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"QSAR descriptor calculation failed: {e}")
        return 1


def _get_toxicity_alerts(alert_set: str, custom_file: Optional[str]) -> Dict[str, Chem.Mol]:
    """Get toxicity alert SMARTS patterns."""
    alerts = {}
    
    if alert_set == 'brenk':
        brenk_alerts = {
            'Alkyl_halide': '[CX4][F,Cl,Br,I]',
            'Aldehyde': '[CH1](=O)',
            'Quaternary_N': '[N+]([C])([C])([C])[C]',
            'Nitro': '[N+](=O)[O-]',
            'Michael_acceptor': '[CX3]=[CX3][CX3]=O',
            'Epoxide': 'C1OC1',
            'Aziridine': 'C1NC1',
            'Thiocarbonyl': '[#6]=[#16]',
            'Crown_ether': '[#8]~[#6]~[#6]~[#8]~[#6]~[#6]~[#8]',
            'Hydrazine': '[NX3][NX3]',
            'Heavy_metals': '[Hg,Pb,As,Cd]',
        }
        alerts.update(brenk_alerts)
    
    elif alert_set == 'pains':
        pains_alerts = {
            'Quinone': 'C1=CC(=O)C=CC1=O',
            'Catechol': 'c1ccc(O)c(O)c1',
            'Rhodanine': 'S1C(=S)NC(=O)C1',
            'Mannich_base': '[#6][CH2][#7]([#6])[#6]',
            'Alkyl_aniline': 'Nc1ccc([#6])cc1',
            'Phenolic_Mannich': 'Oc1ccc(C[#7])cc1',
        }
        alerts.update(pains_alerts)
    
    elif alert_set == 'nih':
        nih_alerts = {
            'Reactive_alkyl_halide': '[CH2X4][Cl,Br,I]',
            'Acid_anhydride': '[CX3](=[OX1])[OX2][CX3](=[OX1])',
            'Isocyanate': '[NX2]=[CX2]=[OX1]',
            'Thiourea': 'SC(=S)N',
            'Alpha_halo_carbonyl': '[CX3](=[OX1])[CX4][F,Cl,Br,I]',
        }
        alerts.update(nih_alerts)
    
    elif alert_set == 'custom' and custom_file:
        try:
            with open(custom_file, 'r') as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        mol = Chem.MolFromSmarts(line)
                        if mol is not None:
                            alerts[f'Custom_{i}'] = mol
        except Exception:
            pass
    
    compiled_alerts = {}
    for name, smarts in alerts.items():
        if isinstance(smarts, str):
            mol = Chem.MolFromSmarts(smarts)
            if mol is not None:
                compiled_alerts[name] = mol
        else:
            compiled_alerts[name] = smarts
    
    return compiled_alerts


def _calculate_structural_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """Calculate structural similarity between two molecules."""
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        from rdkit import DataStructs
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def _generate_sar_report(molecules: List[Chem.Mol], activity_df: pd.DataFrame, 
                        activity_column: str, graceful_exit: GracefulExit) -> str:
    """Generate SAR analysis HTML report."""
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Structure-Activity Relationship Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
            .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .molecule { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Structure-Activity Relationship Analysis</h1>
            <p>Comprehensive SAR analysis report</p>
        </div>
    """
    
    activity_dict = {}
    if 'ID' in activity_df.columns and activity_column in activity_df.columns:
        for _, row in activity_df.iterrows():
            try:
                activity_dict[row['ID']] = float(row[activity_column])
            except (ValueError, TypeError):
                pass
    
    mol_activities = []
    for mol in molecules:
        if mol is not None:
            mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            if mol_id in activity_dict:
                mol_activities.append((mol, activity_dict[mol_id]))
    
    if mol_activities:
        mol_activities.sort(key=lambda x: x[1], reverse=True)
        
        html += f"""
        <div class="section">
            <h2>Activity Summary</h2>
            <p><strong>Total molecules with activity:</strong> {len(mol_activities)}</p>
            <p><strong>Activity range:</strong> {min(act for _, act in mol_activities):.3f} - {max(act for _, act in mol_activities):.3f}</p>
            <p><strong>Mean activity:</strong> {np.mean([act for _, act in mol_activities]):.3f}</p>
        </div>
        """
        
        html += """
        <div class="section">
            <h2>Top 10 Most Active Compounds</h2>
            <table>
                <tr><th>Rank</th><th>ID</th><th>SMILES</th><th>Activity</th></tr>
        """
        
        for i, (mol, activity) in enumerate(mol_activities[:10], 1):
            mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
            smiles = Chem.MolToSmiles(mol)
            html += f"<tr><td>{i}</td><td>{mol_id}</td><td>{smiles}</td><td>{activity:.3f}</td></tr>"
        
        html += "</table></div>"
    
    html += "</body></html>"
    return html


def _perform_free_wilson_analysis(molecules: List[Chem.Mol], activity_df: pd.DataFrame, 
                                 activity_column: str, graceful_exit: GracefulExit) -> Dict:
    """Perform Free-Wilson analysis."""
    
    activity_dict = {}
    if 'ID' in activity_df.columns and activity_column in activity_df.columns:
        for _, row in activity_df.iterrows():
            try:
                activity_dict[row['ID']] = float(row[activity_column])
            except (ValueError, TypeError):
                pass
    
    fragment_activities = {}
    fragment_counts = {}
    
    for mol in molecules:
        if mol is not None:
            mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            if mol_id in activity_dict:
                activity = activity_dict[mol_id]
                
                try:
                    from rdkit.Chem import BRICS
                    fragments = BRICS.BRICSDecompose(mol, returnMols=False)
                    
                    for frag in fragments:
                        clean_frag = frag.replace('[*]', '')
                        if clean_frag:
                            if clean_frag not in fragment_activities:
                                fragment_activities[clean_frag] = []
                                fragment_counts[clean_frag] = 0
                            
                            fragment_activities[clean_frag].append(activity)
                            fragment_counts[clean_frag] += 1
                
                except Exception:
                    continue
    
    fragment_contributions = {}
    for frag, activities in fragment_activities.items():
        if len(activities) >= 2:  # Require at least 2 occurrences
            fragment_contributions[frag] = {
                'mean_activity': np.mean(activities),
                'std_activity': np.std(activities),
                'count': len(activities),
                'contribution': np.mean(activities) - np.mean(list(activity_dict.values()))
            }
    
    sorted_fragments = sorted(fragment_contributions.items(), 
                            key=lambda x: x[1]['contribution'], reverse=True)
    
    results = {
        'total_molecules': len([mol for mol in molecules if mol is not None]),
        'molecules_with_activity': len(activity_dict),
        'total_fragments': len(fragment_activities),
        'significant_fragments': len(fragment_contributions),
        'mean_activity': np.mean(list(activity_dict.values())) if activity_dict else 0,
        'top_positive_fragments': dict(sorted_fragments[:10]),
        'top_negative_fragments': dict(sorted_fragments[-10:])
    }
    
    return results


def _get_qsar_descriptors(descriptor_set: str, include_3d: bool) -> Dict[str, callable]:
    """Get QSAR descriptor functions."""
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    
    descriptors = {}
    
    if descriptor_set in ['all', 'constitutional']:
        constitutional = {
            'MolWt': Descriptors.MolWt,
            'HeavyAtomMolWt': Descriptors.HeavyAtomMolWt,
            'ExactMolWt': Descriptors.ExactMolWt,
            'NumHeavyAtoms': Descriptors.HeavyAtomCount,
            'NumAtoms': lambda mol: mol.GetNumAtoms(),
            'NumHeteroatoms': Descriptors.NumHeteroatoms,
            'FractionCsp3': Descriptors.FractionCSP3,
            'NumRotatableBonds': Descriptors.NumRotatableBonds,
            'NumHBD': Descriptors.NumHDonors,
            'NumHBA': Descriptors.NumHAcceptors,
        }
        descriptors.update(constitutional)
    
    if descriptor_set in ['all', 'topological']:
        topological = {
            'BertzCT': Descriptors.BertzCT,
            'BalabanJ': Descriptors.BalabanJ,
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
        }
        descriptors.update(topological)
    
    if descriptor_set in ['all', 'geometric']:
        geometric = {
            'TPSA': Descriptors.TPSA,
            'LabuteASA': Descriptors.LabuteASA,
        }
        descriptors.update(geometric)
        
        if include_3d:
            geometric_3d = {
                'PMI1': rdMolDescriptors.PMI1,
                'PMI2': rdMolDescriptors.PMI2,
                'PMI3': rdMolDescriptors.PMI3,
                'NPR1': rdMolDescriptors.NPR1,
                'NPR2': rdMolDescriptors.NPR2,
                'RadiusOfGyration': rdMolDescriptors.RadiusOfGyration,
                'InertialShapeFactor': rdMolDescriptors.InertialShapeFactor,
                'Eccentricity': rdMolDescriptors.Eccentricity,
                'Asphericity': rdMolDescriptors.Asphericity,
                'SpherocityIndex': rdMolDescriptors.SpherocityIndex,
            }
            descriptors.update(geometric_3d)
    
    if descriptor_set in ['all', 'electronic']:
        electronic = {
            'LogP': Crippen.MolLogP,
            'MR': Crippen.MolMR,
            'MaxPartialCharge': Descriptors.MaxPartialCharge,
            'MinPartialCharge': Descriptors.MinPartialCharge,
            'MaxAbsPartialCharge': Descriptors.MaxAbsPartialCharge,
            'MinAbsPartialCharge': Descriptors.MinAbsPartialCharge,
        }
        descriptors.update(electronic)
    
    return descriptors