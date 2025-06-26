# rdkit_cli/commands/fragments.py
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS, Recap, rdMolDescriptors, AllChem
from rdkit.Chem.Fraggle import FraggleSim
from tqdm import tqdm

from ..core.common import (
    GracefulExit, get_parallel_jobs, read_molecules, validate_input_file,
    validate_output_path, write_molecules
)
from ..core.logging import log_success


def add_subparser(subparsers) -> None:
    parser_fragment = subparsers.add_parser(
        'fragment',
        help='Fragment molecules using various algorithms'
    )
    parser_fragment.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input molecular file'
    )
    parser_fragment.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with fragments'
    )
    parser_fragment.add_argument(
        '--method',
        choices=['brics', 'recap', 'mmpa', 'custom'],
        default='brics',
        help='Fragmentation method (default: brics)'
    )
    parser_fragment.add_argument(
        '--max-cuts',
        type=int,
        default=3,
        help='Maximum number of cuts per molecule (default: 3)'
    )
    parser_fragment.add_argument(
        '--min-fragment-size',
        type=int,
        default=3,
        help='Minimum fragment size in heavy atoms (default: 3)'
    )
    parser_fragment.add_argument(
        '--max-fragment-size',
        type=int,
        default=50,
        help='Maximum fragment size in heavy atoms (default: 50)'
    )
    parser_fragment.add_argument(
        '--include-parent',
        action='store_true',
        help='Include parent molecules in output'
    )

    parser_fragment_similarity = subparsers.add_parser(
        'fragment-similarity',
        help='Calculate fragment-based similarity between molecules'
    )
    parser_fragment_similarity.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input molecular file'
    )
    parser_fragment_similarity.add_argument(
        '--reference-frags',
        required=True,
        help='Reference fragment set file'
    )
    parser_fragment_similarity.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file with fragment similarity scores'
    )
    parser_fragment_similarity.add_argument(
        '--method',
        choices=['tanimoto', 'dice', 'overlap'],
        default='tanimoto',
        help='Similarity metric (default: tanimoto)'
    )

    parser_lead_optimization = subparsers.add_parser(
        'lead-optimization',
        help='Generate lead compound variations using fragment replacement'
    )
    parser_lead_optimization.add_argument(
        '-i', '--lead',
        required=True,
        help='Lead molecule file (single molecule)'
    )
    parser_lead_optimization.add_argument(
        '--fragment-library',
        required=True,
        help='Fragment library file'
    )
    parser_lead_optimization.add_argument(
        '--input-file',
        required=True,
        help='Input molecular file'
    )
    parser_lead_optimization.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with optimized compounds'
    )
    parser_lead_optimization.add_argument(
        '--max-products',
        type=int,
        default=100,
        help='Maximum number of replacement compounds to generate (default: 100)'
    )
    parser_lead_optimization.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.7,
        help='Minimum similarity to lead compound (default: 0.7)'
    )


def fragment_molecules(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Fragmenting molecules using {args.method} from {input_path}")
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        all_fragments = []
        fragment_counts = {}
        
        with tqdm(total=len(molecules), desc="Fragmenting molecules", ncols=80, colour='blue') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                    
                    if args.include_parent:
                        mol.SetProp("_Name", f"{mol_id}_parent")
                        all_fragments.append(mol)
                    
                    fragments = _fragment_molecule(mol, args.method, args.max_cuts, args.min_fragment_size, args.max_fragment_size)
                    
                    for j, frag in enumerate(fragments):
                        frag_smiles = Chem.MolToSmiles(frag)
                        fragment_counts[frag_smiles] = fragment_counts.get(frag_smiles, 0) + 1
                        
                        frag.SetProp("_Name", f"{mol_id}_frag_{j+1}")
                        frag.SetProp("Parent_ID", mol_id)
                        frag.SetProp("Fragment_Count", str(fragment_counts[frag_smiles]))
                        all_fragments.append(frag)
                
                except Exception as e:
                    logger.debug(f"Failed to fragment molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        write_molecules(all_fragments, output_path)
        unique_fragments = len(set(fragment_counts.keys()))
        log_success(f"Generated {len(all_fragments)} fragments ({unique_fragments} unique), saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Molecule fragmentation failed: {e}")
        return 1


def fragment_similarity(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        reference_path = validate_input_file(args.reference_frags)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Calculating fragment similarity using reference fragments from {reference_path}")
        
        reference_fragments = read_molecules(reference_path)
        reference_fragment_smiles = set()
        for frag in reference_fragments:
            if frag is not None:
                reference_fragment_smiles.add(Chem.MolToSmiles(frag))
        
        logger.info(f"Loaded {len(reference_fragment_smiles)} unique reference fragments")
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        results = []
        
        with tqdm(total=len(molecules), desc="Calculating fragment similarity", ncols=80, colour='green') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                    
                    mol_fragments = _fragment_molecule(mol, 'brics', 3, 3, 50)
                    mol_fragment_smiles = set()
                    for frag in mol_fragments:
                        mol_fragment_smiles.add(Chem.MolToSmiles(frag))
                    
                    similarity = _calculate_fragment_similarity(
                        mol_fragment_smiles, reference_fragment_smiles, args.method
                    )
                    
                    results.append({
                        'ID': mol_id,
                        'SMILES': Chem.MolToSmiles(mol),
                        'Fragment_Similarity': round(similarity, 4),
                        'Num_Fragments': len(mol_fragment_smiles),
                        'Common_Fragments': len(mol_fragment_smiles.intersection(reference_fragment_smiles))
                    })
                
                except Exception as e:
                    logger.debug(f"Failed to process molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        results.sort(key=lambda x: x['Fragment_Similarity'], reverse=True)
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        log_success(f"Calculated fragment similarity for {len(results)} molecules, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Fragment similarity calculation failed: {e}")
        return 1


def lead_optimization(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        lead_path = validate_input_file(args.input_file)
        fragments_path = validate_input_file(args.fragment_library)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Optimizing lead compound using fragments from {fragments_path}")
        
        lead_molecules = read_molecules(lead_path)
        if not lead_molecules:
            logger.error("No lead molecule found")
            return 1
        
        lead_mol = lead_molecules[0]
        logger.info(f"Lead molecule: {Chem.MolToSmiles(lead_mol)}")
        
        fragment_library = read_molecules(fragments_path)
        logger.info(f"Loaded {len(fragment_library)} fragments")
        
        optimized_compounds = []
        generated_count = 0
        
        lead_fragments = _fragment_molecule(lead_mol, 'brics', 3, 3, 50)
        
        with tqdm(total=min(args.max_products, len(fragment_library) * len(lead_fragments)), 
                  desc="Generating compounds", ncols=80, colour='cyan') as pbar:
            
            for lead_frag in lead_fragments:
                if graceful_exit.exit_now or generated_count >= args.max_products:
                    break
                
                for replacement_frag in fragment_library:
                    if graceful_exit.exit_now or generated_count >= args.max_products:
                        break
                    
                    try:
                        if replacement_frag is None:
                            continue
                        
                        new_compound = _replace_fragment(lead_mol, lead_frag, replacement_frag)
                        
                        if new_compound is not None:
                            similarity = _calculate_molecule_similarity(lead_mol, new_compound)
                            
                            if similarity >= args.similarity_threshold:
                                new_compound.SetProp("_Name", f"optimized_{generated_count+1}")
                                new_compound.SetProp("Lead_Similarity", str(round(similarity, 4)))
                                optimized_compounds.append(new_compound)
                                generated_count += 1
                    
                    except Exception as e:
                        logger.debug(f"Failed to generate compound: {e}")
                    
                    pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        write_molecules(optimized_compounds, output_path)
        log_success(f"Generated {len(optimized_compounds)} optimized compounds, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Lead optimization failed: {e}")
        return 1


def _fragment_molecule(mol: Chem.Mol, method: str, max_cuts: int, min_frag_size: int, max_frag_size: int = 50) -> List[Chem.Mol]:
    """Fragment a molecule using the specified method."""
    fragments = []
    
    try:
        if method == 'brics':
            frag_smiles = BRICS.BRICSDecompose(mol, maxCuts=max_cuts, returnMols=False)
            for smiles in frag_smiles:
                clean_smiles = smiles.replace('[*]', '')
                if clean_smiles:
                    frag_mol = Chem.MolFromSmiles(clean_smiles)
                    if frag_mol is not None and min_frag_size <= frag_mol.GetNumHeavyAtoms() <= max_frag_size:
                        fragments.append(frag_mol)
        
        elif method == 'recap':
            recap_tree = Recap.RecapDecompose(mol)
            leaves = recap_tree.GetLeaves()
            for smiles in leaves.keys():
                clean_smiles = smiles.replace('[*]', '')
                if clean_smiles:
                    frag_mol = Chem.MolFromSmiles(clean_smiles)
                    if frag_mol is not None and min_frag_size <= frag_mol.GetNumHeavyAtoms() <= max_frag_size:
                        fragments.append(frag_mol)
        
        elif method == 'mmpa':
            fragments = _mmpa_fragment(mol, max_cuts, min_frag_size, max_frag_size)
        
        elif method == 'custom':
            fragments = _custom_fragment(mol, max_cuts, min_frag_size, max_frag_size)
    
    except Exception:
        pass
    
    return fragments


def _mmpa_fragment(mol: Chem.Mol, max_cuts: int, min_frag_size: int, max_frag_size: int) -> List[Chem.Mol]:
    """Fragment molecule using matched molecular pairs analysis approach."""
    fragments = []
    
    try:
        for i in range(mol.GetNumBonds()):
            if len(fragments) >= max_cuts:
                break
            
            bond = mol.GetBondWithIdx(i)
            if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing():
                atom1_idx = bond.GetBeginAtomIdx()
                atom2_idx = bond.GetEndAtomIdx()
                
                frag_mol = Chem.FragmentOnBonds(mol, [i])
                if frag_mol is not None:
                    frag_smiles = Chem.MolToSmiles(frag_mol)
                    frag_parts = frag_smiles.split('.')
                    
                    for part in frag_parts:
                        clean_part = part.replace('[*]', '')
                        if clean_part:
                            part_mol = Chem.MolFromSmiles(clean_part)
                            if part_mol is not None and min_frag_size <= part_mol.GetNumHeavyAtoms() <= max_frag_size:
                                fragments.append(part_mol)
    
    except Exception:
        pass
    
    return fragments


def _custom_fragment(mol: Chem.Mol, max_cuts: int, min_frag_size: int, max_frag_size: int) -> List[Chem.Mol]:
    """Fragment molecule using custom rules."""
    fragments = []
    
    try:
        pattern_names = [
            'Ester', 'Amide', 'Ether', 'Amine', 'Aromatic_Ring'
        ]
        
        patterns = {
            'Ester': Chem.MolFromSmarts('[C](=O)[O][C]'),
            'Amide': Chem.MolFromSmarts('[C](=O)[N]'),
            'Ether': Chem.MolFromSmarts('[C][O][C]'),
            'Amine': Chem.MolFromSmarts('[C][N]'),
            'Aromatic_Ring': Chem.MolFromSmarts('c1ccccc1')
        }
        
        for pattern_name, pattern in patterns.items():
            if pattern is not None:
                matches = mol.GetSubstructMatches(pattern)
                for match in matches[:max_cuts]:
                    frag_mol = Chem.DeleteSubstructs(mol, pattern)
                    if frag_mol is not None and min_frag_size <= frag_mol.GetNumHeavyAtoms() <= max_frag_size:
                        fragments.append(frag_mol)
    
    except Exception:
        pass
    
    return fragments


def _calculate_fragment_similarity(frags1: Set[str], frags2: Set[str], method: str) -> float:
    """Calculate similarity between two fragment sets."""
    if not frags1 or not frags2:
        return 0.0
    
    intersection = len(frags1.intersection(frags2))
    
    if method == 'tanimoto':
        union = len(frags1.union(frags2))
        return intersection / union if union > 0 else 0.0
    
    elif method == 'dice':
        return (2.0 * intersection) / (len(frags1) + len(frags2))
    
    elif method == 'overlap':
        smaller_set_size = min(len(frags1), len(frags2))
        return intersection / smaller_set_size if smaller_set_size > 0 else 0.0
    
    return 0.0


def _replace_fragment(parent_mol: Chem.Mol, old_frag: Chem.Mol, new_frag: Chem.Mol) -> Optional[Chem.Mol]:
    """Replace a fragment in the parent molecule with a new fragment."""
    try:
        old_frag_smiles = Chem.MolToSmiles(old_frag)
        new_frag_smiles = Chem.MolToSmiles(new_frag)
        
        parent_smiles = Chem.MolToSmiles(parent_mol)
        
        if old_frag_smiles in parent_smiles:
            new_smiles = parent_smiles.replace(old_frag_smiles, new_frag_smiles, 1)
            new_mol = Chem.MolFromSmiles(new_smiles)
            
            if new_mol is not None:
                Chem.SanitizeMol(new_mol)
                return new_mol
    
    except Exception:
        pass
    
    return None


def _calculate_molecule_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """Calculate Tanimoto similarity between two molecules."""
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        from rdkit import DataStructs
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    except Exception:
        return 0.0