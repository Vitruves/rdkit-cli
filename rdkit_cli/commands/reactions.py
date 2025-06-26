# rdkit_cli/commands/reactions.py
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions
from tqdm import tqdm

from ..core.common import (
    GracefulExit, get_parallel_jobs, read_molecules, validate_input_file,
    validate_output_path, write_molecules, get_molecules_from_args, save_dataframe_with_format_detection
)
from ..core.logging import log_success


def add_subparser(subparsers) -> None:
    parser_reaction_search = subparsers.add_parser(
        'reaction-search',
        help='Search for reactions matching a query pattern'
    )
    parser_reaction_search.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input reaction file (RDF format)'
    )
    parser_reaction_search.add_argument(
        '--query-smarts',
        required=True,
        help='Query reaction SMARTS pattern'
    )
    parser_reaction_search.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with matching reactions'
    )

    parser_reaction_apply = subparsers.add_parser(
        'reaction-apply',
        help='Apply reaction transformations to substrate molecules'
    )
    parser_reaction_apply.add_argument(
        '-i', '--input-file',
        help='Input substrate molecules file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_reaction_apply.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_reaction_apply.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_reaction_apply.add_argument(
        '--reaction-smarts',
        required=True,
        help='Reaction SMARTS pattern to apply'
    )
    parser_reaction_apply.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with reaction products'
    )
    parser_reaction_apply.add_argument(
        '--max-products',
        type=int,
        default=100,
        help='Maximum products per substrate (default: 100)'
    )

    parser_reaction_enumerate = subparsers.add_parser(
        'reaction-enumerate',
        help='Enumerate combinatorial libraries using reactions'
    )
    parser_reaction_enumerate.add_argument(
        '-i', '--building-blocks',
        help='Building blocks file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_reaction_enumerate.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_reaction_enumerate.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_reaction_enumerate.add_argument(
        '--reactions',
        required=True,
        help='Reaction database file'
    )
    parser_reaction_enumerate.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with enumerated library'
    )
    parser_reaction_enumerate.add_argument(
        '--max-compounds',
        type=int,
        default=10000,
        help='Maximum compounds to generate (default: 10000)'
    )

    parser_retrosynthesis = subparsers.add_parser(
        'retrosynthesis',
        help='Perform retrosynthetic analysis'
    )
    parser_retrosynthesis.add_argument(
        '-i', '--target',
        help='Target molecule file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_retrosynthesis.add_argument(
        '-S', '--smiles',
        help='Direct target SMILES string'
    )
    parser_retrosynthesis.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_retrosynthesis.add_argument(
        '--reaction-db',
        required=True,
        help='Reaction database file'
    )
    parser_retrosynthesis.add_argument(
        '-o', '--output',
        required=True,
        help='Output JSON file with synthetic routes'
    )
    parser_retrosynthesis.add_argument(
        '--max-depth',
        type=int,
        default=3,
        help='Maximum retrosynthetic depth (default: 3)'
    )


def search(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Searching for reactions matching '{args.query_smarts}' in {input_path}")
        
        query_reaction = AllChem.ReactionFromSmarts(args.query_smarts)
        if query_reaction is None:
            logger.error(f"Invalid reaction SMARTS: {args.query_smarts}")
            return 1
        
        reactions = _read_reactions(input_path)
        logger.info(f"Loaded {len(reactions)} reactions")
        
        matching_reactions = []
        
        with tqdm(total=len(reactions), desc="Searching reactions", ncols=80, colour='blue') as pbar:
            for reaction in reactions:
                if graceful_exit.exit_now:
                    break
                
                try:
                    if reaction is not None and _reaction_matches_query(reaction, query_reaction):
                        matching_reactions.append(reaction)
                
                except Exception as e:
                    logger.debug(f"Failed to process reaction: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        _write_reactions(matching_reactions, output_path)
        log_success(f"Found {len(matching_reactions)} matching reactions, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Reaction search failed: {e}")
        return 1


def apply(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Applying reaction '{args.reaction_smarts}' to substrates from {input_path}")
        
        reaction = AllChem.ReactionFromSmarts(args.reaction_smarts)
        if reaction is None:
            logger.error(f"Invalid reaction SMARTS: {args.reaction_smarts}")
            return 1
        
        substrates = read_molecules(input_path)
        logger.info(f"Loaded {len(substrates)} substrate molecules")
        
        products = []
        
        with tqdm(total=len(substrates), desc="Applying reactions", ncols=80, colour='green') as pbar:
            for i, substrate in enumerate(substrates):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if substrate is None:
                        continue
                    
                    substrate_id = substrate.GetProp("_Name") if substrate.HasProp("_Name") else f"substrate_{i+1}"
                    
                    reaction_products = reaction.RunReactants((substrate,))
                    
                    for j, product_set in enumerate(reaction_products[:args.max_products]):
                        for k, product in enumerate(product_set):
                            try:
                                Chem.SanitizeMol(product)
                                product.SetProp("_Name", f"{substrate_id}_product_{j+1}_{k+1}")
                                product.SetProp("Substrate_ID", substrate_id)
                                products.append(product)
                            except Exception:
                                pass
                
                except Exception as e:
                    logger.debug(f"Failed to process substrate {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        write_molecules(products, output_path)
        log_success(f"Generated {len(products)} reaction products, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Reaction application failed: {e}")
        return 1


def enumerate(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        building_blocks_path = validate_input_file(args.building_blocks)
        reactions_path = validate_input_file(args.reactions)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Enumerating library using building blocks from {building_blocks_path}")
        
        building_blocks = read_molecules(building_blocks_path)
        reactions = _read_reactions(reactions_path)
        
        logger.info(f"Loaded {len(building_blocks)} building blocks and {len(reactions)} reactions")
        
        library_compounds = []
        compound_count = 0
        
        with tqdm(total=min(args.max_compounds, len(building_blocks) * len(reactions)), 
                  desc="Enumerating library", ncols=80, colour='cyan') as pbar:
            
            for reaction in reactions:
                if graceful_exit.exit_now or compound_count >= args.max_compounds:
                    break
                
                try:
                    if reaction is None:
                        continue
                    
                    num_reactants = reaction.GetNumReactantTemplates()
                    
                    if num_reactants == 1:
                        for bb in building_blocks:
                            if graceful_exit.exit_now or compound_count >= args.max_compounds:
                                break
                            
                            try:
                                products = reaction.RunReactants((bb,))
                                for product_set in products:
                                    for product in product_set:
                                        try:
                                            Chem.SanitizeMol(product)
                                            product.SetProp("_Name", f"compound_{compound_count+1}")
                                            library_compounds.append(product)
                                            compound_count += 1
                                            pbar.update(1)
                                            if compound_count >= args.max_compounds:
                                                break
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                    
                    elif num_reactants == 2:
                        for bb1 in building_blocks:
                            if graceful_exit.exit_now or compound_count >= args.max_compounds:
                                break
                            for bb2 in building_blocks:
                                if graceful_exit.exit_now or compound_count >= args.max_compounds:
                                    break
                                
                                try:
                                    products = reaction.RunReactants((bb1, bb2))
                                    for product_set in products:
                                        for product in product_set:
                                            try:
                                                Chem.SanitizeMol(product)
                                                product.SetProp("_Name", f"compound_{compound_count+1}")
                                                library_compounds.append(product)
                                                compound_count += 1
                                                pbar.update(1)
                                                if compound_count >= args.max_compounds:
                                                    break
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                
                except Exception as e:
                    logger.debug(f"Failed to process reaction: {e}")
        
        if graceful_exit.exit_now:
            return 130
        
        write_molecules(library_compounds, output_path)
        log_success(f"Enumerated {len(library_compounds)} library compounds, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Library enumeration failed: {e}")
        return 1


def retrosynthesis(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        target_path = validate_input_file(args.target)
        reaction_db_path = validate_input_file(args.reaction_db)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Performing retrosynthetic analysis for target from {target_path}")
        
        target_molecules = read_molecules(target_path)
        if not target_molecules:
            logger.error("No target molecule found")
            return 1
        
        target = target_molecules[0]
        logger.info(f"Target molecule: {Chem.MolToSmiles(target)}")
        
        reactions = _read_reactions(reaction_db_path)
        logger.info(f"Loaded {len(reactions)} retrosynthetic reactions")
        
        synthetic_routes = []
        
        with tqdm(total=args.max_depth, desc="Analyzing retrosynthesis", ncols=80, colour='magenta') as pbar:
            routes = _find_synthetic_routes(target, reactions, args.max_depth, graceful_exit)
            synthetic_routes.extend(routes)
            pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        result = {
            'target_smiles': Chem.MolToSmiles(target),
            'num_routes': len(synthetic_routes),
            'max_depth': args.max_depth,
            'routes': synthetic_routes
        }
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        log_success(f"Found {len(synthetic_routes)} synthetic routes, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Retrosynthetic analysis failed: {e}")
        return 1


def _read_reactions(file_path: Path) -> List[rdChemReactions.ChemicalReaction]:
    """Read reactions from RDF file."""
    reactions = []
    
    try:
        if file_path.suffix.lower() == '.rdf':
            supplier = rdChemReactions.ReactionMolSupplier(str(file_path))
            for reaction in supplier:
                if reaction is not None:
                    reactions.append(reaction)
        else:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        reaction = AllChem.ReactionFromSmarts(line)
                        if reaction is not None:
                            reactions.append(reaction)
    
    except Exception:
        pass
    
    return reactions


def _write_reactions(reactions: List[rdChemReactions.ChemicalReaction], file_path: Path) -> None:
    """Write reactions to file."""
    try:
        if file_path.suffix.lower() == '.rdf':
            writer = rdChemReactions.ReactionMolWriter(str(file_path))
            for reaction in reactions:
                writer.write(reaction)
            writer.close()
        else:
            with open(file_path, 'w') as f:
                for i, reaction in enumerate(reactions):
                    smarts = rdChemReactions.ReactionToSmarts(reaction)
                    f.write(f"{smarts}\n")
    
    except Exception:
        pass


def _reaction_matches_query(reaction: rdChemReactions.ChemicalReaction, 
                           query_reaction: rdChemReactions.ChemicalReaction) -> bool:
    """Check if reaction matches query pattern."""
    try:
        reaction_smarts = rdChemReactions.ReactionToSmarts(reaction)
        query_smarts = rdChemReactions.ReactionToSmarts(query_reaction)
        
        return query_smarts in reaction_smarts
    
    except Exception:
        return False


def _find_synthetic_routes(target: Chem.Mol, reactions: List[rdChemReactions.ChemicalReaction], 
                          max_depth: int, graceful_exit: GracefulExit, 
                          current_depth: int = 0) -> List[Dict]:
    """Find synthetic routes using recursive retrosynthetic analysis."""
    routes = []
    
    if current_depth >= max_depth or graceful_exit.exit_now:
        return routes
    
    target_smiles = Chem.MolToSmiles(target)
    
    for reaction in reactions:
        if graceful_exit.exit_now:
            break
        
        try:
            if reaction.GetNumProductTemplates() != 1:
                continue
            
            product_template = reaction.GetProductTemplate(0)
            
            if target.HasSubstructMatch(product_template):
                reactant_templates = [reaction.GetReactantTemplate(i) 
                                    for i in range(reaction.GetNumReactantTemplates())]
                
                route = {
                    'depth': current_depth,
                    'target': target_smiles,
                    'reaction_smarts': rdChemReactions.ReactionToSmarts(reaction),
                    'precursors': []
                }
                
                for i, template in enumerate(reactant_templates):
                    precursor_smiles = Chem.MolToSmiles(template)
                    route['precursors'].append(precursor_smiles)
                    
                    if current_depth < max_depth - 1:
                        precursor_mol = Chem.MolFromSmiles(precursor_smiles)
                        if precursor_mol is not None:
                            sub_routes = _find_synthetic_routes(
                                precursor_mol, reactions, max_depth, 
                                graceful_exit, current_depth + 1
                            )
                            if sub_routes:
                                route[f'precursor_{i}_routes'] = sub_routes
                
                routes.append(route)
        
        except Exception:
            continue
    
    return routes[:10]