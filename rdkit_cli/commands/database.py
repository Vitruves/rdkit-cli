# rdkit_cli/commands/database.py
import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from tqdm import tqdm

from ..core.common import (
    GracefulExit, get_parallel_jobs, read_molecules, validate_input_file,
    validate_output_path, write_molecules
)
from ..core.logging import log_success


def add_subparser(subparsers) -> None:
    parser_db_create = subparsers.add_parser(
        'db-create',
        help='Create molecular database from input file'
    )
    parser_db_create.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input molecular file'
    )
    parser_db_create.add_argument(
        '--db-file',
        required=True,
        help='Output SQLite database file'
    )
    parser_db_create.add_argument(
        '--index-fps',
        action='store_true',
        help='Create fingerprint indices for fast similarity search'
    )
    parser_db_create.add_argument(
        '--fp-type',
        choices=['morgan', 'rdkit', 'maccs'],
        default='morgan',
        help='Fingerprint type for indexing (default: morgan)'
    )

    parser_db_search = subparsers.add_parser(
        'db-search',
        help='Search database for similar molecules'
    )
    parser_db_search.add_argument(
        '--db-file',
        required=True,
        help='SQLite database file'
    )
    parser_db_search.add_argument(
        '--query',
        required=True,
        help='Query molecule (SMILES or structure file)'
    )
    parser_db_search.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with search results'
    )
    parser_db_search.add_argument(
        '--similarity',
        type=float,
        default=0.7,
        help='Minimum similarity threshold (default: 0.7)'
    )
    parser_db_search.add_argument(
        '--max-results',
        type=int,
        default=100,
        help='Maximum number of results (default: 100)'
    )

    parser_db_filter = subparsers.add_parser(
        'db-filter',
        help='Filter database using molecular property criteria'
    )
    parser_db_filter.add_argument(
        '--db-file',
        required=True,
        help='SQLite database file'
    )
    parser_db_filter.add_argument(
        '--filters',
        required=True,
        help='Filter criteria (lipinski, druglike, or custom JSON file)'
    )
    parser_db_filter.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with filtered molecule IDs'
    )

    parser_db_export = subparsers.add_parser(
        'db-export',
        help='Export molecules from database'
    )
    parser_db_export.add_argument(
        '--db-file',
        required=True,
        help='SQLite database file'
    )
    parser_db_export.add_argument(
        '-o', '--output',
        required=True,
        help='Output molecular file'
    )
    parser_db_export.add_argument(
        '--format',
        choices=['sdf', 'smiles', 'csv'],
        default='sdf',
        help='Output format (default: sdf)'
    )
    parser_db_export.add_argument(
        '--ids',
        help='Comma-separated list of molecule IDs to export'
    )


def create(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        db_path = Path(args.db_file)
        
        logger.info(f"Creating molecular database from {input_path}")
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        if db_path.exists():
            db_path.unlink()
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE molecules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mol_id TEXT UNIQUE,
                smiles TEXT,
                mol_block TEXT,
                mw REAL,
                logp REAL,
                tpsa REAL,
                hbd INTEGER,
                hba INTEGER,
                rotbonds INTEGER,
                heavy_atoms INTEGER
            )
        """)
        
        if args.index_fps:
            cursor.execute("""
                CREATE TABLE fingerprints (
                    mol_id TEXT,
                    fp_type TEXT,
                    fingerprint BLOB,
                    FOREIGN KEY (mol_id) REFERENCES molecules (mol_id)
                )
            """)
            cursor.execute("CREATE INDEX idx_fp_type ON fingerprints (fp_type)")
        
        conn.commit()
        
        inserted_count = 0
        
        with tqdm(total=len(molecules), desc="Building database", ncols=80, colour='blue') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                    smiles = Chem.MolToSmiles(mol)
                    mol_block = Chem.MolToMolBlock(mol)
                    
                    from rdkit.Chem import Descriptors, Crippen
                    mw = Descriptors.MolWt(mol)
                    logp = Crippen.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    rotbonds = Descriptors.NumRotatableBonds(mol)
                    heavy_atoms = mol.GetNumHeavyAtoms()
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO molecules 
                        (mol_id, smiles, mol_block, mw, logp, tpsa, hbd, hba, rotbonds, heavy_atoms)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (mol_id, smiles, mol_block, mw, logp, tpsa, hbd, hba, rotbonds, heavy_atoms))
                    
                    if args.index_fps:
                        fp = _generate_fingerprint(mol, args.fp_type)
                        if fp is not None:
                            cursor.execute("""
                                INSERT INTO fingerprints (mol_id, fp_type, fingerprint)
                                VALUES (?, ?, ?)
                            """, (mol_id, args.fp_type, fp.ToBinary()))
                    
                    inserted_count += 1
                
                except Exception as e:
                    logger.debug(f"Failed to process molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            conn.close()
            return 130
        
        conn.commit()
        conn.close()
        
        log_success(f"Created database with {inserted_count} molecules, saved to {db_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Database creation failed: {e}")
        return 1


def search(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        db_path = Path(args.db_file)
        if not db_path.exists():
            logger.error(f"Database file not found: {db_path}")
            return 1
        
        output_path = validate_output_path(args.output)
        
        logger.info(f"Searching database {db_path} for similar molecules")
        
        if Path(args.query).exists():
            query_molecules = read_molecules(Path(args.query))
            if not query_molecules:
                logger.error("No query molecule found")
                return 1
            query_mol = query_molecules[0]
        else:
            query_mol = Chem.MolFromSmiles(args.query)
            if query_mol is None:
                logger.error(f"Invalid query SMILES: {args.query}")
                return 1
        
        query_fp = _generate_fingerprint(query_mol, 'morgan')
        if query_fp is None:
            logger.error("Failed to generate fingerprint for query")
            return 1
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT mol_id, fingerprint FROM fingerprints WHERE fp_type = 'morgan'")
        rows = cursor.fetchall()
        
        logger.info(f"Comparing against {len(rows)} fingerprints")
        
        similarities = []
        
        with tqdm(total=len(rows), desc="Calculating similarities", ncols=80, colour='green') as pbar:
            for mol_id, fp_binary in rows:
                if graceful_exit.exit_now:
                    break
                
                try:
                    from rdkit import DataStructs
                    fp = DataStructs.CreateFromBinaryText(fp_binary)
                    similarity = DataStructs.TanimotoSimilarity(query_fp, fp)
                    
                    if similarity >= args.similarity:
                        similarities.append((mol_id, similarity))
                
                except Exception as e:
                    logger.debug(f"Failed to calculate similarity for {mol_id}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            conn.close()
            return 130
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = similarities[:args.max_results]
        
        if not similarities:
            logger.warning("No similar molecules found")
            conn.close()
            return 0
        
        result_mol_ids = [mol_id for mol_id, _ in similarities]
        placeholders = ','.join(['?' for _ in result_mol_ids])
        
        cursor.execute(f"""
            SELECT mol_id, smiles, mol_block, mw, logp, tpsa 
            FROM molecules 
            WHERE mol_id IN ({placeholders})
        """, result_mol_ids)
        
        molecule_data = {row[0]: row for row in cursor.fetchall()}
        conn.close()
        
        results = []
        for mol_id, similarity in similarities:
            if mol_id in molecule_data:
                data = molecule_data[mol_id]
                results.append({
                    'ID': data[0],
                    'SMILES': data[1],
                    'Similarity': round(similarity, 4),
                    'MW': round(data[3], 2),
                    'LogP': round(data[4], 2),
                    'TPSA': round(data[5], 2)
                })
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        log_success(f"Found {len(results)} similar molecules, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Database search failed: {e}")
        return 1


def filter_db(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        db_path = Path(args.db_file)
        if not db_path.exists():
            logger.error(f"Database file not found: {db_path}")
            return 1
        
        output_path = validate_output_path(args.output)
        
        logger.info(f"Filtering database {db_path} using {args.filters} criteria")
        
        conn = sqlite3.connect(str(db_path))
        
        if args.filters == 'lipinski':
            query = """
                SELECT mol_id FROM molecules 
                WHERE mw <= 500 AND logp <= 5 AND hbd <= 5 AND hba <= 10
            """
        elif args.filters == 'druglike':
            query = """
                SELECT mol_id FROM molecules 
                WHERE mw BETWEEN 150 AND 500 
                AND logp BETWEEN -2 AND 5 
                AND tpsa <= 140 
                AND rotbonds <= 10
                AND heavy_atoms >= 10
            """
        else:
            try:
                filter_path = Path(args.filters)
                if filter_path.exists():
                    with open(filter_path, 'r') as f:
                        filter_criteria = json.load(f)
                    
                    conditions = []
                    for prop, criteria in filter_criteria.items():
                        if 'min' in criteria:
                            conditions.append(f"{prop} >= {criteria['min']}")
                        if 'max' in criteria:
                            conditions.append(f"{prop} <= {criteria['max']}")
                    
                    if conditions:
                        query = f"SELECT mol_id FROM molecules WHERE {' AND '.join(conditions)}"
                    else:
                        query = "SELECT mol_id FROM molecules"
                else:
                    logger.error(f"Unknown filter type or file not found: {args.filters}")
                    return 1
            except Exception as e:
                logger.error(f"Failed to parse filter criteria: {e}")
                return 1
        
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        
        filtered_ids = [row[0] for row in results]
        
        with open(output_path, 'w') as f:
            for mol_id in filtered_ids:
                f.write(f"{mol_id}\n")
        
        log_success(f"Filtered to {len(filtered_ids)} molecules, IDs saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Database filtering failed: {e}")
        return 1


def export(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        db_path = Path(args.db_file)
        if not db_path.exists():
            logger.error(f"Database file not found: {db_path}")
            return 1
        
        output_path = validate_output_path(args.output)
        
        logger.info(f"Exporting molecules from database {db_path}")
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        if args.ids:
            mol_ids = [id.strip() for id in args.ids.split(',')]
            placeholders = ','.join(['?' for _ in mol_ids])
            cursor.execute(f"""
                SELECT mol_id, smiles, mol_block 
                FROM molecules 
                WHERE mol_id IN ({placeholders})
            """, mol_ids)
        else:
            cursor.execute("SELECT mol_id, smiles, mol_block FROM molecules")
        
        rows = cursor.fetchall()
        conn.close()
        
        logger.info(f"Exporting {len(rows)} molecules")
        
        if args.format == 'smiles':
            with open(output_path, 'w') as f:
                for mol_id, smiles, _ in rows:
                    f.write(f"{smiles}\t{mol_id}\n")
        
        elif args.format == 'csv':
            data = []
            for mol_id, smiles, _ in rows:
                data.append({'ID': mol_id, 'SMILES': smiles})
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        
        else:  # sdf
            molecules = []
            for mol_id, smiles, mol_block in rows:
                try:
                    if mol_block:
                        mol = Chem.MolFromMolBlock(mol_block)
                    else:
                        mol = Chem.MolFromSmiles(smiles)
                    
                    if mol is not None:
                        mol.SetProp("_Name", mol_id)
                        molecules.append(mol)
                
                except Exception:
                    continue
            
            write_molecules(molecules, output_path)
        
        log_success(f"Exported {len(rows)} molecules to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Database export failed: {e}")
        return 1


def _generate_fingerprint(mol: Chem.Mol, fp_type: str):
    """Generate fingerprint for database storage."""
    try:
        if fp_type == 'morgan':
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        elif fp_type == 'rdkit':
            return Chem.RDKFingerprint(mol, fpSize=2048)
        elif fp_type == 'maccs':
            return rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        else:
            return None
    except Exception:
        return None