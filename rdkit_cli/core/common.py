# rdkit_cli/core/common.py
import logging
import os
import signal
import sys
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from .config import config


class GracefulExit:
    """Handle SIGINT gracefully."""
    
    def __init__(self) -> None:
        self.exit_now = False
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame) -> None:
        self.exit_now = True
        logger = logging.getLogger("rdkit_cli")
        logger.info("Received interrupt signal, finishing current operation...")


def validate_input_file(file_path: Union[str, Path]) -> Path:
    """Validate input file exists and is readable.
    
    Args:
        file_path: Path to input file
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file is not readable
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Input file is not readable: {path}")
    return path


def validate_output_path(file_path: Union[str, Path], create_dirs: bool = True) -> Path:
    """Validate output path and create directories if needed.
    
    Args:
        file_path: Path to output file
        create_dirs: Whether to create parent directories
        
    Returns:
        Validated Path object
        
    Raises:
        PermissionError: If directory is not writable
    """
    path = Path(file_path)
    
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.parent.exists() and not os.access(path.parent, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {path.parent}")
    
    return path


def detect_file_format(file_path: Path) -> str:
    """Detect file format from extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        Detected format string
        
    Raises:
        ValueError: If format cannot be detected
    """
    suffix = file_path.suffix.lower()
    format_map = {
        '.smi': 'smiles',
        '.smiles': 'smiles',
        '.sdf': 'sdf',
        '.mol': 'mol',
        '.mol2': 'mol2',
        '.csv': 'csv',
        '.tsv': 'csv',
        '.xlsx': 'excel',
        '.parquet': 'parquet',
        '.pkl': 'pickle',
        '.pickle': 'pickle',
        '.rdf': 'rdf',
        '.rxn': 'rxn',
    }
    
    if suffix not in format_map:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    return format_map[suffix]


def read_molecules(file_path: Path, format_hint: Optional[str] = None) -> List[Chem.Mol]:
    """Read molecules from file with automatic format detection.
    
    Args:
        file_path: Path to input file
        format_hint: Optional format override
        
    Returns:
        List of RDKit molecule objects
        
    Raises:
        ValueError: If file format is unsupported or file is corrupted
    """
    format_type = format_hint or detect_file_format(file_path)
    
    molecules = []
    
    if format_type == 'smiles':
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t') if '\t' in line else line.split()
                smiles = parts[0]
                mol_id = parts[1] if len(parts) > 1 else f"mol_{line_num}"
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mol.SetProp("_Name", mol_id)
                    molecules.append(mol)
    
    elif format_type == 'sdf':
        supplier = Chem.SDMolSupplier(str(file_path))
        molecules = [mol for mol in supplier if mol is not None]
    
    elif format_type == 'mol':
        mol = Chem.MolFromMolFile(str(file_path))
        if mol is not None:
            molecules = [mol]
    
    elif format_type == 'csv':
        df = pd.read_csv(file_path)
        molecules = _read_molecules_from_dataframe(df)
    
    elif format_type == 'parquet':
        df = pd.read_parquet(file_path)
        molecules = _read_molecules_from_dataframe(df)
    
    else:
        raise ValueError(f"Reading format '{format_type}' not implemented")
    
    return molecules


def write_molecules(molecules: List[Chem.Mol], file_path: Path, 
                   format_hint: Optional[str] = None) -> None:
    """Write molecules to file with automatic format detection.
    
    Args:
        molecules: List of RDKit molecule objects
        file_path: Path to output file
        format_hint: Optional format override
        
    Raises:
        ValueError: If file format is unsupported
    """
    format_type = format_hint or detect_file_format(file_path)
    
    if format_type == 'smiles':
        with open(file_path, 'w') as f:
            for mol in molecules:
                smiles = Chem.MolToSmiles(mol)
                mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                f.write(f"{smiles}\t{mol_id}\n")
    
    elif format_type == 'sdf':
        writer = Chem.SDWriter(str(file_path))
        for mol in molecules:
            writer.write(mol)
        writer.close()
    
    elif format_type == 'csv':
        df = _molecules_to_dataframe(molecules)
        df.to_csv(file_path, index=False)
    
    elif format_type == 'parquet':
        df = _molecules_to_dataframe(molecules)
        df.to_parquet(file_path, index=False)
    
    else:
        raise ValueError(f"Writing format '{format_type}' not implemented")


def get_parallel_jobs(jobs: Optional[int] = None) -> int:
    """Get number of parallel jobs with fallback to config and CPU count.
    
    Args:
        jobs: Requested number of jobs
        
    Returns:
        Number of parallel jobs to use
    """
    if jobs is not None and jobs > 0:
        return jobs
    
    config_jobs = config.get("default_jobs")
    if config_jobs and config_jobs > 0:
        return config_jobs
    
    return os.cpu_count() or 1


def _read_molecules_from_dataframe(df: pd.DataFrame) -> List[Chem.Mol]:
    """Read molecules from a pandas DataFrame.
    
    Args:
        df: DataFrame containing molecular data
        
    Returns:
        List of RDKit molecule objects
        
    Raises:
        ValueError: If no SMILES column is found
    """
    molecules = []
    
    # Find SMILES column
    smiles_col = None
    for col in df.columns:
        if 'smiles' in col.lower():
            smiles_col = col
            break
    
    if smiles_col is None:
        raise ValueError("No SMILES column found in DataFrame")
    
    # Find ID column
    id_col = None
    for col in df.columns:
        if col.lower() in ['id', 'name', 'title', 'compound_id', 'mol_id']:
            id_col = col
            break
    
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        if pd.isna(smiles) or not smiles.strip():
            continue
            
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is not None:
            # Set molecule name/ID
            if id_col is not None and pd.notna(row[id_col]):
                mol.SetProp("_Name", str(row[id_col]))
            else:
                mol.SetProp("_Name", f"mol_{idx+1}")
            
            # Set other properties
            for col, val in row.items():
                if col != smiles_col and col != id_col and pd.notna(val):
                    # Convert numeric types to string for RDKit properties
                    if isinstance(val, (int, float)):
                        mol.SetProp(col, str(val))
                    else:
                        mol.SetProp(col, str(val))
            
            molecules.append(mol)
    
    return molecules


def _molecules_to_dataframe(molecules: List[Chem.Mol]) -> pd.DataFrame:
    """Convert molecules to a pandas DataFrame.
    
    Args:
        molecules: List of RDKit molecule objects
        
    Returns:
        DataFrame with SMILES and molecular properties
    """
    data = []
    
    for mol in molecules:
        if mol is None:
            continue
            
        row = {
            'SMILES': Chem.MolToSmiles(mol),
            'ID': mol.GetProp("_Name") if mol.HasProp("_Name") else "",
        }
        
        # Add all molecule properties
        for prop in mol.GetPropNames():
            if not prop.startswith('_'):
                try:
                    val = mol.GetProp(prop)
                    # Try to convert to numeric if possible
                    try:
                        if '.' in val:
                            val = float(val)
                        else:
                            val = int(val)
                    except (ValueError, TypeError):
                        # Keep as string
                        pass
                    row[prop] = val
                except:
                    # Skip properties that can't be retrieved
                    pass
        
        data.append(row)
    
    return pd.DataFrame(data)


def read_csv_with_molecules(file_path: Path, smiles_col: str = None, id_col: str = None) -> pd.DataFrame:
    """Read CSV file and add molecule objects as a column.
    
    Args:
        file_path: Path to CSV file
        smiles_col: Name of SMILES column (auto-detected if None)
        id_col: Name of ID column (auto-detected if None)
        
    Returns:
        DataFrame with added 'Molecule' column containing RDKit Mol objects
    """
    df = pd.read_csv(file_path)
    
    # Auto-detect SMILES column if not specified
    if smiles_col is None:
        for col in df.columns:
            if 'smiles' in col.lower():
                smiles_col = col
                break
        if smiles_col is None:
            raise ValueError("No SMILES column found and none specified")
    
    # Auto-detect ID column if not specified
    if id_col is None:
        for col in df.columns:
            if col.lower() in ['id', 'name', 'title', 'compound_id', 'mol_id']:
                id_col = col
                break
    
    # Add molecule objects
    molecules = []
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        if pd.isna(smiles) or not str(smiles).strip():
            molecules.append(None)
            continue
            
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is not None and id_col is not None and pd.notna(row[id_col]):
            mol.SetProp("_Name", str(row[id_col]))
        molecules.append(mol)
    
    df['Molecule'] = molecules
    return df


def read_parquet_with_molecules(file_path: Path, smiles_col: str = None, id_col: str = None) -> pd.DataFrame:
    """Read Parquet file and add molecule objects as a column.
    
    Args:
        file_path: Path to Parquet file
        smiles_col: Name of SMILES column (auto-detected if None)
        id_col: Name of ID column (auto-detected if None)
        
    Returns:
        DataFrame with added 'Molecule' column containing RDKit Mol objects
    """
    df = pd.read_parquet(file_path)
    
    # Auto-detect SMILES column if not specified
    if smiles_col is None:
        for col in df.columns:
            if 'smiles' in col.lower():
                smiles_col = col
                break
        if smiles_col is None:
            raise ValueError("No SMILES column found and none specified")
    
    # Auto-detect ID column if not specified
    if id_col is None:
        for col in df.columns:
            if col.lower() in ['id', 'name', 'title', 'compound_id', 'mol_id']:
                id_col = col
                break
    
    # Add molecule objects
    molecules = []
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        if pd.isna(smiles) or not str(smiles).strip():
            molecules.append(None)
            continue
            
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is not None and id_col is not None and pd.notna(row[id_col]):
            mol.SetProp("_Name", str(row[id_col]))
        molecules.append(mol)
    
    df['Molecule'] = molecules
    return df


def write_molecules_to_csv(molecules: List[Chem.Mol], file_path: Path, 
                          include_descriptors: bool = False) -> None:
    """Write molecules to CSV with optional descriptor calculation.
    
    Args:
        molecules: List of RDKit molecule objects
        file_path: Path to output CSV file
        include_descriptors: Whether to calculate and include basic descriptors
    """
    df = _molecules_to_dataframe(molecules)
    
    if include_descriptors:
        from rdkit.Chem import Descriptors
        
        # Add basic descriptors
        descriptors = []
        for mol in molecules:
            if mol is None:
                descriptors.append({})
                continue
                
            desc = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'HeavyAtoms': Descriptors.HeavyAtomCount(mol)
            }
            descriptors.append(desc)
        
        desc_df = pd.DataFrame(descriptors)
        df = pd.concat([df, desc_df], axis=1)
    
    df.to_csv(file_path, index=False)


def write_molecules_to_parquet(molecules: List[Chem.Mol], file_path: Path, 
                              include_descriptors: bool = False) -> None:
    """Write molecules to Parquet with optional descriptor calculation.
    
    Args:
        molecules: List of RDKit molecule objects
        file_path: Path to output Parquet file
        include_descriptors: Whether to calculate and include basic descriptors
    """
    df = _molecules_to_dataframe(molecules)
    
    if include_descriptors:
        from rdkit.Chem import Descriptors
        
        # Add basic descriptors
        descriptors = []
        for mol in molecules:
            if mol is None:
                descriptors.append({})
                continue
                
            desc = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'HeavyAtoms': Descriptors.HeavyAtomCount(mol)
            }
            descriptors.append(desc)
        
        desc_df = pd.DataFrame(descriptors)
        df = pd.concat([df, desc_df], axis=1)
    
    df.to_parquet(file_path, index=False)


def get_molecules_from_args(args, required_if_no_smiles: bool = True) -> List[Chem.Mol]:
    """Get molecules from command line arguments (file or direct SMILES).
    
    Args:
        args: Parsed command line arguments
        required_if_no_smiles: If True, require input_file when no SMILES provided
        
    Returns:
        List of RDKit molecule objects
        
    Raises:
        ValueError: If neither input file nor SMILES are provided
    """
    molecules = []
    
    # Handle direct SMILES input
    if hasattr(args, 'smiles') and args.smiles:
        smiles_list = [s.strip() for s in args.smiles.split(',')]
        for i, smiles in enumerate(smiles_list):
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mol.SetProp("_Name", f"smiles_{i+1}")
                    molecules.append(mol)
                else:
                    logger = logging.getLogger("rdkit_cli")
                    logger.warning(f"Invalid SMILES: {smiles}")
        return molecules
    
    # Handle file input
    if hasattr(args, 'input_file') and args.input_file:
        input_path = validate_input_file(args.input_file)
        
        # For CSV/Parquet files, use custom SMILES column if specified
        if input_path.suffix.lower() in ['.csv', '.parquet']:
            if input_path.suffix.lower() == '.csv':
                if hasattr(args, 'smiles_column') and args.smiles_column:
                    df = read_csv_with_molecules(input_path, smiles_col=args.smiles_column)
                else:
                    df = read_csv_with_molecules(input_path)
            else:  # parquet
                if hasattr(args, 'smiles_column') and args.smiles_column:
                    df = read_parquet_with_molecules(input_path, smiles_col=args.smiles_column)
                else:
                    df = read_parquet_with_molecules(input_path)
            
            molecules = [mol for mol in df['Molecule'] if mol is not None]
        else:
            molecules = read_molecules(input_path)
        
        return molecules
    
    # Neither SMILES nor file provided
    if required_if_no_smiles:
        raise ValueError("Either --input-file or --smiles must be provided")
    
    return []


def save_dataframe_with_format_detection(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame with automatic format detection from file extension.
    
    Args:
        df: DataFrame to save
        output_path: Output file path with extension
    """
    output_format = detect_file_format(output_path)
    if output_format == 'csv':
        df.to_csv(output_path, index=False)
    elif output_format == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        # Default to CSV for unsupported formats
        df.to_csv(output_path, index=False)