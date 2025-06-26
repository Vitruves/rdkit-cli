# tests/conftest.py
"""Pytest configuration and fixtures for RDKit CLI tests."""

import tempfile
from pathlib import Path
from typing import List

import pytest
from rdkit import Chem

from rdkit_cli.core.common import write_molecules


@pytest.fixture
def sample_molecules():
    """Create sample molecules for testing."""
    smiles_list = [
        "CCO",  # ethanol
        "CC(C)O",  # isopropanol
        "c1ccccc1",  # benzene
        "CCc1ccccc1",  # ethylbenzene
        "CC(=O)O",  # acetic acid
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(C)(C)c1ccc(O)cc1",  # BHT
        "Clc1ccc(Cl)cc1",  # dichlorobenzene
        "NC(=O)c1ccccc1"  # benzamide
    ]
    
    molecules = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol.SetProp("_Name", f"mol_{i+1}")
            molecules.append(mol)
    
    return molecules


@pytest.fixture
def sample_sdf_file(sample_molecules, tmp_path):
    """Create a temporary SDF file with sample molecules."""
    sdf_file = tmp_path / "test_molecules.sdf"
    write_molecules(sample_molecules, sdf_file)
    return sdf_file


@pytest.fixture
def sample_smiles_file(sample_molecules, tmp_path):
    """Create a temporary SMILES file with sample molecules."""
    smiles_file = tmp_path / "test_molecules.smi"
    
    with open(smiles_file, 'w') as f:
        for mol in sample_molecules:
            smiles = Chem.MolToSmiles(mol)
            mol_id = mol.GetProp("_Name")
            f.write(f"{smiles}\t{mol_id}\n")
    
    return smiles_file


@pytest.fixture
def sample_csv_file(sample_molecules, tmp_path):
    """Create a temporary CSV file with sample molecules and properties."""
    import pandas as pd
    from rdkit.Chem import Descriptors, Crippen
    
    data = []
    for mol in sample_molecules:
        mol_id = mol.GetProp("_Name")
        smiles = Chem.MolToSmiles(mol)
        
        data.append({
            'ID': mol_id,
            'SMILES': smiles,
            'MW': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'activity': float(hash(smiles) % 100) / 10.0  # Mock activity data
        })
    
    df = pd.DataFrame(data)
    csv_file = tmp_path / "test_molecules.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def graceful_exit_mock():
    """Mock GracefulExit for testing."""
    class MockGracefulExit:
        def __init__(self):
            self.exit_now = False
    
    return MockGracefulExit()


@pytest.fixture
def mock_args():
    """Create mock arguments for testing."""
    class MockArgs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    return MockArgs