#!/usr/bin/env python3
"""
Example workflow demonstrating RDKit CLI capabilities.

This script creates sample data and demonstrates various operations.
"""

import tempfile
from pathlib import Path
from rdkit import Chem
import subprocess
import sys


def create_sample_data(output_dir: Path):
    """Create sample molecular data for demonstration."""
    # Sample SMILES data
    sample_smiles = [
        ("CCO", "ethanol"),
        ("CC(C)O", "isopropanol"), 
        ("c1ccccc1", "benzene"),
        ("CCc1ccccc1", "ethylbenzene"),
        ("CC(=O)O", "acetic_acid"),
        ("CC(=O)Oc1ccccc1C(=O)O", "aspirin"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine"),
        ("CC(C)(C)c1ccc(O)cc1", "BHT"),
        ("Clc1ccc(Cl)cc1", "dichlorobenzene"),
        ("NC(=O)c1ccccc1", "benzamide"),
        ("CCN(CC)CC", "triethylamine"),
        ("CCCCCCCCCCCCCCO", "tetradecanol"),
        ("c1ccc2c(c1)cccn2", "quinoline"),
        ("CC1=CC=C(C=C1)C2=CC=C(C=C2)C", "p-bitolyl"),
        ("CCCCCCCCCCCCCCCC(=O)O", "palmitic_acid")
    ]
    
    # Create SMILES file
    smiles_file = output_dir / "sample_molecules.smi"
    with open(smiles_file, 'w') as f:
        for smiles, name in sample_smiles:
            f.write(f"{smiles}\t{name}\n")
    
    print(f"Created sample SMILES file: {smiles_file}")
    return smiles_file


def run_command(cmd_args, description):
    """Run a CLI command and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: rdkit-cli {' '.join(cmd_args)}")
    print('='*60)
    
    result = subprocess.run(
        [sys.executable, "-m", "rdkit_cli.main"] + cmd_args,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout[:200])
    else:
        print("✗ FAILED")
        print("Error:", result.stderr[:200])
    
    return result.returncode == 0


def main():
    """Main workflow demonstration."""
    print("RDKit CLI Workflow Demonstration")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Working directory: {temp_path}")
        
        # Step 1: Create sample data
        smiles_file = create_sample_data(temp_path)
        
        # Step 2: Convert SMILES to SDF
        sdf_file = temp_path / "molecules.sdf"
        success = run_command([
            "convert",
            "-i", str(smiles_file),
            "-o", str(sdf_file)
        ], "Converting SMILES to SDF format")
        
        if not success:
            print("Failed to convert SMILES to SDF. Check RDKit installation.")
            return
        
        # Step 3: Get file information
        run_command([
            "info",
            "-i", str(sdf_file)
        ], "Getting file information")
        
        # Step 4: Calculate descriptors
        desc_file = temp_path / "descriptors.csv"
        run_command([
            "descriptors",
            "-i", str(sdf_file),
            "-o", str(desc_file),
            "--descriptor-set", "basic"
        ], "Calculating molecular descriptors")
        
        # Step 5: Generate fingerprints
        fp_file = temp_path / "fingerprints.pkl"
        run_command([
            "fingerprints",
            "-i", str(sdf_file),
            "-o", str(fp_file),
            "--fp-type", "morgan"
        ], "Generating Morgan fingerprints")
        
        # Step 6: Calculate similarity matrix
        sim_file = temp_path / "similarity.csv"
        run_command([
            "similarity-matrix",
            "-i", str(sdf_file),
            "-o", str(sim_file),
            "--fp-type", "morgan"
        ], "Calculating similarity matrix")
        
        # Step 7: Cluster molecules
        cluster_file = temp_path / "clusters.csv"
        run_command([
            "cluster",
            "-i", str(sdf_file),
            "-o", str(cluster_file),
            "--method", "butina",
            "--threshold", "0.6"
        ], "Clustering molecules")
        
        # Step 8: Sample diverse molecules
        diverse_file = temp_path / "diverse.sdf"
        run_command([
            "sample",
            "-i", str(sdf_file),
            "-o", str(diverse_file),
            "--count", "5",
            "--method", "diverse"
        ], "Selecting diverse molecules")
        
        # Step 9: Search for substructures
        benzene_matches = temp_path / "benzene_matches.sdf"
        run_command([
            "substructure-search",
            "-i", str(sdf_file),
            "--query", "c1ccccc1",
            "-o", str(benzene_matches)
        ], "Searching for benzene rings")
        
        # Step 10: Calculate statistics
        stats_file = temp_path / "statistics.json"
        run_command([
            "stats",
            "-i", str(sdf_file),
            "-o", str(stats_file),
            "--include-descriptors"
        ], "Calculating dataset statistics")
        
        print(f"\n{'='*60}")
        print("Workflow completed!")
        print("Generated files:")
        for file in temp_path.glob("*"):
            if file.is_file():
                print(f"  - {file.name} ({file.stat().st_size} bytes)")
        print("="*60)


if __name__ == "__main__":
    main()