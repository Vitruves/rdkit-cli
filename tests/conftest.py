"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
import tempfile


# Sample molecules for testing
SAMPLE_SMILES = [
    ("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("benzene", "c1ccccc1"),
    ("ethanol", "CCO"),
    ("acetone", "CC(=O)C"),
]

INVALID_SMILES = [
    "not_a_smiles",
    "C(C(C)",
    "",
]


@pytest.fixture
def sample_molecules():
    """Return list of (name, smiles) tuples."""
    return SAMPLE_SMILES


@pytest.fixture
def tmp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_csv(tmp_dir):
    """Create a sample CSV file with molecules."""
    import pandas as pd

    csv_path = tmp_dir / "sample.csv"
    df = pd.DataFrame(SAMPLE_SMILES, columns=["name", "smiles"])
    df = df[["smiles", "name"]]  # Reorder columns
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_csv_with_invalid(tmp_dir):
    """Create a CSV file with some invalid molecules."""
    csv_path = tmp_dir / "sample_invalid.csv"
    all_smiles = SAMPLE_SMILES + [("invalid", "not_a_smiles")]
    with open(csv_path, "w") as f:
        f.write("smiles,name\n")
        for name, smiles in all_smiles:
            f.write(f"{smiles},{name}\n")
    return csv_path


@pytest.fixture
def sample_smi(tmp_dir):
    """Create a sample SMI file."""
    smi_path = tmp_dir / "sample.smi"
    with open(smi_path, "w") as f:
        for name, smi in SAMPLE_SMILES:
            f.write(f"{smi} {name}\n")
    return smi_path


@pytest.fixture
def output_csv(tmp_dir):
    """Return a path for CSV output."""
    return tmp_dir / "output.csv"


@pytest.fixture
def output_smi(tmp_dir):
    """Return a path for SMI output."""
    return tmp_dir / "output.smi"


@pytest.fixture
def output_svg(tmp_dir):
    """Return a path for SVG output."""
    return tmp_dir / "output.svg"


@pytest.fixture
def output_png(tmp_dir):
    """Return a path for PNG output."""
    return tmp_dir / "output.png"


@pytest.fixture
def output_sdf(tmp_dir):
    """Return a path for SDF output."""
    return tmp_dir / "output.sdf"


@pytest.fixture
def output_dir(tmp_dir):
    """Return a path for output directory."""
    d = tmp_dir / "output_dir"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture
def cli_runner():
    """Return a function to run CLI commands."""
    from rdkit_cli.cli import main

    def run(args):
        if isinstance(args, str):
            args = args.split()
        return main(args)

    return run
