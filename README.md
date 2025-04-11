# RDKit CLI

A comprehensive command-line interface for RDKit, enabling efficient molecular manipulation, analysis, and visualization through simple commands.

## Features

- File I/O in various formats (SMILES, SDF, CSV, TSV, MOL)
- Molecular descriptor calculation
- Fingerprint generation and comparison
- SMILES manipulation and standardization
- 2D/3D conformer generation and manipulation
- Filtering and sorting by molecular properties
- Molecule visualization and export
- Batch processing with parallel execution

## Installation

### From Source

Prerequisites:
- CMake 3.12+
- C++17 compiler
- RDKit
- Boost (program_options, filesystem)
- OpenMP

Build steps:

```bash
git clone <>
cd rdkit-cli
./build.sh
```

To install system-wide:

```bash
./build.sh --install
```

## Usage

Basic usage:

```bash
rdkit-cli --file input.smi --output output.csv
```

For detailed help:

```bash
rdkit-cli --help
```

### Example Commands

Calculate molecular descriptors:
```bash
rdkit-cli --file molecules.smi --calculate-logp LogP --calculate-tpsa TPSA --output results.csv
```

Generate fingerprints and calculate similarity:
```bash
rdkit-cli --file molecules.sdf --fp-morgan MorganFP 2 1024 --output fingerprints.csv
```

Filter by properties:
```bash
rdkit-cli --file compounds.csv --smiles-col SMILES --lipinski-filter Lipinski --output filtered.csv
```

Generate conformers and minimize energy:
```bash
rdkit-cli --file actives.smi --generate-conformers 10 --minimize-energy MMFF94 --output conformers.sdf
```

Visualize molecules:
```bash
rdkit-cli --file hits.sdf --export-svg ./images 300 300
```

## License

MIT License