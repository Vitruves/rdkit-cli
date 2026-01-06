# rdkit-cli

A comprehensive, high-performance CLI tool wrapping RDKit functionality for cheminformatics workflows.

## Features

- **14 Command Categories**: descriptors, fingerprints, filter, convert, standardize, similarity, conformers, reactions, scaffold, enumerate, fragment, diversity, mcs, depict
- **Multiple Input/Output Formats**: CSV, TSV, SMI, SDF, Parquet
- **Parallel Processing**: Efficient multi-core support via ProcessPoolExecutor
- **Ninja-style Progress**: Real-time progress display with speed and ETA

## Installation

```bash
pip install rdkit-cli
```

Or with uv:

```bash
uv add rdkit-cli
```

## Quick Start

```bash
# Compute molecular descriptors
rdkit-cli descriptors compute -i molecules.csv -o descriptors.csv -d MolWt,MolLogP,TPSA

# Generate fingerprints
rdkit-cli fingerprints compute -i molecules.csv -o fingerprints.csv --type morgan

# Filter by drug-likeness
rdkit-cli filter druglike -i molecules.csv -o filtered.csv --rule lipinski

# Standardize molecules
rdkit-cli standardize -i molecules.csv -o standardized.csv --cleanup --neutralize

# Similarity search
rdkit-cli similarity search -i library.csv -o hits.csv --query "c1ccccc1" --threshold 0.7
```

## Commands

### descriptors

Compute molecular descriptors.

```bash
# List available descriptors
rdkit-cli descriptors list
rdkit-cli descriptors list --all

# Compute specific descriptors
rdkit-cli descriptors compute -i input.csv -o output.csv -d MolWt,MolLogP,TPSA

# Compute all descriptors
rdkit-cli descriptors compute -i input.csv -o output.csv --all
```

### fingerprints

Generate molecular fingerprints.

```bash
# List available fingerprint types
rdkit-cli fingerprints list

# Compute Morgan fingerprints (default)
rdkit-cli fingerprints compute -i input.csv -o output.csv --type morgan

# With options
rdkit-cli fingerprints compute -i input.csv -o output.csv \
    --type morgan --radius 3 --bits 4096 --use-chirality
```

Supported types: morgan, maccs, rdkit, atompair, torsion, pattern

### filter

Filter molecules by various criteria.

```bash
# Substructure filter
rdkit-cli filter substructure -i input.csv -o output.csv --smarts "c1ccccc1"
rdkit-cli filter substructure -i input.csv -o output.csv --smarts "c1ccccc1" --exclude

# Property filter
rdkit-cli filter property -i input.csv -o output.csv --rule "MolWt < 500"

# Drug-likeness filters
rdkit-cli filter druglike -i input.csv -o output.csv --rule lipinski
rdkit-cli filter druglike -i input.csv -o output.csv --rule veber
rdkit-cli filter druglike -i input.csv -o output.csv --rule ghose

# PAINS filter
rdkit-cli filter pains -i input.csv -o output.csv
```

### convert

Convert between molecular file formats.

```bash
# Auto-detect formats from extensions
rdkit-cli convert -i molecules.csv -o molecules.sdf

# Explicit format specification
rdkit-cli convert -i molecules.csv -o molecules.smi --out-format smi
```

Supported formats: csv, tsv, smi, sdf, parquet

### standardize

Standardize and canonicalize molecules.

```bash
# Basic standardization
rdkit-cli standardize -i input.csv -o output.csv

# With options
rdkit-cli standardize -i input.csv -o output.csv \
    --cleanup --neutralize --fragment-parent
```

### similarity

Compute molecular similarity.

```bash
# Similarity search
rdkit-cli similarity search -i library.csv -o hits.csv \
    --query "CCO" --threshold 0.7

# Similarity matrix
rdkit-cli similarity matrix -i molecules.csv -o matrix.csv \
    --metric tanimoto

# Clustering
rdkit-cli similarity cluster -i molecules.csv -o clustered.csv \
    --cutoff 0.5
```

### conformers

Generate and optimize 3D conformers.

```bash
# Generate conformers
rdkit-cli conformers generate -i input.csv -o output.sdf --num 10

# Optimize conformers
rdkit-cli conformers optimize -i input.sdf -o optimized.sdf --force-field mmff
```

### reactions

Apply chemical reactions and transformations.

```bash
# SMIRKS transformation
rdkit-cli reactions transform -i input.csv -o output.csv \
    --smirks "[OH:1]>>[O-:1]"

# Reaction enumeration
rdkit-cli reactions enumerate -i reactants.csv -o products.csv \
    --template "reaction.rxn"
```

### scaffold

Extract molecular scaffolds.

```bash
# Murcko scaffolds
rdkit-cli scaffold murcko -i input.csv -o scaffolds.csv

# Generic scaffolds
rdkit-cli scaffold murcko -i input.csv -o scaffolds.csv --generic

# Scaffold decomposition
rdkit-cli scaffold decompose -i input.csv -o decomposed.csv
```

### enumerate

Enumerate molecular variants.

```bash
# Stereoisomers
rdkit-cli enumerate stereoisomers -i input.csv -o isomers.csv --max-isomers 32

# Tautomers
rdkit-cli enumerate tautomers -i input.csv -o tautomers.csv --max-tautomers 50

# Canonical tautomer
rdkit-cli enumerate canonical-tautomer -i input.csv -o canonical.csv
```

### fragment

Fragment molecules.

```bash
# BRICS fragmentation
rdkit-cli fragment brics -i input.csv -o fragments.csv

# RECAP fragmentation
rdkit-cli fragment recap -i input.csv -o fragments.csv

# Functional group extraction
rdkit-cli fragment functional-groups -i input.csv -o groups.csv

# Fragment frequency analysis
rdkit-cli fragment analyze -i fragments.csv -o analysis.csv
```

### diversity

Analyze and select diverse molecules.

```bash
# Pick diverse subset
rdkit-cli diversity pick -i input.csv -o diverse.csv -k 100

# Analyze diversity
rdkit-cli diversity analyze -i input.csv
```

### mcs

Find Maximum Common Substructure.

```bash
# Find MCS across molecules
rdkit-cli mcs -i molecules.csv -o mcs_result.csv

# With options
rdkit-cli mcs -i molecules.csv -o mcs_result.csv \
    --timeout 60 --atom-compare elements
```

### depict

Generate molecular depictions.

```bash
# Single molecule
rdkit-cli depict single --smiles "c1ccccc1" -o benzene.svg

# Batch depiction
rdkit-cli depict batch -i molecules.csv -o images/ -f svg

# Grid image
rdkit-cli depict grid -i molecules.csv -o grid.svg --mols-per-row 4
```

## Global Options

| Option | Description |
|--------|-------------|
| `-n, --ncpu N` | Number of CPUs (-1 = all, default: -1) |
| `-i, --input FILE` | Input file |
| `-o, --output FILE` | Output file |
| `--smiles-column COL` | SMILES column name (default: "smiles") |
| `--name-column COL` | Name column (optional) |
| `--no-header` | Input has no header row |
| `-q, --quiet` | Suppress progress output |
| `-V, --version` | Show version |
| `-h, --help` | Show help |

## Input/Output Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | .csv | Comma-separated, with header |
| TSV | .tsv | Tab-separated, with header |
| SMI | .smi | SMILES format, space-separated |
| SDF | .sdf | Structure-Data File |
| Parquet | .parquet | Apache Parquet format |

## Examples

### Cheminformatics Pipeline

```bash
# 1. Standardize input molecules
rdkit-cli standardize -i raw.csv -o std.csv --cleanup --neutralize

# 2. Filter by drug-likeness
rdkit-cli filter druglike -i std.csv -o druglike.csv --rule lipinski

# 3. Compute descriptors
rdkit-cli descriptors compute -i druglike.csv -o desc.csv -d MolWt,MolLogP,TPSA,HBD,HBA

# 4. Select diverse subset
rdkit-cli diversity pick -i druglike.csv -o diverse.csv -k 500

# 5. Generate depictions
rdkit-cli depict grid -i diverse.csv -o library.svg --mols-per-row 10
```

### Similarity Screening

```bash
# Search for similar compounds
rdkit-cli similarity search -i library.csv -o hits.csv \
    --query "CC(=O)Oc1ccccc1C(=O)O" \
    --threshold 0.6 \
    --type morgan

# Cluster results
rdkit-cli similarity cluster -i hits.csv -o clustered.csv --cutoff 0.4
```

### Scaffold Analysis

```bash
# Extract scaffolds
rdkit-cli scaffold murcko -i library.csv -o scaffolds.csv

# Analyze scaffold diversity
rdkit-cli diversity analyze -i scaffolds.csv --smiles-column scaffold
```

## Development

```bash
# Clone repository
git clone https://github.com/vitruves/rdkit-cli
cd rdkit-cli

# Install with dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=rdkit_cli
```

## License

Apache 2.0
