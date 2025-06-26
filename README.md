# RDKit CLI

A comprehensive command-line interface for RDKit cheminformatics operations with extensive CSV and Parquet support.

## Installation

```bash
# Install with uv (recommended)
uv add rdkit-cli

# Install from source
pip install -e .

# Install with development dependencies
pip install -e .[dev]
```

## Input/Output Support

RDKit CLI supports multiple input and output formats:

### Input Formats

- **SDF** (`.sdf`) - Recommended for 3D structures, conformers, and molecules with properties
- **SMILES** (`.smi`, `.smiles`) - Best for large datasets, simple structure lists
- **MOL** (`.mol`) - Single molecule files
- **CSV** (`.csv`) - Tabular data with SMILES column, great for datasets with metadata
- **Parquet** (`.parquet`) - Efficient binary format for large datasets with fast I/O
- **Direct SMILES** - Use `-S "CCO,CCN,CCC"` for quick operations

### Output Formats

- **CSV** (`.csv`) - Human-readable, compatible with Excel and data analysis tools
- **Parquet** (`.parquet`) - Compact, fast loading, preserves data types
- **SDF** (`.sdf`) - 3D structures with properties
- **SMILES** (`.smi`) - Simple structure lists

### Input Method Selection Guide

**Use SDF when:**

- Working with 3D conformers or optimized structures
- Need to preserve molecular properties and metadata
- Performing 3D-based operations (alignment, shape similarity)

**Use SMILES when:**

- Working with large datasets (>10K molecules)
- Only need 2D structure information
- Fast processing is priority

**Use CSV when:**

- Have molecular data with additional columns (activity, properties)
- Need to specify custom SMILES column names
- Working with spreadsheet-like data

**Use Parquet when:**

- Working with very large datasets (>100K molecules)
- Need fastest I/O performance
- Want to preserve exact data types and compression

**Use Direct SMILES when:**

- Quick testing or single-molecule operations
- Don't have a file ready
- Prototyping workflows

## Core I/O Operations

### File Format Conversion

```bash
# Convert between formats
rdkit-cli convert -i molecules.smi -o molecules.sdf
rdkit-cli convert -i data.csv -c smiles_column -o structures.parquet

# With direct SMILES input
rdkit-cli convert -S "CCO,CCN,CCC" -o molecules.sdf
```

### Molecular Structure Operations

```bash
# Standardize structures
rdkit-cli standardize -i molecules.sdf -o clean.sdf --remove-salts --neutralize

# Validate molecules
rdkit-cli validate -i molecules.sdf -o valid.sdf --strict

# Split datasets
rdkit-cli split -i large_dataset.sdf -o train.sdf --test-file test.sdf --ratio 0.8

# Merge files
rdkit-cli merge -i "file1.sdf,file2.sdf" -o combined.sdf

# Remove duplicates
rdkit-cli deduplicate -i molecules.sdf -o unique.sdf --method inchi
```

## Descriptors & Properties

### Molecular Descriptors

```bash
# Basic descriptors to CSV
rdkit-cli descriptors -i molecules.sdf -o descriptors.csv

# Custom descriptor set to Parquet (fastest for large datasets)
rdkit-cli descriptors -i molecules.sdf -o descriptors.parquet --descriptor-set druglike

# Specific descriptors
rdkit-cli descriptors -i molecules.sdf -o results.csv --descriptors "MolWt,LogP,TPSA"

# With direct SMILES and custom column
rdkit-cli descriptors -S "CCO,CCN,CCC" -o quick_descriptors.csv
rdkit-cli descriptors -i data.csv -c molecule_smiles -o results.parquet
```

### Physicochemical Properties

```bash
# Basic physicochemical properties
rdkit-cli physicochemical -i molecules.sdf -o props.csv

# With drug-like filters
rdkit-cli physicochemical -i molecules.sdf -o props.csv --include-druglike-filters --include-qed
```

### ADMET Properties

```bash
# Basic ADMET predictions
rdkit-cli admet -i molecules.sdf -o admet.csv

# All available models
rdkit-cli admet -i molecules.sdf -o admet.csv --models all
```

## Fingerprints & Similarity

### Fingerprint Generation

```bash
# Morgan fingerprints (most common)
rdkit-cli fingerprints -i molecules.sdf -o fingerprints.pkl --fp-type morgan

# MACCS keys for similarity searching
rdkit-cli fingerprints -i molecules.sdf -o fingerprints.pkl --fp-type maccs

# Custom parameters
rdkit-cli fingerprints -i molecules.sdf -o fingerprints.pkl --fp-type morgan --radius 3 --n-bits 4096

# Direct SMILES input
rdkit-cli fingerprints -S "CCO,CCN,CCC" -o test_fps.pkl
```

### Similarity Analysis

```bash
# Find similar molecules
rdkit-cli similarity -i query.smi --database molecules.sdf -o similar.csv --threshold 0.7

# Direct query SMILES
rdkit-cli similarity -S "CCO" --database molecules.sdf -o similar.csv

# Similarity matrix for clustering
rdkit-cli similarity-matrix -i molecules.sdf -o matrix.csv --fp-type morgan

# Molecular clustering
rdkit-cli cluster -i molecules.sdf -o clusters.csv --method butina --threshold 0.8

# Diversity selection
rdkit-cli diversity-pick -i molecules.sdf -o diverse.sdf --count 100 --method maxmin
```

## Substructure Analysis

### Substructure Searching

```bash
# Search for substructures
rdkit-cli substructure-search -i molecules.sdf --query "c1ccccc1" -o hits.sdf

# Count matches
rdkit-cli substructure-search -i molecules.sdf --query "C" -o counts.csv --count-matches

# Direct SMILES input
rdkit-cli substructure-search -S "CCO,CCCC" --query "C" -o hits.sdf
```

### SMARTS Filtering

```bash
# Filter with SMARTS patterns
rdkit-cli substructure-filter -i molecules.sdf --smarts-file patterns.txt -o filtered.sdf --mode exclude
```

### Scaffold Analysis

```bash
# Analyze scaffolds
rdkit-cli scaffold-analysis -i molecules.sdf -o scaffolds.csv --include-counts

# Extract Murcko scaffolds
rdkit-cli murcko-scaffolds -i molecules.sdf -o scaffolds.sdf --generic --unique-only

# Functional group analysis
rdkit-cli functional-groups -i molecules.sdf -o groups.csv --hierarchy ifg
```

## 3D Structure & Conformers

### Conformer Generation

```bash
# Generate 3D conformers (use SDF input/output for 3D)
rdkit-cli conformers -i molecules.smi -o conformers.sdf --num-confs 10 --method etkdg

# With optimization
rdkit-cli conformers -i molecules.smi -o optimized.sdf --optimize --ff mmff94

# Direct SMILES input
rdkit-cli conformers -S "CCO,CCN" -o quick_conformers.sdf --num-confs 5
```

### Molecular Alignment

```bash
# Align to template
rdkit-cli align-molecules -i molecules.sdf --template template.sdf -o aligned.sdf --align-mode mcs
```

### Shape Similarity

```bash
# 3D shape similarity (requires 3D structures)
rdkit-cli shape-similarity -i molecules.sdf --reference "CCO" -o similarity.csv --threshold 0.3
```

## Fragment Analysis

### Molecular Fragmentation

```bash
# BRICS fragmentation
rdkit-cli fragment -i molecules.sdf -o fragments.sdf --method brics --min-fragment-size 3

# RECAP fragmentation
rdkit-cli fragment -i molecules.sdf -o fragments.sdf --method recap --include-parent

# Direct SMILES input
rdkit-cli fragment -S "CCO,CCN,CCC" -o fragments.sdf --method brics
```

### Fragment-based Analysis

```bash
# Fragment similarity
rdkit-cli fragment-similarity -i molecules.sdf --reference-frags fragments.smi -o similarity.csv

# Lead optimization
rdkit-cli lead-optimization -i lead.sdf --fragment-library fragments.smi -o optimized.sdf --max-products 50
```

## Machine Learning Support

### Feature Generation

```bash
# Morgan fingerprints for ML
rdkit-cli ml-features -i molecules.sdf -o features.csv --feature-type morgan_fp

# Combined features
rdkit-cli ml-features -i molecules.sdf -o features.csv --feature-type combined

# Direct SMILES input
rdkit-cli ml-features -S "CCO,CCN,CCC" -o features.csv --feature-type descriptors
```

### Dataset Preparation

```bash
# Split for ML training
rdkit-cli ml-split -i dataset.csv --train-file train.csv --test-file test.csv --split-ratio 0.8

# Stratified splitting
rdkit-cli ml-split -i dataset.csv --train-file train.csv --test-file test.csv --stratify activity_class
```

## Specialized Analysis

### Toxicity Screening

```bash
# Screen for toxicity alerts
rdkit-cli toxicity-alerts -i molecules.sdf -o alerts.csv --alert-set brenk

# Multiple alert sets
rdkit-cli toxicity-alerts -i molecules.sdf -o alerts.csv --alert-set pains

# Direct SMILES input
rdkit-cli toxicity-alerts -S "CCO,CCN,CCC" -o alerts.csv --alert-set nih
```

### SAR Analysis

```bash
# Structure-activity relationships
rdkit-cli sar-analysis -i molecules.sdf --activity-file activities.csv -o sar_report.html

# Matched molecular pairs
rdkit-cli matched-pairs -i molecules.sdf --activity-file activities.csv -o pairs.csv --max-pairs 1000

# Free-Wilson analysis
rdkit-cli free-wilson -i molecules.sdf --activity-file activities.csv -o free_wilson.json

# QSAR descriptors
rdkit-cli qsar-descriptors -i molecules.sdf -o qsar.csv --include-3d
```

## Database Operations

### Database Creation

```bash
# Create molecular database
rdkit-cli db-create -i molecules.sdf --db-file molecules.db --index-fps

# With specific fingerprint type
rdkit-cli db-create -i molecules.sdf --db-file molecules.db --index-fps --fp-type morgan
```

### Database Operations

```bash
# Search database
rdkit-cli db-search --db-file molecules.db --query "CCO" -o results.csv --similarity 0.7

# Filter by properties
rdkit-cli db-filter --db-file molecules.db --filters lipinski -o filtered_ids.csv

# Export molecules
rdkit-cli db-export --db-file molecules.db -o molecules.sdf --format sdf
```

## Utilities

### File Information

```bash
# Get file statistics
rdkit-cli info -i molecules.sdf

# Detailed statistics with descriptors
rdkit-cli stats -i molecules.sdf -o stats.json --include-descriptors

# Direct SMILES input
rdkit-cli info -S "CCO,CCN,CCC"
```

### Molecular Sampling

```bash
# Random sampling
rdkit-cli sample -i molecules.sdf -o sample.sdf --count 1000 --method random

# Diverse sampling
rdkit-cli sample -i molecules.sdf -o diverse.sdf --count 100 --method diverse

# Systematic sampling
rdkit-cli sample -i molecules.sdf -o systematic.sdf --count 500 --method systematic
```

### Performance & Configuration

```bash
# Benchmark operations
rdkit-cli benchmark -i molecules.sdf --operations "descriptors,fingerprints"

# Configuration management
rdkit-cli config --list
rdkit-cli config --set default_fps morgan
rdkit-cli config --set default_radius 2
rdkit-cli config --reset
```

## Advanced Usage Tips

### Working with Large Datasets

- Use Parquet format for datasets >100K molecules
- Use `--skip-errors` flag to handle problematic molecules
- Set parallel jobs with `-j N` for faster processing
- Consider sampling large datasets before expensive operations

### Custom SMILES Columns

```bash
# Specify custom column names in CSV/Parquet files
rdkit-cli descriptors -i data.csv -c molecule_structure -o results.parquet
rdkit-cli descriptors -i compounds.parquet -c canonical_smiles -o descriptors.csv
```

### Chaining Operations

```bash
# Convert and calculate descriptors
rdkit-cli convert -i molecules.smi -o molecules.sdf
rdkit-cli descriptors -i molecules.sdf -o descriptors.csv --descriptor-set druglike

# Filter and analyze
rdkit-cli substructure-filter -i molecules.sdf --smarts-file filters.txt -o clean.sdf
rdkit-cli descriptors -i clean.sdf -o clean_descriptors.csv
```

### Error Handling

- Use `--skip-errors` to continue processing when individual molecules fail
- Check file formats with `rdkit-cli info` before processing
- Validate structures with `rdkit-cli validate` to identify problems

## Command Reference

### Global Options

- `-v, --verbose`: Enable verbose logging
- `--debug`: Enable debug logging with detailed information
- `-j N, --jobs N`: Number of parallel jobs (auto-detected by default)

### Common Input Options

- `-i FILE, --input-file FILE`: Input file path
- `-S SMILES, --smiles SMILES`: Direct SMILES string(s) (comma-separated)
- `-c COL, --smiles-column COL`: SMILES column name for CSV/Parquet files

### Output Formats

- CSV: Human-readable, Excel compatible
- Parquet: Fast, compressed, preserves data types
- SDF: 3D structures with properties
- SMILES: Simple structure lists
- JSON: Structured data (reports, statistics)

## Requirements

- Python ≥ 3.9
- RDKit ≥ 2024.3.1
- pandas ≥ 2.0.0
- numpy ≥ 1.24.0
- pyarrow ≥ 14.0.0 (for Parquet support)
- scikit-learn ≥ 1.3.0
- matplotlib ≥ 3.7.0

See `pyproject.toml` for complete dependency list.

## Configuration

Configuration is stored in `~/.config/rdkit-cli/config.json`:

```json
{
    "default_fps": "morgan",
    "default_radius": 2,
    "default_jobs": 4,
    "default_threshold": 0.7
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

For development:

```bash
# Install development dependencies
uv add --dev pytest pytest-cov pytest-mock black isort mypy ruff

# Run tests
pytest

# Code formatting
black rdkit_cli/
isort rdkit_cli/
ruff check rdkit_cli/
```
