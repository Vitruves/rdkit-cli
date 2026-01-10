# rdkit-cli

A comprehensive, high-performance CLI tool wrapping RDKit functionality for cheminformatics workflows.

## Features

- **29 Command Categories**: align, conformers, convert, deduplicate, depict, descriptors, diversity, enumerate, filter, fingerprints, fragment, info, mcs, merge, mmp, props, protonate, reactions, rgroup, rings, rmsd, sample, sascorer, scaffold, similarity, split, standardize, stats, validate
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

### stats

Calculate dataset statistics.

```bash
# Basic statistics
rdkit-cli stats -i molecules.csv -o stats.json --format json

# Specific properties
rdkit-cli stats -i molecules.csv -p MolWt,LogP,TPSA

# List available properties
rdkit-cli stats -i molecules.csv --list-properties
```

### split

Split files into smaller chunks.

```bash
# Split into N files
rdkit-cli split -i large.csv -o chunks/ -c 10

# Split by chunk size
rdkit-cli split -i large.csv -o chunks/ -s 1000

# With custom prefix
rdkit-cli split -i large.csv -o chunks/ -c 5 --prefix molecules
```

### sample

Randomly sample molecules.

```bash
# Sample by count
rdkit-cli sample -i molecules.csv -o sample.csv -k 100 --seed 42

# Sample by fraction
rdkit-cli sample -i molecules.csv -o sample.csv -f 0.1

# Memory-efficient streaming (reservoir sampling)
rdkit-cli sample -i huge.csv -o sample.csv -k 1000 --stream
```

### deduplicate

Remove duplicate molecules.

```bash
# Deduplicate by canonical SMILES (default)
rdkit-cli deduplicate -i molecules.csv -o unique.csv

# Deduplicate by InChIKey
rdkit-cli deduplicate -i molecules.csv -o unique.csv -b inchikey

# Deduplicate by scaffold
rdkit-cli deduplicate -i molecules.csv -o unique.csv -b scaffold

# Keep last occurrence instead of first
rdkit-cli deduplicate -i molecules.csv -o unique.csv --keep last
```

### validate

Validate molecular structures.

```bash
# Basic validation
rdkit-cli validate -i molecules.csv -o validated.csv

# Output only valid molecules
rdkit-cli validate -i molecules.csv -o valid.csv --valid-only

# With constraints
rdkit-cli validate -i molecules.csv -o validated.csv \
    --max-atoms 100 --max-rings 8

# Check allowed elements
rdkit-cli validate -i molecules.csv -o validated.csv \
    --allowed-elements C,H,N,O,S,F,Cl

# Check stereo and show summary
rdkit-cli validate -i molecules.csv -o validated.csv \
    --check-stereo --summary
```

### info

Quick molecule information from SMILES.

```bash
# Basic info
rdkit-cli info "CCO"

# JSON output
rdkit-cli info "c1ccccc1" --json

# Shows: formula, MW, LogP, TPSA, stereocenters, Lipinski violations, InChI/InChIKey
```

### merge

Combine multiple molecule files.

```bash
# Merge two files
rdkit-cli merge -i file1.csv file2.csv -o merged.csv

# Merge with deduplication
rdkit-cli merge -i file1.csv file2.csv -o merged.csv --dedupe

# Track source file
rdkit-cli merge -i file1.csv file2.csv -o merged.csv --source-column source
```

### sascorer

Calculate synthetic accessibility and drug-likeness scores.

```bash
# SA Score only (default)
rdkit-cli sascorer -i molecules.csv -o scores.csv

# Include QED score
rdkit-cli sascorer -i molecules.csv -o scores.csv --qed

# Include Natural Product-likeness score
rdkit-cli sascorer -i molecules.csv -o scores.csv --npc

# All scores
rdkit-cli sascorer -i molecules.csv -o scores.csv --qed --npc
```

### rgroup

R-group decomposition around a core structure.

```bash
# Decompose around benzene core
rdkit-cli rgroup -i molecules.csv -o decomposed.csv --core "c1ccc([*:1])cc1"

# Multiple attachment points
rdkit-cli rgroup -i molecules.csv -o decomposed.csv \
    --core "c1ccc([*:1])cc([*:2])1"
```

### rings

Ring system analysis.

```bash
# Extract ring systems
rdkit-cli rings extract -i molecules.csv -o rings.csv

# Ring information (counts, sizes, aromaticity)
rdkit-cli rings info -i molecules.csv -o ring_info.csv

# Frequency analysis
rdkit-cli rings frequency -i molecules.csv -o ring_freq.csv
```

### align

3D molecular alignment.

```bash
# Align to reference structure (MCS-based)
rdkit-cli align -i probes.sdf -o aligned.sdf -r reference.sdf

# Open3DAlign method
rdkit-cli align -i probes.sdf -o aligned.sdf -r reference.sdf --method o3a
```

### rmsd

RMSD calculations between 3D structures.

```bash
# Compare to reference
rdkit-cli rmsd compare -i molecules.sdf -o results.csv -r reference.sdf

# Pairwise RMSD matrix
rdkit-cli rmsd matrix -i molecules.sdf -o matrix.csv

# Conformer RMSD analysis
rdkit-cli rmsd conformers -i multi_conf.sdf -o conf_rmsd.csv
```

### mmp

Matched Molecular Pairs analysis.

```bash
# Fragment molecules for MMP
rdkit-cli mmp fragment -i molecules.csv -o fragments.csv

# Find matched pairs
rdkit-cli mmp pairs -i fragments.csv -o pairs.csv

# Apply MMP transformation
rdkit-cli mmp transform -i molecules.csv -o transformed.csv \
    -t "[c:1][CH3]>>[c:1][NH2]"
```

### protonate

Protonation state enumeration.

```bash
# Enumerate at physiological pH
rdkit-cli protonate -i molecules.csv -o protonated.csv --ph 7.4

# Neutralize charged molecules
rdkit-cli protonate -i molecules.csv -o neutral.csv --neutralize

# Enumerate all states
rdkit-cli protonate -i molecules.csv -o states.csv --enumerate-all
```

### props

Property column operations.

```bash
# Add a column
rdkit-cli props add -i molecules.csv -o output.csv -c series -v "series_A"

# Rename a column
rdkit-cli props rename -i molecules.csv -o output.csv --from name --to mol_name

# Drop columns
rdkit-cli props drop -i molecules.csv -o output.csv -c col1,col2

# Keep only specific columns
rdkit-cli props keep -i molecules.csv -o output.csv -c smiles,name,MolWt

# List columns
rdkit-cli props list -i molecules.csv
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
# 1. Validate and filter input
rdkit-cli validate -i raw.csv -o validated.csv --valid-only

# 2. Deduplicate
rdkit-cli deduplicate -i validated.csv -o unique.csv -b inchikey

# 3. Standardize molecules
rdkit-cli standardize -i unique.csv -o std.csv --cleanup --neutralize

# 4. Filter by drug-likeness
rdkit-cli filter druglike -i std.csv -o druglike.csv --rule lipinski

# 5. Compute descriptors
rdkit-cli descriptors compute -i druglike.csv -o desc.csv -d MolWt,MolLogP,TPSA,HBD,HBA

# 6. Get dataset statistics
rdkit-cli stats -i druglike.csv -o stats.json --format json

# 7. Select diverse subset
rdkit-cli diversity pick -i druglike.csv -o diverse.csv -k 500

# 8. Generate depictions
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

### Large Dataset Processing

```bash
# Sample from a huge dataset
rdkit-cli sample -i huge_library.csv -o sample.csv -k 10000 --stream

# Split for parallel processing
rdkit-cli split -i library.csv -o batches/ -c 10

# Process batches in parallel (using xargs)
ls batches/*.csv | xargs -P 4 -I {} rdkit-cli descriptors compute -i {} -o {}.desc.csv -d MolWt,LogP
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
