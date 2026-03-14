# rdkit-cli

[![PyPI version](https://img.shields.io/pypi/v/rdkit-cli.svg)](https://pypi.org/project/rdkit-cli/)
[![Python versions](https://img.shields.io/pypi/pyversions/rdkit-cli.svg)](https://pypi.org/project/rdkit-cli/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/rdkit-cli?period=total&units=international_system&left_color=black&right_color=green&left_text=downloads)](https://pepy.tech/projects/rdkit-cli)
[![License](https://img.shields.io/pypi/l/rdkit-cli.svg)](https://github.com/vitruves/rdkit-cli/blob/main/LICENSE)

A high-performance CLI for cheminformatics workflows, powered by native RDKit (C++ under the hood).

**29 commands** | **5 I/O formats** (CSV, TSV, SMI, SDF, Parquet) | **multi-core parallel processing** | **~80ms startup**

## Installation

```bash
pip install rdkit-cli
```

## Quick Start

```bash
# Quick molecule info — no files needed
rdkit-cli info "c1ccccc1"

# Compute descriptors
rdkit-cli descriptors compute -i molecules.csv -o desc.csv -d MolWt,MolLogP,TPSA

# Filter by drug-likeness
rdkit-cli filter druglike -i molecules.csv -o filtered.csv --rule lipinski

# Similarity search
rdkit-cli similarity search -i library.csv -o hits.csv --query "c1ccccc1" --threshold 0.7

# Standardize structures
rdkit-cli standardize -i molecules.csv -o std.csv --cleanup --neutralize
```

## Commands

```
Usage: rdkit-cli [-h] [-V] <command> ...

Commands:
    align          Align 3D molecules to a reference
    conformers     Generate and optimize 3D conformers
    convert        Convert between molecular file formats
    deduplicate    Remove duplicate molecules
    depict         Generate molecular depictions (SVG/PNG)
    descriptors    Compute molecular descriptors
    diversity      Analyze and select diverse molecules
    enumerate      Enumerate stereoisomers and tautomers
    filter         Filter by substructure, properties, drug-likeness, PAINS
    fingerprints   Compute fingerprints (Morgan, MACCS, RDKit, AtomPair, Torsion)
    fragment       BRICS/RECAP fragmentation and functional groups
    info           Quick molecule information from SMILES
    mcs            Find Maximum Common Substructure
    merge          Merge multiple molecule files
    mmp            Matched Molecular Pairs analysis
    props          Property column operations (add, rename, drop, keep)
    protonate      Enumerate protonation states
    reactions      Apply SMIRKS transformations and enumerate products
    rgroup         R-group decomposition around a core
    rings          Ring system analysis and extraction
    rmsd           Calculate RMSD between 3D structures
    sample         Randomly sample molecules (reservoir sampling supported)
    sascorer       Synthetic accessibility, QED, and NP-likeness scores
    scaffold       Extract Murcko scaffolds
    similarity     Search, matrix, and clustering
    split          Split files into smaller chunks
    standardize    Standardize and canonicalize molecules
    stats          Calculate dataset statistics
    validate       Validate molecular structures

Use 'rdkit-cli <command> --help' for command-specific options.
```

## Global Options

| Option | Description |
|--------|-------------|
| `-i, --input FILE` | Input file |
| `-o, --output FILE` | Output file |
| `-n, --ncpu N` | Number of CPUs (-1 = all, default: 1; auto-scales for heavy commands) |
| `--smiles-column COL` | SMILES column name (default: "smiles") |
| `--name-column COL` | Name column (optional) |
| `--no-header` | Input has no header row |
| `-q, --quiet` | Suppress progress output |

## Example Pipeline

```bash
# Validate → deduplicate → standardize → filter → describe → pick diverse subset
rdkit-cli validate -i raw.csv -o valid.csv --valid-only
rdkit-cli deduplicate -i valid.csv -o unique.csv -b inchikey
rdkit-cli standardize -i unique.csv -o std.csv --cleanup --neutralize
rdkit-cli filter druglike -i std.csv -o druglike.csv --rule lipinski
rdkit-cli descriptors compute -i druglike.csv -o desc.csv -d MolWt,MolLogP,TPSA,HBD,HBA
rdkit-cli diversity pick -i druglike.csv -o diverse.csv -k 500
```

## Formats

| Format | Extension |
|--------|-----------|
| CSV | `.csv` |
| TSV | `.tsv` |
| SMILES | `.smi` |
| SDF | `.sdf` |
| Parquet | `.parquet` |

Formats are auto-detected from file extensions. Override with `--in-format` / `--out-format`.

## Performance

- **Native RDKit**: C++ computation with Python bindings — no performance penalty
- **Smart parallelism**: defaults to single-threaded for fast commands (avoids IPC overhead), auto-scales to all cores for heavy workloads (`descriptors --all`). Override with `-n -1`
- **Lazy imports**: ~80ms startup time regardless of installed packages
- **Streaming**: Memory-efficient reservoir sampling for large datasets

**Benchmarks** — 27K molecules, Apple M-series (8 cores):

| Command | Time | Throughput |
|---------|------|------------|
| `fingerprints compute --type morgan` | 3.1s | ~8,700 mol/s |
| `descriptors compute -d MolWt,MolLogP,TPSA` | 6.4s | ~4,200 mol/s |
| `filter druglike --rule lipinski` | 6.9s | ~3,900 mol/s |
| `standardize --cleanup --uncharge` | 7.0s | ~3,900 mol/s |
| `descriptors compute --all` (auto-parallel) | 55s | ~490 mol/s |

## Development

```bash
git clone https://github.com/vitruves/rdkit-cli
cd rdkit-cli
uv sync --dev
uv run pytest
```

## License

Apache 2.0
