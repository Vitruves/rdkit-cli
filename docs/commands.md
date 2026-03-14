# Command Reference

Full usage examples for all `rdkit-cli` commands. For quick reference, see the [README](../README.md).

## Table of Contents

- [align](#align)
- [conformers](#conformers)
- [convert](#convert)
- [deduplicate](#deduplicate)
- [depict](#depict)
- [descriptors](#descriptors)
- [diversity](#diversity)
- [enumerate](#enumerate)
- [filter](#filter)
- [fingerprints](#fingerprints)
- [fragment](#fragment)
- [info](#info)
- [mcs](#mcs)
- [merge](#merge)
- [mmp](#mmp)
- [props](#props)
- [protonate](#protonate)
- [reactions](#reactions)
- [rgroup](#rgroup)
- [rings](#rings)
- [rmsd](#rmsd)
- [sample](#sample)
- [sascorer](#sascorer)
- [scaffold](#scaffold)
- [similarity](#similarity)
- [split](#split)
- [standardize](#standardize)
- [stats](#stats)
- [validate](#validate)

---

## align

Align 3D molecules to a reference structure.

```bash
# MCS-based alignment (default)
rdkit-cli align -i probes.sdf -o aligned.sdf -r reference.sdf

# Open3DAlign method
rdkit-cli align -i probes.sdf -o aligned.sdf -r reference.sdf --method o3a
```

## conformers

Generate and optimize 3D conformers. Auto-scales to all cores.

```bash
# Generate conformers
rdkit-cli conformers generate -i input.csv -o output.sdf --num 10

# Optimize conformers
rdkit-cli conformers optimize -i input.sdf -o optimized.sdf --force-field mmff
```

## convert

Convert between molecular file formats.

```bash
# Auto-detect formats from extensions
rdkit-cli convert -i molecules.csv -o molecules.sdf

# Explicit format specification
rdkit-cli convert -i molecules.csv -o molecules.smi --out-format smi
```

Supported formats: csv, tsv, smi, sdf, parquet

## deduplicate

Remove duplicate molecules.

```bash
# By canonical SMILES (default)
rdkit-cli deduplicate -i molecules.csv -o unique.csv

# By InChIKey
rdkit-cli deduplicate -i molecules.csv -o unique.csv -b inchikey

# By scaffold
rdkit-cli deduplicate -i molecules.csv -o unique.csv -b scaffold

# Keep last occurrence instead of first
rdkit-cli deduplicate -i molecules.csv -o unique.csv --keep last
```

## depict

Generate molecular depictions.

```bash
# Single molecule
rdkit-cli depict single --smiles "c1ccccc1" -o benzene.svg

# Batch depiction
rdkit-cli depict batch -i molecules.csv -o images/ -f svg

# Grid image
rdkit-cli depict grid -i molecules.csv -o grid.svg --mols-per-row 4
```

## descriptors

Compute molecular descriptors. Auto-scales to all cores when using `--all` or `--category`.

```bash
# List available descriptors
rdkit-cli descriptors list
rdkit-cli descriptors list --all

# Compute specific descriptors
rdkit-cli descriptors compute -i input.csv -o output.csv -d MolWt,MolLogP,TPSA

# Compute all descriptors (auto-parallel)
rdkit-cli descriptors compute -i input.csv -o output.csv --all
```

## diversity

Analyze and select diverse molecules.

```bash
# Pick diverse subset
rdkit-cli diversity pick -i input.csv -o diverse.csv -k 100

# Analyze diversity
rdkit-cli diversity analyze -i input.csv
```

## enumerate

Enumerate molecular variants.

```bash
# Stereoisomers
rdkit-cli enumerate stereoisomers -i input.csv -o isomers.csv --max-isomers 32

# Tautomers
rdkit-cli enumerate tautomers -i input.csv -o tautomers.csv --max-tautomers 50

# Canonical tautomer
rdkit-cli enumerate canonical-tautomer -i input.csv -o canonical.csv
```

## filter

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

## fingerprints

Compute molecular fingerprints.

```bash
# List available types
rdkit-cli fingerprints list

# Morgan fingerprints (default)
rdkit-cli fingerprints compute -i input.csv -o output.csv --type morgan

# With options
rdkit-cli fingerprints compute -i input.csv -o output.csv \
    --type morgan --radius 3 --bits 4096 --use-chirality
```

Supported types: morgan, maccs, rdkit, atompair, torsion, pattern

## fragment

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

## info

Quick molecule information from SMILES.

```bash
# Basic info
rdkit-cli info "CCO"

# JSON output
rdkit-cli info "c1ccccc1" --json

# Shows: formula, MW, LogP, TPSA, stereocenters, Lipinski violations, InChI/InChIKey
```

## mcs

Find Maximum Common Substructure.

```bash
# Find MCS across molecules
rdkit-cli mcs -i molecules.csv -o mcs_result.csv

# With options
rdkit-cli mcs -i molecules.csv -o mcs_result.csv \
    --timeout 60 --atom-compare elements
```

## merge

Combine multiple molecule files.

```bash
# Merge two files
rdkit-cli merge -i file1.csv file2.csv -o merged.csv

# Merge with deduplication
rdkit-cli merge -i file1.csv file2.csv -o merged.csv --dedupe

# Track source file
rdkit-cli merge -i file1.csv file2.csv -o merged.csv --source-column source
```

## mmp

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

## props

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

## protonate

Protonation state enumeration.

```bash
# Enumerate at physiological pH
rdkit-cli protonate -i molecules.csv -o protonated.csv --ph 7.4

# Neutralize charged molecules
rdkit-cli protonate -i molecules.csv -o neutral.csv --neutralize

# Enumerate all states
rdkit-cli protonate -i molecules.csv -o states.csv --enumerate-all
```

## reactions

Apply chemical reactions and transformations.

```bash
# SMIRKS transformation
rdkit-cli reactions transform -i input.csv -o output.csv \
    --smirks "[OH:1]>>[O-:1]"

# Reaction enumeration
rdkit-cli reactions enumerate -i reactants.csv -o products.csv \
    --template "reaction.rxn"
```

## rgroup

R-group decomposition around a core structure.

```bash
# Decompose around benzene core
rdkit-cli rgroup -i molecules.csv -o decomposed.csv --core "c1ccc([*:1])cc1"

# Multiple attachment points
rdkit-cli rgroup -i molecules.csv -o decomposed.csv \
    --core "c1ccc([*:1])cc([*:2])1"
```

## rings

Ring system analysis.

```bash
# Extract ring systems
rdkit-cli rings extract -i molecules.csv -o rings.csv

# Ring information (counts, sizes, aromaticity)
rdkit-cli rings info -i molecules.csv -o ring_info.csv

# Frequency analysis
rdkit-cli rings frequency -i molecules.csv -o ring_freq.csv
```

## rmsd

RMSD calculations between 3D structures.

```bash
# Compare to reference
rdkit-cli rmsd compare -i molecules.sdf -o results.csv -r reference.sdf

# Pairwise RMSD matrix
rdkit-cli rmsd matrix -i molecules.sdf -o matrix.csv

# Conformer RMSD analysis
rdkit-cli rmsd conformers -i multi_conf.sdf -o conf_rmsd.csv
```

## sample

Randomly sample molecules.

```bash
# Sample by count
rdkit-cli sample -i molecules.csv -o sample.csv -k 100 --seed 42

# Sample by fraction
rdkit-cli sample -i molecules.csv -o sample.csv -f 0.1

# Memory-efficient streaming (reservoir sampling)
rdkit-cli sample -i huge.csv -o sample.csv -k 1000 --stream
```

## sascorer

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

## scaffold

Extract molecular scaffolds.

```bash
# Murcko scaffolds
rdkit-cli scaffold murcko -i input.csv -o scaffolds.csv

# Generic scaffolds
rdkit-cli scaffold murcko -i input.csv -o scaffolds.csv --generic

# Scaffold decomposition
rdkit-cli scaffold decompose -i input.csv -o decomposed.csv
```

## similarity

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

## split

Split files into smaller chunks.

```bash
# Split into N files
rdkit-cli split -i large.csv -o chunks/ -c 10

# Split by chunk size
rdkit-cli split -i large.csv -o chunks/ -s 1000

# With custom prefix
rdkit-cli split -i large.csv -o chunks/ -c 5 --prefix molecules
```

## standardize

Standardize and canonicalize molecules.

```bash
# Basic standardization
rdkit-cli standardize -i input.csv -o output.csv

# With cleanup and uncharging
rdkit-cli standardize -i input.csv -o output.csv --cleanup --uncharge

# With fragment parent
rdkit-cli standardize -i input.csv -o output.csv --cleanup --fragment-parent
```

## stats

Calculate dataset statistics.

```bash
# Basic statistics
rdkit-cli stats -i molecules.csv -o stats.json --format json

# Specific properties
rdkit-cli stats -i molecules.csv -p MolWt,LogP,TPSA

# List available properties
rdkit-cli stats -i molecules.csv --list-properties
```

## validate

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

---

## Example Pipelines

### Cheminformatics Pipeline

```bash
rdkit-cli validate -i raw.csv -o valid.csv --valid-only
rdkit-cli deduplicate -i valid.csv -o unique.csv -b inchikey
rdkit-cli standardize -i unique.csv -o std.csv --cleanup --uncharge
rdkit-cli filter druglike -i std.csv -o druglike.csv --rule lipinski
rdkit-cli descriptors compute -i druglike.csv -o desc.csv -d MolWt,MolLogP,TPSA,HBD,HBA
rdkit-cli stats -i druglike.csv -o stats.json --format json
rdkit-cli diversity pick -i druglike.csv -o diverse.csv -k 500
rdkit-cli depict grid -i diverse.csv -o library.svg --mols-per-row 10
```

### Similarity Screening

```bash
rdkit-cli similarity search -i library.csv -o hits.csv \
    --query "CC(=O)Oc1ccccc1C(=O)O" --threshold 0.6 --type morgan
rdkit-cli similarity cluster -i hits.csv -o clustered.csv --cutoff 0.4
```

### Scaffold Analysis

```bash
rdkit-cli scaffold murcko -i library.csv -o scaffolds.csv
rdkit-cli diversity analyze -i scaffolds.csv --smiles-column scaffold
```

### Large Dataset Processing

```bash
# Stream-sample from a huge file
rdkit-cli sample -i huge_library.csv -o sample.csv -k 10000 --stream

# Split for external parallel processing
rdkit-cli split -i library.csv -o batches/ -c 10
ls batches/*.csv | xargs -P 4 -I {} rdkit-cli descriptors compute -i {} -o {}.desc.csv -d MolWt,LogP
```
