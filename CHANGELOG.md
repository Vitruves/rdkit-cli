# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2026-04-03

### Added

- **stereo**: New command for stereochemistry analysis — CIP label assignment (R/S, E/Z), stereocenter perception, enhanced stereo group inspection, and stereo cleanup/canonicalization
- **energy**: New command for force field energy calculations — single-point MMFF/UFF energy and structure minimization with convergence reporting
- **pharmacophore**: New command for pharmacophore feature perception (Donor, Acceptor, Aromatic, etc.) and 2D pharmacophore fingerprint similarity search
- **fingerprints**: Added Avalon, MHFP (MinHash), and 2D pharmacophore (Gobbi) fingerprint types
- **descriptors**: Added 42 Molecular Quantum Numbers (MQN) and 10 3D shape descriptors (PMI, NPR, Asphericity, Eccentricity, SpherocityIndex, PBF); new `--mqn`, `--3d`, and `--generate-conformers` flags
- **similarity**: Added 8 metrics (AllBit, Asymmetric, BraunBlanquet, Kulczynski, McConnaughey, OnBit, RogotGoldberg, Tversky with alpha/beta); new `shape` subcommand for 3D shape similarity (Tanimoto, Protrude, Tversky)
- **filter**: Expanded structural alert catalogs — `--catalog` option supports PAINS, PAINS_A/B/C, Brenk, NIH, ZINC, and all combined; added `alerts` subcommand as alias
- **conformers**: Added `constrained` subcommand for template-constrained 3D embedding and `torsion` subcommand for dihedral angle scanning with energy profiles
- **reactions**: Added `map` subcommand for atom-atom mapping inspection (text/JSON) and `fingerprint` subcommand for reaction difference/structural fingerprints
- **scaffold**: Added `network` subcommand for scaffold network construction (CSV/JSON output) using rdScaffoldNetwork
- **props**: Added `charges` subcommand for Gasteiger partial charges and `crippen` subcommand for per-atom LogP/MR contributions
- **fragment**: Added `brics-build` subcommand for recombining BRICS fragments into new molecules
- **depict**: Added `highlight` subcommand for SMARTS-based atom/bond highlighting with custom RGB colors

### Changed

- Total command count increased from 29 to 32 (stereo, energy, pharmacophore)
- Total fingerprint types increased from 6 to 9
- Total descriptor count increased from ~133 to ~185
- Similarity metrics increased from 5 to 13

## [0.3.1] - 2026-03-14

### Changed

- **Migrated to MorganGenerator API**: replaced deprecated `GetMorganFingerprintAsBitVect` / `GetHashedMorganFingerprint` with `rdFingerprintGenerator.GetMorganGenerator` across fingerprints, similarity, diversity, and sascorer modules. Also migrated AtomPair and TopologicalTorsion fingerprints.
- **Smart parallelism defaults**: global default changed from all cores (`-1`) to single-threaded (`1`), avoiding IPC overhead on fast commands. Heavy workloads (`descriptors --all`, `descriptors --category`) auto-scale to all cores. Users can always override with `-n -1`.
- **Optimized README**: replaced verbose per-command docs with compact help-style reference and benchmark table. Full command docs moved to `docs/commands.md`.

### Removed

- Dead `_fgs` reference in fragment module (unused `GetMorganFingerprint` assignment)

## [0.3.0] - 2026-01-10

### Added

- **info**: Quick molecule information from SMILES (formula, MW, LogP, TPSA, stereocenters, Lipinski violations, InChI/InChIKey)
- **merge**: Combine multiple molecule files with optional deduplication and source tracking
- **sascorer**: Calculate Synthetic Accessibility (SA) Score, Natural Product-likeness (NPC), and QED scores
- **rgroup**: R-group decomposition around a core SMARTS pattern with labeled attachment points
- **rings**: Ring system analysis - extract ring systems (fused, spiro, bridged) and analyze frequencies
- **align**: 3D molecular alignment to a reference structure (MCS-based or Open3DAlign)
- **rmsd**: RMSD calculations between 3D structures (compare to reference, pairwise matrix, conformer analysis)
- **mmp**: Matched Molecular Pairs analysis - fragment molecules, find pairs, apply transformations
- **protonate**: Protonation state enumeration at specified pH with neutralization option
- **props**: Property column operations - add, rename, drop, keep columns in molecule files

### Changed

- Total command count increased from 19 to 29

## [0.2.0] - 2026-01-06

### Added

- **stats**: Calculate dataset statistics (MolWt, LogP, TPSA, etc. with min/max/mean/median/stdev)
- **split**: Split files into smaller chunks (by number of chunks or chunk size)
- **sample**: Randomly sample molecules (by count or fraction, with reservoir sampling for large files)
- **deduplicate**: Remove duplicate molecules (by SMILES, InChI, InChIKey, or scaffold)
- **validate**: Validate molecular structures (valence, kekulization, stereo, element constraints)

### Changed

- Commands are now displayed in alphabetical order in help output
- Total command count increased from 14 to 19

## [0.1.0] - 2026-01-06

### Added

- Initial release with 14 command categories
- **descriptors**: Compute molecular descriptors (200+ available)
- **fingerprints**: Generate molecular fingerprints (morgan, maccs, rdkit, atompair, torsion, pattern)
- **filter**: Filter molecules by substructure, properties, drug-likeness (Lipinski/Veber/Ghose), PAINS
- **convert**: Convert between molecular file formats (CSV, TSV, SMI, SDF, Parquet)
- **standardize**: Standardize and canonicalize molecules
- **similarity**: Similarity search, matrix computation, and clustering
- **conformers**: Generate and optimize 3D conformers
- **reactions**: SMIRKS transformations and reaction enumeration
- **scaffold**: Murcko scaffold extraction and decomposition
- **enumerate**: Stereoisomer and tautomer enumeration
- **fragment**: BRICS/RECAP fragmentation and functional group analysis
- **diversity**: MaxMin diversity picking and diversity analysis
- **mcs**: Maximum Common Substructure finding
- **depict**: SVG/PNG molecular depictions (single, batch, grid)

### Features

- Multi-core parallel processing via ProcessPoolExecutor
- Ninja-style progress display with speed and ETA
- Support for multiple I/O formats (CSV, TSV, SMI, SDF, Parquet)
- Automatic format detection from file extensions
- Lazy imports for fast CLI startup (~0.08s)
- Comprehensive test suite (182 tests)

### Dependencies

- rdkit>=2024.3.1
- rich-argparse>=1.4.0
- pandas>=2.0.0
- pyarrow>=14.0.0
- numpy>=1.24.0
