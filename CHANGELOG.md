# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
