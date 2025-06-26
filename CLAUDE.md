# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RDKit CLI is a comprehensive command-line interface for RDKit cheminformatics operations, written in Python. The project has transitioned from a C++ implementation to a Python-based modular architecture.

## Development Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Run the CLI
rdkit-cli --help
```

### Code Quality and Testing
```bash
# Format code
black rdkit_cli/
isort rdkit_cli/

# Type checking
mypy rdkit_cli/

# Linting
flake8 rdkit_cli/

# Run tests (when test directory exists)
pytest
pytest --cov=rdkit_cli --cov-report=term-missing
```

### Building and Distribution
```bash
# Build package
python -m build

# Install locally
pip install .
```

## Architecture Overview

### Core Structure
The application follows a modular command-based architecture:

- **`rdkit_cli/main.py`**: Entry point and command routing with argparse-based CLI
- **`rdkit_cli/core/`**: Core utilities embedded in `__init__.py`:
  - **Logging**: Colorized logging with timestamps using `colorlog`
  - **Configuration**: JSON-based config management in `~/.config/rdkit-cli/`
  - **Common utilities**: File validation, molecule I/O, graceful exit handling
- **`rdkit_cli/commands/`**: Modular command implementations

### Command Architecture
Each command module implements:
- `add_subparser(subparsers)`: Register CLI subcommands
- Command handler functions that accept `(args, graceful_exit: GracefulExit)`
- Progress tracking with `tqdm`
- Proper error handling and logging

### Key Patterns
- **Graceful Exit**: All operations support `Ctrl+C` interruption via `GracefulExit`
- **Progress Tracking**: Long operations use `tqdm` with colored progress bars
- **Validation**: Input/output paths validated before processing
- **Logging**: Structured logging with different verbosity levels
- **Configuration**: Persistent user preferences in JSON config

## Command Categories

### I/O Operations (`io_ops.py`)
- `convert`: File format conversion (SDF, SMILES, CSV, etc.)
- `standardize`: Molecular standardization (salt removal, neutralization)
- `validate`: Structure validation and filtering
- `split`: Dataset splitting into chunks
- `merge`: Multiple file merging
- `deduplicate`: Duplicate removal using various hash methods

### Descriptors (`descriptors.py`)
- `descriptors`: Calculate molecular descriptors with predefined sets
- `physicochemical`: Drug-like properties with filters (Lipinski, Veber, Egan)
- `admet`: ADMET property prediction using simplified models

### Fingerprints (`fingerprints.py`)
- `fingerprints`: Generate various fingerprint types (Morgan, RDKit, MACCS)
- `similarity`: Molecular similarity search
- `similarity-matrix`: Pairwise similarity calculations
- `cluster`: Fingerprint-based clustering (Butina, hierarchical, k-means)
- `diversity-pick`: Diverse subset selection (MaxMin, sphere exclusion)

### Additional Planned Modules
The architecture supports extensive additional functionality as outlined in `STRUCTURE.md`:
- Substructure analysis and scaffold decomposition  
- 3D conformer generation and alignment
- Fragment analysis and lead optimization
- Reaction processing and retrosynthesis
- Molecular optimization and docking preparation
- Visualization and reporting
- Database operations with SQLite backend
- Machine learning feature extraction
- Specialized analysis (toxicity alerts, SAR analysis)

## Development Guidelines

### Code Style
- **Black**: Line length 88, Python 3.9+ target
- **isort**: Black-compatible import sorting
- **Type hints**: Required for all function definitions
- **Documentation**: Docstrings for public functions

### Dependencies
- **Core**: RDKit, pandas, numpy, click
- **UI**: tqdm (progress), colorlog (logging), rich-argparse (help)
- **ML**: scikit-learn (clustering), joblib (parallelization)
- **Dev**: pytest, black, isort, mypy, flake8

### Error Handling
- Validate inputs early using `validate_input_file()` and `validate_output_path()`
- Use structured logging at appropriate levels
- Support `--skip-errors` flags for batch processing
- Return appropriate exit codes (0=success, 1=error, 130=interrupted)

### Performance Considerations
- Use `get_parallel_jobs()` for CPU-bound operations
- Process large datasets in chunks when memory constrained
- Implement graceful interruption for long-running operations
- Cache expensive computations when appropriate

## Configuration System

Users can customize default behavior:
```bash
rdkit-cli config --set default_fps morgan
rdkit-cli config --set default_radius 2
rdkit-cli config --list
```

Configuration stored in `~/.config/rdkit-cli/config.json` with defaults for fingerprint parameters, chunk sizes, parallel jobs, and memory limits.