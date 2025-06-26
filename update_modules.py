#!/usr/bin/env python3
"""Script to systematically add CSV/Parquet support to all RDKit CLI modules."""

import re
from pathlib import Path

def update_module_args(file_path: Path):
    """Update argument parsing to include -S and -c options."""
    content = file_path.read_text()
    
    # Pattern to find input file arguments that need updating
    input_patterns = [
        (
            r"(\t+parser_\w+\.add_argument\(\s*\n\s*'-i', '--input-file',\s*\n\s*required=True,\s*\n\s*help='[^']*'\s*\n\s*\))",
            lambda m: m.group(1).replace("required=True,", "").replace("help='Input molecular file'", "help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'") + 
                     f"\n\t{m.group(1).split('.add_argument')[0]}.add_argument(\n\t\t'-S', '--smiles',\n\t\thelp='Direct SMILES string(s) - comma-separated for multiple'\n\t)" +
                     f"\n\t{m.group(1).split('.add_argument')[0]}.add_argument(\n\t\t'-c', '--smiles-column',\n\t\thelp='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'\n\t)"
        ),
        # Also handle cases with different help text patterns
        (
            r"(\t+parser_\w+\.add_argument\(\s*\n\s*'-i', '--input-file',\s*\n\s*required=True,\s*\n\s*help='[^']*file[^']*'\s*\n\s*\))",
            lambda m: m.group(1).replace("required=True,", "").replace("'Input molecular file'", "'Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'").replace("'Input file'", "'Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'") + 
                     f"\n\t{m.group(1).split('.add_argument')[0]}.add_argument(\n\t\t'-S', '--smiles',\n\t\thelp='Direct SMILES string(s) - comma-separated for multiple'\n\t)" +
                     f"\n\t{m.group(1).split('.add_argument')[0]}.add_argument(\n\t\t'-c', '--smiles-column',\n\t\thelp='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'\n\t)"
        )
    ]
    
    for pattern, replacement in input_patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    return content

def update_module_imports(file_path: Path):
    """Update imports to include helper functions."""
    content = file_path.read_text()
    
    # Update common imports
    import_patterns = [
        (
            r"from \.\.core\.common import \(\s*\n\s*([^)]+)\s*\n\)",
            lambda m: f"from ..core.common import (\n\t{m.group(1).strip()}, get_molecules_from_args, save_dataframe_with_format_detection\n)"
        )
    ]
    
    for pattern, replacement in import_patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    return content

def update_module_functions(file_path: Path):
    """Update module functions to use helper functions."""
    content = file_path.read_text()
    
    # Replace input file reading patterns
    function_patterns = [
        # Replace input_path = validate_input_file(args.input_file) followed by molecules = read_molecules(input_path)
        (
            r"(\s+)input_path = validate_input_file\(args\.input_file\)\s*\n\s*([^=\n]*)\s*\n\s*molecules = read_molecules\(input_path\)",
            r"\1molecules = get_molecules_from_args(args)\n\1\2"
        ),
        # Replace df.to_csv(output_path, index=False) with format detection
        (
            r"(\s+)df\.to_csv\(([^,]+), index=False\)",
            r"\1save_dataframe_with_format_detection(df, \2)"
        )
    ]
    
    for pattern, replacement in function_patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    return content

def main():
    """Update all command modules."""
    commands_dir = Path("rdkit_cli/commands")
    
    # List of module files to update
    modules = [
        "substructure.py",
        "conformers.py", 
        "fragments.py",
        "reactions.py",
        "optimization.py",
        "visualization.py",
        "database.py",
        "ml_support.py",
        "specialized.py",
        "utils.py"
    ]
    
    for module in modules:
        module_path = commands_dir / module
        if module_path.exists():
            print(f"Updating {module}...")
            
            # Read current content
            original_content = module_path.read_text()
            
            # Apply updates
            updated_content = update_module_imports(module_path)
            updated_content = update_module_args(Path("temp"))  # Using temp path since we're working with content
            updated_content = update_module_functions(Path("temp"))
            
            # Write back if changed
            if updated_content != original_content:
                module_path.write_text(updated_content)
                print(f"  ✓ Updated {module}")
            else:
                print(f"  - No changes needed for {module}")
        else:
            print(f"  ! Module {module} not found")

if __name__ == "__main__":
    main()