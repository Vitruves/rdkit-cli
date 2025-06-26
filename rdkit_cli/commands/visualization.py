# rdkit_cli/commands/visualization.py
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from tqdm import tqdm

from ..core.common import (
    GracefulExit, get_parallel_jobs, read_molecules, validate_input_file,
    validate_output_path
)
from ..core.logging import log_success


def add_subparser(subparsers) -> None:
    parser_visualize = subparsers.add_parser(
        'visualize',
        help='Generate molecular structure images'
    )
    parser_visualize.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input molecular file'
    )
    parser_visualize.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for images'
    )
    parser_visualize.add_argument(
        '--format',
        choices=['png', 'svg', 'pdf'],
        default='png',
        help='Image format (default: png)'
    )
    parser_visualize.add_argument(
        '--size',
        default='300x300',
        help='Image size in pixels (default: 300x300)'
    )
    parser_visualize.add_argument(
        '--highlight-atoms',
        help='Comma-separated list of atom indices to highlight'
    )

    parser_grid_image = subparsers.add_parser(
        'grid-image',
        help='Create grid image of multiple molecules'
    )
    parser_grid_image.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input molecular file'
    )
    parser_grid_image.add_argument(
        '-o', '--output',
        required=True,
        help='Output image file'
    )
    parser_grid_image.add_argument(
        '--mols-per-row',
        type=int,
        default=4,
        help='Number of molecules per row (default: 4)'
    )
    parser_grid_image.add_argument(
        '--img-size',
        default='200x200',
        help='Size of individual molecule images (default: 200x200)'
    )
    parser_grid_image.add_argument(
        '--max-mols',
        type=int,
        default=100,
        help='Maximum number of molecules to include (default: 100)'
    )

    parser_plot_descriptors = subparsers.add_parser(
        'plot-descriptors',
        help='Create plots of molecular descriptors'
    )
    parser_plot_descriptors.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input CSV file with descriptors'
    )
    parser_plot_descriptors.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for plots'
    )
    parser_plot_descriptors.add_argument(
        '--plot-type',
        choices=['scatter', 'histogram', 'boxplot', 'correlation', 'pca'],
        default='scatter',
        help='Type of plot to generate (default: scatter)'
    )
    parser_plot_descriptors.add_argument(
        '--x',
        help='X-axis descriptor for scatter plots'
    )
    parser_plot_descriptors.add_argument(
        '--y',
        help='Y-axis descriptor for scatter plots'
    )

    parser_report = subparsers.add_parser(
        'report',
        help='Generate comprehensive HTML report'
    )
    parser_report.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input molecular file'
    )
    parser_report.add_argument(
        '-o', '--output',
        required=True,
        help='Output HTML report file'
    )
    parser_report.add_argument(
        '--include-descriptors',
        action='store_true',
        help='Include molecular descriptors in report'
    )
    parser_report.add_argument(
        '--include-images',
        action='store_true',
        help='Include molecular structure images'
    )
    parser_report.add_argument(
        '--max-structures',
        type=int,
        default=50,
        help='Maximum number of structures to include (default: 50)'
    )


def visualize(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating molecular images from {input_path}")
        
        width, height = map(int, args.size.split('x'))
        
        highlight_atoms = None
        if args.highlight_atoms:
            highlight_atoms = [int(x.strip()) for x in args.highlight_atoms.split(',')]
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        generated_count = 0
        
        with tqdm(total=len(molecules), desc="Generating images", ncols=80, colour='blue') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                    
                    if args.format == 'svg':
                        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
                    else:
                        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
                    
                    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
                    drawer.FinishDrawing()
                    
                    if args.format == 'svg':
                        svg_text = drawer.GetDrawingText()
                        output_file = output_dir / f"{mol_id}.svg"
                        with open(output_file, 'w') as f:
                            f.write(svg_text)
                    else:
                        png_data = drawer.GetDrawingText()
                        output_file = output_dir / f"{mol_id}.png"
                        with open(output_file, 'wb') as f:
                            f.write(png_data)
                    
                    generated_count += 1
                
                except Exception as e:
                    logger.debug(f"Failed to generate image for molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        log_success(f"Generated {generated_count} molecular images in {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Molecular visualization failed: {e}")
        return 1


def grid_image(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Creating grid image from {input_path}")
        
        width, height = map(int, args.img_size.split('x'))
        
        molecules = read_molecules(input_path)
        molecules = molecules[:args.max_mols]
        logger.info(f"Creating grid with {len(molecules)} molecules")
        
        mol_names = []
        for i, mol in enumerate(molecules):
            if mol is not None:
                mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                mol_names.append(mol_id)
            else:
                mol_names.append(f"invalid_{i+1}")
        
        try:
            img = Draw.MolsToGridImage(
                molecules,
                molsPerRow=args.mols_per_row,
                subImgSize=(width, height),
                legends=mol_names,
                useSVG=False
            )
            
            img.save(output_path)
            
            log_success(f"Created grid image with {len(molecules)} molecules, saved to {output_path}")
            return 0
        
        except Exception as e:
            logger.error(f"Failed to create grid image: {e}")
            return 1
        
    except Exception as e:
        logger.error(f"Grid image generation failed: {e}")
        return 1


def plot_descriptors(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating descriptor plots from {input_path}")
        
        df = pd.read_csv(input_path)
        logger.info(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_columns:
            logger.error("No numeric columns found for plotting")
            return 1
        
        plt.style.use('seaborn-v0_8')
        
        if args.plot_type == 'scatter':
            if not args.x or not args.y:
                logger.error("Scatter plot requires --x and --y arguments")
                return 1
            
            if args.x not in df.columns or args.y not in df.columns:
                logger.error(f"Columns {args.x} or {args.y} not found in data")
                return 1
            
            plt.figure(figsize=(10, 8))
            plt.scatter(df[args.x], df[args.y], alpha=0.6)
            plt.xlabel(args.x)
            plt.ylabel(args.y)
            plt.title(f'{args.y} vs {args.x}')
            plt.grid(True, alpha=0.3)
            
            output_file = output_dir / f"scatter_{args.x}_vs_{args.y}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        elif args.plot_type == 'histogram':
            n_cols = 3
            n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for i, col in enumerate(numeric_columns):
                if i < len(axes):
                    axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
            
            for i in range(len(numeric_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            output_file = output_dir / "histograms.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        elif args.plot_type == 'boxplot':
            n_cols = 3
            n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for i, col in enumerate(numeric_columns):
                if i < len(axes):
                    axes[i].boxplot(df[col].dropna())
                    axes[i].set_title(f'Boxplot of {col}')
                    axes[i].set_ylabel(col)
                    axes[i].grid(True, alpha=0.3)
            
            for i in range(len(numeric_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            output_file = output_dir / "boxplots.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        elif args.plot_type == 'correlation':
            corr_matrix = df[numeric_columns].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title('Correlation Matrix of Descriptors')
            plt.tight_layout()
            
            output_file = output_dir / "correlation_matrix.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        elif args.plot_type == 'pca':
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            data_clean = df[numeric_columns].dropna()
            
            if len(data_clean) < 2:
                logger.error("Insufficient data for PCA analysis")
                return 1
            
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_clean)
            
            pca = PCA()
            pca_result = pca.fit_transform(data_scaled)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax1.set_title('PCA Score Plot')
            ax1.grid(True, alpha=0.3)
            
            ax2.bar(range(1, min(11, len(pca.explained_variance_ratio_) + 1)), 
                   pca.explained_variance_ratio_[:10])
            ax2.set_xlabel('Principal Component')
            ax2.set_ylabel('Explained Variance Ratio')
            ax2.set_title('PCA Explained Variance')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_file = output_dir / "pca_analysis.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        log_success(f"Generated {args.plot_type} plots in {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Descriptor plotting failed: {e}")
        return 1


def report(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Generating HTML report from {input_path}")
        
        molecules = read_molecules(input_path)
        molecules = molecules[:args.max_structures]
        logger.info(f"Generating report for {len(molecules)} molecules")
        
        html_content = _generate_html_report(molecules, args, graceful_exit)
        
        if graceful_exit.exit_now:
            return 130
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        log_success(f"Generated HTML report with {len(molecules)} molecules, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return 1


def _generate_html_report(molecules: List[Chem.Mol], args, graceful_exit: GracefulExit) -> str:
    """Generate HTML report content."""
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RDKit CLI Molecular Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
            .molecule { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
            .mol-image { float: left; margin-right: 15px; }
            .mol-info { overflow: hidden; }
            .descriptor-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            .descriptor-table th, .descriptor-table td { 
                border: 1px solid #ddd; padding: 8px; text-align: left; 
            }
            .descriptor-table th { background-color: #f2f2f2; }
            .summary { background-color: #e8f4fd; padding: 15px; margin: 20px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>RDKit CLI Molecular Report</h1>
            <p>Generated report containing molecular structures and properties</p>
        </div>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Total Molecules:</strong> {num_molecules}</p>
            <p><strong>Include Descriptors:</strong> {include_descriptors}</p>
            <p><strong>Include Images:</strong> {include_images}</p>
        </div>
    """.format(
        num_molecules=len(molecules),
        include_descriptors=args.include_descriptors,
        include_images=args.include_images
    )
    
    for i, mol in enumerate(molecules):
        if graceful_exit.exit_now:
            break
        
        try:
            if mol is None:
                continue
            
            mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
            smiles = Chem.MolToSmiles(mol)
            
            html += f"""
            <div class="molecule">
                <h3>Molecule: {mol_id}</h3>
            """
            
            if args.include_images:
                try:
                    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
                    drawer.DrawMolecule(mol)
                    drawer.FinishDrawing()
                    svg_text = drawer.GetDrawingText()
                    
                    html += f"""
                    <div class="mol-image">
                        {svg_text}
                    </div>
                    """
                except Exception:
                    pass
            
            html += f"""
                <div class="mol-info">
                    <p><strong>SMILES:</strong> {smiles}</p>
            """
            
            if args.include_descriptors:
                try:
                    from rdkit.Chem import Descriptors, Crippen
                    
                    descriptors = {
                        'Molecular Weight': round(Descriptors.MolWt(mol), 2),
                        'LogP': round(Crippen.MolLogP(mol), 2),
                        'TPSA': round(Descriptors.TPSA(mol), 2),
                        'Heavy Atoms': Descriptors.HeavyAtomCount(mol),
                        'H Donors': Descriptors.NumHDonors(mol),
                        'H Acceptors': Descriptors.NumHAcceptors(mol),
                        'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
                        'Aromatic Rings': Descriptors.NumAromaticRings(mol)
                    }
                    
                    html += """
                    <table class="descriptor-table">
                        <tr><th>Property</th><th>Value</th></tr>
                    """
                    
                    for prop, value in descriptors.items():
                        html += f"<tr><td>{prop}</td><td>{value}</td></tr>"
                    
                    html += "</table>"
                
                except Exception:
                    pass
            
            html += """
                </div>
                <div style="clear: both;"></div>
            </div>
            """
        
        except Exception:
            continue
    
    html += """
    </body>
    </html>
    """
    
    return html