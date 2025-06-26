# rdkit_cli/commands/ml_support.py
import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from tqdm import tqdm

from ..core.common import (
    GracefulExit, get_parallel_jobs, read_molecules, validate_input_file,
    validate_output_path, get_molecules_from_args, save_dataframe_with_format_detection
)
from ..core.logging import log_success


def add_subparser(subparsers) -> None:
    parser_ml_features = subparsers.add_parser(
        'ml-features',
        help='Generate machine learning features from molecules'
    )
    parser_ml_features.add_argument(
        '-i', '--input-file',
        help='Input molecular file (SDF, SMILES, MOL, CSV, Parquet)'
    )
    parser_ml_features.add_argument(
        '-S', '--smiles',
        help='Direct SMILES string(s) - comma-separated for multiple'
    )
    parser_ml_features.add_argument(
        '-c', '--smiles-column',
        help='Name of SMILES column in CSV/Parquet files (auto-detected if not specified)'
    )
    parser_ml_features.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file with features'
    )
    parser_ml_features.add_argument(
        '--feature-type',
        choices=['morgan_fp', 'rdkit_fp', 'maccs', 'descriptors', 'combined'],
        default='morgan_fp',
        help='Type of features to generate (default: morgan_fp)'
    )
    parser_ml_features.add_argument(
        '--radius',
        type=int,
        default=2,
        help='Radius for Morgan fingerprints (default: 2)'
    )
    parser_ml_features.add_argument(
        '--n-bits',
        type=int,
        default=2048,
        help='Number of bits for fingerprints (default: 2048)'
    )

    parser_ml_split = subparsers.add_parser(
        'ml-split',
        help='Split dataset for machine learning'
    )
    parser_ml_split.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input CSV file with features and targets'
    )
    parser_ml_split.add_argument(
        '-o', '--train-file',
        required=True,
        help='Output training set file'
    )
    parser_ml_split.add_argument(
        '--test-file',
        required=True,
        help='Output test set file'
    )
    parser_ml_split.add_argument(
        '--split-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser_ml_split.add_argument(
        '--stratify',
        help='Column name for stratified splitting'
    )
    parser_ml_split.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser_ml_validate = subparsers.add_parser(
        'ml-validate',
        help='Validate machine learning model performance'
    )
    parser_ml_validate.add_argument(
        '-i', '--model-file',
        required=True,
        help='Input model file (pickle format)'
    )
    parser_ml_validate.add_argument(
        '--test-data',
        required=True,
        help='Test dataset CSV file'
    )
    parser_ml_validate.add_argument(
        '-o', '--output',
        required=True,
        help='Output validation results JSON file'
    )
    parser_ml_validate.add_argument(
        '--target-column',
        default='target',
        help='Name of target column (default: target)'
    )
    parser_ml_validate.add_argument(
        '--task-type',
        choices=['regression', 'classification'],
        default='regression',
        help='ML task type (default: regression)'
    )


def features(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Generating {args.feature_type} features from {input_path}")
        
        molecules = read_molecules(input_path)
        logger.info(f"Loaded {len(molecules)} molecules")
        
        features_data = []
        
        with tqdm(total=len(molecules), desc="Generating features", ncols=80, colour='blue') as pbar:
            for i, mol in enumerate(molecules):
                if graceful_exit.exit_now:
                    break
                
                try:
                    if mol is None:
                        continue
                    
                    mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                    features = _generate_ml_features(mol, args.feature_type, args.radius, args.n_bits)
                    
                    if features is not None:
                        row = {'ID': mol_id, 'SMILES': Chem.MolToSmiles(mol)}
                        
                        if isinstance(features, dict):
                            row.update(features)
                        else:
                            for j, feature_value in enumerate(features):
                                row[f'feature_{j}'] = feature_value
                        
                        features_data.append(row)
                
                except Exception as e:
                    logger.debug(f"Failed to generate features for molecule {i}: {e}")
                
                pbar.update(1)
        
        if graceful_exit.exit_now:
            return 130
        
        if not features_data:
            logger.error("No features generated")
            return 1
        
        df = pd.DataFrame(features_data)
        df.to_csv(output_path, index=False)
        
        n_features = len(df.columns) - 2  # subtract ID and SMILES columns
        log_success(f"Generated {n_features} features for {len(features_data)} molecules, saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        return 1


def split_data(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        input_path = validate_input_file(args.input_file)
        train_path = validate_output_path(args.train_file)
        test_path = validate_output_path(args.test_file)
        
        logger.info(f"Splitting dataset from {input_path}")
        
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        stratify_column = None
        if args.stratify and args.stratify in df.columns:
            stratify_column = df[args.stratify]
        
        train_df, test_df = train_test_split(
            df,
            train_size=args.split_ratio,
            random_state=args.random_seed,
            stratify=stratify_column
        )
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        log_success(f"Split dataset: {len(train_df)} training, {len(test_df)} test samples")
        log_success(f"Training set saved to {train_path}")
        log_success(f"Test set saved to {test_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Dataset splitting failed: {e}")
        return 1


def validate(args, graceful_exit: GracefulExit) -> int:
    logger = logging.getLogger("rdkit_cli")
    
    try:
        model_path = validate_input_file(args.model_file)
        test_data_path = validate_input_file(args.test_data)
        output_path = validate_output_path(args.output)
        
        logger.info(f"Validating model from {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        test_df = pd.read_csv(test_data_path)
        logger.info(f"Loaded test data with {len(test_df)} samples")
        
        if args.target_column not in test_df.columns:
            logger.error(f"Target column '{args.target_column}' not found in test data")
            return 1
        
        feature_columns = [col for col in test_df.columns 
                          if col not in ['ID', 'SMILES', args.target_column]]
        
        if not feature_columns:
            logger.error("No feature columns found in test data")
            return 1
        
        X_test = test_df[feature_columns].values
        y_test = test_df[args.target_column].values
        
        logger.info(f"Making predictions on {len(X_test)} samples with {len(feature_columns)} features")
        
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return 1
        
        results = {
            'model_file': str(model_path),
            'test_data_file': str(test_data_path),
            'task_type': args.task_type,
            'n_samples': len(y_test),
            'n_features': len(feature_columns),
            'target_column': args.target_column
        }
        
        if args.task_type == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
            
            results.update({
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2_score': float(r2),
                'predictions': {
                    'mean': float(np.mean(y_pred)),
                    'std': float(np.std(y_pred)),
                    'min': float(np.min(y_pred)),
                    'max': float(np.max(y_pred))
                },
                'targets': {
                    'mean': float(np.mean(y_test)),
                    'std': float(np.std(y_test)),
                    'min': float(np.min(y_test)),
                    'max': float(np.max(y_test))
                }
            })
            
            logger.info(f"Regression metrics - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
        
        else:  # classification
            accuracy = accuracy_score(y_test, y_pred)
            
            try:
                from sklearn.metrics import precision_recall_fscore_support
                precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                results.update({
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'class_distribution': {
                        str(cls): int(count) for cls, count in zip(*np.unique(y_test, return_counts=True))
                    }
                })
                
                logger.info(f"Classification metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            except Exception:
                results.update({
                    'accuracy': float(accuracy)
                })
                logger.info(f"Classification accuracy: {accuracy:.4f}")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = {
                feature_columns[i]: float(importances[i]) 
                for i in range(len(feature_columns))
            }
            sorted_importance = dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True)[:20])
            results['top_features'] = sorted_importance
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        log_success(f"Model validation completed, results saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return 1


def _generate_ml_features(mol: Chem.Mol, feature_type: str, radius: int, n_bits: int):
    """Generate ML features for a molecule."""
    try:
        if feature_type == 'morgan_fp':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return list(fp.ToBitString())
        
        elif feature_type == 'rdkit_fp':
            fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
            return list(fp.ToBitString())
        
        elif feature_type == 'maccs':
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            return list(fp.ToBitString())
        
        elif feature_type == 'descriptors':
            from rdkit.Chem import Descriptors, Crippen
            
            descriptors = {
                'MW': Descriptors.MolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'HeavyAtoms': mol.GetNumHeavyAtoms(),
                'NumRings': Descriptors.RingCount(mol),
                'AromaticRings': Descriptors.NumAromaticRings(mol),
                'FractionCsp3': Descriptors.FractionCSP3(mol),
                'LabuteASA': Descriptors.LabuteASA(mol),
                'BertzCT': Descriptors.BertzCT(mol),
                'Chi0v': Descriptors.Chi0v(mol),
                'Chi1v': Descriptors.Chi1v(mol),
                'Chi2v': Descriptors.Chi2v(mol),
                'Chi3v': Descriptors.Chi3v(mol),
                'Chi4v': Descriptors.Chi4v(mol),
                'Kappa1': Descriptors.Kappa1(mol),
                'Kappa2': Descriptors.Kappa2(mol),
                'Kappa3': Descriptors.Kappa3(mol)
            }
            return descriptors
        
        elif feature_type == 'combined':
            from rdkit.Chem import Descriptors, Crippen
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=1024)
            fp_bits = list(fp.ToBitString())
            
            descriptors = {
                'MW': Descriptors.MolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'HeavyAtoms': mol.GetNumHeavyAtoms(),
                'NumRings': Descriptors.RingCount(mol),
                'AromaticRings': Descriptors.NumAromaticRings(mol),
                'FractionCsp3': Descriptors.FractionCSP3(mol)
            }
            
            combined_features = {}
            combined_features.update(descriptors)
            for i, bit in enumerate(fp_bits):
                combined_features[f'fp_{i}'] = int(bit)
            
            return combined_features
        
        else:
            return None
    
    except Exception:
        return None