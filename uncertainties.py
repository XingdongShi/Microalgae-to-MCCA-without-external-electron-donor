#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty analysis for microalgae biorefinery

Based on succinic project but adapted for microalgae system structure
"""

import numpy as np
import biosteam as bst
import pandas as pd
from scipy.stats import spearmanr
import os
from .model_utils import MicroalgaeModel
from .system import microalgae_mcca_sys, microalgae_tea
from .lca import create_microalgae_lca
from biosteam.evaluation import Metric

def create_model():
    """Create evaluation model with metrics for microalgae system"""
    # Get system components
    u = microalgae_mcca_sys.flowsheet.unit
    s = microalgae_mcca_sys.flowsheet.stream
    
    # Find main product and boiler
    main_product = s.caproic_acid_product
    main_product_chemical_IDs = ['CaproicAcid']
    
    # Find boiler
    boiler = None
    for unit in microalgae_mcca_sys.units:
        if hasattr(unit, 'natural_gas') or 'BT' in unit.ID:
            boiler = unit
            break
    
    # Create LCA object
    lca = create_microalgae_lca(microalgae_mcca_sys, main_product, main_product_chemical_IDs, boiler)
    
    # Define metrics
    metrics = [
        Metric('MFSP', lambda: microalgae_tea.solve_price(main_product), '$/kg'),
        Metric('TCI', lambda: microalgae_tea.TCI/1e6, 'MM$'),
        Metric('VOC', lambda: microalgae_tea.VOC/1e6, 'MM$/y'),
        Metric('FOC', lambda: microalgae_tea.FOC/1e6, 'MM$/y'),
        Metric('GWP', lambda: lca.GWP, 'kg CO2-eq/kg'),
        Metric('FEC', lambda: lca.FEC, 'MJ/kg'),
    ]
    
    # Create namespace for parameter loading
    namespace_dict = {
        'microalgae_sys': microalgae_mcca_sys,
        'microalgae_tea': microalgae_tea,
        'u': u,
        's': s,
        'lca': lca,
        'bst': bst,
        'np': np,
        # Add chemical streams for easier access
        'miaoalgae': s.microalgae,  # Based on Excel file typo
        'GlucoAmylase': None,  # Will be set dynamically
        'AlphaAmylase': None,  # Will be set dynamically
        'Yeast': None,
        'Octanol': None,
        'base_fermentation': None,
        'FGD_lime': None,
        'PowerUtility': bst.PowerUtility,
    }
    
    # Try to find chemical streams
    for stream in microalgae_mcca_sys.feeds:
        if 'microalgae' in stream.ID.lower():
            namespace_dict['miaoalgae'] = stream
            break
    
    # Create model
    model = MicroalgaeModel(microalgae_mcca_sys, metrics=metrics, namespace_dict=namespace_dict)
    
    return model, namespace_dict

def run_uncertainty_analysis(n_samples=1000):
    """Run Monte Carlo uncertainty analysis"""
    print("Creating model...")
    model, namespace_dict = create_model()
    
    # Load parameter distributions
    current_dir = os.path.dirname(__file__)
    param_file = os.path.join(current_dir, 'parameter_distributions.xlsx')
    
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter distributions file not found: {param_file}")
    
    print(f"Loading parameter distributions from {param_file}...")
    model.load_parameter_distributions(param_file, namespace_dict)
    
    print(f"Loaded {len(model.parameters)} parameters:")
    for i, param in enumerate(model.parameters):
        print(f"  {i+1}. {param.name}")
    
    # Run Monte Carlo simulation
    print(f"\nRunning Monte Carlo simulation with {n_samples} samples...")
    np.random.seed(3221)  # For reproducibility
    
    try:
        samples = model.sample(n_samples, 'L')
        model.load_samples(samples)
        model.evaluate()
        
        # Get results
        results_df = model.table
        print(f"Simulation completed successfully. Results shape: {results_df.shape}")
        
        # Calculate Spearman correlations
        print("\nCalculating Spearman correlations...")
        correlations = {}
        for metric_name in ['MFSP', 'GWP', 'FEC']:
            if metric_name in results_df.columns:
                metric_correlations = {}
                for param in model.parameters:
                    param_values = samples[:, param.index]
                    metric_values = results_df[metric_name].values
                    # Remove NaN values
                    valid_indices = ~(np.isnan(param_values) | np.isnan(metric_values))
                    if np.sum(valid_indices) > 10:  # Need at least 10 valid points
                        try:
                            corr, p_value = spearmanr(param_values[valid_indices], 
                                                    metric_values[valid_indices])
                            if not np.isnan(corr):
                                metric_correlations[param.name] = corr
                        except:
                            continue
                correlations[metric_name] = metric_correlations
        
        return results_df, correlations, model
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        print("Trying with baseline evaluation...")
        
        # Try baseline evaluation
        baseline_sample = model.get_baseline_sample()
        baseline_result = model(baseline_sample)
        print("Baseline evaluation:")
        for key, value in baseline_result.items():
            print(f"  {key}: {value}")
        
        return None, {}, model

if __name__ == '__main__':
    try:
        results, correlations, model = run_uncertainty_analysis(n_samples=100)  # Start with smaller sample
        
        if results is not None:
            print("\nUncertainty analysis completed!")
            print(f"Results shape: {results.shape}")
            
            # Print summary statistics
            print("\nSummary statistics:")
            for col in results.columns:
                if col in ['MFSP', 'GWP', 'FEC', 'TCI', 'VOC', 'FOC']:
                    print(f"{col}: mean={results[col].mean():.4f}, std={results[col].std():.4f}")
            
            # Print top correlations for MFSP
            if 'MFSP' in correlations and correlations['MFSP']:
                print("\nTop correlations with MFSP:")
                mfsp_corr = correlations['MFSP']
                sorted_corr = sorted(mfsp_corr.items(), key=lambda x: abs(x[1]), reverse=True)
                for param, corr in sorted_corr[:10]:
                    print(f"  {param}: {corr:.3f}")
        else:
            print("Uncertainty analysis failed, but baseline evaluation successful.")
            
    except Exception as e:
        print(f"Error in uncertainty analysis: {e}")
        import traceback
        traceback.print_exc()