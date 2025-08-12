#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main analysis runner for microalgae biorefinery

Runs uncertainty analysis and TRY analysis based on microalgae system structure
"""

import os
import pandas as pd
from .uncertainties import run_uncertainty_analysis
from .TRY_analysis import run_TRY_analysis, analyze_TRY_results

def create_results_directory():
    """Create results directory if it doesn't exist"""
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    return results_dir

def save_correlations(correlations, results_dir):
    """Save correlation results to file"""
    corr_file = os.path.join(results_dir, 'correlations.txt')
    with open(corr_file, 'w') as f:
        f.write("Spearman Correlations for Microalgae Biorefinery\n")
        f.write("=" * 50 + "\n\n")
        
        for metric, corr_dict in correlations.items():
            if corr_dict:  # Only write if there are correlations
                f.write(f"{metric} correlations:\n")
                f.write("-" * 20 + "\n")
                sorted_corr = sorted(corr_dict.items(), key=lambda x: abs(x[1]), reverse=True)
                for param, corr in sorted_corr:
                    f.write(f"  {param:<40}: {corr:>8.4f}\n")
                f.write("\n")
    
    print(f"Correlations saved to: {corr_file}")

def run_quick_baseline():
    """Run a quick baseline evaluation to test the system"""
    print("Running quick baseline evaluation...")
    try:
        from .uncertainties import create_model
        model, namespace_dict = create_model()
        
        # Try baseline evaluation
        baseline_sample = model.get_baseline_sample()
        baseline_result = model(baseline_sample)
        
        print("Baseline evaluation successful!")
        print("Baseline results:")
        for key, value in baseline_result.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"Baseline evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all analyses for microalgae biorefinery"""
    print("=" * 60)
    print("Microalgae Biorefinery Analysis Suite")
    print("=" * 60)
    
    # Create results directory
    results_dir = create_results_directory()
    
    # Test baseline first
    print("\n1. Testing baseline evaluation...")
    if not run_quick_baseline():
        print("Baseline evaluation failed. Please check system setup.")
        return
    
    # Run uncertainty analysis
    print("\n2. Running uncertainty analysis...")
    try:
        unc_results, correlations, model = run_uncertainty_analysis(n_samples=500)
        
        if unc_results is not None:
            # Save uncertainty results
            unc_file = os.path.join(results_dir, 'uncertainty_results.csv')
            unc_results.to_csv(unc_file, index=False)
            print(f"Uncertainty results saved to: {unc_file}")
            
            # Save correlations
            if correlations:
                save_correlations(correlations, results_dir)
            
            # Print summary
            print("\nUncertainty Analysis Summary:")
            print(f"  Samples: {len(unc_results)}")
            print(f"  Parameters: {len(model.parameters)}")
            
            for col in ['MFSP', 'GWP', 'FEC', 'TCI']:
                if col in unc_results.columns:
                    mean_val = unc_results[col].mean()
                    std_val = unc_results[col].std()
                    print(f"  {col}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print("Uncertainty analysis failed.")
            
    except Exception as e:
        print(f"Error in uncertainty analysis: {e}")
    
    # Run TRY analysis
    print("\n3. Running TRY analysis...")
    try:
        try_results = run_TRY_analysis()
        
        if try_results is not None:
            # Save and analyze TRY results
            try_file = os.path.join(results_dir, 'TRY_analysis_results.csv')
            analyze_TRY_results(try_results, try_file)
            
            print("\nTRY Analysis Summary:")
            print(f"  Successful evaluations: {len(try_results)}")
            
            if 'MFSP' in try_results.columns:
                min_mfsp = try_results['MFSP'].min()
                max_mfsp = try_results['MFSP'].max()
                print(f"  MFSP range: {min_mfsp:.4f} - {max_mfsp:.4f} $/kg")
        else:
            print("TRY analysis failed.")
            
    except Exception as e:
        print(f"Error in TRY analysis: {e}")
    
    print(f"\n4. Analysis completed!")
    print(f"   Results saved in '{results_dir}' directory.")
    print("=" * 60)

if __name__ == '__main__':
    main() 