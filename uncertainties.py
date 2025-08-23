#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat August 23 13:00:00 2025

Microalgae biorefinery to produce medium chain fatty acids 
by anaerobic fermentation without external electron donor addition- uncertainty analysis

References
----------
[1] BioSTEAM Documentation: 
    https://biosteam.readthedocs.io/en/latest/API/evaluation/Model.html
[2] Cortes-Peña et al., BioSTEAM: A Fast and Flexible Platform for the Design, 
    Simulation, and Techno-Economic Analysis of Biorefineries under Uncertainty. 
    ACS Sustainable Chem. Eng. 2020, 8 (8), 3302–3310.
[3] succinic biorefineries project:
    https://github.com/BioSTEAMDevelopmentGroup/Bioindustrial-Park/tree/master/biorefineries/succinic

@author: Xingdong Shi
@version: 0.0.1
"""

from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd
import biosteam as bst
# import contourplots
from ._chemicals import chems
from .tea import microalgae_tea as create_tea_for_system
from .lca import create_microalgae_lca
from importlib import import_module
from .model_utils import MicroalgaeModel
from biosteam.evaluation import Metric
from datetime import datetime
from biosteam.utils import TicToc
import os

microalgae_filepath = os.path.dirname(__file__)
microalgae_results_filepath = os.path.join(microalgae_filepath, 'analyses', 'results')

# Create results directory if it doesn't exist
if not os.path.exists(microalgae_results_filepath):
    os.makedirs(microalgae_results_filepath)

def _get_system_creators():
    systems = []  # (label, creator_fn)
    create_baseline = getattr(import_module('.system', __package__), 'create_microalgae_MCCA_production_sys')
    systems.append(('baseline', create_baseline))
    create_no_yeast = getattr(import_module('.system_no_yeast', __package__), 'create_microalgae_MCCA_production_no_yeast_sys')
    systems.append(('no_yeast', create_no_yeast))
    create_noCyeast = getattr(import_module('.system_noCyeast', __package__), 'create_microalgae_MCCA_noCyeast_production_sys')
    systems.append(('noCyeast', create_noCyeast))
    create_ethanol = getattr(import_module('.system_ethanol', __package__), 'create_microalgae_MCCA_production_sys_ethanol')
    systems.append(('ethanol', create_ethanol))
    return systems

def build_context_for_system(microalgae_mcca_sys, sys_label='baseline'):
    u = microalgae_mcca_sys.flowsheet.unit
    s = microalgae_mcca_sys.flowsheet.stream
    main_product = getattr(s, 'caproic_acid_product', None)
    if main_product is None:
        raise AttributeError('caproic_acid_product not found in flowsheet streams.')
    boiler = next((unit for unit in microalgae_mcca_sys.units if ('BT' in unit.ID)), None)
    lca = create_microalgae_lca(microalgae_mcca_sys, main_product, ['CaproicAcid'], boiler)
    tea_obj = create_tea_for_system(microalgae_mcca_sys)
    metrics = [
        Metric('MPSP', lambda: tea_obj.solve_price(main_product), '$/kg'),
        Metric('TCI', lambda: tea_obj.TCI/1e6, 'MM$'),
        Metric('VOC', lambda: tea_obj.VOC/1e6, 'MM$/y'),
        Metric('FOC', lambda: tea_obj.FOC/1e6, 'MM$/y'),
        Metric('GWP', lambda: lca.GWP, 'kg CO2-eq/kg'),
        Metric('FEC', lambda: lca.FEC, 'MJ/kg'),
    ]
    namespace_dict = {
        'microalgae_sys': microalgae_mcca_sys,
        'microalgae_tea': tea_obj,
        'u': u,
        's': s,
        'lca': lca,
        'bst': bst,
        'np': np,
        'PowerUtility': bst.PowerUtility,
    }
    # Provide system-specific alias names expected by parameter distribution 'Load statement'
    if sys_label == 'baseline':
        namespace_dict['microalgae_mcca_sys'] = microalgae_mcca_sys
        namespace_dict['microalgae_tea_baseline'] = tea_obj
    elif sys_label == 'no_yeast':
        namespace_dict['microalgae_mcca_sys_no_yeast'] = microalgae_mcca_sys
        namespace_dict['microalgae_tea_no_yeast'] = tea_obj
        namespace_dict['microalgae_no_yeast_tea'] = tea_obj
    elif sys_label == 'noCyeast':
        namespace_dict['microalgae_mcca_noCyeast_sys'] = microalgae_mcca_sys
        namespace_dict['microalgae_tea_noCyeast'] = tea_obj
        namespace_dict['microalgae_noCyeast_tea'] = tea_obj
    elif sys_label == 'ethanol':
        namespace_dict['microalgae_mcca_sys_ethanol'] = microalgae_mcca_sys
        namespace_dict['microalgae_tea_ethanol'] = tea_obj
    # No universal aliases; only system-specific names are exposed to match each system's distribution file.
    model = MicroalgaeModel(microalgae_mcca_sys, metrics=metrics, namespace_dict=namespace_dict)
    return model, tea_obj, lca, namespace_dict

 

# %% 

N_simulations_per_mode = 2000 # 2000 Monte Carlo

percentiles = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]

notification_interval = 10

results_dict = {'Baseline':{'MPSP':{}, 'GWP100a':{}, 'FEC':{}, 
                            'GWP Breakdown':{}, 'FEC Breakdown':{},},
                'Uncertainty':{'MPSP':{}, 'GWP100a':{}, 'FEC':{}},
                'Sensitivity':{'Spearman':{'MPSP':{}, 'GWP100a':{}, 'FEC':{}}},}

modes = [
            'baseline',
         ]

# Per-system parameter distribution files (Excel). Falls back to baseline file if specific one is missing.
parameter_distributions_files = {
    'baseline': 'parameter_distributions.xlsx',
    'no_yeast': 'parameter_distributions_no_yeast.xlsx',
    'noCyeast': 'parameter_distributions_noCyeast.xlsx',
    'ethanol': 'parameter_distributions_ethanol.xlsx',
}

#%%

timer = TicToc('timer')
timer.tic()

# Set seed to make sure each time the same set of random numbers will be used
np.random.seed(3221)

systems_to_run = _get_system_creators()
if not systems_to_run:
    raise RuntimeError('No microalgae systems available to analyze.')

for (sys_label, creator_fn) in systems_to_run:
    print(f"\n===== Running uncertainty analysis for system: {sys_label} =====")
    bst.settings.set_thermo(chems)
    microalgae_mcca_sys = creator_fn()
    microalgae_mcca_sys.simulate()
    model, tea_obj, lca, namespace_dict = build_context_for_system(microalgae_mcca_sys, sys_label=sys_label)
    mode = modes[0]
    # Choose distribution file for this system label
    fname = parameter_distributions_files.get(sys_label, parameter_distributions_files['baseline'])
    parameter_distributions_filename = os.path.join(microalgae_filepath, fname)
    if not os.path.isfile(parameter_distributions_filename):
        raise FileNotFoundError(f"Parameter distributions file not found for '{sys_label}': {parameter_distributions_filename}")
    
    model.parameters = ()
    model.load_parameter_distributions(parameter_distributions_filename, namespace_dict)
    
    
    parameters = model.get_parameters()
    
    samples = model.sample(N=N_simulations_per_mode, rule='L')
    model.load_samples(samples)
    
    
    model.exception_hook = 'warn'
    baseline_initial = model.metrics_at_baseline()
    baseline = pd.DataFrame(data=np.array([[i for i in baseline_initial.values],]), 
                            columns=baseline_initial.keys())
    
    results_dict['Baseline']['MPSP'][mode] = tea_obj.solve_price(microalgae_mcca_sys.flowsheet.stream.caproic_acid_product)
    print(f"MPSP: ${results_dict['Baseline']['MPSP'][mode]:.2f}/kg")
    results_dict['Baseline']['GWP100a'][mode] = tot_GWP = lca.GWP
    print(f"GWP: {results_dict['Baseline']['GWP100a'][mode]:.4f} kg CO2-eq/kg")
    results_dict['Baseline']['FEC'][mode] = tot_FEC = lca.FEC
    print(f"FEC: {results_dict['Baseline']['FEC'][mode]:.4f} MJ/kg")
          
    model.evaluate(notify=notification_interval, autoload=None, autosave=None, file=None)
    
        
    # Baseline results
    baseline_end = model.metrics_at_baseline()
    dateTimeObj = datetime.now()
    minute = '0' + str(dateTimeObj.minute) if len(str(dateTimeObj.minute))==1 else str(dateTimeObj.minute)
    file_to_save = os.path.join(microalgae_results_filepath,
        f'_microalgae_{sys_label}_{dateTimeObj.year}.{dateTimeObj.month}.{dateTimeObj.day}-{dateTimeObj.hour}.{minute}'\
        + f'_{N_simulations_per_mode}sims')
    
    baseline.index = ('initial', )
    baseline.to_excel(file_to_save+'_'+mode+'_0_baseline.xlsx')
    
    # Parameters
    parameters = model.get_parameters()
    index_parameters = len(model.get_baseline_sample())
    parameter_values = model.table.iloc[:, :index_parameters].copy()
    
    #%%
    
    # TEA results
    for index_TEA, i in enumerate(model.metrics):
        if hasattr(i, 'element') and i.element == 'LCA': 
            break
    else:
        index_TEA = len(model.metrics) - 2  # Assume last 2 are LCA metrics
        
    index_TEA = index_parameters + index_TEA
    TEA_results = model.table.iloc[:, index_parameters:index_TEA].copy()
    TEA_percentiles = TEA_results.quantile(q=percentiles)
    
    # LCA_results
    LCA_results = model.table.iloc[:, index_TEA::].copy()
    LCA_percentiles = LCA_results.quantile(q=percentiles)
    
    # # Spearman's rank correlation
    table = model.table
    model.table = model.table.dropna()
    
    spearman_results = model.spearman()
    spearman_results.columns = pd.Index([i.name_with_units for i in model.metrics])
    
    model.table = table
    
    # Calculate the cumulative probabilities of each parameter
    probabilities = {}
    for i in range(index_parameters):
        p = parameters[i]
        p_values = parameter_values.iloc[:, 2*i]
        probabilities[p.name] = p.distribution.cdf(p_values)
        parameter_values.insert(loc=2*i+1, 
                          column=(parameter_values.iloc[:, 2*i].name[0], 'Probability'), 
                          value=probabilities[p.name],
                          allow_duplicates=True)
    
    #%%
    '''Output to Excel'''
    with pd.ExcelWriter(file_to_save+'_'+mode+'_1_full_evaluation.xlsx') as writer:
        parameter_values.to_excel(writer, sheet_name='Parameters')
        TEA_results.to_excel(writer, sheet_name='TEA results')
        TEA_percentiles.to_excel(writer, sheet_name='TEA percentiles')
        LCA_results.to_excel(writer, sheet_name='LCA results')
        LCA_percentiles.to_excel(writer, sheet_name='LCA percentiles')
        spearman_results.to_excel(writer, sheet_name='Spearman')
        model.table.to_excel(writer, sheet_name='Raw data')
    
    # Extract results for plotting
    # Find the MPSP column
    mpsp_col = None
    for col in model.table.columns:
        if 'MPSP' in str(col) or 'price' in str(col).lower():
            mpsp_col = col
            break
    
    # Find GWP column
    gwp_col = None
    for col in model.table.columns:
        if 'GWP' in str(col):
            gwp_col = col
            break
    
    # Find FEC column
    fec_col = None
    for col in model.table.columns:
        if 'FEC' in str(col):
            fec_col = col
            break
    
    if mpsp_col is not None:
        results_dict['Uncertainty']['MPSP'][mode] = model.table[mpsp_col]
    else:
        results_dict['Uncertainty']['MPSP'][mode] = [results_dict['Baseline']['MPSP'][mode]] * len(model.table)
    
    if gwp_col is not None:
        results_dict['Uncertainty']['GWP100a'][mode] = model.table[gwp_col]
    else:
        results_dict['Uncertainty']['GWP100a'][mode] = [results_dict['Baseline']['GWP100a'][mode]] * len(model.table)
    
    if fec_col is not None:
        results_dict['Uncertainty']['FEC'][mode] = model.table[fec_col]
    else:
        results_dict['Uncertainty']['FEC'][mode] = [results_dict['Baseline']['FEC'][mode]] * len(model.table)
    
    # Spearman correlations for sensitivity analysis
    df_rho, df_p = model.spearman_r()
    
    if mpsp_col is not None and mpsp_col in df_rho.columns:
        results_dict['Sensitivity']['Spearman']['MPSP'][mode] = df_rho[mpsp_col]
    else:
        results_dict['Sensitivity']['Spearman']['MPSP'][mode] = pd.Series()
    
    if gwp_col is not None and gwp_col in df_rho.columns:
        results_dict['Sensitivity']['Spearman']['GWP100a'][mode] = df_rho[gwp_col]
    else:
        results_dict['Sensitivity']['Spearman']['GWP100a'][mode] = pd.Series()
    
    if fec_col is not None and fec_col in df_rho.columns:
        results_dict['Sensitivity']['Spearman']['FEC'][mode] = df_rho[fec_col]
    else:
        results_dict['Sensitivity']['Spearman']['FEC'][mode] = pd.Series()
            
#%% Clean up NaN values for plotting
metrics = ['MPSP', 'GWP100a', 'FEC']
for mode in modes:
    for metric in metrics:
        median_val = 1.5  # Default fallback value
        if len(results_dict['Uncertainty'][metric][mode]) > 0:
            median_val = np.nanmedian(results_dict['Uncertainty'][metric][mode])
            if np.isnan(median_val):
                median_val = 1.5
        
        for i in range(len(results_dict['Uncertainty'][metric][mode])):
            if np.isnan(results_dict['Uncertainty'][metric][mode].iloc[i]):
                results_dict['Uncertainty'][metric][mode].iloc[i] = median_val

# %% Plots - temporarily disabled due to contourplots dependency
# MPSP_units = r"$\mathrm{\$}\cdot\mathrm{kg}^{-1}$"
# GWP_units = r"$\mathrm{kg}$"+" "+ r"$\mathrm{CO}_{2}\mathrm{-eq.}\cdot\mathrm{kg}^{-1}$"
# FEC_units = r"$\mathrm{MJ}\cdot\mathrm{kg}^{-1}$"

scenario_name_labels = ['Baseline']

def get_small_range(num, offset):
    return(num-offset, num+offset)

print(f'\nAnalysis completed. Timer: {timer.toc():.2f} seconds')
