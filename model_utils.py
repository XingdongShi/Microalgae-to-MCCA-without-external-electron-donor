#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model utilities for microalgae biorefinery uncertainty analysis

Based on succinic project's model_utils.py but adapted for microalgae system structure
"""

import biosteam as bst
from biosteam.evaluation import Model
from pandas import DataFrame, read_excel
import chaospy as shape
import numpy as np

def codify(statement):
    """Convert load statement to executable code"""
    if isinstance(statement, str):
        return statement
    return str(statement)

class MicroalgaeModel(Model):
    """Extended Model class for microalgae project with parameter distribution loading"""
    
    def __init__(self, system, metrics=None, specification=None, 
                 parameters=None, retry_evaluation=True, exception_hook='warn',
                 namespace_dict={}):
        Model.__init__(self, system=system, specification=specification, 
                     parameters=parameters, retry_evaluation=retry_evaluation, exception_hook=exception_hook)
        self.namespace_dict = namespace_dict
        # Set metrics after initialization
        if metrics is not None:
            self.metrics = metrics
    
    def load_parameter_distributions(self, distributions, namespace_dict=None):
        """Load parameter distributions from Excel file"""
        if namespace_dict is None:
            namespace_dict = self.namespace_dict
            
        df = distributions
        if type(df) is not DataFrame:
            df = read_excel(distributions)
            
        create_function = self.create_function
        param = self.parameter
        
        for i, row in df.iterrows():
            name = row['Parameter name']
            element = row['Element']
            kind = row['Kind']
            units = row['Units']
            baseline = row['Baseline']
            shape_data = row['Shape']
            lower, midpoint, upper = row['Lower'], row['Midpoint'], row['Upper']
            load_statements = codify(row['Load statement'])
            
            # Skip blank parameters
            if name == 'Blank parameter':
                continue
                
            D = None
            if shape_data.lower() in ['triangular', 'triangle']:
                D = shape.Triangle(lower, midpoint, upper)
            elif shape_data.lower() in ['uniform']:
                D = shape.Uniform(lower, upper)
            
            if D is not None:
                try:
                    param(name=name, 
                          setter=create_function(load_statements, namespace_dict), 
                          element=element, 
                          kind=kind, 
                          units=units,
                          baseline=baseline, 
                          distribution=D)
                except Exception as e:
                    print(f"Warning: Failed to load parameter '{name}': {e}")
    
    def create_function(self, code, namespace_dict):
        """Create parameter setter function from code string"""
        def wrapper_fn(statement):
            def f(x):
                namespace_dict['x'] = x
                try:
                    exec(statement, namespace_dict)
                except Exception as e:
                    print(f"Warning: Failed to execute '{statement}': {e}")
            return f
        function = wrapper_fn(code)
        return function 