"""Configuration viewer module for the multiple attention model.

This module provides functions to load, view, and analyze model configurations
and evaluation metrics from saved experiments.
"""

import sys
import pickle
import fire
import glob
import json
import os

def loadconfig(a):
    """Load configuration from a saved experiment.
    
    Args:
        a (str): Name of the experiment to load.
        
    Returns:
        train_config: Loaded configuration object.
    """
    name = 'runs/{}/config.pkl'.format(a)
    with open(name, 'rb') as f:
        a = pickle.load(f)
    return a

def printconfig(a):
    """Print all configuration parameters.
    
    Args:
        a (train_config): Configuration object to print.
    """
    for i in vars(a):
        print(i, getattr(a, i))

def cateval(a):
    """Load evaluation metrics from JSON files.
    
    Args:
        a (str): Name of the experiment to load metrics for.
        
    Returns:
        list: List of evaluation metrics from all JSON files.
    """
    files = glob.glob('evaluations/%s/metrics-*.json' % a)
    rt = []
    for i in files:
        with open(i) as f:
            rt.append(json.load(f))
    return rt

# Filter function to extract frame accuracy from evaluation metrics
filter1 = lambda x: x['ff']['all']['frame_acc']

def main():
    """Main function to find and display the best performing experiment.
    
    This function:
    1. Lists all experiments in the runs directory
    2. Loads configurations and evaluation metrics
    3. Filters for experiments with attention layer 'b5'
    4. Finds the experiment with the highest frame accuracy
    5. Prints the configuration of the best experiment
    """
    l = os.listdir('runs')
    d = {}
    for i in l:
        conf = loadconfig(i)
        if conf.attention_layer == 'b5':
            v = list(map(filter1, cateval(i)))
            if v:
                d[i] = max(v)
    print(d)
    n = max(d, key=lambda x: d[x])
    print(n, d[n])
    printconfig(loadconfig(n))

if __name__ == "__main__":
    fire.Fire()


