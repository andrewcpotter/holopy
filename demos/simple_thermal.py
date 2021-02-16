#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:05:39 2021

@author: acpotter
"""
import sys
sys.path.append("..") # import one subdirectory up in files

# standard imports
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time as time
import pickle
import random

# 3rd party packages
import qiskit as qk
import qiskit.providers.aer.noise as noise
#import networkx as nx
import tenpy as tp

# custom
from networks.networks import IsoMPS
from networks.isonetwork import QKParamCircuit
import circuits.basic_circuits as circuits

#%% Hyper-parameters
## Model (xxz chain) ##
J = 1.0
Delta = 1.5
hz=0.0

## Ansatz ##
# bulk sites
nphys = 1 # physical qubits
nbond = 1 # bond qubits
l_uc = 1 # length of unit cell
L = 2 # number of unit cells
N = l_uc * L # number of tensors (total chain length)
n_param_per_site = 4 # 3 for xxz-circ, 1 for random prep probability
# left-boundary:
n_param_bdry = 3 # generic 1q gate
# circuit label (in case running multiple circuits in parallel)
label = 'Z'

## Simulation ##
shots=100
perr2q = 0.0 # 2-qubit errors
perr1q = 0.0 # 1-qubit errors

#%% Temporary definitions (will become function inputs)
mbasis = ['x']*N

#%% Setup Circuits
# Quantum registers
preg = qk.QuantumRegister(nphys)
breg = qk.QuantumRegister(nbond)

# Classical registers 
# for random state prep (separate for each qubit in chain):
prep_reg = [qk.ClassicalRegister(nphys) for j in range(N)] 
# for measurement outcomes:
mreg = [qk.ClassicalRegister(nphys) for j in range(N)]

# Circuit Parameters
param_names = [[qk.circuit.Parameter(label + 'lb'+str(j)) 
           for j in range(n_param_bdry)]]
for j in range(l_uc):
    param_names += [[qk.circuit.Parameter(label+str(j))
                     for j in range(n_param_per_site)]]
param_names_flat = [name for site in param_names for name in site]
          
#[qk.circuit.Parameter(label+str(j)) for j in range(n_params)]#qk.circuit.ParameterVector(label,length=n_params)


