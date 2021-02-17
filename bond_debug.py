#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:28:37 2021

@author: Aaron
"""

import numpy as np
import qiskit as qk
import networkx as nx
import tenpy as tp
from scipy.optimize import minimize
import pickle

import networks.networks as nw
from networks.isonetwork import QKParamCircuit
import circuits.basic_circuits as circuits

### Burn In Debugging


def ising_impo(J=1.0, h=1.0):
    site = tp.networks.site.SpinHalfSite(conserve=None)
    #Id, Sp, Sm, Sz = site.Id, site.Sp, site.Sm, 2*site.Sz
    #Sx = Sp + Sm
    Id, Sx, Sz = site.Id, site.Sigmax, site.Sigmaz
    W = [[Id,Sx,-h*Sz], 
         [None,None,-J*Sx], 
         [None,None,Id]]
    H = tp.networks.mpo.MPO.from_grids([site], [W], bc='infinite', IdL=0, IdR=-1)
    return H


def energy_tp(param_vals,*args):
    """
    function to calculate energy using MPO/MPS contraction in tenpy
    inputs:
        - param_vals = dict {parameter:value}
        - *args, 
            args[0] should be psi: state as IsoMPS
            args[1] should be H_mpo: Hamiltonian as MPO
        (input made this way to be compatible w/ scipy.optimize)
    outputs:
        - float, <psi|H|psi> computed w/ tenpy
    """
    # parse inputs
    psi=args[0] # state as isoMPS    
    H_mpo = args[1] # Hamiltonian as tenpy MPO
    param_dict = dict(zip(param_names,param_vals))
    
    # convert state from holoPy isoMPS to tenpy MPS
    psi_tp = psi.to_tenpy(param_dict,L=np.inf)
    
    # compute energy
    E = (H_mpo.expectation_value(psi_tp)).real
    return E





########## Runnning the code

nb = 2 # number of bond qubits
L = 1  # number of unit cells (?)
l_uc = 1 # sites per unit cell

preg = qk.QuantumRegister(1,'p') # physical qubits
breg = qk.QuantumRegister(nb,'b') # bond qubits
creg = qk.ClassicalRegister(L*l_uc+nb,'m') # classical register to hold measurement outcomes

circs = []

param_names= [] # list of circuit parameters
for j in range(l_uc):
    # circuit to initialize state (start in state 101010101...)
    init_circ = qk.QuantumCircuit(preg)
    init_circ.h(preg)
    #if j%2==0: init_circ.x(preg)
    
    circ_tmp,params_tmp = circuits.star_circ(preg,
                                             breg,
                                             label='[c{}]'.format(j),
                                             circ_type='xxz')   
    circs+=[circ_tmp]
    param_names+=params_tmp

# setup circuit-generated isoMPS
psi = nw.IsoMPS(preg,breg,circs,L=L)

# try:
    # (opt_vals,opt_params) = pickle.load(open( "./debug_param_dict.pkl", "rb" ) )
# except:
H_mpo = ising_impo(1.0,0.5)
x0 = 0.03*np.random.randn(len(param_names))
opt_result = minimize(energy_tp, # function to minimize
                      x0, # starting point for parameters 
                      args=(psi,H_mpo), # must take form (isoMPS,tenpy MPO)
                      method='BFGS'
                     )
opt_vals = opt_result.x
opt_params = dict(zip(param_names,opt_vals))
    # pickle.dump( (opt_vals,opt_params), open( "./debug_param_dict.pkl", "wb" ) )
    

transfer_eigs, density_matrices = psi.get_bond_steady_state(params = opt_params, debug=True)
psi.set_burn_in_params(density_matrices[0])