#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 22:12:41 2020

@author: acpotter
"""

import numpy as np
import cirq
import sympy 

import mps
import holovqa as holopy
import tenpy

#%% Hyperparameters
J=1.0
g=1.0

#%% Setup qubits
nphys = 1
nbond = 1
qp = [cirq.NamedQubit('p'+str(j)) for j in range(nphys)] # physical qubits
qb = [cirq.NamedQubit('b'+str(j)) for j in range(nbond)] # bond qubits

#%% Setup Tenpy model
site = SpinHalfSite(conserve=None)
Id, Sp, Sm, Sz = site.Id, site.Sp, site.Sm, 2*site.Sz
Sx = Sp + Sm
W = [[Id,Sx,g*Sz], [None,None,-J*Sx], [None,None,Id]]
H = MPO.from_grids([site], [W], bc='infinite', IdL=0, IdR=-1)

#%% setup variational circuit
# define variational parameters as sympy symbols
n_var_params = 15 # number of variational parameters
vpsym = [sympy.Symbol('theta'+str(j)) for j in range(n_var_params)] # list of symbols for circuit parameters

# construct cirq circuit with symbolic parameters
c = cirq.Circuit()
c.append([cirq.H(qp[0]), # arbitrary SU(4) circuit using Cartan decomposition
          # 1q on p qubit
          cirq.rx(vpsym[0])(qp[0]),
          cirq.rz(vpsym[1])(qp[0]),
          cirq.rx(vpsym[2])(qp[0]),
          # 1q on bond qubit
          cirq.rx(vpsym[3])(qb[0]),
          cirq.rz(vpsym[4])(qb[0]),
          cirq.rx(vpsym[5])(qb[0]),
          # cartan sub algebra part
          cirq.XX(qb[0],qp[0]) ** vpsym[6],
          cirq.YY(qb[0],qp[0]) ** vpsym[7],
          cirq.ZZ(qb[0],qp[0]) ** vpsym[8], 
          # 1q on p qubit
          cirq.rx(vpsym[9])(qp[0]),
          cirq.rz(vpsym[10])(qp[0]),
          cirq.rx(vpsym[11])(qp[0]),
          # 1q on bond qubit
          cirq.rx(vpsym[12])(qb[0]),
          cirq.rz(vpsym[13])(qb[0]),
          cirq.rx(vpsym[14])(qb[0]),
          ])

def energy(params,circuit):
    """
    computes energy of parameterized circuit ansatz
    inputs:
      params, np.array() of length = n_var_params, variational parameter values
      circuit, cirq.QuantumCircuit(), parameterized circuit
    outputs:
      holo_E, float, energy of iMPS generated by circuit
    """
    # setup circuit (resolve parameters to numerical values)
    resolver = cirq.ParamResolver({'theta'+str(j):params[j] for j in range(n_var_params)})
    resolved_circuit = cirq.resolve_parameters(circuit, resolver)   
    u = resolved_circuit.unitary(qubit_order = qp+qb)
    unitary = u.reshape([2**nphys,2**nbond,2**nphys,2**nbond])
    # Convert to MPS and compute <H>    
    # change the order of indices to [p, vL, vR] = [p_out, b_in, b_out] 
    # (with p_in = 0 to go from unitary to isometry)
    B = np.swapaxes(unitary[:,:,0,:],1,2)
    psi = MPS.from_Bflat([site], [B], bc='infinite', dtype=complex, form=None)
    psi.canonical_form()
    psi.convert_form(psi.form)
    holo_E = (H.expectation_value(psi)).real
    
    return holo_E

#%% Test single unitary
param_vals = 0.5*np.ones(15)
resolver = cirq.ParamResolver({'theta'+str(j):param_vals[j] for j in range(n_var_params)})
resolved_circuit = cirq.resolve_parameters(c, resolver)   
u = resolved_circuit.unitary(qubit_order = qp+qb)
unitary = u.reshape([2**nphys,2**nbond,2**nphys,2**nbond])