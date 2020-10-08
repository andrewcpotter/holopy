#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:44:21 2020

@author: acpotter
"""

import numpy as np
import cirq
import sympy 

import mps
import holovqa as holopy
import tenpy

#%% test/debug
# setup ansatz
nphys = 1
nbond = 1
n_var_params = 15 # number of variational parameters
vpsym = [sympy.Symbol('x'+str(j)) for j in range(n_var_params)] # list of symbols for circuit parameters

#%% Construct circuit
# setup qubit registers
qp = [cirq.NamedQubit('p'+str(j)) for j in range(nphys)] # physical qubits
qb = [cirq.NamedQubit('b'+str(j)) for j in range(nbond)] # bond qubits
qubits = [qp,qb]
psi = holopy.HoloMPS(qp,qb,vpsym)

# construct cirq circuit with symbolic parameters
c=psi.sites[0].circuit
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

# resolve parameters and evaluate unitaries
vpvals = 0.5*np.ones(n_var_params) # capriciously chosen parameter values
#params = dict(zip(vpsym,vpvals))
tensors = psi.tensors(vpvals)

# convert to custom mps
psi_mps = psi.as_mps(vpvals)
print('<psi|psi>={}'.format(psi_mps.expect()))

#%% Setup MPO from TenPY model
J=1.0
g=1.0
site = tenpy.networks.site.SpinHalfSite(conserve=None)
Id, Sp, Sm, Sz = site.Id, site.Sp, site.Sm, 2*site.Sz
Sx = Sp + Sm
W = [[Id,Sx,g*Sz], [None,None,-J*Sx], [None,None,Id]]
H = tenpy.networks.mpo.MPO.from_grids([site], [W], bc='infinite', IdL=0, IdR=-1)

# compute energy
tenpy_mps = psi.to_tenpy(vpvals,L=np.inf)
holo_E = (H.expectation_value(tenpy_mps)).real
print('Tenpy Ising model energy = {:.4}'.format(holo_E))