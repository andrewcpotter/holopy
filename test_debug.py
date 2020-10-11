#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:14:12 2020

@author: acpotter
"""
import numpy as np
import qiskit as qk
import networkx as nx

from isonetwork import IsoTensor,IsoNetwork

#%% qiskit debugging
nphys = 2
nbond = 2
L = 3

# setup registers
preg = qk.QuantumRegister(nphys,'p')
breg = qk.QuantumRegister(nbond,'b')
qregs = [preg,breg]
cregs = [qk.ClassicalRegister(nphys,'m'+str(j)) for j in range(L)]


# setup variational parameters
nvp = 3 # number of variational parameters
plabels = [qk.circuit.Parameter('x'+str(j))for j in range(nvp)]
pvals = 0.37*np.arange(L) 
paramdict = dict(zip(plabels,pvals))

# setup some arbitrary circuit for each node 
# (same architecture for each node, different parameter value(s))
circs=[qk.QuantumCircuit() for j in range(L)]
for j in [2,0,1]:#range(L):
    for reg in qregs: circs[j].add_register(reg)
    circs[j].cx(preg[0],breg[0])
    circs[j].rz(plabels[j],breg[0])
    circs[j].cx(preg[0],breg[0])

# setup nodes and network
meas_list = [[(preg,cregs[j],qk.QuantumCircuit())] for j in range(L)]
nodes = [IsoTensor('n'+str(j),
                   qregs,
                   circs[j],
                   [plabels[j]],
                   meas_list[j], # list of measurements, specified as tuples
                   circ_type='qiskit') for j in range(L)]
edges = [
        (nodes[0],nodes[1],{'qreg':breg}),
        (nodes[1],nodes[2],{'qreg':breg})
        ]
#meas_dict = {nodes[j]:[(preg,cregs[j],qk.QuantumCircuit())] for j in range(len(cregs))}
param_assignments = {nodes[j]:[plabels[j]] for j in range(L)}
tn = IsoNetwork(nodes,edges,qregs,param_assignments)
tn.construct_circuit(paramdict).draw('mpl',scale=0.5)