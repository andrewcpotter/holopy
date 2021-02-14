"""
Predefined (parameterized) qiskit circuits for holovqe

Created: 2/14/2021, AC Potter
"""
import sys
sys.path.append("..") # import one subdirectory up in files
from networks.isonetwork import QKParamCircuit

import qiskit as qk 


#%% Arbitrary SU(4) circuit
## Will become inputs ##
qp = qk.QuantumRegister(1) # physical qubit
qb = qk.QuantumRegister(1) # bond qubit 



