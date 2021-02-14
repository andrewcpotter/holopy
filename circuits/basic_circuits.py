"""
Predefined (parameterized) qiskit circuits for holovqe

Created: 2/14/2021, AC Potter
"""
# standard imports
import sys
sys.path.append("..") # import one subdirectory up in files
import numpy as np
import qiskit as qk 

# holopy imports
from networks.isonetwork import QKParamCircuit

#%% Two-qubit circuits
def add_su4_circ(circ,q1,q2,params):
    """
    inputs:
        - q1,2 qubits
        - params, qiskit ParameterVector object or list of qk Parameters
    returns: 
        - QKParamCircuit object
    """
    
    # 1q gates
    # physical qubit
    circ.rx(params[0],q1)
    circ.rz(params[1],q1)
    circ.rx(params[2],q1)
    # bond qubit
    circ.rx(params[2],q2)
    circ.rz(params[3],q2)
    circ.rx(params[4],q2)
    
    # two qubit gates
    # xx-rotation
    [circ.h(q) for q in [q1,q2]]
    circ.cx(q1,q2)
    circ.rz(params[5],q2)
    circ.cx(q1,q2)
    [circ.h(q) for q in [q1,q2]]
    # yy-rotation
    [circ.rx(np.pi/2,q) for q in [q1,q2]]
    circ.cx(q1,q2)
    circ.rz(params[6],q2)
    circ.cx(q1,q2)
    [circ.rx(-np.pi/2,q) for q in [q1,q2]]
    # zz-rotation
    circ.cx(q1,q2)
    circ.rz(params[7],q2)
    circ.cx(q1,q2)
    
    # 1q gates
    # physical qubit
    circ.rx(params[8],q1)
    circ.rz(params[9],q1)
    circ.rx(params[10],q1)
    # bond qubit
    circ.rx(params[11],q2)
    circ.rz(params[12],q2)
    circ.rx(params[13],q2)
    

def add_xxz_circ(circ,q1,q2,params):
    """
    inputs:
        - q1,2 qubits
        - params, qiskit ParameterVector object or list of qk Parameters
    returns: 
        - QKParamCircuit object
    """
    
    # two qubit gates
    # xx-rotation
    [circ.h(q) for q in [q1,q2]]
    circ.cx(q1,q2)
    circ.rz(params[0],q2)
    circ.cx(q1,q2)
    [circ.h(q) for q in [q1,q2]]
    # yy-rotation
    [circ.rx(np.pi/2,q) for q in [q1,q2]]
    circ.cx(q1,q2)
    circ.rz(params[1],q2)
    circ.cx(q1,q2)
    [circ.rx(-np.pi/2,q) for q in [q1,q2]]
    # zz-rotation
    circ.cx(q1,q2)
    circ.rz(params[2],q2)
    circ.cx(q1,q2)
    
    

#%% Multi-qubit circuits
def star_circ(qp,qb,label,circ_type='su4'):
    """
    sequentially interacts qp with each q in qb
    inputs:
        - qp, quantum register w/ 1 physical qubit
        - qb, quantum register w/ N bond qubits
        - label, str, label for circuit
        - circ_type, str, 'su4','xxz',etc...
    outputs:
        - parameterized circuit
        - list of parameters
    """
    nb = len(qb) # number of bond qubits
    circ = qk.QuantumCircuit(qp,qb)
    
    # parse number of parameters
    if circ_type=='su4': 
        n_params = nb*15
        params = qk.circuit.ParameterVector(label,length=n_params)
        for i in range(nb):
            add_su4_circ(circ,qp[0],qb[i],params[15*i:15*i+15])
        
    elif circ_type=='xxz':
        n_params = 3*nb
        params = qk.circuit.ParameterVector(label,length=n_params)
        for i in range(nb):
            add_xxz_circ(qp[0],qb[i],params[15*i:15*i+15])
    else:
        raise NotImplementedError(circ_type+' not implemented')
    
    param_circ = QKParamCircuit(circ,params)
    return param_circ,params
        


#%% test/debug
# qp = qk.QuantumRegister(1) # physical qubit
# qb = qk.QuantumRegister(2) # bond qubit 
# label = 'c1' # label for circuit
# circ,params = star_circ(qp, qb, label)
# circ.circ.draw('mpl',scale=0.4)