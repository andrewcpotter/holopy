#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:20:36 2020

@author: acpotter
"""
#%% -- IMPORTS -- 
# external packages
import numpy as np
import qiskit as qk
import networkx as nx
import tenpy

# custom things
from isonetwork import IsoTensor, IsoNetwork, QKParamCircuit
import mps


#%%
class IsoMPS(IsoNetwork):
    """
    MPS defined by 
        - number of physical and bond qubits (sets up associated quantum registers accordingly)
        - l_uc - length of unit cell
        - L number of times to repeat unit cell
        - circuits for each site in the unit cell, and initial state of bond-qubits
    """
               
    def __init__(self,
                 preg, 
                 breg,
                 pcircs,
                 **kwargs):
        """
        inputs:
            preg, list of bond qubit registers
            breg, list of bond qubit registers
                (for qiskit: register= quantum register)
            l_uc, int, number of sites in unit cell
            pcircs, list, of parameterized circuit objects:
                pcircs[0] - boundary circuit (acting only on bond-qubits)
                pcircs[1...l_uc] for each site in unit-cell
            param_names,list of sympy symbols, parameterized gate parameters (shared by all tensors)
            L, int (default=1), Length of System (number of times to repeat unit cell)
            bdry_circ, boundary vector circuit for prepping initial state of bond-qubits
            circuit_format, str, (default='cirq'), type of circuit editor/simulator used
            measurement_circuit, list of circuits to be performed on physical register
        """

        self.l_uc = len(pcircs) # length of unit cell

#        self.n_params = len(param_names)
        
        # parse kwargs that don't depend on circuit_format
        if 'circuit_format' in kwargs.keys():
            self.circuit_format = kwargs['circuit_format']
        else: 
            self.circuit_format = 'qiskit'

        if 'L' in kwargs.keys():
            self.L = kwargs['L']
        else:
            self.L=1 
        
        if self.circuit_format == 'qiskit':
            # setup classical registers for measurement outcomes
            self.cregs = [[qk.ClassicalRegister(len(preg)) for i in range(self.l_uc)] for j in range(self.L)]                          
                             
            self.nphys = len(preg) # number of physical qubits
            self.nbond = len(breg) # number of bond qubits
            self.qregs = [preg,breg]
            if 'boundary_circuit' in kwargs.keys():
                bdry_circ = kwargs['boundary_circuit']
            else:
                bdry_circ = QKParamCircuit(qk.QuantumCircuit(), []) 
            if 'bases' in kwargs.keys():
                if 'FH' in kwargs.keys():
                    self.FH = kwargs['FH']
                else:
                    self.FH = False
                self.measurement_circuit = self.measurement(kwargs['bases'], self.FH, preg)
            else:
                self.measurement_circuit = [[qk.QuantumCircuit() for i in range(self.l_uc)]for j in range(self.L)]
                
            # make the MPS/tensor-train -- same qubits used by each tensor
            self.bdry_tensor = IsoTensor('v_L',
                                         [breg],
                                         bdry_circ)
            self.sites= [[IsoTensor('A'+str(x)+str(y),
                                               [preg,breg],
                                               pcircs[y],
                                               meas_list=[(preg,
                                                           self.cregs[x][y],
                                                           self.measurement_circuit[x][y])])
                          for y in range(self.l_uc)]
                         for x in range(self.L)]
            
            # setup IsoNetwork
            # make a flat list of nodes
            self.nodes = [self.bdry_tensor]
            for x in range(self.L): self.nodes += self.sites[x]
            
            self.edges = [(self.nodes[i],self.nodes[i+1],{'qreg':breg}) for i in range(len(self.nodes)-1)]
            
            
            # construct graph and check that is a DAG
            # check for repeated node names
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(self.nodes)
            self.graph.add_edges_from(self.edges)        
    
            # check that graph is directed & acyclic (DAG)
            if nx.algorithms.dag.is_directed_acyclic_graph(self.graph) != True:
                raise RuntimeError('Graph must be directed and acyclic')
                
            # store node information    
            # self.creg_dict = creg_dict
            self.node_names = [node.name for node in self.nodes]
            if len(self.node_names) != len(set(self.node_names)):
                raise ValueError('Tensor nodes must have unique names')
    
            # store variational parameter info
            self.param_assignments = {}
            for node in self.nodes:
                self.param_assignments[node]=node.param_names
            
            # topologically sort nodes in order of execution
            self.sorted_nodes = [node for node in nx.topological_sort(self.graph)]

        else:
            raise NotImplementedError('only qiskit implemented')
            
    ## cpu simulation ##  
    def left_bdry_vector(self,params):
        """
        computes full unitaries for each state (any initial state for physicalqubit)
        inputs:
            params, dictionary of parameters {'name':numerical-value}
        returns:
            bdry_vec, unitary correspond to boundary
            ulist, list of unitaries for tensors in unit cell
        """
        bvec_l = self.bdry_tensor.unitary(params)[:,0] # boundary circuit tensor 
        return bvec_l
    
    def unitaries(self,params):
        """
        computes full unitaries for each state (any initial state for physicalqubit)
        inputs:
            params, dictionary of parameters {'name':numerical-value}
        returns:
            ulist, list of rank-4 tensors for each site in unit cell
        """
        ulist = [self.sites[j].unitary(params) for j in range(self.l_uc)]
        return ulist
    
    def tensors(self,params):
        """
        computes tensors for fixed initial state of physical qubit = |0>
        inputs:
            params, dictionary of parameters {'name':numerical-value}
        returns:
            tensors, list of rank-3 tensors for each site in unit cell
        """
        tensors = [self.sites[j].unitary(params)[:,:,0,:] for j in range(self.l_uc)]
        return tensors
    
    ## Convert to other format(s) ##
    def to_tenpy(self,params,L=1):
        """
        inputs:
            params, dictionary of parameters {'name':numerical-value}
            L, int, number of repetitions of unit cell, 
                set to np.inf for iMPS
            TODO: add any other args needed to specify, symmetries, site-type etc...
        outputs:
            tenpy MPS object created from cirq description
        """
        site = tenpy.networks.site.SpinHalfSite(conserve=None)
        if (L==np.inf) and (self.l_uc==1) and (self.nphys==1):
            B = np.swapaxes(self.tensors(params)[0],1,2)
            psi = tenpy.networks.mps.MPS.from_Bflat([site], 
                                                [B], 
                                                bc='infinite', 
                                                dtype=complex, 
                                                form=None)
            
        else:
            B_arrs = [np.swapaxes(tensor,1,2) for tensor in self.tensors(params)]
            B_arrs[0] = B_arrs[0][:,0:1,:]
            B_arrs[-1] = B_arrs[-1][:,:,0:1]
            psi = tenpy.networks.mps.MPS.from_Bflat([site]*L,
                                                    B_arrs, 
                                                    bc = 'finite', 
                                                    dtype=complex, 
                                                    form=None)    
        psi.canonical_form()
        psi.convert_form(psi.form)
        return psi    
    
    def as_mps(self,params,L=1):
        """
        converts to custom MPS class object
        inputs:
            params, dictionary of parameters {'name':numerical-value}
            L, int, number of repetitions of unit cell, 
                set to np.inf for iMPS
        outputs:
            custom MPS object created from cirq description
        """
        tensors = self.tensors(params)
        bvecl = self.left_bdry_vector(params)
        state = mps.MPS(tensors,L=L,bdry_vecs=[bvecl,None], rcf = True)
        return state
    
    def as_mpo(self,params):
        """
        converts to custom MPO class object
        inputs:
            params, dictionary of parameters {'name':numerical-value}
        outputs:
            custom MPS object created from cirq description
        """
        tensors = self.compute_unitaries(params)
        bvecl = self.compute_left_bdry_vector(params)
        op = mps.MPO(tensors,L=self.L,bdry_vecs=[bvecl,None], rcf = True)
        return op
        
    ##  correlation function sampling ##
    def measurement(self, bases, FH, preg):
        """
        let's aim at generating a measurement circuit here
        basis: measurement basis for each site
            possible formats: 
                - cirq circuit for physical qubits that maps physical qubits to measurement basis
                - string of basis as 
        possible backends:  
            'tenpy' - uses 
            'qasm' - output qasm script to measure
        inputs:
            options: dictionary with entries specifying:
                burn-in length, 
                unit cell length, 
                basis to measure in for each site,
                number of samples to take (could be infinite for cpu-simulations)
                backend: whether to run as 
                pauli basis or fermi?
                
        process: first specify the circuit according to the measurement needed
        need to add the choice of measurement circuit in the 
        shots specified in the qsam thing?
        we need to qsam each time we sample?
        """
        if self.circuit_format == 'qiskit':
            mc_total = []
            if FH == False:
                 #measurement circuit
        #check whether the input string is a list 
                if self.L != len(bases):
                    raise ValueError('bases must have same length with L')
                for base_uc in bases:
                    if len(base_uc) != self.l_uc:
                        raise ValueError('base must be a string with same length as l_uc ')   
                    mc_uc = []
                    for base in base_uc:
                        qc = qk.QuantumCircuit()
                        for reg in self.qregs: qc.add_register(reg)
                        if base == 'x':
                            for i in range(len(preg)):
                                qc.h(preg[i])
                        if base == 'y':
                            for i in range(len(preg)):    
                                qc.h(preg[i])
                                qc.sdg(preg[i])
                        mc_uc.append(qc)
                    mc_total.append(mc_uc)
            else:
                # now bases is a string with total length L * l_uc * len(preg)
                # explicitly list the pauli string for each site (already consider the JW-string outside)
                for k in range(self.L):
                    mc1 = []
                    for j in range(self.l_uc):
                        qc = qk.QuantumCircuit()
                        for reg in self.qregs: qc.add_register(reg)
                        for i in range(len(preg)):
                            base = bases[k * self.l_uc + j * len(preg) + i]
                            if base == 'x':
                                qc.h(preg[i])
                            elif base == 'y':
                                qc.h(preg[i])
                                qc.sdg(preg[i])
                        mc1.append(qc)
                    mc_total.append(mc1)     
            return mc_total
        else:
            raise NotImplementedError('only qiskit implemented')
        