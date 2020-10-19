#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 14:03:30 2020

@author: acpotter
"""
#%% -- IMPORTS -- 
import numpy as np
#import cirq
import qiskit as qk
#import sympy 
import networkx as nx

import mps
import tenpy

#%% 
class ParamCircuit(object):
    """
    Parameterized circuit
    Circuit + parameters
    """
    def __init__(self,circ,param_names):
        self.circ=circ
        self.param_names = param_names
        
    def bind_parameters(self,params):
        raise NotImplementedError()
    
    def bind_from_array(self,param_vals):
        """
        input: param_vals, np.array of values, must be same length as self.param_names
        """
        params = dict(zip(self.param_names,param_vals))
        return self.bind_from_array(params)

    
class QKParamCircuit(ParamCircuit):
    """
    ParamCircuit implemented with qiskit
    """
    def __init__(self,circ,param_names):
        self.circ=circ
        self.param_names = param_names
        self.circuit_format='qiskit'
    
    def bind_from_array(self,params):
        """
        sets named parameters to particular values
        input:
            params: dictionary {parameter name: numerical value}
        output:
            circuit with parameters resolved
        """
        return self.circ.bind_parameters(params)
    
        
            
#%% -- ISOTENSOR CLASS -- 
class IsoTensor(object):
    """
        node of an isometric tensor-network, generated by parameterized cirq unitary
        works equally for tensor network state (TNS) or operator (TNO); 
        for TNS: physical register implicitly assumed to start from reference state: |00..0> 
        
        Intention: circuit object intended to be easily adaptable to work equally with cirq, qiskit, etc...
    """
    def __init__(self,
                 name, # label for the tensor
                 qregs, # listof quantum registers
                 pcirc, # parameterized circuit object
                 #param_names, # list of circuit parameter names (str's)
                 meas_list=[], # list of tuples: (qreg, creg, measurement circuit)
                 circuit_format:str='qiskit' # string specifying circuit type
                 ):
        self.name=name
        self.qregs=qregs
        self.regdims = [2**len(reg) for reg in qregs]
        self.circ= pcirc.circ
        self.param_names = pcirc.param_names
#        self.param_names=param_names
        self.circuit_format=circuit_format
        self.meas_list=meas_list
        
    def __str__(self):
        return self.name
    
    def __rep__(self):
        return self.name

    ## Resolve Circuit Parameters ##
    def resolve_circuit(self,params,include_measurements=True):
        """
        resolves parameters in circuit
        inputs:
            params: dictionary of parameter names and values
            include_measurements, bool, whether or not to include measurement and reset
        outputs:
            resolved circuit
        """
        if self.circuit_format == 'qiskit':
            cres = self.circ.bind_parameters(params)
            if include_measurements:
                for qreg,creg,mcirc in self.meas_list:
                    cres.add_register(creg)
                    cres.combine(mcirc) # add the measurement circuit
                    cres.measure(qreg,creg)
                    cres.reset(qreg)
            return cres
        else:
            raise NotImplementedError()
        
    def bind_params(self,params):
        """
        inputs:
            - params: dictionary {'name':value} for parameters in circuit
        outputs:
            - circuit with symbolic parameters set to numerical values
        """
        if self.circuit_format == 'qiskit':
            return self.circ.bind_parameters(params)
        else:
            raise NotImplementedError()
            

    ## Compute unitaries ##
    def unitary(self,params):
        """
        inputs:
            - params: dictionary {'name':value} for parameters in circuit
        outputs: 
            - unitary for circuit, as numpy array with shape regdims (output legs),regdims (input legs)
        """
        if self.circuit_format == 'qiskit':
            return self.unitary_qiskit(params)
        elif self.circuit_format == 'cirq':
            return self.unitary_cirq(params)
        else:
            raise NotImplementedError('only qiskit implemented')
        
    
    def unitary_qiskit(self,params):
        """
        inputs:
            - params, dictionary {parameter:value} for parameters in circuit
                note: parameter key type depends on type of circuit
                for qiskit: parameter keys are qiskit circuit parameters
                for cirq: they are sympy symbols
        """
        # setup unitary simulator and compute unitary
        bound_circ = self.circ.bind_parameters(params)
        simulator = qk.Aer.get_backend('unitary_simulator')
        result = qk.execute(bound_circ,simulator).result()
        u = result.get_unitary(bound_circ) 

        # need to re-size and re-order to be ampatible with expected indexing
        # note: qiskit writes bases in opposite order of usual convention
        # e.g. for 3-qubit register: [q0,q1,q2], 
        # the state 011 refers to: q0=1, q1=1, q2=0
        u = u.reshape(self.regdims[::-1]+self.regdims[::-1]) # reshape as tensor
        nreg = len(self.qregs)
        old_order = list(range(2*nreg))
        new_order = old_order.copy()
        new_order[0:nreg] = old_order[0:nreg][::-1]
        new_order[nreg::] = old_order[nreg::][::-1]
        u = np.moveaxis(u,old_order,new_order)
        return u
        
    def unitary_cirq(self,params):
        """ unitary constructor for cirq-based circuits """
        qubit_order = [q for qreg in self.qregs for q in qreg] # order to return the qubit unitary
        # resolve the symbolic circuit parameters to numerical values
        resolver = cirq.ParamResolver(params)
        resolved_circuit = cirq.resolve_parameters(self.circuit, resolver)   
        u = resolved_circuit.unitary(qubit_order = qubit_order)
        return u.reshape(self.regdims) # reshape as a multi-l
    


    
                 
#%%
class IsoNetwork(object):
    """
    NetworkX directed graph with:
        nodes = IsoTensors
        edges have list of qubits
    
    To Do: 
        - add global measurement register names list
        - create to_qasm function that traverses the grapha and assembles
            together the qasm files for each node, adding the appropriate header
            and defining qubits and measurement registers one time in the beginning
    """
    def __init__(self,nodes=[],
                 edges=[],
                 qregs=[],
                 circuit_format='qiskit'
                 ):
        """
        nodes, list of IsoTensors
        edges, list of tuples (output node, input node, list of qubits passed along edge)
        qregs, list of qubit registers 
            (for cirq: each qubit register is list of qubits, 
             for qiskit, each qreg is a QuantumRegister object)
        cregs, list of classical registers 
        # meas_dict, dictionary of classical registers to 
        #     hold measurement values for each node that gets measured
        #     keys=MeasurementNode, values = list of tuples: 
        #         (qreg to be measured, creg that stores outcome, circuit to transform qubits to measurement basis)
        #     note: keys of this define which nodes get measured
        param_assignments, 
            dict with key = node, value = list of parameter objects for that node
            for qiskit: parameters are inbuilt circuit parameter
            for cirq: parameters are sympy symbols
        measurement_nodes, list of IsoTensors that get measured
            i.e. have at least one output leg that terminates in a measurement
            actual basis for measurement only specified at qasm output/simulator step
        """         
        self.circuit_format=circuit_format
        
        # construct graph and check that is a DAG
        # check for repeated node names
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)        

        # check that graph is directed & acyclic (DAG)
        if nx.algorithms.dag.is_directed_acyclic_graph(self.graph) != True:
            raise RuntimeError('Graph must be directed and acyclic')
            
        # store node information    
        self.nodes = nodes
        self.qregs = qregs
        # self.creg_dict = creg_dict
        self.node_names = [node.name for node in nodes]
        if len(self.node_names) != len(set(self.node_names)):
            raise ValueError('Tensor nodes must have unique names')

        # store variational parameter info
        self.param_assignments = {}
        for node in nodes:
            self.param_assignments[node]=node.param_names
#        self.param_assignments = param_assignments
        
        # topologically sort nodes in order of execution
        self.sorted_nodes = [node for node in nx.topological_sort(self.graph)]
        
            
    
    ## Circuit Construction Methods ##
    def construct_circuit(self,param_dict):
        """
        input: 
            param_dict, dict of {parameter:value}
        output:
            circuit
        """
        if self.circuit_format=='qiskit':
            return self.construct_cirquit_qiskit(param_dict)
        else:
            raise NotImplementedError
        
    def construct_cirquit_qiskit(self,param_dict):
        """
        construct circuit for network using qiskit
        """        
        self.circ = qk.QuantumCircuit()
        # add quantum and classical registers
        for reg in self.qregs: self.circ.add_register(reg)
        #for reg in list(self.creg_dict.values()): self.circ.add_register(reg)
        
        for node in self.sorted_nodes:
            node_dict = {k:param_dict[k] for k in self.param_assignments[node]}
            node_circ = node.resolve_circuit(node_dict)
            self.circ = self.circ.combine(node_circ)
        
        return self.circ
            
    def to_qasm(self,param_dict):
        if self.circuit_format=='qiskit':
            return self.construct_circuit(param_dict).qasm()
        else:
            raise NotImplementedError()
    
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
            
            if 'boundary_circuit' in kwargs.keys():
                bdry_circ = kwargs['boundary_circuit']
            else:
                bdry_circ = QKParamCircuit(qk.QuantumCircuit(), [])
                
            # make the MPS/tensor-train -- same qubits used by each tensor
            self.bdry_tensor = IsoTensor('v_L',
                                         [breg],
                                         bdry_circ)
            self.sites= [[IsoTensor('A'+str(x)+str(y),
                                               [preg,breg],
                                               pcircs[y],
                                               meas_list=[(preg,
                                                           self.cregs[x][y],
                                                           qk.QuantumCircuit())])
                          for y in range(self.l_uc)]
                         for x in range(self.L)]
            
            # setup IsoNetwork
            # make a flat list of nodes
            self.nodes = [self.bdry_tensor]
            for x in range(self.L): self.nodes += self.sites[x]
            
            self.edges = [(self.nodes[i],self.nodes[i+1],{'qreg':breg}) for i in range(len(self.nodes)-1)]
            self.qregs = [preg,breg]
            
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
    def sample_correlations(self,L,bases,N_samples):
        """
        basis: measurement basis for each site
            possible formats: 
                - cirq circuit for physical qubits that maps physical qubits to measurement basis
                - string of 
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
                
        """
        raise NotImplementedError
