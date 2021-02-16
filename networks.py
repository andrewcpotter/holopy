#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:20:36 2020
@author: acpotter
"""
#%% -- IMPORTS -- 
import sys
sys.path.append("..") # import one subdirectory up in files
# external packages
import numpy as np
import qiskit as qk
import networkx as nx
import tenpy

from scipy.optimize import minimize
from scipy import linalg as la
from scipy.sparse import linalg as lin
import bond_circuits as bcircs

# custom things
from networks.isonetwork import IsoTensor, IsoNetwork, QKParamCircuit
import mps.mps as mps


def bond_qk_unitary(circuit, params):
    """
    this is meant to turn a parametrized circuit into a unitary ndarray
    """
    # setup unitary simulator and compute unitary
    bound_circ = circuit.bind_parameters(params)
    simulator = qk.Aer.get_backend('unitary_simulator')
    result = qk.execute(bound_circ,simulator).result()
    u = result.get_unitary(bound_circ) 

    # need to re-size and re-order to be compatible with expected indexing
    # note: qiskit writes bases in opposite order of usual convention
    # e.g. for 3-qubit register: [q0,q1,q2], 
    # the state 011 refers to: q0=1, q1=1, q2=0
    
    ### AARON: what is this? if need to uncomment, will include "nbond" as input argument
    
    # u = u.reshape(self.regdims[::-1]+self.regdims[::-1]) # reshape as tensor
    ### nreg = len(self.qregs)
    # old_order = list(range(2*nbond))
    # new_order = old_order.copy()
    # new_order[0:nbond] = old_order[0:nbond][::-1]
    # new_order[nbond::] = old_order[nbond::][::-1]
    # u = np.moveaxis(u,old_order,new_order)
    return u

def basic_cost_function(qk_circuit, circ_params, target_matrix):
    # chi = len(target_matrix[0,:])
    unitary = bond_qk_unitary(qk_circuit, circ_params)
    print(" 'unitary' has shape {} ".format(np.shape(unitary)))
    pure_state = unitary[:,0]
    trace_norm = np.vdot(pure_state,np.dot(target_matrix,pure_state))
    return -trace_norm


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
        #whether to prepare the state as a thermal distribution
        
        if 'thermal' in kwargs.keys():
            self.thermal = kwargs['thermal']
            if 'thermal_prob' in kwargs.keys(): # this should be a prob. list in shape of l*l_uc*n_phys
                self.thermal_prob = kwargs['thermal_prob']
            else: raise ValueError('thermal state requires probability distribution')
        else:
            self.thermal = False
            self.thermal_prob =[[0 for i in range(self.l_uc)] for j in range(self.L)] 

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
                self.measurement_circuit = self.measurement(kwargs['bases'], preg, self.FH)
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
                                                            self.measurement_circuit[x][y])],
                                            thermal=self.thermal, 
                                    thermal_prob=self.thermal_prob[x][y])
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
        
        ### for bond qubit prep (from Aaron)
        self.bond_prep_circ, self.bond_prep_params = bcircs.bond_star_circ(self.breg, "burn in")
            
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
        ulist=[]
        for j in range(self.l_uc): # start at one to skip boundary vector
            site = self.sites[0][j] 
            site_param_names = site.param_names
            site_param_vals = [params[k] for k in site_param_names]
            site_param_dict = dict(zip(site_param_names,site_param_vals))
            ulist += [site.unitary(site_param_dict)]
#        ulist = [self.sites[j][0].unitary(params) for j in range(self.l_uc)]
        return ulist
    
    def tensors(self,params):
        """
        computes tensors for fixed initial state of physical qubit = |0>
        inputs:
            params, dictionary of parameters {'name':numerical-value}
        returns:
            tensors, list of rank-3 tensors for each site in unit cell
        """
        tensors=[]
        for j in range(self.l_uc): # start at one to skip boundary vector
            site = self.sites[0][j] 
            site_param_names = site.param_names
            site_param_vals = [params[k] for k in site_param_names]
            site_param_dict = dict(zip(site_param_names,site_param_vals))
            tensors += [site.unitary(site_param_dict)[:,:,0,:]]
        #tensors = [self.sites[j][0].unitary(params)[:,:,0,:] for j in range(self.l_uc)]
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
        # elif (L==np.inf):
        #     B_arrs = [np.swapaxes(tensor,1,2) for tensor in self.tensors(params)[0]]
        #     psi = tenpy.networks.mps.MPS.from_Bflat([site]*self.l_uc,
        #                                             B_arrs, 
        #                                             bc = 'infinite', 
        #                                             dtype=complex, 
        #                                             form=None)  
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
    def measurement(self, bases, preg, FH=False):
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
                    raise ValueError('bases must have same length as L')
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
                            # extract pauli basis for a single qubit from the string
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
    
    
    ### Below code is from Aaron for bond prep        
    
    
    # @property ##if we want the method to create new class attributes dynamically when needed? seems unnecessary.
    def get_bond_steady_state(self, num_eigs = 0, debug = False):
        """
    
        Parameters
        ----------
        probs : list, optional
            DESCRIPTION: The default is [], an empty list.
            NOTE: this argument should be eliminated in favor of modifying the IsoMPS.tensors() method
        num_eigs : int, optional
            DESCRIPTION: The default is 0, and the method will find all eigenvalues/eigenmodes of the transfer matrix using full diagonalization. 
            For num_eigs = k with 0 < k < chi^2 - 1, the method will find the largest k eigenvalues using sparse methods.
        debug : bool, optional
            DESCRIPTION: If Debug is set to True, various sanity checks will be performed and output printed.
            The default is False (no debugging)
        Returns
        -------
        density_matrices : list of ndarrays
            DESCRIPTION: A list of chi by chi density matrices for the bond state that are eigenmodes of the transfer matrix, in order of descending eigenvalue.
            The first entry corresponds to the steady state.
        lambdas : list
            DESCRIPTION: a list of eigenvalues of the transfer matrix in descending order. The first entry should be 1.

        """
        ### NOTE: probs is for the thermal case; ideally, this should be built into the "tensors" method itself
        
        ### num_eigs = 0 will compute ALL eigenvalues of transfer matrix. Seems reasonable for the bond sizes we're dealing with.
        if num_eigs >= (4**self.nbond)-1:
            num_eigs = 0
        
        tensors = self.tensors()
        
        chi = len(tensors[0][:,0,0])
        
        #### Sanity checks:
        ########################
        if debug:
            print("Debugging enabeled")
            print()
            print("Found {} tensors".format(len(tensors)))
            print()
            t_shape = None
            for dummy1,B_tensor in enumerate(tensors):
                trans = np.tensordot(B_tensor,B_tensor.conj,axes=([1],[1]))
                ### check that all tensors have same shape
                if dummy1 > 0:
                    assert np.shape(trans) == t_shape
                t_shape = np.shape(trans)
                dumb_thing = np.tensordot(B_tensor,B_tensor.conj,axes=([1,2],[1,2]))
                dumb_thing -= np.eye(len(dumb_thing[:,0]))
                print("things that should (probably) be zero: {}, {}, {}".format(np.average(dumb_thing),np.std(dumb_thing),max(np.abs(dumb_thing).tolist())))
                print()
        
        
        ### NOTE: can probably change dtype to float (or double) below?
        transfer_matrix = np.zeros((chi,chi,chi,chi),dtype=complex)
        
        for foo1,tensor in enumerate(tensors):
            t_mat = np.tensordot(tensor,tensor.conj,axes=([1],[1]))
            #### NOTE: Assuming tensor indices have format [left_bond, physical, right_bond]
            ### t_mat indices SHOULD be [left_lower,right_lower,left_upper,right_upper], right?
            t_mat = np.swapaxes(t_mat,1,2)
            ### t_mat indices should NOW be [left_lower,left_upper,right_lower,right_upper]
            if foo1 == 0:
                transfer_matrix[:,:,:,:] = t_mat[:,:,:,:]
            else:
                transfer_matrix[:,:,:,:] = np.tensordot(transfer_matrix,t_mat,axes=([2,3],[0,1]))[:,:,:,:]
        
        #### Sanity checks:
        ########################
        if debug:
            dumb_thing = np.tensordot(transfer_matrix,np.eye(chi),axes=([2,3],[0,1])) - np.eye(chi)
            print("If right canonical, should be zero: {}, {}, {}".format(np.average(dumb_thing),np.std(dumb_thing),max(np.abs(dumb_thing).tolist())))
            print()
            
            
        ### Unless I fucked up the indices, transfer_matrix is now the transfer matrix of the unit cell.
                
        
        ### Now we reshape and diagonalize the transfer matrix. Split into two steps out of caution.
        dummy = np.reshape(transfer_matrix,(chi**2,chi,chi))
        transfer2 = np.reshape(dummy,(chi**2,chi**2))
        ### complex conjugation is **just in case**
        ### transfer2 has indices [left_bonds,right_bonds], it is now a linear operator acting on the space of bond density matrices
        ### we want LEFT eigenvectors of this object (right?)
        
        #### Sanity checks:
        ########################
        if debug:
            dumb_thing = np.reshape(np.reshape(transfer2,(chi,chi,chi,chi)) - transfer_matrix,(chi**4))
            print("If I understand reshape, should be zero: {}, {}, {}".format(np.average(dumb_thing),np.std(dumb_thing),max(np.abs(dumb_thing).tolist())))
            print()
            
        
        lambdas = []
        density_matrices = []
        
        if num_eigs == 0:
            eigs, vecs = la.eig(transfer2, left=True, right=False)
            ### if it's Hermitian we can just use eigh instead?
            ### eigenvalues will be returned in ascending order
            for foo2 in range(len(eigs)-1,-1,-1):
                lambdas.append(eigs[foo2])
                density_matrices.append(np.reshape(vecs[foo2],(chi,chi)))
                ### the density matrices sould be Hermitian, so the reshape should be fine...
                
        else:
            right_transfer = np.transpose(transfer2).conj()
            eigs, vecs = lin.eigs(right_transfer, k=num_eigs, which='LM')
            ### if it's Hermitian we can just use eigsh instead?
            ### eigenvalues will be returned in ascending order??? WE SHOULD CHECK.
            for foo3 in range(len(eigs)-1,-1,-1):
                lambdas.append(eigs[foo3])
                density_matrices.append(np.reshape(vecs[foo3],(chi,chi)))
        
        #### Sanity checks:
        ########################
        if debug:
            print("First and last eigenvales are {} and {}".format(lambdas[0],lambdas[-1]))
            print("The first eigenvalue should be 1...")
            tguy = np.zeros((chi**2,chi**2),dtype=complex)
            idguy = np.zeros((chi**2,chi**2),dtype=complex)
            for dumdum in range(len(lambdas)):
                dm = density_matrices[dumdum]
                check1 = np.trace(dm)
                print("Density matrix {} has trace {}".format(dumdum,check1))
                dumb_thing = dm - np.transpose(dm.conj())
                print("If density matrices are Hermitean, should be zero: {}, {}, {}".format(np.average(dumb_thing),np.std(dumb_thing),max(np.abs(dumb_thing).tolist())))
                dumb_thing = dm - np.matmul(dm,dm)
                print("If density matrices are idempotent, should be zero: {}, {}, {}".format(np.average(dumb_thing),np.std(dumb_thing),max(np.abs(dumb_thing).tolist())))
                tguy += lambdas[dumdum]*np.outer(dm,dm)
                idguy += np.outer(dm,dm)
                print()
            dumb_thing = np.reshape(tguy - transfer2,(chi**4))
            print("If eigenmodes reproduce transfer matrix, should be zero: {}, {}, {}".format(np.average(dumb_thing),np.std(dumb_thing),max(np.abs(dumb_thing).tolist())))
            print()
            dumb_thing = np.reshape(idguy - np.eye(chi**2),(chi**4))
            print("If eigenmodes form complete set, should be zero: {}, {}, {}".format(np.average(dumb_thing),np.std(dumb_thing),max(np.abs(dumb_thing).tolist())))
            print()
        
        ### OPTIONAL:
        # self.steady_state_dm = density_matrices[0]
        
        ### Instead of returning lists, can make this method a "@property" of class
        ### the method then adds/sets the class properties lambdas, [bond_]density_matrices
        return (lambdas, density_matrices)
    
    def set_burn_in_params(self, steady_state, fun=basic_cost_function, args = None):
        """
        
    
        Parameters
        ----------
        steady_state : ndarray
            A chi by chi density matrix for the bond register representing the steady state of the holoMPS transfer matrix
            Should be the first entry in the "density_matrices" output of BondSteadyState() function.
        fun : callable function
            A callable "cost function" fun( circuit, parameters, steady_state_density_matrix, optional arguments). 
            The first three arguments *MUST* be 1) circuit parameters, 2) the circuit 3) the target steady state DM for the bond register.
            We will *MINIMIZE* the cost function using an optimizer.
            The default is "basic_cost_function"
        args : tuple, optional
            A tuple of additional arguments to be fed into the cost function, fun( circuit, parameters, ssdm). IF NEEDED.
            These arguments come *AFTER* the required three arguments as detailed above. Tuple must be correclty ordered. 
            The default is None.
    
        Returns
        -------
        burn_in_params : list
            Returns the list of optimized parameters for the bond circuit
            Same format as self.bond_params
    
        """
        ### The lines below are apparently a useful trick to mask optional args of cost function so it's easier to reference as fun(a,b) in the future
        if args is not None:
            fun = lambda  bond_params, bond_circ, ssdm, fun=fun: fun(bond_params, bond_circ, ssdm, *args)
        ### now fun is a three-argument function of some circuit (acting on the bond space) and some steady state density matrix.
        
        ### use some optimizer to minimize fun by varying parameters of circ
            
        ### OPTIMIZATION
        burn_in_params = minimize(fun, self.bond_prep_params, args=(self.bond_prep_circ,steady_state),method='BFGS')
        
        self.bond_prep_params = burn_in_params
