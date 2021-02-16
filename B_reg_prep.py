#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14, 2021 lmao this my valentine

@author: Aaron
"""
### %% -- IMPORTS -- 
### external packages
import numpy as np
import qiskit as qk

### Special to these functions:
from scipy.optimize import minimize
from scipy import linalg as la
from scipy.sparse import linalg as lin
import bond_circuits as bcircs


#### TO DO: add this to the IsoMPS class in networks.py
#### TO DO: include this in the equivalent class for thermal states. 
#### NOTE: for thermal states, the information about the initial states should be included in the "tensors" method for convenience / modularity.


##### Might want to make a bond_circuits.py script????
##### basically, a clone of basic_circuits.py with no physical register?
#### these functions could call that?


def bond_qk_unitary(circuit, params):
    """
    inputs:
        - params, dictionary {parameter:value} for parameters in circuit
            note: parameter key type depends on type of circuit
            for qiskit: parameter keys are qiskit circuit parameters
            for cirq: they are sympy symbols
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


########## NEW CLASS FUNCTION:
    # @property ##if we want the method to create new class attributes dynamically when needed? seems unnecessary.
    def BondSteadyState(self, probs = [], num_eigs = 0, debug = False):
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
            for dummy1,B_tensor in enumerate(tensors):
                trans = np.tensordot(tensor,tensor.conj,axes=([1],[1]))
                ### check that all tensors have same shape
                if dummy1 > 0:
                    assert np.shape(trans) == t_shape
                t_shape = np.shape(trans)
                dumb_thing = np.tensordot(tensor,tensor.conj,axes=([1,2],[1,2]))
                dumb_thing -= np.eye(len(dum_thing[:,0]))
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

    

##### DEBUG !!!!, imported from bond_circuits.py
    self.bond_prep_circ, self.bond_prep_params = bcircs.bond_star_circ(self.breg, "burn in")

    

### function that takes the steady state density matrix, a parametrized circuit, and a cost function as an argument and finds the best pure state approximation to steady state

### fun is a callable cost function to be defined, with format fun(circuit, steady_state_density_matrix, optional arguments as needed)
        
### For now, we can use -np.abs(Trace[ steady_state circ |0><0| circ^{\dagg} ]) as the cost function. By default, minimize the cost function
        

    def Burn_In_Unitary(self, steady_state, fun, args = None):
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
        
        return burn_in_params
        
    
    
    






### given the best approximation using a unitary, density matrices, lambdas, and the maximum number of burn in layers, returns the error
### Advanced: may want to check if there is only a slight increase in error for using less than the max number of burn in layers?
    















#### use class methods for alternative instantiation
# @classmethod
# def From_Tenpy(cls, iMPS):
        
        
        
        
        
        
        
        
        
        