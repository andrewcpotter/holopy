#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14, 2021 lmao this my valentine

@author: Aaron
"""
### %% -- IMPORTS -- 
### external packages
import numpy as np
### Special to this function:
from scipy import linalg as la
from scipy.sparse import linalg as lin



#### TO DO: add this to the IsoMPS class in networks.py
#### TO DO: include this in the equivalent class for thermal states. 
#### NOTE: for thermal states, the information about the initial states should be included in the "tensors" method for convenience / modularity.

########## NEW CLASS FUNCTION:
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
            eigs, vecs = lin.eigs(night_transfer, k=num_eigs, which='LM')
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

    



##### FUNCTIONS that use the output of the above class function
    

### function that takes the steady state density matrix, a parametrized circuit, and a cost function as an argument and finds the best pure state approximation to steady state

### fun is a callable cost function to be defined, with format fun(circuit, steady_state_density_matrix, optional arguments as needed)
        

### For now, we can use -np.abs(Trace[ steady_state circ |0><0| circ^{\dagg} ]) as the cost function. By default, minimize the cost function
        

def Burn_In_Unitary(steady_state, circuit, fun, args = None):
    ### The lines below are apparently a useful trick to mask optional args of cost function so it's easier to reference as fun(a,b) in the future
    if args is not None:
        fun = lambda circ, ssdm, fun=fun: fun(circ, ssdm, *args)
    ### now fun is a two-argument function of some circuit (acting on the bond space) and some steady state density matrix.
        
    ### use some optimizer to minimize fun by varying parameters of circ
        
    ### good_circ = circuit with ideal parameters
        
    return good_circ
        
    
    
    






### given the best approximation using a unitary, density matrices, lambdas, and the maximum number of burn in layers, returns the error
### Advanced: may want to check if there is only a slight increase in error for using less than the max number of burn in layers?
    















#### use class methods for alternative instantiation
# @classmethod
# def From_Tenpy(cls, iMPS):
        
        
        
        
        
        
        
        
        
        