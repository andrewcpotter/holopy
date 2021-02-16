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
    def BondSteadyState(self, probs = [], num_eigs = 0):
        """
    
        Parameters
        ----------
        probs : list, optional
            DESCRIPTION: The default is [], an empty list.
            NOTE: this argument should be eliminated in favor of modifying the IsoMPS.tensors() method
        num_eigs : int, optional
            DESCRIPTION: The default is 0, and the method will find all eigenvalues/eigenmodes of the transfer matrix using full diagonalization. 
            For num_eigs = k with 0 < k < chi^2 - 1, the method will find the largest k eigenvalues using sparse methods.

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
                
        ### Unless I fucked up the indices, transfer_matrix is now the transfer matrix of the unit cell.
                
        
        ### Now we reshape and diagonalize the transfer matrix. Split into two steps out of caution.
        dummy = np.reshape(transfer_matrix,(chi**2,chi,chi))
        transfer2 = np.reshape(dummy,(chi**2,chi**2))
        ### complex conjugation is **just in case**
        ### transfer2 has indices [left_bonds,right_bonds], it is now a linear operator acting on the space of bond density matrices
        ### we want LEFT eigenvectors of this object (right?)
        
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
                
        ##### TO DO: Add sanity checks:
        ### 1) is the leading eigenvalue 1?
        ### 2) Are the density matrices hermitian, trace 1, and square to themselves?
        ### 3) Are the eigenvalues sorted correctly?
        ### 4) For benchmarking, we can check that summing over \lambda_n \rho_n reproduces the original Transfer matrix.
                
        return (density_matrices,lambdas)

    








#### use class methods for alternative instantiation
# @classmethod
# def From_Tenpy(cls, iMPS):
        
        
        
        
        
        
        
        
        
        