

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  29 04:04:53 2021

@author: shahin75
"""

# thermal state imports
import numpy as np
import random
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.networks.site import SpinHalfSite
import scipy.linalg as la

class thermal_state(object):
    
    """
    Represents thermal states (in the forms of Matrix Product Density Operator 
    and random-hologrphic Matrix Product State) and is used for finite-temperature 
    simulations.   
    """

    def __init__(self, tensor, N):
        """
        Parameters
        --------------
        N: int
            Number of sites in the main network chain (= L * l_uc, where L is number of 
            repetitions of the unit cell and l_uc is the length of unit-cell).
        tensor: numpy.ndarray
            Bulk rank-4 tensors of the main chain.
            tensor index ordering: physical-out, bond-out, physical-in, bond-in
            (with "in/out" referring to the right canonical form ordering)               
        """
        self.N = N
        self.tensor = tensor
        # tensor dimensions (consistent with rank-4 structure)
        self.d = tensor[:,0,0,0].size # physical leg dimension (assumes rank-4 structures)
        self.chi = tensor[0,:,0,0].size # bond leg dimension (assumes rank-4 structures)

    def network_from_cells(self, network_type, 
                           L, chi_MPO=None, 
                           params=None, bdry_vecs=None, 
                           method=None, T=None):      
        """
        Returns network of finite thermal-holographic Matrix Product State (random-holoMPS), finite 
        holo-MPS, finite holographic Matrix Product Operator (holoMPO), or MPO of a given model.
        --------------
        Inputs:
              --the input assumes the list of unitary tensors at each unit-cell or rank-4 numpy.ndarray--       
          network_type: str
             One of "random_state", "circuit_MPS", "circuit_MPO", or "MPO" options.
          L: int
             Length (number) of repetitions of unit cell in the main network chain.
          chi_MPO: int
             Bond leg dimension for MPO-based structures. 
          params: numpy.ndarray
             Optimized parameters for unitary structure and probability weights.
          bdry_vecs: list
            List of left (first element) and right (second element) boundary vectors.
            (must be set to [None,None] by default, which gives left and right boundary vectors = |0> 
            for MPO-based structures. For holoMPS-based structures, the default [None,None]
            would give left boundary = |0> while the right boundary is traced over). 
          method: str 
            One of "thermal_state_class" or "tenpy" options. (if set to "tenpy", the returned structure
            would be one of physics-TenPy networks). This option is currently only available for 
            "random_state", "circuit_MPS", and "MPO" options. 
           T: float
            Temperature (for thermal-holoMPS option).
        Note:
          -For random_state, circuit_MPS and circuit_MPO options, the original circuit with 
           parameters must be inserted as args. In this case, the returned list of bulk tensors
           includes rank-3 numpy.ndarray for random_state/circuit_MPS and rank-4 numpy.ndarray for
           circuit_MPO.
          -For holoMPS-based structures, the index ordering is: site, physical_out, bond-in, bond-out
           while for holoMPO-based structures, the index ordering is: physical-out, bond-out,
           physical-in, bond-in (with "in/out" referring to right canonical form ordering).
          -For MPO structures constructed by "thermal_state_class method", the unit cell tensor of MPO 
           network must be inserted as arg (e.g. Hamiltonian unit cell). In this case, the bulk tensors 
           would be rank-4 numpy.ndarray (consistent with final structure of MPO). For "tenpy"-method-based
           structures, the list of bulk tensors must be inserted (see TeNPy docs for more detail).     
          -Tracing over right boundary for holoMPS-based structures is appropriate for 
           holographic simulations. 
          -Set bdry_vecs to None by default for "tenpy" method. Set method to None for holoMPO-based 
           structures.
        """
        
        # for circuit-based structures:
        # both circuit and params must be included
        if network_type == 'random_state' or network_type == 'circuit_MPS' or network_type == 'circuit_MPO':
            
            l_uc = len(self) # length of unit-cell
            N = l_uc * L # number of sites
            unitary_list = L * self # list of unitaries
            
            # if network_type is set to random-holoMPS:
            if network_type == 'random_state':
                
                # defining tensor dimensions
                tensor = np.swapaxes(unitary_list[0][:,:,0,:],1,2) # change to MPS-based structure
                d = tensor[:,0,0].size # physical leg dimension (for random state)
                chi = tensor[0,:,0].size # bond leg dimension (for random state)
                
                if T == 0:
                    tensor_list1 = [np.swapaxes(unitary[:,:,0,:],1,2) for unitary in unitary_list]
                else:
                    # list of variational probability weights and random selections at each site
                    probs_list = L * thermal_state.prob_list(self,params,T)
                    random_list = [random.choice(p) for p in probs_list]
                    index_list = [probs_list[j].index(random_list[j]) for j in range(N)]
                    tensor_list1 = [np.swapaxes(unitary[:,:,j,:],1,2) for unitary,j in zip(unitary_list,
                                                                                           index_list)]      

            # if network_type is set to holoMPS:
            elif network_type == 'circuit_MPS':
                
                # defining tensor dimensions
                tensor = np.swapaxes(unitary_list[0][:,:,0,:],1,2) # change to MPS-based structure
                d = tensor[:,0,0].size # physical leg dimension (for random state)
                chi = tensor[0,:,0].size # bond leg dimension (for random state)
                # bulk tensors of holoMPS structure
                tensor_list1 = [np.swapaxes(unitary[:,:,0,:],1,2) for unitary in unitary_list]  

            # if network_type is set to circuit_MPO 
            # this option assumes original, circuit-based MPO structures (e.g. holoMPO)
            elif network_type == 'circuit_MPO':
                
                # defining tensor dimensions (consistent with rank-4 structures)
                # index ordering consistent with holographic-based MPO structures
                d = unitary_list[0][:,0,0,0].size # physical leg dimension (for MPO)
                chi = unitary_list[0][0,:,0,0].size # bond leg dimension (for MPO)
                tensor_list1 = unitary_list
            
            # testing boundary conditions 
            
            if network_type == 'random_state' or network_type == 'circuit_MPS':
                # specific to holoMPS-based structures
                
                if method == 'tenpy':
                    # based on previous circuit file
                    tensor_list1[0] = tensor_list1[0][:,0:1,:]
                    tensor_list1[-1] = tensor_list1[-1][:,:,0:1]
                    site = SpinHalfSite(None) 
                    M = MPS.from_Bflat([site]*N, tensor_list1, bc='finite', dtype=complex, form=None)
                    MPS.canonical_form_finite(M,renormalize=True,cutoff=0.0)
                
                elif method == 'thermal_state_class':
                    bdry = []
                    # if boundary vectors are not specified for holoMPS-based structures:     
                    # checking left boundary vector
                    # if left boundary vector not specified, set to (1,0,0,0...)
                    if np.array(bdry_vecs[0] == None).all():
                        bdry += [np.zeros(chi)]
                        bdry[0][0] = 1
                    else:
                        if bdry_vecs[0].size != chi:
                            raise ValueError('left boundary vector different size than bulk tensors')
                        bdry += [bdry_vecs[0]]
                
                    # checking right boundary vector (special to holoMPS-based structures)
                    if np.array(bdry_vecs[1] == None).all():
                        bdry += [None]
                    else:
                        if bdry_vecs[1].size != chi:
                            raise ValueError('right boundary vector different size than bulk tensors')
                        bdry += [bdry_vecs[1]]
                    
                    # if both boundary vectors are specified
                    for j in range(2):
                        if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi:
                            bdry.append(bdry_vecs[j]) 
                    
                    M = [[bdry[0]],tensor_list1,[bdry[1]]] # final state structure
                    
                else: 
                    raise ValueError('only one of "thermal_state_class" or "tenpy" options')
                        
            elif network_type == 'circuit_MPO': # specific to holoMPO-based structures
                bdry = []
                for j in range(2):
                    # if both boundary vectors are specified 
                    if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi:
                        bdry.append(bdry_vecs[j])
                    
                    # if boundary vectors not specified, set to (1,0,0,0...)
                    elif np.array(bdry_vecs[j] == None).all():
                        bdry += [np.zeros(chi)]
                        bdry[j][0] = 1
                    else:
                        if bdry_vecs[j].size != chi:
                            raise ValueError('boundary vectors different size than bulk tensors')
                        bdry += [bdry_vecs[j]]         
            
                M = [[bdry[0]],tensor_list1,[bdry[1]]] # final state structure
                      
        # if network_type is set to MPO: 
        # this option assumes genuine MPO_based structures (e.g. Hamiltonian MPO)  
        elif network_type == 'MPO':  
            
            if method == 'tenpy': # tenpy-based MPO
                site = SpinHalfSite(None)
                M = MPO.from_grids([site]*L, self, bc = 'finite', IdL=0, IdR=-1)  
                
            elif method == 'thermal_state_class':               
                # only bulk tensors of the main chain must be included (w/out params)
                tensor_list1 = [self]*L
                # testing boundary conditions
                bdry = []
                for j in range(2):
                    # if both boundary vectors are specified 
                    if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi_MPO:
                        bdry.append(bdry_vecs[j])
                
                    # if boundary vectors not specified, set to (1,0,0,0...)
                    elif np.array(bdry_vecs[j] == None).all():
                        bdry += [np.zeros(chi_MPO)]
                        bdry[j][0] = 1
                    else:
                        if bdry_vecs[j].size != chi_MPO:
                            raise ValueError('boundary vectors different size than bulk tensors')
                        bdry += [bdry_vecs[j]]
                        
                M = [[bdry[0]],tensor_list1,[bdry[1]]] # final state structure
            else: 
                raise ValueError('only one of "thermal_state_class" or "tenpy" options')              
        else:
            raise ValueError('only one of "random_state", "circuit_MPS", "circuit_MPO", "MPO" options')
            
        return M

    def prob_list(self, params, T):
        """ 
        Returns list of variational probability weights (based on Boltzmann ditribution)
        for thermal-holographic matrix product state or thermal density matrix 
        product operator for each unit-cell.
        --------------
        Inputs:
            --the input assumes the list of unitary tensors at each unit-cell--
          params: numpy.ndarray
             Optimized parameters for unitary structure and probability weights.
          T: float
             Temperature.  
        """
        # tensor dimensions (consistent with rank-4 structure)
        # physical leg dimension list (for all circuits in unit-cell)
        d_list = [unitary[:,0,0,0].size for unitary in self] 
        # bond leg dimension (for all circuits in unit-cell)
        chi_list = [unitary[0,:,0,0].size for unitary in self]
        l_uc = len(self) # length of unit-cell
        d_tot = sum(d_list) # total physical leg dimension * l_uc
        prob_params = params[:d_tot]
        
        if T != 0: # checking temperature
            exc_list= [np.exp(-k/T) for k in prob_params] # list of boltzmann weights
            # grouping boltzmann weights for each site
            group_list = [exc_list[j:j+d] for d,j in zip(d_list,range(0,len(exc_list),d))] 
            z_list = [sum(j) for j in group_list] # list of partition functions 
            norm_list = []
            for j in range(len(group_list)): # normalizing weights
                for k in group_list[j]:
                    norm_list.append(k/z_list[j])
            # final probs for unit-cell
            prob_list = [norm_list[j:j+d] for d,j in zip(d_list,range(0,len(norm_list),d))]  
            
        else:
            prob_list = [(np.zeros(d)).tolist() for d in d_list]
            # setting probability of |0> to 1 at T = 0
            for j in range(len(prob_list)): 
                prob_list[j][0] += 1
        return prob_list

    def density_matrix(self, params, 
                       T, L, 
                       bdry_vecs=None):      
        """
        Returns thermal Matrix Product Density Operator (MPDO).
        --------------
        Inputs:
            --the input assumes the list of unitary tensors at each unit-cell--
          params: numpy.ndarray
             Optimized parameters for unitary structure and probability weights.
          L: int
             Length (number) of repetitions of unit cell in the main network chain. 
          T: float
             Tempreture.
          prob_list: list 
             List of probability weights of each physical state (the length of prob_list 
             should match the physical leg dimension). If set to None, it would call
             thermal_based prob_list fuction to compute probability weights for density
             matrix.
          bdry_vecs: list
            List of left (first element) and right (second element) boundary vectors.
          Note:
            -If bdry_vecs is set to None, the infinite version of MPDO (iMPDO) would be returned.
             In this case, the steady-state state of density matrix is computed based on MPDO 
             transfer-matrix structure.
            -If bdry_vecs is set to [None,None], the left and right boundary = id would be returned,
             where id is the identity element with appropriate dimension consistent with the bond 
             dimension chi.
        """
        
        # tensor dimensions (consistent with rank-4 structure)
        # index ordering consistent with holographic-based MPO structures
        d = self[0][:,0,0,0].size # physical leg dimension
        chi = self[0][0,:,0,0].size # bond leg dimension
        l_uc = len(self) # length of unit-cell
        N = l_uc * L # number of sites
        
        # construction of state
        if bdry_vecs == None:
            # constructing state for iMPDO
            state = thermal_state.network_from_cells(self,'circuit_MPO',
                                                     L,None,
                                                     params,[None,None], 
                                                     None,T)
        else: # constructing state for MPDO
            state = thermal_state.network_from_cells(self,'circuit_MPO',
                                                     L,None,
                                                     params,bdry_vecs,
                                                     None,T)
        # constructing the probability weights chain
        probs_list = L * thermal_state.prob_list(self,params,T)
        #p_matrix_chain = [np.diag(p) for p in probs_list]
        
        # contractions of density matrix:
        contractions = [] 
        for j in range(N):
            W1 = np.einsum('abcd,c,eick->abdeik',state[1][j],probs_list[j],state[1][j].conj())
            #W1 = np.einsum('abcd,ck->abdk',state[1][j],p_matrix_chain[j])
            # reordering and reshaping indices
            W2 = np.einsum('abdeik->abiedk',W1)
            #W2 = np.einsum('abdk,cjki->abdcji',W1,state[1][j].conj())
            #W3 = np.einsum('abdcji->abjcdi',W2)
            contractions.append(W2)

        if bdry_vecs != None:
            # left boundary contraction
            bvecl = np.eye(chi)
            bvecr = np.eye(chi) 
            method = 'MPDO' # structure label
        else:
            # finding steady-state of transfer matrix of iMPDO 
            # finding tranfer matrix:
            if l_uc == 1: # if only one unitary within each unit-cell
                transfer_mat = np.einsum('abad->bd',np.reshape(contractions[0],[d,chi**2,d,chi**2]))
                new_contractions =  contractions
            else: # for l_uc > 1
                W_list = [np.reshape(contractions[j],[d,chi**2,d,chi**2]) for j in range(l_uc)]
                W_list1 = [W_list[0]]
                for j in range(1,len(W_list)):
                    tensor = np.reshape(np.einsum('acdijk->aijckd',
                             np.einsum('abcd,ijkb->acdijk',W_list1[0],W_list[j])),
                             [d**2,chi**2,d**2,chi**2])
                    W_list1 = []
                    W_list1.append(tensor)
                transfer_mat = np.einsum('abad->bd',W_list1[0]) # tranfer matrix
                new_contractions = contractions[:-1]
            eig_vals,eig_vecs = la.eig(transfer_mat)
            idx = np.where(np.abs(1-abs(eig_vals))<1e-5)[0][0] # index of steady-state
            steady_den = np.reshape(eig_vecs[:,idx],[chi,chi]) # steady-state density matrix
            bvecl = steady_den/np.trace(steady_den) # normalization of steady-state
            #v = eig_vecs[:,idx] # steady-state
            method = 'iMPDO' # structure label
            bvecr = np.eye(chi) 
            
        density_matrix = [[bvecl],new_contractions,[bvecr],method] ## last element = structure label
        
        return density_matrix

    def network_site_contraction(self, state_type, chi_MPO=None, MPO=None):
        """
        Returns a list of contractions of the newtork at each site for <MPS|MPS> 
        (tranfer-matrix-like structures), <MPS|MPO|MPS>, or <MPO|DMPO> networks.
        MPS: holographic-matrix-product-state-based structures.
        MPO: matrix product operator.
        MPDO: matrix product density operator.
        --------------
        Inputs:
          --the input assumes thermal_state_class-based holoMPS and MPDO structures--
          state_type: str
             One of "random_state", "circuit_MPS", or "density_matrix" options. 
          chi_MPO: int
             Bond leg dimension for MPO-based structure. 
          MPO: thermal_state_class-based MPO structure.  
             Set to None for pure wave function simulations for MPS states.
        Note:
          -If MPO is not inserted for holoMPS states, the function computes transfer matrices 
           for the state wave fucntion at each site.
          -Length of MPO structure might be less than length of state.
          -The output would be returned as a list of contraction matrices computed for each 
           unit cell at each site.
        """     
        contraction_list = []
        # for holoMPS and random holoMPS-based structures: 
        if state_type == 'random_state' or state_type == 'circuit_MPS':
            
            # tensor dimensions (consistent with rank-3 structures)
            # index ordering consistent with holoMPS-based structures
            tensor = self[1][0]
            d = tensor[:,0,0].size # physical leg dimension 
            chi = tensor[0,:,0].size # bond leg dimension 
            N = len(self[1]) # number of sites (L * l_uc) in the main network chain (for holoMPS state).
            
            # contracted (transfer) matrices for the wave function: 
            # (w/out MPO inserted)
            if MPO == None: 
                # site contractions for state and its dual
                for j in range(N):
                    # contraction state/dual state
                    tensor1 = np.tensordot(self[1][j].conj(),self[1][j],axes=[0,0])
                    # reshaping into matrix 
                    # contraction (transfer) matrix at each site
                    tensor2 = np.reshape(np.swapaxes(np.swapaxes(np.swapaxes(tensor1,0,3),0,1),2,3),
                                         [chi**2,chi**2]) 
                    contraction_list.append(tensor2)                 
     
            # contracted matrices w/ MPO inserted
            else: 
                N_MPO = len(MPO[1]) # number of sites for squeezed MPO structure.
                # site contractions for state/MPO/dual-state
                for j in range(N_MPO):
    
                    # contractions with state
                    chi_s = chi_MPO * chi
                    tensor1 = np.tensordot(MPO[1][j],self[1][j],axes=[2,0])
                    tensor2 = np.reshape(np.swapaxes(np.swapaxes(tensor1,1,2),2,3),[d,chi_s,chi_s])

                    # contractions with dual state
                    chi_tot = chi_s * chi # total bond dimension
                    tensor3 = np.tensordot(self[1][j].conj(),tensor2,axes=[0,0])
                    # contracted matrices at each site
                    tensor4 = np.reshape(np.swapaxes(np.swapaxes(np.swapaxes(tensor3,0,3),0,1),2,3),
                                         [chi_tot,chi_tot])
                    contraction_list.append(tensor4)
                
                # contraction of rest of state and its dual if N_MPO different than N
                for j in range(N-N_MPO):
                    # contraction state/dual state
                    tensor5 = np.tensordot(self[1][j].conj(),self[1][j],axes=[0,0])
                    # reshaping into matrix 
                    # contraction (transfer) matrix at each site
                    tensor6 = np.reshape(np.swapaxes(np.swapaxes(np.swapaxes(tensor5,0,3),0,1),2,3),
                                         [chi**2,chi**2]) 
                    contraction_list.append(tensor6)    
   
        # contracted network structure at each site for density matrix
        # must include MPO (e.g. Hamiltonian MPO)
        elif state_type == 'density_matrix':
            
            N = len(self[1]) # number of sites in main network chain.
            N_MPO = len(MPO[1]) # number of sites for inserted MPO structure.
            # tensor dimensions 
            d = np.shape(self[1][0])[0] # density matrix physical leg dimension
            d_MPO = MPO[1][0][:,0,0,0].size # MPO physical leg dimension
            chi = np.shape(self[1][0])[1] # density matrix bond leg dimension
            chi_MPO = MPO[1][0][0,:,0,0].size # MPO bond leg dimension
            chi_tot = chi**2 * chi_MPO # total bond leg dimension
            tensor_list = []
            
            # confirming the label of structure
            if self[3] == 'MPDO' or self[3] == 'iMPDO': 
                
                if self[3] == 'MPDO': # left and right contractions for finite structure
                    s_left = np.reshape(np.einsum('abcdee->abcd',self[1][0]),[d,chi**2,d])
                    #s_left = np.reshape(np.einsum('abcdej,ej->abcd',self[1][0],self[0][0]),[d,chi**2,d])
                    s_right = np.reshape(np.einsum('aeebcd->abcd',self[1][-1]),[d,d,chi**2])
                    #s_right = np.reshape(np.einsum('aejbcd,ej->abcd',self[1][-1],self[2][0]),[d,d,chi**2])
            
                elif self[3] == 'iMPDO':  # left and right contractions infinite structure 
                    s_left = np.reshape(np.einsum('abcdee,ee->abcd',self[1][0],self[0][0]),[d,chi**2,d])
                    #s_right = np.reshape(np.einsum('aeebcd,ee->abcd',self[1][-1],self[3][0]),[d_s,d_s,chi_s**2])
                    s_right = np.reshape(np.einsum('aeebcd->abcd',self[1][-1]),[d,d,chi**2])
                
                # MPO boundary contractions
                MPO_left = np.einsum('abcd,d->abc',MPO[1][0],MPO[0][0])
                MPO_right = np.einsum('abcd,b->acd',MPO[1][-1],MPO[2][0]) 
                # contractions of boundary MPO and state and reshaping         
                left_tensor = np.einsum('bcde->dbec',np.einsum('abc,dea->bcde',s_left,MPO_left))
                right_tensor = np.einsum('bcde->dbce',np.einsum('abc,dae->bcde',s_right,MPO_right))
                bvecl = np.reshape(np.einsum('jbkj->bk',left_tensor),[chi_tot])
                bvecr = np.reshape(np.einsum('aack->ck',right_tensor),[chi_tot]) 
                
                # bulk contractions
                for j in range(1,N-1):
                    # MPO and density matrix contractions
                    tensor1 = np.einsum('abcd,ijak->bcdijk',np.reshape(self[1][j],[d,chi**2,d,chi**2]),
                                        MPO[1][j])
                    #tensor1 = np.einsum('abcd,cijk->abdijk',self[1][j],MPO[1][j]) 
                    tensor2 = np.einsum('bcdijk->ibjcdk',tensor1) # index reordering
                    #tensor2 = np.einsum('abdijk->abijdk',tensor1)
                    tensor3 = np.reshape(tensor2,[d,chi_tot,d_MPO,chi_tot])
                    # tracing over p_out and p_in
                    tensor4 = np.trace(tensor3,axis1=0,axis2=2)
                    tensor_list.append(np.reshape(tensor4,[chi_tot,chi_tot]))

                contraction_list = [bvecl] + tensor_list + [bvecr]
            
            else:
                raise ValueError('structure label must be "MPDO" or "iMPDO"')
        else:
            raise ValueError('only one of "random_state", "circuit_MPS", or "density_matrix" options')
            
        return contraction_list

    def expectation_value(self, state_type,
                          L, chi_MPO=None,
                          MPO=None, params=None, 
                          method=None, T=None):
        """
         Returns the numerical result of full contractions of <MPS|MPS>,
         <MPS|MPO|MPS>, or <MPO|DMPO> networks.
         MPS: holographic-matrix-product-state-based structures.
         MPO: matrix product operator.
         DMPO: density matrix product operator.
        --------------
        Inputs:
          --the input assumes thermal_state_class-based holoMPS, DMPO structures,
                 and the list of unitary tensors at each unit-cell--
          state_type: str
             One of "random_state", "circuit_MPS", or "density_matrix" options. 
          chi_MPO: int
             Bond leg dimension for MPO-based structures.
          MPO: thermal_state_class-based MPO structure.
             Set to None for pure wave function simulations.
          L: int
             Length (number) of repetitions of unit cell in the main network chain.
          params: numpy.ndarray
             Optimized parameters for unitary structure and probability weights.
             (only required for density matrix-method II).
          method: str
             One of "method_I" or "method_II" options. This option is available 
             for "density_matrix".
          T: float
           Temperature       
        Note:
          -Left boundary condition is set by the given holoMPS boundary vectors, and the right 
           boundary is averaged over (as consistent with holographic-based simulations).
          -If MPO is not inserted (for MPS structures), the function computes the expectation value 
           for the state wave fucntion (<MPS|MPS>).
          -The expectation value could be computed with two different methods using density matrix-
           based structures. If method is set to "method_II", the list of unitary tensors at each
           unit-cell must be passed as input.
          -If "method_II" is selected for density matrix operations, the MPO must have the same
           index ordering as TenPy (virutal left, virtual right, physical out, physical in).
        """ 
        
        # for holoMPS and random holoMPS-based structures:
        if state_type == 'random_state' or state_type == 'circuit_MPS':
            
            # list of contracted matrices
            con_mat = thermal_state.network_site_contraction(self,state_type,
                                                             chi_MPO,MPO) 
            # accumulation of contracted matrices defined at each site
            con_mat0 = con_mat[0]
            for j in range(1,len(con_mat)):
                con_mat0 = con_mat[j] @ con_mat0
            
            # tensor dimensions (consistent with rank-3 structure)
            # index ordering consistent with holoMPS-based structures
            tensor = self[1][0]
            d = tensor[:,0,0].size # physical leg dimension (for holoMPS)
            chi = tensor[0,:,0].size # bond leg dimension (for holoMPS)
    
            # w/out MPO inserted
            if MPO == None:
                bvecl = np.kron(self[0][0].conj(),self[0][0]) # left boundary contraction
                # right boundary contraction:
                if np.array(self[2][0] == None).all():
                    # summing over right vector if right boundary condition is not specified 
                    con_mat_on_rvec = np.reshape(con_mat0 @ bvecl,[chi**2])
                    rvec = np.reshape(np.eye(chi),[chi**2])
                    expect_val = np.dot(rvec,con_mat_on_rvec)
                else:
                    bvecr = np.kron(self[2][0].conj(),self[2][0])
                    expect_val = bvecr.conj().T @ con_mat0 @ bvecl
        
           # w/ MPO inserted
            else:
                # left boundary contraction
                bvecl = np.kron(self[0][0].conj(),np.kron(MPO[0][0],self[0][0])) 
                # right boundary constraction:
                if np.array(self[2][0] == None).all():
                    # summing over right vectors if right boundary condition is not specified
                    # employ the specified right boundary vector of MPO.                 
                    con_vleft = np.reshape((con_mat0 @ bvecl),[chi,chi_MPO,chi]) # con_mat on left vector
                    MPO_rvec_contracted = np.reshape(np.tensordot(MPO[2][0],con_vleft,axes=[0,1]),[chi**2])
                    rvec = np.reshape(np.eye(chi),[chi**2])
                    expect_val = np.dot(rvec,MPO_rvec_contracted)
                else:
                    bvecr = np.kron(self[2][0].conj(),np.kron(MPO[2][0],self[2][0]))
                    expect_val = bvecr.conj().T @ con_mat0 @ bvecl

        # for density-matrix-based structures:
        # must include MPO (e.g. Hamiltonian MPO)
        elif state_type == 'density_matrix': 
   
            # first method of contracting density matrix structures
            if method == 'method_I':
                chi_MPO = MPO[1][0][0,:,0,0].size # MPO bond leg dimension
                chi = np.shape(self[1][0])[1] # density matrix bond leg dimension
                # list of contracted matrices
                con_mat = thermal_state.network_site_contraction(self,state_type,
                                                                 chi_MPO,MPO) 
                con_mat0 = con_mat[1]
                # boundary vectors
                bvecl = con_mat[0]
                bvecr = con_mat[-1]
                
                # checking structure label
                if self[3] == "MPDO": # finite structures
                    if len(con_mat) == 2:
                        expect_val = (bvecr.T @ bvecl)/chi
                    else:
                        #M = np.linalg.matrix_power(con_mat0,len(con_mat)-2)
                        for j in range(2,len(con_mat)-1):
                            con_mat0 = con_mat[j] @ con_mat0
                        expect_val = (bvecr.T @ con_mat0 @ bvecl)/chi
                elif self[3] == "iMPDO": # infinite structures
                    if len(con_mat) == 2:
                        expect_val = bvecr.T @ bvecl
                    else: # case of l_uc > 1
                        #M = np.linalg.matrix_power(con_mat0,len(con_mat)-2)
                        for j in range(2,len(con_mat)-1):
                            con_mat0 = con_mat[j] @ con_mat0
                        expect_val = bvecr.T @ con_mat0 @ bvecl
                else:
                    raise ValueError('structure label must be "MPDO" or "iMPDO"')
               
            # second method of contracting density matrix structures
            elif method == 'method_II':  
                
                l_uc = len(self) # length of each unit-cell
                N = L * l_uc # total number of sites
                # changing index ordering to: p_out, b_out, p_in, b_in
                MPO_list = [np.einsum('abcd->cadb',MPO[1][0]) for j in range(l_uc)] 
                # list of probabilities for each unit-cell
                prob_list = thermal_state.prob_list(self,params,T)
                # list of contracted matrices
                con_mat = [np.einsum('arbs,bick,c,ajcl->rijskl',MPO_list[m],self[m],
                                     prob_list[m],self[m].conj()) for m in range(l_uc)]
                # contraction with left boundary vector 
                bvecl = np.einsum('rijskk,s->rij',con_mat[0],MPO[0][0])
                
                for n in range(1,N):
                    bvecl = np.einsum('rijskl,skl->rij',con_mat[n % l_uc],bvecl)
                expect_val = np.einsum('rii,r',bvecl,MPO[2][0])
            
            else:
                ValueError('only one of "method_I" or "method_II" options')
        else:
            raise ValueError('only one of "random_state", "circuit_MPS", or "density_matrix" options')
          
        return (expect_val).real

    def entropy(self):
        """
        Returns the von Neumann entropy of a given probability 
        weight list (in the form of Shannon entropy) for each unit-cell.
        --------------
        --the input assumes thermal_state_class-based prob_list--
        """
        # physical leg dimension list (for all circuits in unit-cell)
        d_list = [len(j) for j in self]
        l_uc = len(self) # unit-cell length 
        
        # avoiding NaN in numpy.log() function
        new_prob_list = [np.array(j)[np.array(j) > 1.e-30] for j in self] 
        s_list1 = []
        for j in range(len(new_prob_list)):
            for p in new_prob_list[j]:
                s_list1.append(-p*np.log(p)) # converting to form of Shannon entropy
        
        # list of entropies at each site (within unit-cell)
        s_list2 = [sum(s_list1[j:j+d]) for d,j in zip(d_list,range(0,len(s_list1),d))]
        s = sum(s_list2)/l_uc # entropy
        return s

    def free_energy(self, params, 
                    state_type, L, 
                    Hamiltonian, T,
                    chi_H=None, bdry_vecs1=None, 
                    bdry_vecs2=None, method=None, 
                    N_sample=None,
                    S=None):
        """
        Returns the Helmholtz free energy of a thermal density matrix structure 
        or thermal holographic matrix product state.
        --------------
        Inputs:
        --the input assumes the list of unitary tensors at each unit-cell--
        state_type: str
           One of "density_matrix" or "random_state"  options.
        params: numpy.ndarray
           Optimized parameters for unitary structure and probability weights.
        L: int
           Length (number) of repetitions of unit cell in the main network chain.
        Hamiltonian: numpy.ndarray 
           The Hamiltonian MPO of model.  
        T: float
           Temperature
        chi_H: int
           Bond leg dimension for Hamiltonian MPO structure.
        bdry_vecs1 and bdry_vecs2: list
           List of left (first element) and right (second element) boundary vectors for 
           state and Hamiltonian networks, respectively (set to [None,None] by default
           for finite structures and None for infinte structures).
        method: str
           For densit matrix: one of "method_I" or "method_II" options.
           For random-holoMPS: one of 'thermal_state_class', 'tenpy', or 'MPO_tenpy' options.
           ("MPO_tenpy" is set when MPO structure is already constructed in TenPy).
        N_sample: int
           Number of samples for averaging energy of thermal random holoMPS (only for 
           "random_state" option.  
        S: float
           Optional entropy setting (if set to None, returns thermal_state-based entropy).
        """   
        l_uc = len(self) # length of unit-cell
        N = l_uc * L # total number of sites
        
        # for density-matrix-based structures:
        if state_type == 'density_matrix':
            
            if S == None:
                S = thermal_state.entropy(thermal_state.prob_list(self,params,T)) # entropy
             
            if method == 'method_I': # first method of computing density matrix structures free energy
                
                density_mat = thermal_state.density_matrix(self,params,
                                                           T,L,
                                                           bdry_vecs1) # density matrix
                MPO_Hamiltonian = thermal_state.network_from_cells(Hamiltonian,'MPO',
                                                                   N,chi_H,
                                                                   None,bdry_vecs2,
                                                                   'thermal_state_class',T) # Hamiltonian MPO
                E = thermal_state.expectation_value(density_mat,'density_matrix',
                                                    L,chi_H,
                                                    MPO_Hamiltonian,params,
                                                    'method_I',T) # energy of system          
                F = E - T*S # total Helmholtz free energy 
            elif method == 'method_II':
                
                MPO_Hamiltonian = thermal_state.network_from_cells(Hamiltonian,'MPO',
                                                                   N,chi_H,
                                                                   None,bdry_vecs2,
                                                                   'thermal_state_class',T) # Hamiltonian MPO
                E = thermal_state.expectation_value(self,'density_matrix',
                                                    L,chi_H,
                                                    MPO_Hamiltonian,params,
                                                    'method_II',T) # energy of system            
                F = (E/(N-l_uc)) - T*S # Helmholtz free energy  
                
            else:
                ValueError('only one of "method_I" or "method_II" options')
            
        # for random-holoMPS-based structures:
        elif state_type == 'random_state':
            
            N = l_uc * L # total number of sites
            if S == None:
                S = thermal_state.entropy(thermal_state.prob_list(self,params,T)) # entropy
            
             # computing free energy using thermal state library built-in functions
            if method == 'thermal_state_class':
                random_state = thermal_state.network_from_cells(self,'random_state',
                                                                L,None,
                                                                params,bdry_vecs1,
                                                                'thermal_state_class',T) # random_state MPS
                MPO_Hamiltonian = thermal_state.network_from_cells(Hamiltonian,'MPO',
                                                                   N,chi_H,
                                                                   None,bdry_vecs2,
                                                                   'thermal_state_class',T) # Hamiltonian MPO
                # sampling over different runs
                Es = [thermal_state.expectation_value(random_state,'random_state',
                                                      L,chi_H,
                                                      MPO_Hamiltonian,params,
                                                      method,T) for j in range(N_sample)] 
                E = np.mean(Es) # energy of system
                F = (E/N) - T*S # Helmholtz free energy   
            
            elif method == 'tenpy':
                MPO_Hamiltonian = thermal_state.network_from_cells(Hamiltonian,'MPO',
                                                                   N,chi_H,
                                                                   None,bdry_vecs2,
                                                                   'tenpy',T) # Hamiltonian MPO
                # sampling over different runs
                Es = [(MPO_Hamiltonian.expectation_value(random_state)).real for j in range(N_sample)] 
                E =  np.mean(Es) # energy of system
                F = (E/N) - T*S # Helmholtz free energy
            
            elif method == 'MPO_tenpy':
                random_state = thermal_state.network_from_cells(self,'random_state',
                                                                L,None,
                                                                params,bdry_vecs1,
                                                                'tenpy',T) # random_state MPS
                # sampling over different runs
                Es = [(Hamiltonian.expectation_value(random_state)).real for j in range(N_sample)] 
                E =  np.mean(Es) # energy of system
                F = (E/N) - T*S # Helmholtz free energy
                
            else: 
                raise ValueError('only one of "thermal_state_class", "tenpy", or MPO_tenpy options')
                
        else:
            raise ValueError('only one of "random_state" or "density_matrix" options')
        return F
