"""
"""

import sys
sys.path.append("..") # import one subdirectory up in files

import numpy as np

from networks.networks import IsoMPS

#%% XXZ Tenpy MPO
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.tools.params import asConfig
from tenpy.networks.site import SpinHalfSite
#__all__ = ['XXZModel', 'XXChain']
class XXZModel(CouplingMPOModel):
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity')
        assert conserve != 'Sz'
        if conserve == 'best':
            conserve = 'parity'
            if self.verbose >= 1.:
                print(self.name + ": set conserve to", conserve)
        site = SpinHalfSite(conserve=conserve)
        return site
    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1.))
        d = np.asarray(model_params.get('d', 1.))
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(J, u1, 'Sigmax', u2, 'Sigmax', dx)
            self.add_coupling(J, u1, 'Sigmay', u2, 'Sigmay', dx)
            self.add_coupling(J*d, u1, 'Sigmaz', u2, 'Sigmaz', dx)
        # done
class XXZChain(XXZModel, NearestNeighborModel):
    def __init__(self, model_params):
        model_params = asConfig(model_params, self.__class__.__name__)
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)

def xxz_mpo(J,Delta,hz=0):
    model_params = dict(J=J, d=Delta, bc_MPS='infinite', conserve=None, verbose=False)
    return XXZChain(model_params).H_MPO

#%% (Free) Energy calculators
def energy_tp(param_vals,*args):
    """
    function to calculate energy using MPO/MPS contraction in tenpy
    inputs:
        - param_vals = dict {parameter:value}
        - *args, 
            args[0] should be psi: state as IsoMPS
            args[1] should be H_mpo: Hamiltonian as MPO
        (input made this way to be compatible w/ scipy.optimize)
    outputs:
        - float, <psi|H|psi> computed w/ tenpy
    """
    # parse inputs
    psi=args[0] # state as isoMPS    
    H_mpo = args[1] # Hamiltonian as tenpy MPO
    
    param_dict = dict(zip(psi.param_list,param_vals))
    
    # convert state from holoPy isoMPS to tenpy MPS
    psi_tp = psi.to_tenpy(param_dict,L=np.inf)
    
    # compute energy
    E = (H_mpo.expectation_value(psi_tp)).real
    return E

# def energy_tp_corr(param_vals,*args):
#     psi=args[0]
#     J,Delta,h = args[1]
#     psi_tp=psi.to_tenpy(opt_params,L=np.inf)
#     Cxx_tp = np.mean([4*psi_tp.correlation_function("Sx", "Sx", 
#                                                  sites1=[j], 
#                                                  sites2=[j+1],
#                                                  opstr=None, str_on_first=False, hermitian=False, autoJW=False)
#                       for j in range(l_uc)])
#     Cyy_tp = np.mean([4*psi_tp.correlation_function("Sy", "Sy", 
#                                                  sites1=[j], 
#                                                  sites2=[j+1],
#                                                  opstr=None, str_on_first=False, hermitian=False, autoJW=False)
#                       for j in range(l_uc)])
#     Czz_tp = np.mean([4*psi_tp.correlation_function("Sz", "Sz", 
#                                                  sites1=[j], 
#                                                  sites2=[j+1],
#                                                  opstr=None, str_on_first=False, hermitian=False, autoJW=False)
#                       for j in range(l_uc)])
#     return J*(Cxx_tp+Cyy_tp+Delta*Czz_tp)

def entropy(prob_list,L):
    # entropy function taken from Shahin's code; L should be viewed as L*l_uc in his convention
    """
    Returns the von Neumann entropy (per site) of a given list 
    probability weight list (the form of Shannon entropy).
    --------------
    --the input assumes thermal_state_class-based prob_list--
    L: int
         Length (number) of repetitions of unit cell in the main network chain.   
    """
    new_prob_list = [np.array(j)[np.array(j) > 1.e-30] for j in prob_list] # avoiding NaN in numpy.log() function
    s_list1 = []
    d=1
    for j in range(len(new_prob_list)):
        for p in new_prob_list[j]:
            s_list1.append(-p*np.log(p)) # converting to form of Shannon entropy
            
    s_list2 = [sum(s_list1[j:j+d]) for j in range(0,len(s_list1),d)]
    s_tot = sum(s_list2)/L # average entropy of chain
    return s_tot

def free_energy_tp(tot_param_vals,*args):
    #we need to define psi each time we call the function under our current circuit construction;
    #tot_param_vals is just a list (array) of prams with first L*l_uc terms being the prob list on 
    #each site and the rest being the regular parameters
    L,l_uc = args[0]
    H_mpo = args[1]
    prob, param_vals = np.split(tot_param_vals,[L*l_uc])
    prob = np.reshape(prob,(L,l_uc,1))
    psi = IsoMPS(preg,breg,site_pcircs,boundary_circuit=bond_prep_pcirc,L=L,thermal = True,thermal_prob=prob)
    E = energy_tp(param_vals,psi,H_mpo)
    S = entropy(prob,L*l_uc)
    F = E - T*S
    return F