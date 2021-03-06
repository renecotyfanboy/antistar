    # -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 15:31:33 2020

@author: simd9
"""

import numpy as np
import astropy.units as u

class pion_decay:
    
    E_max = 70.77483833786383*u.MeV
    N = 3.82
    
    def spectrum(E):
    
        E0 = 938.28
        alpha_1 = 4.155
        beta_1 = -33.494
        alpha_2 = 1.673
        beta_2 = -19.982
        alpha_3 = -0.03273
        beta_3 = -0.006921
        
        #Section efficace totale
        N = 3.82 #gamma/annihilation p+/p-
        
        if isinstance(E,u.Quantity):
            E_loc = E/u.MeV
            return N*((E0-E_loc)**alpha_1*np.exp(beta_1)+(E0-E_loc)**alpha_2*np.exp(beta_2)+beta_3*np.exp(alpha_3*E_loc))/u.MeV
        else :
            E_loc = E
            if E>E0:
                return 0
            else:
                return N*(np.sign(E0-E_loc)*(np.abs(E0-E_loc))**alpha_1*np.exp(beta_1)+np.sign(E0-E_loc)*(np.abs(E0-E_loc))**alpha_2*np.exp(beta_2)+beta_3*np.exp(alpha_3*E_loc))
        
        
        
#%% Demo
@np.vectorize
def spectrum(E):
    
    E0 = 938.28
    alpha_1 = 4.155
    beta_1 = -33.494
    alpha_2 = 1.673
    beta_2 = -19.982
    alpha_3 = -0.03273
    beta_3 = -0.006921
    
    #Section efficace totale
    #N = 3.82 #gamma/annihilation p+/p-
    E_loc = E
    if E>E0:
        return 0
    else:
        return (np.sign(E0-E_loc)*(np.abs(E0-E_loc))**alpha_1*np.exp(beta_1)+np.sign(E0-E_loc)*(np.abs(E0-E_loc))**alpha_2*np.exp(beta_2)+beta_3*np.exp(alpha_3*E_loc))

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from matplotlib import rc
    
    rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    rc('text', usetex=True)
    
    fig, ax = plt.subplots(figsize=(3,3))
    
    E_span = np.linspace(1*u.MeV,938*u.MeV,10000)
    F_span = pion_decay.spectrum(E_span)
    
    ax.loglog(E_span,F_span,color='black')
    
    ax.set_xlabel(r'$E_\gamma$  [MeV]')
    ax.set_ylabel(r'$F(E_\gamma)$  [ph.MeV-1]')
    ax.vlines(pion_decay.E_max/u.MeV, 1e-9, pion_decay.spectrum(pion_decay.E_max)*u.MeV,linestyle=':',color='black')
    ax.hlines(pion_decay.spectrum(pion_decay.E_max)*u.MeV,3,pion_decay.E_max/u.MeV,linestyle=':',color='black')
    
    ax.grid(True, which="both")
    ax.set_xlim(left=3,right=2000)  
    ax.set_ylim(bottom=5e-6,top=5e-2)
    plt.title(r'$p-\bar{p}$ annihilation')
    plt.tight_layout()
    plt.savefig('fig.png',dpi=600, transparent = True)