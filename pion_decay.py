# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 15:31:33 2020

@author: simd9
"""

#test push

import numpy as np
import astropy.units as u

class pion_decay:
    
    def spectrum(E):
    
        E_loc = E/u.MeV
        E0 = 938.28
        alpha_1 = 4.155
        beta_1 = -33.494
        alpha_2 = 1.673
        beta_2 = -19.982
        alpha_3 = -0.03273
        beta_3 = -0.006921
        N = 3.82
        
        return N*(np.sign(E0-E_loc)*(np.abs(E0-E_loc))**alpha_1*np.exp(beta_1)+np.sign(E0-E_loc)*(np.abs(E0-E_loc))**alpha_2*np.exp(beta_2)+beta_3*np.exp(alpha_3*E_loc))/u.MeV
    
    def plot():
        
        pass