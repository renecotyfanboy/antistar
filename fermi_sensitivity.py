# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 21:29:45 2020

@author: simd9
"""

import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d

#%% Class Definition

class Fermi_sensitivity:
    
    def __init__(self,string):
        
        if string in ['l0b0','l0b30','l0b90','l120b45']:
        
            energy = np.array([3.62418640e+01, 4.76026992e+01, 6.25248461e+01, 8.21246788e+01,
                               1.07868524e+02, 1.41682362e+02, 1.86095916e+02, 2.44431906e+02,
                               3.21054637e+02, 4.21696503e+02, 5.53886848e+02, 7.27515257e+02,
                               9.55571433e+02, 1.25511700e+03, 1.64856192e+03, 2.16534108e+03,
                               2.84411641e+03, 3.73566928e+03, 4.90669964e+03, 6.44481605e+03,
                               8.46508998e+03, 1.11186646e+04, 1.46040625e+04, 1.91820376e+04,
                               2.51950828e+04, 3.30930536e+04, 4.34668226e+04, 5.70924851e+04,
                               7.49894209e+04, 9.84965576e+04, 1.29372540e+05, 1.69927300e+05,
                               2.23194871e+05, 2.93160371e+05, 3.85058146e+05, 5.05763365e+05,
                               6.64306376e+05, 8.72548294e+05])
    
            l0b0 = np.array([3.42282456e-11, 2.27380418e-11, 1.61082140e-11, 1.24250418e-11,
                               9.94847546e-12, 8.21981032e-12, 6.79152118e-12, 5.76391562e-12,
                               5.02929474e-12, 4.38830255e-12, 3.82900590e-12, 3.34099257e-12,
                               2.91517736e-12, 2.67266562e-12, 2.46273857e-12, 2.26930043e-12,
                               2.09105608e-12, 1.92681210e-12, 1.77546882e-12, 1.69994930e-12,
                               1.65422227e-12, 1.60972525e-12, 1.56642517e-12, 1.53631722e-12,
                               1.57878504e-12, 1.62242679e-12, 1.66727490e-12, 1.80731720e-12,
                               1.96137528e-12, 2.18587089e-12, 2.50515749e-12, 2.94757848e-12,
                               3.56746823e-12, 4.54170627e-12, 6.05636673e-12, 8.78970967e-12,
                               1.32314095e-11, 1.99176313e-11])
                        
            l0b30 = np.array([8.17521589e-12, 5.43084804e-12, 3.60774698e-12, 2.50881217e-12,
                               1.85868156e-12, 1.44759573e-12, 1.13363629e-12, 9.36653590e-13,
                               7.73898964e-13, 6.39424877e-13, 5.57617632e-13, 4.86548314e-13,
                               4.24536901e-13, 3.73868659e-13, 3.44502791e-13, 3.17443493e-13,
                               2.92509593e-13, 2.74229484e-13, 2.66852971e-13, 2.59674878e-13,
                               2.52689870e-13, 2.58861372e-13, 2.66016977e-13, 2.73370382e-13,
                               2.96025453e-13, 3.21259049e-13, 3.54239626e-13, 4.05982831e-13,
                               4.65284080e-13, 5.61121645e-13, 6.79128190e-13, 8.65838348e-13,
                               1.13026220e-12, 1.54334961e-12, 2.24222643e-12, 3.37528965e-12,
                               5.08092315e-12, 7.64846362e-12])
                        
            l0b90 = np.array([5.95082126e-12, 3.95316851e-12, 2.62611505e-12, 1.74454497e-12,
                               1.20748961e-12, 8.78880084e-13, 6.53141980e-13, 5.11009120e-13,
                               3.99806365e-13, 3.29117184e-13, 2.71929185e-13, 2.30017816e-13,
                               2.00701653e-13, 1.75121885e-13, 1.59450291e-13, 1.46926116e-13,
                               1.36544315e-13, 1.32871402e-13, 1.29297287e-13, 1.29387965e-13,
                               1.32964587e-13, 1.36640075e-13, 1.46230134e-13, 1.58694982e-13,
                               1.72222350e-13, 1.96553382e-13, 2.25263614e-13, 2.61339380e-13,
                               3.16300292e-13, 3.82819745e-13, 4.86694931e-13, 6.25085239e-13,
                               8.43727883e-13, 1.18499250e-12, 1.75819704e-12, 2.64666591e-12,
                               3.98410433e-12, 5.99738986e-12])
            
            l120b45 = np.array([4.68111989e-12, 3.10969779e-12, 2.06579207e-12, 1.37542498e-12,
                               9.66832974e-13, 7.16289023e-13, 5.46404601e-13, 4.27499293e-13,
                               3.40666264e-13, 2.81471476e-13, 2.32562482e-13, 1.99303602e-13,
                               1.73902018e-13, 1.51737909e-13, 1.38097588e-13, 1.27250583e-13,
                               1.17255565e-13, 1.13758159e-13, 1.10698173e-13, 1.08815150e-13,
                               1.11823085e-13, 1.14914168e-13, 1.20425894e-13, 1.30691155e-13,
                               1.41831440e-13, 1.58672094e-13, 1.81849068e-13, 2.08411465e-13,
                               2.51030437e-13, 3.03823330e-13, 3.80941224e-13, 4.86896800e-13,
                               6.53161687e-13, 9.09747236e-13, 1.34135549e-12, 2.01918202e-12,
                               3.03953431e-12, 4.57550073e-12])
            
            entries = {'l0b0':l0b0,'l0b30':l0b30,'l0b90':l0b90,'l120b45':l120b45}
            
            self.energy = energy*(u.MeV)
            self.bounds = (self.energy.min(),self.energy.max())
            self.E2flux = entries[string]*(u.erg*u.cm**(-2)*u.s**(-1))
            self.interpE2flux = interp1d(self.energy,self.E2flux,kind='cubic')

        else : 
            
            print("Utiliser l0b0, l0b30, l0b90,l120b45")
            
    def SED(self,E):
        
        flux = self.interpE2flux(E)*(u.erg*u.cm**(-2)*u.s**(-1))/E**2
        
        return flux.to(u.cm**(-2)*u.s**(-1)/(u.MeV))

    def flux(self,E):
        
        flux = self.interpE2flux(E)*(u.erg*u.cm**(-2)*u.s**(-1))
        
        return flux.to(u.MeV*u.cm**(-2)*u.s**(-1))
#%%

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    
    for string in ['l0b0','l0b30','l0b90','l120b45']:
    
        sensitivity = Fermi_sensitivity(string)
        bounds = sensitivity.bounds
        E_span = np.linspace(bounds[0],bounds[1],10000)
        plt.loglog(E_span,(sensitivity.flux(E_span)).to(u.erg*u.s**(-1)*u.cm**(-2)),label=string)
    
    plt.xlabel(r'$E_\gamma \:$ (MeV)')
    plt.ylabel(r'$\Phi_\gamma \:$ (erg.cm-2.s-1)')
    plt.grid(True,which="both",ls="-")
    plt.legend()