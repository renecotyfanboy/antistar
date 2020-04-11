# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:23:57 2020

@author: simd9
"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from scipy.stats import t

class Source_analyser:
    
    bands_center = np.sqrt([50*100,100*300,300*1000,1e3*3e3,3e3*10e3,10e3*30e3,30e3*300e3])
    band_error = np.vstack((np.abs(bands_center-np.array([50,100,300,1000,3000,10000,30000])),np.abs(bands_center-np.array([100,300,1000,3000,10000,30000,300000]))))
    
    def __init__(self,source):
        
        self.source = source
        self.spectrum_type = source.field('SpectrumType')[0]
        self.specs = self.get_specs()
        self.spectrum = self.get_func()
        
    def get_specs(self):
        
        specs = {}
        
        if self.spectrum_type == 'PowerLaw':
            
            specs['E0'] = self.source.field('Pivot_Energy')*u.MeV
            specs['K'] = self.source.field('PL_Flux_Density')*(1/u.cm**2/u.s/u.MeV)
            specs['gamma'] = self.source.field('PL_Index')
        
        else :
            
            print("Implémenter les autres types de spectres !")
            
        return specs
    
    def get_func(self):
        
        if self.spectrum_type == 'PowerLaw':
            
            def f(E):                
                
                E0 = self.specs['E0']
                K = self.specs['K']
                gamma = self.specs['gamma']
                
                return K*(E/E0)**(-gamma)*E**2
            
            return f
    
    def plot_nufnu(self):
        
        plt.loglog()
        yerr = (self.source.field('Unc_Flux_Band')[:,:,1]/self.source.field('Flux_Band')*self.source.field('nuFnu_Band')).reshape((7,))
        plt.errorbar(self.bands_center,self.source.field('nuFnu_Band')[0,:],xerr = self.band_error,yerr=yerr,fmt='none',color='black')
        
    def plot_model(self):
        
        plt.loglog()
        E_span = np.linspace(30,50000,10000)*u.MeV
        confprob = 0.60
        dof = 2
        tval = t.ppf(1.0 - (1.0 - confprob)/2, dof)
        sig = 0
        
        if self.spectrum_type == 'PowerLaw':
            
            sig += abs(self.source.field('Unc_PL_Flux_Density')*self.spectrum(E_span)/self.source.field('PL_Flux_Density'))
            sig += abs(self.source.field('Unc_PL_Index')*self.spectrum(E_span)*np.log(E_span/(self.source.field('Pivot_Energy')*u.MeV)))
            sig = tval*sig.to(u.erg/u.cm**2/u.s)
            
            plt.plot(E_span,(self.spectrum(E_span)).to(u.erg/u.cm**2/u.s),color='black',linestyle='--')
            plt.fill_between(E_span,(self.spectrum(E_span)+sig).to(u.erg/u.cm**2/u.s),(self.spectrum(E_span)-sig).to(u.erg/u.cm**2/u.s),color='lightgrey')
            
    def plot_all(self):
        
        self.plot_model()
        self.plot_nufnu()
        
        plt.xlabel(r'$E_\gamma$ [MeV]')
        plt.ylabel(r'$\nu F_\nu$ [erg.cm-2.s-1]')
        plt.xlim(45,300000)
        plt.ylim(0.09e-11,10e-11)