# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 22:52:10 2020

@author: simd9
"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from fermi_sensitivity import Fermi_sensitivity
from pion_decay import pion_decay

#%%

class Antiobject:
    
    sensitivity = Fermi_sensitivity('l0b30')
    bounds = sensitivity.bounds
    E_span = np.linspace(bounds[0],938*u.MeV,10000)
    F_span = pion_decay.spectrum(E_span)
    
    def __init__(self,**kwargs):
        
        self.proton_flux = kwargs.get('proton_flux',None)
        self.effective_surface = kwargs.get('effective_surface',None)
        self.gamma_luminosity = kwargs.get('gamma_luminosity',None)
        self.earth_distance = kwargs.get('earth_distance',None)
        
        if self.gamma_luminosity is None :
            self.flux_on_earth = (self.E_span**2*self.F_span*self.proton_flux*self.effective_surface/(4*np.pi*self.earth_distance**2)).to(u.MeV*u.cm**(-2)*u.s**(-1))
        
        else :
            self.flux_on_earth = (self.E_span**2*self.F_span/(4*np.pi*self.earth_distance**2)*self.gamma_luminosity/pion_decay.N).to(u.MeV*u.cm**(-2)*u.s**(-1))
        
        
        self.magnitude = np.log10(self.flux_on_earth/self.sensitivity.flux(self.E_span))
        self.max = np.max(self.magnitude)
        self.is_observable = self.max > 1
        
    def plot(self):
        
        plt.figure(figsize=(10,5))
        
        plt.subplot(121)
        plt.title(r'Flux $\gamma$ reçu sur Terre')
        plt.loglog(self.E_span,self.flux_on_earth,color='black')
        plt.xlabel(r'$E_\gamma$ (MeV)')
        plt.ylabel(r'$\Phi_\gamma$ (MeV.cm-2.s-1)')
        
        plt.subplot(122)
        plt.title(r'Ordre de magnitude par rapport à la sensibilité')
        plt.xlabel(r'$E_\gamma$ (MeV)')
        plt.ylabel('Magnitude')
        plt.semilogx(self.E_span,self.magnitude,color='black')

#%%

antijupiter = Antiobject(proton_flux = 2e8*(1/5.2)**2*(u.cm**(-2)*u.s**(-1)),
                         effective_surface = 2*np.pi*(7e7*u.m)**2,
                         earth_distance = 5.91e11*u.cm)

antiasteroid = Antiobject(proton_flux = 2e8*(1/4)**2*(u.cm**(-2)*u.s**(-1)),
                          effective_surface = 2*np.pi*(1*u.km)**2,
                          earth_distance = 3*u.AU)

antistar = Antiobject(gamma_luminosity = (1/3)*1e36*(10)**2*(1/10)**3/u.s,
                      earth_distance = 150*u.pc)