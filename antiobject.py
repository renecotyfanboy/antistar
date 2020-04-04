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
    
    def __init__(self,settings):
        
        self.sensitivity = Fermi_sensitivity('l0b30')
        self.bounds = self.sensitivity.bounds
        self.E_span = np.linspace(self.bounds[0],938*u.MeV,10000)
        self.F_span = pion_decay.spectrum(self.E_span)
        self.proton_flux = settings['proton_flux'] 
        self.effective_surface = settings['effective_surface'] 
        self.earth_distance = settings['earth_distance'] 
        self.flux_on_earth = (self.E_span**2*self.F_span*self.proton_flux*self.effective_surface/(4*np.pi*self.earth_distance**2)).to(u.MeV*u.cm**(-2)*u.s**(-1))
        self.magnitude = np.log10(self.flux_on_earth/self.sensitivity.flux(self.E_span))
        self.max = np.max(self.magnitude)
        
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

jupiter_settings = {'proton_flux' : 2e8*(1/5.2)**2*(u.cm**(-2)*u.s**(-1)),
                    'effective_surface' : 2*np.pi*(7e7*u.m)**2,
                    'earth_distance' : 5.91e11*u.cm}
   
asteroid_settings = {'proton_flux' : 2e8*(1/4)**2*(u.cm**(-2)*u.s**(-1)),
                    'effective_surface' : 2*np.pi*(1*u.km)**2,
                    'earth_distance' : 3*u.AU}
     
antijupiter = Antiobject(asteroid_settings)
antijupiter.plot()

