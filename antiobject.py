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
    E_span = np.linspace(bounds[0],925*u.MeV,10000)
    F_span = pion_decay.spectrum(E_span)
    
    def __init__(self,**kwargs):
        
        self.constructing_properties = kwargs
        self.computed_properties = {}
        
        # Initialization of antiobject properties
        self.proton_flux = kwargs.get('proton_flux',None)
        self.radius = kwargs.get('radius',None)
        self.mass = kwargs.get('mass',None)
        self.velocity = kwargs.get('velocity',None)
        self.effective_surface = kwargs.get('effective_surface',None)
        self.gamma_luminosity = kwargs.get('gamma_luminosity',None)
        self.earth_distance = kwargs.get('earth_distance',None)
        self.name = kwargs.get('name',None)
        
        if self.name is not None:
            self.constructing_properties.pop('name')
            
        # Gamma Luminosity computation with given data
        if self.gamma_luminosity is None :
            
            if self.mass is not None and self.velocity is not None:
                
                self.gamma_luminosity = (1/3)*1e36*(self.mass/u.M_sun)**2*(self.velocity/(10*u.km/u.s))**(-3)/(u.s)
                self.computed_properties["gamma_luminosity"] = self.gamma_luminosity
                
            elif self.proton_flux is not None:
                
                if self.effective_surface is None:
                    
                    self.effective_surface = 2*np.pi*self.radius**2
                    
                self.gamma_luminosity = (pion_decay.N*self.proton_flux*self.effective_surface).to(1/u.s)
                self.computed_properties["gamma_luminosity"] = self.gamma_luminosity
          
        # Earth flux computation
        self.flux_on_earth = (self.E_span**2*0.5*self.F_span/(4*np.pi*self.earth_distance**2)*self.gamma_luminosity/pion_decay.N).to(u.erg*u.cm**(-2)*u.s**(-1))
        self.magnitude = np.log10(self.flux_on_earth/self.sensitivity.flux(self.E_span))
        self.max = np.max(self.magnitude)
        self.is_observable = self.max > 1
        self.computed_properties["max_magnitude"] = self.max
        
    @classmethod
    def antistar_factory(cls,mass=10*u.M_sun,velocity=100*(u.km/u.s),earth_distance=30*u.pc,name = 'Antistar'):
        
        return cls(mass = mass,velocity = velocity,earth_distance = earth_distance,name = name)
     
    @classmethod
    def antiplanet_factory(cls,proton_flux = 2e8*(1/5.2)**2*(u.cm**(-2)*u.s**(-1)),radius = 7e7*u.m,earth_distance = 591e6*u.km,name = 'Antijupiter'):
    
        return cls(proton_flux = proton_flux,radius = radius,earth_distance = earth_distance,name = name)    
    
    
    
    def plot(self):
        
        from matplotlib import rc

        # activate latex text rendering
        rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
        rc('text', usetex=True)
        
        plt.figure(figsize=(9,4.5), tight_layout=True)
        plt.subplot(121)
        plt.loglog(self.E_span,self.flux_on_earth,color='black',label='Flux on Earth')
        plt.loglog(self.E_span,(self.sensitivity.flux(self.E_span)).to(u.erg*u.cm**(-2)*u.s**(-1)),color='black',linestyle=':',label='Fermi-LAT sensitivity')
        plt.xlabel(r'$E_\gamma$ [{}]'.format(self.E_span.unit.to_string('latex_inline')))
        plt.ylabel(r'$\Phi_\gamma$ [{}]'.format(self.flux_on_earth.unit.to_string('latex_inline')))
        plt.xlim(left=self.bounds[0]/u.MeV,right=937)
        plt.grid(True,which="both",ls="-")
        plt.legend()
        
        plt.subplot(122,adjustable='datalim')
        plt.axis('off')
        plt.table(self.gen_cells(), cellLoc='center',bbox=[0,0,1,1],fontsize=22)
        
    def gen_cells(self):
        
        key_to_latex = {'proton_flux':r'$\Phi_p$',
                        'radius':r'$R$',
                        'mass':r'$M$',
                        'velocity':r'$v$',
                        'effective_surface':r'$A$',
                        'gamma_luminosity':r'$L_\gamma$',
                        'earth_distance':r'$d$',
                        'max_magnitude':'Mag'}
        
        cells = []
        
        if self.name is not None:
            cells.append([r'\textbf{Name}',self.name])
        
        cells.append([r'\textbf{Construction Data}','Value'])
        
        for key,q in self.constructing_properties.items():
            cells.append([key_to_latex[key],'{:.2e} {}'.format(q.value,q.unit.to_string('latex_inline'))])
            
        cells.append([r'\textbf{Computed Data}','Value'])
    
        for key,q in self.computed_properties.items():
            if key != "max_magnitude":
                cells.append([key_to_latex[key],'{:.2e} {}'.format(q.value,q.unit.to_string('latex_inline'))])
            else :
                cells.append([key_to_latex[key],'{:.2} {}'.format(q.value,q.unit.to_string('latex_inline'))])
                
        return cells