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
        
        self.computed_properties = {}
        
        # Initialization of antiobject properties
        self.gamma_luminosity = kwargs.get('gamma_luminosity',None)
        self.constructing_properties = kwargs.get('construction_properties',None)
        self.earth_distance = kwargs.get('earth_distance',None)
        self.type = kwargs.get('type','Custom Build')
        self.name = kwargs.get('name',None)    
        
        # Gamma Luminosity computation with given data
        self.computed_properties["gamma_luminosity"] = self.gamma_luminosity
          
        # Earth flux computation
        self.flux_on_earth = (self.E_span**2*self.F_span/(4*np.pi*self.earth_distance**2)*self.gamma_luminosity/pion_decay.N).to(u.erg*u.cm**(-2)*u.s**(-1))
        self.magnitude = np.log10(self.flux_on_earth/self.sensitivity.flux(self.E_span))
        self.max = np.max(self.magnitude)
        self.is_observable = self.max > 1
        self.computed_properties["max_magnitude"] = self.max
        
    @classmethod
    def antistar_factory(cls,mass=10*u.M_sun,velocity=100*(u.km/u.s),earth_distance=30*u.pc,name='Antistar'):
        
        construction_properties = {'mass':mass,'velocity':velocity,'earth_distance':earth_distance}
        
        gamma_luminosity = 0.5*(1/3)*1e36*(mass/u.M_sun)**2*(velocity/(10*u.km/u.s))**(-3)/(u.s)
        
        return cls(gamma_luminosity=gamma_luminosity,earth_distance=earth_distance,construction_properties = construction_properties,name=name,type = 'Milky Way Antistar')
     
    @classmethod
    def antiplanet_factory(cls,proton_flux = 2e8*(1/5.2)**2*(u.cm**(-2)*u.s**(-1)),radius = 7e7*u.m,earth_distance = 591e6*u.km,name='Antijupiter'):
        
        construction_properties = {'proton_flux':proton_flux,'radius':radius,'earth_distance':earth_distance}
        
        effective_surface = 4*np.pi*radius**2
        gamma_luminosity = (0.5*pion_decay.N*proton_flux*effective_surface).to(1/u.s)
        
        return cls(gamma_luminosity=gamma_luminosity,earth_distance=earth_distance,construction_properties = construction_properties,name=name,type ='Solar Antiobject')   
    
    @classmethod
    def cluster_factory(cls,density=1e-5*u.cm**(-3),temperature=1e8*u.K,fraction = 1e-7,length=2.2*u.Mpc,earth_distance=16.5*u.Mpc,name='Virgo Cluster'):
        
        volume = (4/3)*np.pi*length**3
        construction_properties = {'density':density,'temperature':temperature,'fraction':fraction,'length':length,'earth_distance':earth_distance}
        
        gamma_luminosity = (0.5*3e-14*fraction*(1e-8/u.K*temperature)**(-1/2)*density**2*volume*u.cm**3/u.s).to(1/u.s)
    
        return cls(gamma_luminosity=gamma_luminosity,earth_distance=earth_distance,construction_properties = construction_properties,name=name,type ='Galaxy Cluster')
    
    def plot(self):
        
        from matplotlib import rc

        # activate latex text rendering
        rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
        rc('text', usetex=True)
        
        plt.figure(figsize=(9,4.5), tight_layout=True)
        plt.subplot(121)
        plt.loglog(self.E_span,self.flux_on_earth,color='black',label='Flux on Earth')
        plt.loglog(self.E_span,(self.sensitivity.flux(self.E_span)).to(u.erg*u.cm**(-2)*u.s**(-1)),color='black',linestyle=':',label='Fermi-LAT sensitivity \n 10 years point source')
        plt.xlabel(r'$E_\gamma$ [{}]'.format(self.E_span.unit.to_string('latex_inline')))
        plt.ylabel(r'$\nu F_\nu$ [{}]'.format(self.flux_on_earth.unit.to_string('latex_inline')))
        plt.xlim(left=self.bounds[0]/u.MeV,right=937)
        plt.grid(True,which="both",ls="-")
        plt.legend()
        
        plt.subplot(122,adjustable='datalim')
        plt.axis('off')
        plt.table(self.gen_cells(), cellLoc='center',bbox=[0,0,1,1],fontsize=22)
    
    def plot_spectrum(self):
        
        from matplotlib import rc
        
        # activate latex text rendering
        rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
        rc('text', usetex=True)
        
        plt.figure(figsize=(3,3), tight_layout=True)
        plt.loglog(self.E_span,self.flux_on_earth,color='black',label='Flux on Earth')
        plt.loglog(self.E_span,(self.sensitivity.flux(self.E_span)).to(u.erg*u.cm**(-2)*u.s**(-1)),color='black',linestyle=':',label='Fermi-LAT sensitivity \n 10 years point source')
        plt.xlabel(r'$E_\gamma$ [{}]'.format(self.E_span.unit.to_string('latex_inline')))
        plt.ylabel(r'$\nu F_\nu$ [{}]'.format(self.flux_on_earth.unit.to_string('latex_inline')))
        plt.xlim(left=self.bounds[0]/u.MeV,right=937)
        plt.grid(True,which="both",ls="-")
        plt.legend()
        plt.title(r'Antistar spectrum $(10 M_\odot, 30 pc)$')
    
    
    def gen_cells(self):
        
        key_to_latex = {'proton_flux':r'$\Phi_p$',
                        'radius':r'$R$',
                        'mass':r'$M$',
                        'velocity':r'$v$',
                        'effective_surface':r'$A$',
                        'gamma_luminosity':r'$L_\gamma$',
                        'earth_distance':r'$d$',
                        'max_magnitude':'Mag',
                        'density':r'$n$',
                        'temperature':r'$T$',
                        'fraction':r'$f$',
                        'length':r'$L$'}
        
        type_to_formula = {'Galaxy Cluster':r'$L_\gamma \simeq 3.10^{-14}\frac{f}{T_8^{1/2}}\int n^2 dV$',
                           'Milky Way Antistar':r'$L_\gamma \simeq \frac{1}{3} 10^{36} \left(\frac{M}{M_{\odot}}\right)^2\frac{1}{v_6^3}$',
                           'Solar Antiobject':r'$L_\gamma \simeq \frac{1}{2}N_\gamma \Phi_p 4\pi R^2$',
                           'Custom' : r'\textit{Custom formula}'}
        cells = []
        
        if self.name is not None:
            cells.append([r'\textbf{Name}',self.name])
        
        cells.append([r'\textbf{Construction Type}',self.type])
        
        for key,q in self.constructing_properties.items():
            
            if isinstance(q,u.Quantity):
                cells.append([key_to_latex[key],'{:.2e} {}'.format(q.value,q.unit.to_string('latex_inline'))])
            else:
                cells.append([key_to_latex[key],'{:.2e}'.format(q)])
        cells.append([r'\textbf{Computed Data}',type_to_formula[self.type]])
    
        for key,q in self.computed_properties.items():
            if key != "max_magnitude":
                cells.append([key_to_latex[key],'{:.2e} {}'.format(q.value,q.unit.to_string('latex_inline'))])
            else :
                cells.append([key_to_latex[key],'{:.2} {}'.format(q.value,q.unit.to_string('latex_inline'))])
                
        return cells