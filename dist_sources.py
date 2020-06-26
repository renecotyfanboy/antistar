# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
from sympy import init_printing
import astropy.units as u
from tqdm import tqdm
from matplotlib import rc
from scipy.integrate import trapz
from data_filter import get_sources
from antiobjects.pion_decay import spectrum
from astropy.coordinates import SkyCoord,Distance
from astroquery.gaia import Gaia
# activate latex text rendering
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)
init_printing(use_latex=True,forecolor="White")

final_sources, excluded_sources  = get_sources(TS_max=9)

#%% Constantes

def calc_dist(source,mass=1*u.M_sun):

    velocity=100*(u.km/u.s)
    gamma_luminosity = 0.5*(1/3)*1e36*(mass/u.M_sun)**2*(velocity/(10*u.km/u.s))**(-3)/(u.s)
    E_span = np.linspace(100,1000,10000)
    I_spectrum = spectrum(E_span)
    ref_flux = trapz(I_spectrum*E_span,E_span)
    K_flux = source['Energy_Flux100']*624151/ref_flux*(u.cm**(-2)*u.s**(-1))
    return np.sqrt(4*np.pi*gamma_luminosity/K_flux).to(u.pc)

d = np.zeros((len(final_sources)))*u.pc

for _,source in tqdm(enumerate(final_sources)):
    d[_] = calc_dist(source,mass=0.3*u.M_sun)

stellar_density =  0.14*(u.pc**(-3))    
fraction = max(1/(4/3*np.pi*(d**3)*stellar_density)) 
#%% GAIA

d_gamm = []
d_gaia = []
m_gaia = []

for source in tqdm(final_sources):
    
    #final_sources[2]['Source_Name']
    coord = SkyCoord(l=source['GLON'], b=source['GLAT'], unit=(u.degree, u.degree), frame='galactic')
    radius = u.Quantity(source['Conf_95_SemiMajor'], u.deg)
    j = Gaia.cone_search_async(coord, radius)
    r = j.get_results()
    mask = r['parallax'].mask
    dist = r['dist'][~mask]
    lum = r['lum_val'][~mask]
    parallax_len = np.zeros_like(dist)*u.pc
    parallax_len = Distance(parallax=r['parallax'][~mask].filled().data*u.mas,allow_negative=True)
    min_index = np.nanargmin(np.abs(parallax_len))
    
    m_gaia.append(lum[min_index]**(1/3.5)*u.M_sun)
    d_gaia.append(parallax_len[min_index])
    d_gamm.append(calc_dist(source,mass = m_gaia[-1]))
    
