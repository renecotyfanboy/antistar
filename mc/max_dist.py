# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:45:18 2020

@author: simd9
"""

import numpy as np
import astropy.units as u
from astropy.io import fits

kmap = fits.getdata('K_map.fits')
min_k = np.nanmin(kmap[kmap>1e-10])*(u.cm**(-2)*u.s**(-1))
max_mass = 1*u.M_sun
velocity=100*(u.km/u.s)
L_gamma = 0.5*(1/3)*1e36*(max_mass/u.M_sun)**2*(velocity/(10*u.km/u.s))**(-3)/(u.s)
dist = np.sqrt(4*np.pi*L_gamma/min_k).to(u.pc)