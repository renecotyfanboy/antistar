# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:33:44 2020

@author: simd9
"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits
from matplotlib.colors import LogNorm
from scipy.integrate import trapz
from astropy.wcs import WCS
from matplotlib import rc
from antiobjects.pion_decay import spectrum
from reproject import reproject_interp,reproject_exact
from matplotlib.ticker import LogFormatter 
from matplotlib import ticker

# activate latex text rendering
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)

K_map = fits.getdata('K_map.fits')*u.cm**(-2)*u.s**(-1)
wcs = WCS(fits.getheader('K_map.fits'),naxis=2)
E_span = np.linspace(50,1000,10000)
I_spectrum = spectrum(E_span)
E_flux = trapz(I_spectrum*E_span,E_span)*u.MeV

integral_map = np.array((K_map*E_flux)/(u.erg*u.cm**(-2)*u.s**(-1)))

integral_map = reproject_exact((integral_map,wcs),
                                wcs,shape_out=(1441,2881),
                                return_footprint=False)

ustr = (u.erg*u.cm**(-2)*u.s**(-1)).to_string(format="latex_inline")
plt.figure(figsize=(5,3.5))
plt.axis('off')
plt.title(r'Integrated flux threshold [{}]'.format(ustr)+'\n'+r'MAMA source spectrum')
plt.imshow(integral_map[85:1441-85,:],
           origin='lower',
           cmap=cm.inferno,
           norm=LogNorm(vmin=1e-7, vmax=1e-5))

formatter = LogFormatter(10, labelOnlyBase=False) 
cb = plt.colorbar(orientation='horizontal')
#cb.ax.minorticks_on()
minorticks = [(1+i)*1e-6 for i in range(5)]
cb.ax.xaxis.set_ticks(minorticks, minor=True)

plt.savefig('sensitivity.png', dpi=600,transparent=True)