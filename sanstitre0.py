#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:40:34 2020

@author: simon
"""
import numpy as np
from astropy.io import fits
from reproject import reproject_interp
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import matplotlib.cm as cm

k_map = fits.getdata('flux_MAMA.fits')
k_map_old = fits.getdata('old_flux_MAMA.fits')
k_map_old[k_map_old<1e-11] = np.nan
WCS_in = WCS(header=fits.getheader('old_flux_MAMA.fits'),naxis=2)
WCS_out = WCS(header=fits.getheader('flux_MAMA.fits'),naxis=2)
k_map_old_reproject,_ = reproject_interp((k_map_old,WCS_in), WCS_out,shape_out= k_map.shape)

ratio = k_map_old_reproject/k_map

plt.figure(figsize=(5,3.5))
plt.axis('off')
plt.title(r'Ratio old/new')
plt.imshow(ratio,
           origin='lower',
           cmap=cm.inferno)

plt.colorbar(orientation='horizontal')
plt.savefig('ratio.png', dpi=600,transparent=True)