# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:35:11 2020

@author: simd9
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from matplotlib import rc
from data_filter import get_sources

# activate latex text rendering
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)

final_sources, excluded_sources  = get_sources(TS_max=9)

glon_span = []
glat_span = []

for _ in range(len(final_sources)):
    
    source = final_sources[_]
    glon_span.append(source['GLON'])
    glat_span.append(source['GLAT'])

glon_span = np.array(glon_span)
glon_span[glon_span>180] -= 360
glon_span *= -np.pi/180
glat_span = np.array(glat_span)*np.pi/180

axes_coords = [0, 0, 1, 1]

fig = plt.figure(figsize=(5,4))

ax_image = fig.add_axes(axes_coords, label="ax image")
img = plt.imread('fermi_sky_map.png',format='float')
ax_image.imshow(img)
ax_image.axis('off')

ax_aitoff = fig.add_axes(axes_coords, projection='aitoff')
ax_aitoff.grid(True)
ax_aitoff.patch.set_alpha(0.)
ax_aitoff.set_xticklabels([])
ax_aitoff.set_yticklabels([])
ax_aitoff.scatter(glon_span,glat_span,color='white',marker='*',s=80)

plt.title(r'4FGL-DR2 significance $< 3 \sigma$ over 1 GeV'
          +'\n'+r'[Fermi-LAT sky 5 years ($E >$ 1 GeV), Galactic coordinates]')

#plt.close()
plt.savefig('custom_skymap_4FGLDR2.png', dpi=600,transparent=True)