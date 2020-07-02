# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:03:17 2020

@author: simd9
"""

import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)

x = da.from_zarr('bigsim.zarr')

glon = x[0,:]
glat= x[1,:]
m = x[2,:]
r = x[3,:]

# plt.subplot(212)
# fig = plt.gcf()
# ax_image = fig.add_axes([0, 0, 1, 1], label="ax image")
# img = plt.imread('fermi_sky_map.png',format='float')
# ax_image.imshow(img)
# ax_image.axis('off')

# ax_aitoff = fig.add_axes([0, 0, 1, 1], projection='aitoff')
# ax_aitoff.grid(True)
# ax_aitoff.patch.set_alpha(0.)
# ax_aitoff.set_xticklabels([])
# ax_aitoff.set_yticklabels([])
# ax_aitoff.scatter(glon.compute()*np.pi/180,glat.compute()*np.pi/180,color='white',marker='*',s=80)

plt.subplot(221)
plt.hist(m.compute(),bins=np.arange(0,2,0.01),density=True)
plt.xlabel(r'$M (M_{\odot})$')
plt.ylabel(r'Density')
plt.subplot(222)
plt.hist(r.compute(),bins=np.arange(0,300,0.1),density=True)
#plt.plot(np.arange(0,300,0.1),(np.arange(0,300,0.1)*0.208/300)**3)
plt.xlabel(r'r (pc)')

plt.subplot(223)
plt.hist(glon.compute(),bins=np.arange(0,360,0.01),density=True)
plt.xlabel(r'Galactic Longitude (deg)')
plt.ylabel(r'Density')
plt.subplot(224)
plt.hist(glat.compute(),bins=np.arange(-90,90,0.01),density=True)
plt.xlabel(r'Galactic Latitude (deg)')