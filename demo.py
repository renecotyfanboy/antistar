# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 04:05:05 2020

@author: simd9
"""

import astropy.units as u
import matplotlib.pyplot as plt
from antiobject import Antiobject

#%% Antiobject construction

antijupiter = Antiobject.antiplanet_factory()

antiasteroid = Antiobject.antiplanet_factory(proton_flux = 2e8*(1/3)**2*(u.cm**(-2)*u.s**(-1)),
                                             radius = 2*u.km,
                                             earth_distance = 2*u.AU,
                                             name = 'Antiasteroid')

antistar = Antiobject.antistar_factory()

#%% Antiobject plotting

antistar.plot()
antijupiter.plot()
antiasteroid.plot()

#plt.savefig("foo.png", dpi=1200)