# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 04:05:05 2020

@author: simd9
"""

import astropy.units as u
from antiobject import Antiobject

antijupiter = Antiobject(proton_flux = 2e8*(1/5.2)**2*(u.cm**(-2)*u.s**(-1)),
                         radius = 7e7*u.m,
                         earth_distance = 591e6*u.km,
                         name = 'Antijupiter')

antiasteroid = Antiobject(proton_flux = 2e8*(1/3)**2*(u.cm**(-2)*u.s**(-1)),
                          radius = 2*u.km,
                          earth_distance = 2*u.AU,
                          name = 'Antiasteroid')

antistar = Antiobject(mass = 10*u.M_sun,
                      velocity = 100*(u.km/u.s),
                      earth_distance = 30*u.pc,
                      name = 'Antistar')

antistar.plot()
antijupiter.plot()
antiasteroid.plot()