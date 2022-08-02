#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 15:24:36 2022

@author: hirshb
"""

#imports
import numpy as np
import pandas as pd
from sim import Simulation

"""
HERE, TO RUN THE SIMULATION, YOU WILL NEED TO DO THE FOLLOWING THINGS:
    
    1. CREATE AN OBJECT OF THE SIMULATION CLASS. A MINIMAL EXAMPLE IS
        >>> mysim = Simulation( dt=0.1E-15, L=11.3E-10, ftype="LJ" )
    
    2. DEFINE THE PARAMETERS FOR THE POTENTIAL. USE A DICTIONARY, FOR EXAMPLE
    FOR THE LJ MODEL OF ARGON, IN SI UNITS:
        >>> params = { "eps":  1.656778224E-21, "sig": 3.4E-10 }
    
THEN, CALLING THE METHODS YOU IMPLEMENTED IN sim.py, YOU NEED TO
    3. READ THE INITIAL XYZ FILE PROVIDED OR SAMPLE INITIAL COORDINATES.
    4. SAMPLE INITIAL MOMENTA FROM MB DISTRIBUTION.
    5. REMOVE COM MOTION.
    6. RUN THE SIMULATION, INCLUDING PRINTING XYZ AND ENERGIES TO FILES.

THE SPECIFIC SIMULATIONS YOU NEED TO RUN, AND THE QUESTIONS YOU NEED TO ANSWER,
ARE DEFINED IN THE COURSE NOTES.
    
NOTE THAT TO CALL A METHOD OF A CLASS FOR THE OBJECT mysim, THE SYNTAX IS
    >>> mysim.funcName( args )
    
FINALLY, YOU SHOULD
    7. ANALYZE YOUR RESULTS. THE SPECIFIC GRAPHS TO PLOT ARE EXPLAINED IN THE
    COURSE NOTES.

NOTE: THE INPUT FILE GIVEN HAS THE COORDINATES IN ANGSTROM.
*YOUR OUTPUT XYZ FILE SHOULD BE PRINTED IN ANGSTROM TOO*, BUT YOU CAN USE
ANY UNITS YOU WANT IN BETWEEN, I SUGGEST USING SI UNITS.

"""
# For part A
R = np.array([[-1.05e-10, 0.0, 0.0]])
mass = np.array([6.6335209e-26])
dt = 1.0e-15

# CREATE AN OBJECT OF THE SIMULATION CLASS.
mysim = Simulation(dt=dt, L=5e-10, Nsteps=5000, R=R, mass=mass, kind=["Ar"], ftype="DoubleWell", printfreq=10,
                 NG=10000000)

# DEFINE THE PARAMETERS FOR THE POTENTIAL.
params_A = { "omega": 7.596337239980638e13 } 
params_B = { "eps":  1.656778224E-21, "sig": 3.4E-10 }
params_Final = { "A": 4.11e20 , "B": 8.22 }
# READ THE INITIAL XYZ FILE PROVIDED OR SAMPLE INITIAL COORDINATES.
# mysim.readXYZ('Ar_init.xyz')
# Set the mass array and scale R to angstrom
# mysim.mass = np.array([6.6335209e-26] * mysim.R.shape[0])
# mysim.R = mysim.R * 1e-10
# SAMPLE INITIAL MOMENTA FROM MB DISTRIBUTION.
# mysim.sampleMB(temp=298.0)
# # REMOVE COM MOTION.
# mysim.removeRCM()
# RUN THE SIMULATION, INCLUDING PRINTING XYZ AND ENERGIES TO FILES.
mysim.runMeta(**params_Final)
mysim.__del__()