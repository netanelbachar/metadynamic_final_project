#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Feb  6 11:55:59 2022

@author: hirshb


WELCOME TO YOUR FIRST PROJECT! THIS BIT OF TEXT IS CALLED A DOCSTRING.
BELOW, I HAVE CREATED A CLASS CALLED "SIMULATION" FOR YOUR CONVENIENCE.
I HAVE ALSO IMPLEMENTED A CONSTRUCTOR, WHICH IS A METHOD THAT IS CALLED 
EVERY TIME YOU CREATE AN OBJECT OF THE CLASS USING, FOR EXAMPLE, 
    
    >>> mysim = Simulation( dt=0.1E-15, L=11.3E-10, ftype="LJ" )

I HAVE ALSO IMPLEMENTED SEVERAL USEFUL METHODS THAT YOU CAN CALL AND USE, 
BUT DO NOT EDIT THEM. THEY ARE: evalForce, dumpXYZ, dumpThermo and readXYZ.

YOU DO NOT NEED TO EDIT THE CLASS ITSELF. 

YOUR JOB IS TO IMPLEMENT THE LIST OF CLASS METHODS DEFINED BELOW WHERE YOU 
WILL SEE THE FOLLOWING TEXT: 

        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################

YOU ARE, HOWEVER, EXPECTED TO UNDERSTAND WHAT ARE THE MEMBERS OF THE CLASS
AND USE THEM IN YOUR IMPLEMENTATION. THEY ARE ALL EXPLAINED IN THE 
DOCSTRING OF THE CONSTRUCTOR BELOW. FOR EXAMPLE, WHENEVER YOU WISH 
TO USE/UPDATE THE MOMENTA OF THE PARTICLES IN ONE OF YOUR METHODS, YOU CAN
ACCESS IT BY USING self.p. 

    >>> self.p = np.zeros( (self.Natoms,3) )
        
FINALLY, YOU WILL NEED TO EDIT THE run.py FILE WHICH RUNS THE SIMULATION.
SEE MORE INSTRUCTIONS THERE.

"""
################################################################
################## NO EDITING BELOW THIS LINE ##################
################################################################

#imports
import numpy as np
import pandas as pd
from scipy.constants import Boltzmann as BOLTZMANN
import matplotlib.pyplot as plt

class Simulation:
    
    def __init__( self, dt, L, Nsteps=0, R=None, mass=None, kind=None, \
                 p=None, F=None, U=None, K=None, seed=937142, ftype=None, \
                 deltaMC=0.5e-10, temp=298, NG = 300, \
                 step=0, printfreq=1000, xyzname="sim.xyz", fac=1.0, \
                 outname="sim.log", debug=False ):
        """
        THIS IS THE CONSTRUCTOR. SEE DETAILED DESCRIPTION OF DATA MEMBERS
        BELOW. THE DESCRIPTION OF EACH METHOD IS GIVEN IN ITS DOCSTRING.

        Parameters
        ----------
        dt : float
            Simulation time step.
            
        L : float
            Simulation box side length.
            
        Nsteps : int, optional
            Number of steps to take. The default is 0.
            
        R : numpy.ndarray, optional
            Particles' positions, Natoms x 3 array. The default is None.
            
        mass : numpy.ndarray, optional
            Particles' masses, Natoms x 1 array. The default is None.
            
        kind : list of str, optional
            Natoms x 1 list with atom type for printing. The default is None.
            
        p : numpy.ndarray, optional
            Particles' momenta, Natoms x 3 array. The default is None.
            
        F : numpy.ndarray, optional
            Particles' forces, Natoms x 3 array. The default is None.
            
        U : float, optional
            Potential energy . The default is None.
            
        K : numpy.ndarray, optional
            Kinetic energy. The default is None.
            
        seed : int, optional
            Big number for reproducible random numbers. The default is 937142.
            
        ftype : str, optional
            String to call the force evaluation method. The default is None.
            
        step : INT, optional
            Current simulation step. The default is 0.
            
        printfreq : int, optional
            PRINT EVERY printfreq TIME STEPS. The default is 1000.
            
        xyzname : TYPE, optional
            DESCRIPTION. The default is "sim.xyz".
            
        fac : float, optional
            Factor to multiply the positions for printing. The default is 1.0.
            
        outname : TYPE, optional
            DESCRIPTION. The default is "sim.log".
            
        debug : bool, optional
            Controls printing for debugging. The default is False.

        Returns
        -------
        None.

        """
        
        #general        
        self.debug=debug 
        self.printfreq = printfreq 
        self.xyzfile = open( xyzname, 'w' ) 
        self.outfile = open( outname, 'w' ) 
        
        #simulation
        self.Nsteps = Nsteps 
        self.dt = dt 
        self.L = L 
        self.seed = seed 
        self.step = step         
        self.fac = fac
        self.NG = NG # Number of steps for Gaussian addition
        
        self.temp = temp
        self.accepted = 0
        self.deltaMC = deltaMC
        self.particle_i = None
        
        #system        
        if R is not None:
            self.R = R        
            self.mass = mass
            self.kind = kind
            self.Natoms = self.R.shape[0]
            self.mass_matrix = np.array( [self.mass,]*3 ).transpose()
        else:
            self.R = np.zeros( (1,3) )
            self.mass = 1.6735575E-27 #H mass in kg as default
            self.kind = ["H"]
            self.Natoms = self.R.shape[0]
            self.mass_matrix = np.array( [self.mass,]*3 ).transpose()
            
        if p is not None:
            self.p = p
            self.K = K
        else:
            self.p = np.zeros( (self.Natoms,3) )
            self.K = 0.0
        
        if F is not None:
            self.F = F
            self.U = U
        else:
            self.F = np.zeros( (self.Natoms,3) )
            self.U = 0.0
               
        #set RNG seed
        np.random.seed( self.seed )
        
        #check force type
        if ( ftype == "LJ" or ftype == "Harm" or ftype == "Anharm" or ftype == "LJMC" or ftype == "DoubleWell"):
            self.ftype = "eval" + ftype
        else:
            raise ValueError("Wrong ftype value - use LJ or Harm or Anharm.")
    
    def __del__( self ):
        """
        THIS IS THE DESCTRUCTOR. NOT USUALLY NEEDED IN PYTHON. 
        JUST HERE TO CLOSE THE FILES.

        Returns
        -------
        None.

        """
        self.xyzfile.close()
        self.outfile.close()
    
    def evalForce( self, **kwargs ):
        """
        THIS FUNCTION CALLS THE FORCE EVALUATION METHOD, BASED ON THE VALUE
        OF FTYPE, AND PASSES ALL OF THE ARGUMENTS (WHATEVER THEY ARE).

        Returns
        -------
        None. Calls the correct method based on self.ftype.

        """
        
        getattr(self, self.ftype)(**kwargs)
            
    def dumpThermo( self ):
        """
        THIS FUNCTION DUMPS THE ENERGY OF THE SYSTEM TO FILE.

        Parameters
        ----------
        R : numpy.ndarray
            Natoms x 3 array with particle coordinates.
        step : int
            Simulation time step.
        file : _io.TextIOWrapper 
            File object to write to. Created using open() in main.py

        Returns
        -------
        None.

        """
        if( self.step == 0 ):
            self.outfile.write( "step K U E \n" )
        
        self.outfile.write( str(self.step) + " " \
                          + "{:.6e}".format(self.K) + " " \
                          + "{:.6e}".format(self.U) + " " \
                          + "{:.6e}".format(self.E) + "\n" )
                
    def dumpXYZ( self ):
        """
        THIS FUNCTION DUMP THE COORDINATES OF THE SYSTEM IN XYZ FORMAT TO FILE.

        Parameters
        ----------
        R : numpy.ndarray
            Natoms x 3 array with particle coordinates.
        step : int
            Simulation time step.
        file : _io.TextIOWrapper 
            File object to write to. Created using open() in main.py

        Returns
        -------
        None.

        """
            
        self.xyzfile.write( str( self.Natoms ) + "\n")
        self.xyzfile.write( "Step " + str( self.step ) + "\n" )
        
        for i in range( self.Natoms ):
            self.xyzfile.write( self.kind[i] + " " + \
                              "{:.6e}".format( self.R[i,0]*self.fac ) + " " + \
                              "{:.6e}".format( self.R[i,1]*self.fac ) + " " + \
                              "{:.6e}".format( self.R[i,2]*self.fac ) + "\n" )
    
    def readXYZ( self, inpname ):
        """
        THIS FUNCTION READS THE INITIAL COORDINATES IN XYZ FORMAT.

        Parameters
        ----------
        R : numpy.ndarray
            Natoms x 3 array with particle coordinates.
        step : int
            Simulation time step.
        file : _io.TextIOWrapper 
            File object to write to. Created using open() in main.py

        Returns
        -------
        None.

        """
           
        df = pd.read_csv( inpname, sep="\s+", skiprows=2, header=None )
        
        self.kind = df[ 0 ]
        self.R = df[ [1,2,3] ].to_numpy()
        self.Natoms = self.R.shape[0]
        
################################################################
################## NO EDITING ABOVE THIS LINE ##################
################################################################
    
    
    def sampleMB( self, temp, removeCM=True ):
        """
        THIS FUNCTIONS SAMPLES INITIAL MOMENTA FROM THE MB DISTRIBUTION.
        IT ALSO REMOVES THE COM MOMENTA, IF REQUESTED.

        Parameters
        ----------
        temp : float
            The temperature to sample from.
        removeCM : bool, optional
            Remove COM velocity or not. The default is True.

        Returns
        -------
        None. Sets the value of self.p.

        """
        
        sigma = (self.mass*BOLTZMANN*temp)**.5
        re_sigma = sigma.reshape(sigma.shape[0], 1)
        self.p = np.random.normal(loc=0, scale=re_sigma, size=(sigma.shape[0],3))
        if removeCM:
            pm = self.p * self.mass.reshape((self.Natoms, 1))
            self.p -= np.sum(pm, axis=0) / (np.sum(self.mass))
        
    def applyPBC( self ):
        """
        THIS FUNCTION APPLIES PERIODIC BOUNDARY CONDITIONS.

        Returns
        -------
        None. Sets the value of self.R.

        """
        
        # "Return" particles if they are out of bounds
        self.R[self.R > self.L/2] -= self.L
        self.R[self.R <= -self.L/2] += self.L
                    
    def removeRCM( self ):
        """
        THIS FUNCTION ZEROES THE CENTERS OF MASS POSITION VECTOR.

        Returns
        -------
        None. Sets the value of self.R.

        """    
        
        # Calculate CoM vector
        massed_pos = self.R * self.mass.reshape((self.Natoms, 1))
        cm_vec = np.sum(massed_pos, axis=0) / (np.sum(self.mass))
        # Update the positions by CoM vector 
        self.R -= cm_vec
             
    def evalLJ( self, eps, sig ):
        """
        THIS FUNCTION EVALUTES THE LENNARD-JONES POTENTIAL AND FORCE.

        Parameters
        ----------
        eps : float
            epsilon LJ parameter.
        sig : float
            sigma LJ parameter.

        Returns
        -------
        None. Sets the value of self.F and self.U.

        """
        
        if( self.debug ):
            print( "Called evalLJ with eps = " \
                  + str(eps) + ", sig= " + str(sig)  )
        
        # self.F = np.zeros((self.Natoms, 3))
        # self.U = 0
        # for i, Ri in enumerate(self.R):
        #     r = self.R[i] - self.R # array of distance vectors from Ri particle
        #     # Apply minimal image convention
        #     r[r >= self.L/2] -= self.L
        #     r[r < -self.L/2] += self.L
        #     # Potential calculation part
        #     dist = (np.sum((r)**2, axis=1))**0.5 # array of distances
        #     dist = dist.reshape((len(dist), 1))
        #     sig_dist = np.divide(sig, dist, out=np.zeros_like(self.R), where= dist!=0.) # to take care of divide by zero 
        #     eps_dist = np.divide(eps, dist, out=np.zeros_like(self.R), where= dist!=0.) # to take care of divide by zero
        #     v = 4 * eps * ( (sig_dist)**12 - (sig_dist)**6 ) # array of potentials
        #     self.U += np.sum(v) # final overall potential
        #     # Force calculation part
        #     r_hat = np.divide(self.R, dist, out=np.zeros_like(self.R), where= dist!=0.)
        #     f = 4 * (eps_dist) * ( 12*(sig_dist)**12 - 6*(sig_dist)**6 ) * r_hat
        #     self.F[i] = np.sum(f, axis=0)
            
        self.F = np.zeros((self.Natoms, 3))
        self.U = 0
        for i in range(self.Natoms-1):
            r = self.R[i] - self.R[i+1:] # array of distance vectors from particle R[i] to other particles
            # Apply minimal image convention
            r[r >= self.L/2] -= self.L
            r[r < -self.L/2] += self.L
            # Potential calculation part
            dist = (np.sum((r)**2, axis=1))**0.5 # array of distances
            dist = dist.reshape((len(dist), 1))
            sig_dist = np.divide(sig, dist, out=np.zeros_like(r), where= dist!=0.) # to take care of divide by zero
            eps_dist = np.divide(eps, dist, out=np.zeros_like(r), where= dist!=0.) # to take care of divide by zero
            v = 4 * eps * ( (sig_dist)**12 - (sig_dist)**6 ) # array of potentials
            self.U += np.sum(v) # final overall potential
            # Force calculation part
            r_hat = np.divide(r, dist, out=np.zeros_like(r), where= dist!=0.) #self.R[i+1:] / dist # array of normals
            f = 4 * (eps_dist) * ( 12*(sig_dist)**12 - 6*(sig_dist)**6 ) * r_hat # array of forces between point R[i] to other points
            self.F[i+1:self.Natoms] -= f # update the other particles force with the current particle i
            self.F[i] += sum(f) # update current particle i force with all the other particles
                        
    def evalHarm( self, omega ):
        """
        THIS FUNCTION EVALUATES THE POTENTIAL AND FORCE FOR A HARMONIC TRAP.

        Parameters
        ----------
        omega : float
            The frequency of the trap.

        Returns
        -------
        None. Sets the value of self.F and self.U.

        """

        if( self.debug ):
            print( "Called evalHarm with omega = " + str(omega) )
        
        v_part = 0.5 * omega**2 * self.R**2 * self.mass.reshape((self.Natoms, 1))
        self.U = np.sum(v_part) # numpy.sum() sums along first axis
        self.F = -omega**2 * self.R * self.mass.reshape((self.Natoms, 1))

    def evalAnharm( self, Lambda ):
        """
        THIS FUNCTION EVALUATES THE POTENTIAL AND FORCE FOR AN ANHARMONIC TRAP.

        Parameters
        ----------
        Lambda : float
            The parameter of the trap U = 0.25 * Lambda * x**4

        Returns
        -------
        None. Sets the value of self.F and self.U.

        """
    
        if( self.debug ):
            print( "Called evalAnharm with Lambda = " + str(Lambda) ) 
            
        v_part = 0.25 * Lambda * self.R**4
        self.U = np.sum(v_part)
        # self.F = # ????
        
    def evalLJMC( self, eps, sig ):
        """
        THIS FUNCTION EVALUTES THE LENNARD-JONES POTENTIAL.

        Parameters
        ----------
        eps : float
            epsilon LJ parameter.
        sig : float
            sigma LJ parameter.

        Returns
        -------
        None. Sets the value of self.U.

        """
        
        if( self.debug ):
            print( "Called evalLJMC with eps = " \
                  + str(eps) + ", sig= " + str(sig)  )
                
        self.F = np.zeros((self.Natoms, 3))
        self.U = 0.0
        if self.particle_i is not None:
            i = self.particle_i
            r = self.R[i] - np.append(self.R[:i], self.R[i+1:], axis=0) # array of distance vectors from particle R[i] to every other particle
            # Apply minimal image convention
            r[r >= self.L/2] -= self.L
            r[r < -self.L/2] += self.L
            # Potential calculation part
            dist = (np.sum((r)**2, axis=1))**0.5 # array of distances
            dist = dist.reshape((len(dist), 1))
            sig_dist = np.divide(sig, dist, out=np.zeros_like(r), where= dist!=0.) # to take care of divide by zero
            v = 4 * eps * ( (sig_dist)**12 - (sig_dist)**6 ) # array of potentials
            self.U += np.sum(v) # final overall potential
            
    def evalDoubleWell( self, A, B ):
        """
        THIS FUNCTION EVALUATES THE POTENTIAL AND FORCE FOR A DOUBLE WELL SURFACE.

        Parameters
        ----------
        A : float
            
        B : float
            

        Returns
        -------
        None. Sets the value of self.U.

        """

        if( self.debug ):
            print( "Called evalDoubleWell with A = " \
                  + str(A) + ", B= " + str(B) )
        
        v_part = A*self.R**4 - B*self.R**2
        self.U = np.sum(v_part) # numpy.sum() sums along first axis
        self.F = -4*A*self.R**3 + 2*B*self.R
        
    def CalcKinE( self ):
        """
        THIS FUNCTIONS EVALUATES THE KINETIC ENERGY OF THE SYSTEM.

        Returns
        -------
        None. Sets the value of self.K.

        """
        
        p_part = self.p**2
        re_mass = self.mass.reshape(self.Natoms, 1)
        self.K = np.sum(p_part / (2*re_mass))
        
    def VVstep( self, **kwargs ):
        """
        THIS FUNCTIONS PERFORMS ONE VELOCITY VERLET STEP.

        Returns
        -------
        None. Sets self.R, self.p.
        """
        
        # First, the momenta are propagated
        self.p += 0.5*self.F*self.dt
        # Then, the positions are updated
        self.R += (self.p / self.mass.reshape(self.Natoms, 1))*self.dt
        self.applyPBC()
        # The new forces are evaluated using the new positions
        self.evalForce(**kwargs)
        # Velocity is then propagated using the new forces
        self.p += 0.5*self.F*self.dt
        
    def MCstep( self, **kwargs ):
        """
        THIS FUNCTIONS PERFORMS ONE METROPOLIS MC STEP IN THE NVT ENSEMBLE.
        YOU WILL NEED TO PROPOSE TRANSLATION MOVES, APPLY  
        PBC, CALCULATE THE CHANGE IN POTENTIAL ENERGY, ACCEPT OR REJECT, 
        AND CALCULATE THE ACCEPTANCE PROBABILITY.

        Returns
        -------
        None. Sets self.R.
        """
        # Save positions and potential energy
        R_old = np.copy(self.R)
        U_old = self.U
        # Pick a random particle from all the particles and move it randomly
        self.particle_i = np.random.randint(low=0, high=len(self.R))
        self.evalForce(**kwargs)
        u_particle = self.U # the potential energy of that particle before it was moved
        self.R[self.particle_i] += np.random.uniform(low=-1.0, high=1.0, size=3) * self.deltaMC
        self.applyPBC()
        # Evaluate and decide if we accept the change
        self.evalForce(**kwargs)
        u_new_particle = self.U # the potential energy of that particle after it was moved
        self.U = U_old + (u_new_particle - u_particle) # the potential energy resulting from moving only that particle
        beta = 1 / (BOLTZMANN*self.temp)
        prob = min((1, np.exp(-beta * (self.U - U_old)) ))
        xsi = np.random.uniform()
        
        if prob < xsi:
            self.R = np.copy(R_old)
            self.U = U_old
        else:
            self.accepted += 1
        
    def runMC( self, **kwargs ):
        """ 
        THIS FUNCTION DEFINES AN MC SIMULATION DOES, GIVEN AN INSTANCE OF 
        THE SIMULATION CLASS. YOU WILL NEED TO LOOP OVER MC STEPS, 
        PRINT THE COORDINATES AND ENERGIES EVERY PRINTFREQ TIME STEPS 
        TO THEIR RESPECTIVE FILES, SIMILARLY TO YOUR MD CODE.

        Returns
        -------
        None.
        """
        
        self.evalForce(**kwargs) # calculate the initial potential energy
        while self.step <= self.Nsteps:
            self.E = self.U + self.K                
            if self.step % self.printfreq == 0:
                self.dumpThermo()
                self.dumpXYZ()
            self.MCstep(**kwargs)
            self.step += 1
            
    def runMeta( self, **kwargs ):
         """ 
         MD with metadynamics implementation

         Returns
         -------
         None.
         """
         
         self.evalForce(**kwargs)
         while self.step <= self.Nsteps:
             self.CalcKinE()
             self.E = self.U + self.K
             if self.step % NG == 0:
                  self.metadynamics()               
                  self.evalMeta() # not final
             if self.step % self.printfreq == 0:
                 self.dumpThermo()
                 self.dumpXYZ()
             self.VVstep(**kwargs)
             self.applyPBC()
             self.step += 1
        
    def run( self, **kwargs ):
        """
        THIS FUNCTION DEFINES A SIMULATION DOES, GIVEN AN INSTANCE OF 
        THE SIMULATION CLASS. YOU WILL NEED TO:
            1. EVALUATE THE FORCES (USE evaluateForce() AND PASS A DICTIONARY
                                    WITH ALL THE PARAMETERS).
            2. PROPAGATE FOR NS TIME STEPS USING THE VELOCITY VERLET ALGORITHM.
            3. APPLY PBC.
            4. CALCULATE THE KINETIC, POTENTIAL AND TOTAL ENERGY AT EACH TIME
            STEP. 
            5. YOU WILL ALSO NEED TO PRINT THE COORDINATES AND ENERGIES EVERY 
        PRINTFREQ TIME STEPS TO THEIR RESPECTIVE FILES, xyzfile AND outfile.

        Returns
        -------
        None.

        """      
        
        self.evalForce(**kwargs)
        while self.step <= self.Nsteps:
            self.CalcKinE()
            self.E = self.U + self.K
            if self.step % self.printfreq == 0:
                self.dumpThermo()
                self.dumpXYZ()
            self.VVstep(**kwargs)
            self.applyPBC()
            self.step += 1
