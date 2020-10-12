# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:58:00 2020

@author: sihan
"""
import numpy as np
import numba


@numba.njit
def write_kinetic_energy(v):
    """
    Calculates and collects the current total kinetic energy to an array
    Input: v: Nx3 nd.array of velocity vector in XYZ directions
    Ouput: Appends current kinetic energy to a running list
    """
    # global K
    # K = np.append(K,0.5* np.sum(v**2))
    K_now = 0.5* np.sum(v**2)
    return K_now

#%%
@numba.njit
def LJ(r):
    """ Returns LJ energy in non-dimensionalized terms
    Inputs: r : float of representing distance between two particles
    """
    return 4*(1/r**12 - 1/r**6)
#%%

@numba.njit
def nrg(x, sum=0):
    """
    Returns forces for a provided position vector 
    Input: X : Nx3 nd.array of cartersian coordinates
    Output: Energy of total system 
    """

     # Total potential energy remains a scalar 
    for i in range(len(x)-1):
        for j in range(i+1,len(x)):
            # Iterate over two distinct particles at a time

            # Difference in the X,Y,Z coordinates
            diff = x[i] - x[j]
            distance = np.sqrt(diff[0]**2+diff[1]**2+diff[2]**2)
            # Add to total potential, current particle pair's energy contribution
            sum+=LJ(distance)

    return sum
#%%
@numba.njit
def get_forces(x, force):
    # force = np.zeros((x.shape[0],x.shape[1]))
    for i in range(len(x)-1):
        for j in range(i+1,len(x)):
        # Iterate over two distinct particles at a time

        # Difference in the X,Y,Z coordinates
            diff = x[i] - x[j]
            # diff = pbc(diff)
            r = np.sqrt(diff[0]**2+diff[1]**2+diff[2]**2)
            
            # 
            # force_one = 48/(r**(13))-24**(r**(7))
            # force[i,0]+=force_one*diff[0]/r
            # force[i,1]+=force_one*diff[1]/r
            # force[i,2]+=force_one*diff[2]/r
            force[i]=force[i]+(48*diff / r**14 -24*diff/r**8)
            
    return force
#%%

def main():
    filename = '10.txt'
    curr_pos = np.loadtxt(filename)
    
    write_kinetic_energy(curr_pos)
    LJ(1)
    ene = nrg(curr_pos)
    
    force = np.zeros((curr_pos.shape[0],curr_pos.shape[1]))
    force = get_forces(curr_pos, force)
    return force,ene
force, ene = main()