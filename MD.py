#!/usr/bin/env python3

# This code requires a module called autograd which performs for automatic differentiation
# use "pip install autograd" and restart your kernel to see the changes
import numpy as np
import autograd.numpy as np2
from autograd import elementwise_grad
import numba
import matplotlib.pyplot as plt

# -------Write current position to a cummulative string---
# @numba.njit
# def write_positions(pos):
#     """ 
#     Writes current position provided through argument to XYZ file format
#     Input: pos : Nx3 nd.array of xyz coordinates
#     Output: XYZ file format string 
#     """

#     curr_string = f'{len(pos)}'+'\nILoveMD\n'
#     for i in range(len(pos)):
#         curr_string = curr_string + 'Particle '
#         for j in range(3):
#             curr_string = curr_string + str(pos[i][j]) + ' '
#         curr_string = curr_string + '\n'
#     # print(curr_string)
#     return curr_string


# -------Write current kinetic energy to a cummulative list---

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


# -------Write current potential energy to a cummulative list---


# @numba.njit
def write_potential_energy(x):
    """
    Calculates and collects the current total potential energy to an array
    Input: v: Nx3 nd.array of position in XYZ directions
    Ouput: Appends current total potential energy to a running list
    """
    # global V
    # V = np.append(V,nrg(x))
    return nrg(x)

@numba.njit
def LJ(r):
    """ Returns LJ energy in non-dimensionalized terms
    Inputs: r : float of representing distance between two particles
    """
    return 4*(1/r**12 - 1/r**6)

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

@numba.njit
def get_forces(x, force):
    # force = np.zeros((x.shape[0],x.shape[1]))
    for i in range(0,len(x)):
        for j in range(0,len(x)):
        # Iterate over two distinct particles at a time
            if i != j:
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

# load the initial position. Replace '10.txt' with a similar file to redo calculations for a different system
# @numba.njit
def main():
    cutoff = 5
    filename = '10.txt'
    curr_pos = np2.loadtxt(filename)
    K = np.array([])
    V = np.array([])
    
    # This function gets the elementwise gradient of total potential energy
    # with respect to each cartesian coordinate 
    
    # i.e. dU/dx1, dU/dy1, dU/dz1, 
    #	   dU/dx2, dU/dy2, dU/dz2
    #       ......  
    #      dU/xn, dU/dyn, dU/dzn
    
    # Initialize velocity to zero taking size from input file
    curr_v = np.zeros(shape=(len(curr_pos),3))
    
    # simulation time settings
    t_min = 0
    t_max = 1
    t_int = 0.002
    t_range = np.arange(t_min,t_max,t_int)
    mass = 1
    
    momentum = np.array([np.sum(curr_v,axis=0)])
    
    total_string = '' # Initialize a single string to hold all the written positions to be written in a file
    
    for t in t_range: #Iterate over all time range. Does not include the simulation of t_max
    
        # Recording steps before the loops
        K = np.append(K,write_kinetic_energy(curr_v))
        
        V = np.append(V, write_potential_energy(curr_pos))
    #     # total_string = total_string + write_positions(curr_pos)
        if t != t_min:
         	momentum = np.append(momentum, [np.sum(curr_v,axis=0)],axis=0)
    #     # Equation of motions
        # energy = nrg(curr_pos)
        
        force = np.zeros((len(curr_pos),3))
        xlr8 = get_forces(curr_pos, force) / mass 
        
        v_temp = curr_v + 0.5 * xlr8 * t_int
        x_new = curr_pos + v_temp * t_int
        
        force = np.zeros((len(curr_pos),3))
        new_xlr8 = get_forces(x_new,force) / mass
    
        # Update with new values
        curr_v = v_temp + 0.5 * new_xlr8 * t_int
        curr_pos = x_new
    
    # Write the total string to a file
    write_filename = filename.split('.')[0] + '.xyz'
    with open(write_filename,'w') as f:
        f.write(total_string)
    
    # print('V shape is ', V.shape)
    # print()
    # print('K shape is ', K.shape)
    
    
    
    
    plt.plot(t_range, K, label='Kinetic Energy')
    plt.plot(t_range, V, label='Potential Energy')
    plt.plot(t_range, V+K, label='Total Energy')
    plt.xlabel('Time step (no units)')
    plt.ylabel('Energy (no units)')
    plt.title('Energy variation with time')
    plt.legend()
    plt.savefig('MD_plot.png',dpi=300)
    plt.cla()
    
    plt.plot(t_range,momentum)
    plt.legend(['X','Y','Z'])
    plt.xlabel('Time step (no units)')
    plt.ylabel('Momentum (no units)')
    plt.title('Momentum variation with time')
    plt.savefig('momentum.png',dpi=300)
main()