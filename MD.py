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
def nrg(x, cutoff=2.5,L=6.8, Sum=0):
    """
    Returns forces for a provided position vector 
    Input: X : Nx3 nd.array of cartersian coordinates
    Output: Energy of total system 
    """

     # Total potential energy remains a scalar 
    for i in range(0, len(x)-1):
        for j in range(i,len(x)):
            # Iterate over two distinct particles at a time
            if i!= j:
                Sum+=0
                # Difference in the X,Y,Z coordinates
                diff = x[i] - x[j]
                diff = periodic_boundary(diff)
                distance = np.sqrt(diff[0]**2+diff[1]**2+diff[2]**2)
                # if (distance < cutoff):
                # # Add to total potential, current particle pair's energy contribution
                Sum+=LJ(distance)+4*(1/cutoff**12-1/cutoff**6)
                # else:
                    # Sum += 0

    return Sum

@numba.njit
def periodic_boundary(x, L=6.8):
    for i in range(3):
        if x[i] < L/2:
            x[i] = x[i] + L
        elif x[i] > L/2:
            x[i] += x[i] - L
        else:
            x[i] = x[i]
    return x
@numba.njit
def get_forces(x, force,cutoff,L):
    # force = np.zeros((x.shape[0],x.shape[1]))
    for i in range(0,len(x)):
        for j in range(0,len(x)):
        # Iterate over two distinct particles at a time
            if i != j:
            # Difference in the X,Y,Z coordinates
                diff = x[i] - x[j]
                # diff = pbc(diff)
                diff= periodic_boundary(diff,L)
                r = np.sqrt(diff[0]**2+diff[1]**2+diff[2]**2)
                # if r < cutoff:
                    
                # 
               
                force[i]=force[i]+(48*diff / r**14 -24*diff/r**8)-(48/cutoff**13-24/cutoff**7)*diff/r
                # else:
                    # force[i] = force[i] + 0
                
            
    return force
@numba.njit
def pressure_calculator(x, force, temperature=100, L = 6.8, cutoff=2.5):
    V = L**3
    KB = 1
    P = len(x)*KB*temperature/V
    
    for i in range(0, len(x)-1):
        for j in range(i+1, len(x)):
            # calculate rij
            diff = x[i] - x[j]
                # diff = pbc(diff)
            diff= periodic_boundary(diff,L)
            r = np.sqrt(diff[0]**2+diff[1]**2+diff[2]**2)
            # if r < cutoff:
            P = P + 1/(3*V)*r*force[i,j]
            # else:
            #     P  = P + 0
    return P

@numba.njit
def Temperature_calculate(K_eng, x):
    KB=1
    T = (2*K_eng)/(3*(len(x)-1)*KB)
    return T
# load the initial position. Replace '10.txt' with a similar file to redo calculations for a different system
# @numba.njit
def main():
    cutoff = 2.5
    filename = '10.txt'
    curr_pos = np2.loadtxt(filename)
    K = np.array([])
    V = np.array([])
    L = 6.8
    E = np.array([])
    Temperature = np.array([])
    Pressure = np.array([])
    # This function gets the elementwise gradient of total potential energy
    # with respect to each cartesian coordinate 
    
    # i.e. dU/dx1, dU/dy1, dU/dz1, 
    #	   dU/dx2, dU/dy2, dU/dz2
    #       ......  
    #      dU/xn, dU/dyn, dU/dzn
    
    # Initialize velocity to zero taking size from input file
    # curr_v = np.zeros(shape=(len(curr_pos),3))
    
    #Initialize velocity in random
    curr_v = np.random.randn(len(curr_pos),3)
    curr_v = curr_v - np.mean(curr_v)
    
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
        #%%
        # Potential energy
        energy = nrg(curr_pos)
        # print(energy)
        E = np.append(E,energy)
        
        # Kinetic Energy
        
        #%%
        force = np.zeros((len(curr_pos),3))
        xlr8 = get_forces(curr_pos, force, cutoff,L) / mass 
        
        v_temp = curr_v + 0.5 * xlr8 * t_int
        x_new = curr_pos + v_temp * t_int
        
        force = np.zeros((len(curr_pos),3))
        new_xlr8 = get_forces(x_new,force,cutoff,L) / mass
    
        # Update with new values
        curr_v = v_temp + 0.5 * new_xlr8 * t_int
        curr_pos = x_new
        
        K_eng = write_kinetic_energy(curr_v)
        print(curr_v[0,0])
        T = Temperature_calculate(K_eng, curr_pos)
        P = pressure_calculator(curr_pos, force, T)
        
        Pressure = np.append(Pressure, P)
        Temperature = np.append(Temperature, T)
    
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
    plt.cla()
    
    plt.plot(t_range,E)
    plt.legend('E')
    plt.xlabel('Time step (no units)')
    plt.ylabel('E (no units)')
    plt.title('E variation with time')
    plt.savefig('E.png',dpi=300)
    
    plt.plot(t_range,Temperature)
    plt.legend('Temperature')
    plt.xlabel('Time step (no units)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature variation with time')
    plt.savefig('Temperature.png',dpi=300)
    
    plt.plot(t_range,Pressure)
    plt.legend('Pressure')
    plt.xlabel('Time step (Pa)')
    plt.ylabel('Pressure (K)')
    plt.title('Pressure variation with time')
    plt.savefig('Pressure.png',dpi=300)
main()