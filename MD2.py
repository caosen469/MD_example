from numba.np.ufunc import parallel
import numpy as np
import random
import math
import numba
from numba import jit
import os
import time
#%%
start = time.time()

def main():
    a = np.loadtxt('liquid256.txt')

    N=len(a) 
    m=1
    totalT = 3
    dt=0.02              
    ts = int(totalT / dt) + 1
    t_range = np.arange(0,totalT+dt,dt)

    position=np.zeros((N,3))
    velocity=np.zeros((N,3))
    mom = np.zeros((ts,3))
    force=np.zeros((N,3))
    K = np.ones(ts)*0
    U = np.ones(ts)*0
    T = np.ones(ts)*0
    P = np.ones(ts)*0
    avgT = np.ones(ts)*0

#%%

    r_cutoff=2.5
    F_cutoff= - (48 / r_cutoff**13 - 24  / r_cutoff**7)
    U_cutoff= 4 * (1 / r_cutoff**12 - 1 / r_cutoff**6)

    L=6.8
    V=L**3

    kB=1.38*math.pow(10,-23)
    epsilon=1.66*math.pow(10,-21)
    sigma=3.4*math.pow(10,-10)

    avgT=cal_avgT(T)

    pend=P[ts-1]
    print(pend)

    H=K+U
    timeavg_K=np.sum(K,axis=0)/ts
    timeavg_U=np.sum(U,axis=0)/ts
    timeavg_T=np.sum(T,axis=0)/ts *epsilon/kB
    timeavg_P=np.sum(P,axis=0)/ts *epsilon/sigma**3
    print(str(timeavg_K)+' '+ str(timeavg_U)+' '+ str(timeavg_T)+' '+ str(timeavg_P))  

    # with open('C:/vscode/python/VMD.xyz','w') as f:
    #    f.write(total_string)

#%%
    for t in range(0,ts):
        if t==0 :
            position = a
            velocity = ini_v(velocity,N)       
            
            U[t] = get_U(position, r_cutoff, U_cutoff, F_cutoff) 
            K[t] = get_K(velocity,m)
            T[t] = get_T(K[t],N)
            P[t] = get_pressure(position,N,T[t],r_cutoff, F_cutoff, V)
            
            mom[t] = momentum(velocity)
            # total_string = write_positions(position)
            
            

        else:
            force = cal_force(position, r_cutoff, F_cutoff)
            velocity = cal_velocity(force,velocity,m,dt) 
            position = cal_position(velocity,position,dt,L) 
            force = cal_force(position,r_cutoff,F_cutoff)
            velocity = cal_velocity(force,velocity,m,dt)         
            
            U[t] = get_U(position, r_cutoff, U_cutoff, F_cutoff) 
            K[t] = get_K(velocity,m)
            T[t] = get_T(K[t],N)
            P[t] = get_pressure(position,N,T[t],r_cutoff, F_cutoff, V)
            
            mom[t] = momentum(velocity)
            # total_string += write_positions(position)
#%%

    import matplotlib.pyplot as plt
    plt.plot(t_range, K, label='Kinetic Energy')
    plt.plot(t_range, U, label='Potential Energy')
    plt.plot(t_range, H, label='Total Energy')
    plt.xlabel('Time step (no units)')
    plt.ylabel('Energy (no units)')
    plt.title('Energy variation with time')
    plt.legend()
    plt.savefig('MD_plot.png',dpi=300)
    plt.cla()

    plt.plot(t_range,T*epsilon/kB)
    plt.xlabel('Time step (no units)')
    plt.ylabel('Temperature (no units)')
    plt.title('Temperature variation with time')
    plt.savefig('temperature.png',dpi=300)
    plt.cla()

    plt.plot(t_range,avgT*epsilon/kB)
    plt.xlabel('Time step (no units)')
    plt.ylabel('Average Temperature (no units)')
    plt.title('Average temperature variation with time')
    plt.savefig('avgtemperature.png',dpi=300)
    plt.cla()

    plt.plot(t_range,P*epsilon/sigma**3)
    plt.xlabel('Time step (no units)')
    plt.ylabel('Pressure (no units)')
    plt.title('Pressure variation with time')
    plt.savefig('pressure.png',dpi=300)
    plt.cla()

    plt.plot(t_range,mom)
    plt.legend(['X','Y','Z'])
    plt.xlabel('Time step (no units)')
    plt.ylabel('Momentum (no units)')
    plt.title('Momentum variation with time')
    plt.savefig('momentum.png',dpi=300)
    end = time.time()

# calculate the distance from the coordinate

@numba.njit(parallel=True)
def norm(r):
    rf=0
    for i in numba.prange(3):
        rf = rf + r[i]*r[i]
    return math.sqrt(rf)

@numba.njit(parallel=True)
def ini_v(v,N):
    for i in range(N):
        for j in range(3):
            v[i][j]=random.uniform(-1,1)
    sum_v=v.sum(axis=0)
    aver_v=sum_v/N
    for i in range(N):
        v[i]=(v[i]-aver_v)*1.78
    return v

@numba.njit(parallel=True)
def pbc(r,L=6.8):
    for k in numba.prange(3):
        if r[k]<-L/2:
            r[k] = r[k] + L
        elif r[k]>L/2:
            r[k] = r[k] - L
        else:
            r[k] = r[k]
    return r

# calculat the lennard jones potential from 
@numba.njit
def LJ(r, U_cutoff,r_cutoff, F_cutoff):
    r6=r**6
    return 4 * (1 / r6**2 - 1 / r6) -U_cutoff - (r - r_cutoff)*F_cutoff

@numba.njit(parallel=True)
def get_U(r, r_cutoff, U_cutoff, F_cutoff):
    u=0
    rf=0
    for i in numba.prange(0,len(r)-1):
        for j in range(i+1,len(r)):
            ri = r[i] - r[j]
            ri = pbc(ri)
            rf = norm(ri) 
            if rf<r_cutoff:
                u = u + LJ(rf, U_cutoff, r_cutoff, F_cutoff)
            else:
                u = u + 0
    return u

@numba.njit(parallel=True)
def get_K(v,m):
    k=0
    for i in numba.prange(0,len(v)):
        a=v[i]
        rf = norm(a)
        k=k + 0.5 * m * rf * rf
    return k

@numba.njit
def get_T(K,N):
    insT = 2 * K / (3 * (N-1))
    return insT

@numba.njit(parallel=True)
def get_pressure(r,N,insT, r_cutoff, F_cutoff, V):
    sumrf=0.0
    for i in numba.prange(N-1):
        for j in range(i+1,N):
            delta_r = r[i] - r[j]
            pbc(delta_r)
            rf = norm(delta_r)
            if rf<r_cutoff:
                Fi = (48 * delta_r / rf**14 - 24 * delta_r / rf**8) + F_cutoff*delta_r/rf
            else:
                Fi = delta_r*0
    
            for k in range(3):
                sumrf=sumrf+delta_r[k]*Fi[k]
    insP=N*insT/V + sumrf/(3*V)
    return insP

@numba.njit(parallel=True)
def cal_force(r, r_cutoff, F_cutoff):
    f=np.zeros((len(r),3))
    for i in numba.prange(0,len(r)):
        for j in range(0,len(r)):
            if i!=j:
                delta_r = r[i] - r[j]
                delta_r = pbc(delta_r)
                rf = norm(delta_r)
                if rf<r_cutoff:
                    f[i] = f[i]  + (48 * delta_r / rf**14 - 24 * delta_r / rf**8) + F_cutoff*delta_r/rf
                else:
                    f[i] += 0
    return f

@numba.njit(parallel=True)
def cal_velocity(force,velocity, m=1, dt=1):
    velocity = velocity + ((force / m) * (dt / 2 ))
    return velocity

@numba.njit(parallel=True)
def cal_position(velocity,position,dt=1,L=6.8):
    rn = position + velocity * dt
    N = velocity.shape[0]
    for i in numba.prange(N):
        for j in range(3):
            if rn[i][j]<0:
                rn[i][j]=rn[i][j]+L
            elif rn[i][j]>L:
                rn[i][j]=rn[i][j]-L
            else:
                rn[i][j]=rn[i][j]
    return rn

@numba.njit
def momentum(velocity):
    
    return np.sum(velocity,axis=0)


def write_positions(r):
    N = r.shape[0]
    curr_string = f'{N}'+'\nMD\n'
    for i in range(N):
        curr_string = curr_string + 'Particle '
        for j in range(3):
            curr_string = curr_string + str(r[i][j]) + ' '
        curr_string = curr_string + '\n'
    return curr_string




@numba.njit(parallel=True)        
def cal_avgT(T, ts = 0):
    countT=0
    avg=np.ones(ts)*0
    for i in numba.prange(ts):
        countT+= T[i]
        avg[i]=countT/(i+1)
    return avg

main()
