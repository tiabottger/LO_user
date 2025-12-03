"""
This will integrate a 1-D NPZD model foreward in time.  The integrations
is done using the backward-implicit method from ROMS.

This framework is designed to work for both the Fennel and Banas models.
"""

import numpy as np
import matplotlib.pyplot as plt
from lo_tools import plotting_functions as pfun
import npzd_equations
from importlib import reload
reload(npzd_equations)

# set the model to use: 'banas', 'fennel', etc.
modname = 'mix0'

# z-coordinates (bottom to top, positive up)
H = 30 # max depth [m]
N = 30 # number of vertical grid cells
Dz = H/N
z_w = np.arange(-H,Dz,Dz)
z_rho = z_w[:-1] + Dz/2
Z = {'Dz': Dz, 'N': N, 'z_rho': z_rho, 'z_w': z_w}

# time
tmax = 20 # max time [days]
dt = 0.01 # time step [days]

# number of time steps
nt = int(np.round(tmax/dt))
# number of time steps between saves of profiles
ntp = int(np.round((tmax/10)/dt))
Ntp = int(np.round((nt/ntp))) + 1 # total number of saved profiles
# number of time steps between saves of net amounts (reservoirs)
ntr = int(np.round((tmax/100)/dt))
Ntr = int(np.round((nt/ntr))) + 1 # total number of saved net amounts

# initialize dict of output arrays
vn_list = ['Phy', 'Chl', 'Zoo', 'SDet', 'LDet', 'NO3', 'NH4']
Omat = np.nan * np.ones((Ntp, N))
V = dict()
for vn in vn_list:
    V[vn] = Omat.copy()
    
# initialize dict of reservoir time series
vnr_list = ['Phy', 'Zoo', 'SDet', 'LDet', 'NO3', 'NH4', 'Lost']
R = dict()
for vn in vnr_list:
    R[vn] = np.nan * np.ones(Ntr)
    
# intial conditions, all [mmol N m-3], except Chl which is [mg Chl m-3]
v = dict()
v['Phy'] = 0.01 * np.ones(N)
v['Chl'] = 2.5 * v['Phy'].copy()
v['Zoo'] = 0.1 * v['Phy'].copy()
v['SDet'] = 0 * np.ones(N)
v['LDet'] = 0 * np.ones(N)
v['NO3'] = 20 * np.ones(N)
v['NH4'] = 0 * np.ones(N)

temp = 10 * np.ones(N) # potential temperature [deg C] vs. z
salt = 32 * np.ones(N) # salinity [psu] vs. z
swrad0 = 500 # surface downward shortwave radiation [W m-3]
Env = {'temp': temp, 'salt': salt, 'swrad0': swrad0}

denitrified = 0
TRvec = []
it = 0
itp = ntp
Itp = 0
itr = ntr
Itr = 0
while it <= nt:
    
    # save output vectors if it is time
    if itp == ntp:
        print('t = %0.2f days' % (it*dt))
        for vn in vn_list:
            V[vn][Itp,:] = v[vn]
        # report on global conservation
        net_N = 0
        for vn in vn_list:
            if vn == 'Chl':
                pass
            else:
                net_N += np.sum(Dz * v[vn])
        net_N += denitrified
        print(' mean N = %0.7f [mmol N m-3]' % (net_N/H))
        itp = 0
        Itp += 1
    # save reservoir output if it is time
    if itr == ntr:
        TRvec.append(it*dt)
        for vn in vnr_list:
            if vn == 'Lost':
                R[vn][Itr] = denitrified
            else:
                R[vn][Itr] = np.sum(Dz * v[vn])
        itr = 0
        Itr += 1
    
    # integrate forward one time step
    v, denitrified = npzd_equations.update_v(v, denitrified, modname, dt, Z, Env)
        
    it += 1
    itp += 1
    itr += 1
        
# plotting
#plt.close('all')
pfun.start_plot(fs=8, figsize=(16,6))

if True:
    # Vertical profiles
    fig, axes = plt.subplots(nrows=1, ncols=len(V.keys()), squeeze=False)
    ii = 0
    for vn in V.keys():
        ax = axes[0,ii]
        vv = V[vn]
        for tt in range(Ntp):
            ax.plot(vv[tt,:], z_rho, lw=(tt+1)/4)
            if vn == 'NO3':
                ax.set_xlim(0, 25)
            if ii == 0:
                ax.set_ylabel('Z [m]')
            if ii > 0:
                ax.set_yticklabels([])
        ax.set_title(vn)
        ii += 1
    fig.suptitle(modname)

if True:
    # time series of integrals
    fig = plt.figure(figsize=(16,6))
    ax = fig.add_subplot(211)
    for vn in vnr_list:
        if vn == 'NO3':
            pass
        else:
            ax.plot(TRvec, R[vn], label=vn, lw=2)
    ax.legend()
    ax.grid(True)
    ax.set_title(modname)
    #ax.set_xlabel('Days')
    ax.set_ylabel('Net N [mmol N m-2]')
    ax = fig.add_subplot(212)
    for vn in ['NO3']:
        ax.plot(TRvec, R[vn], label=vn, lw=2)
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('Days')
    ax.set_ylabel('Net N [mmol N m-2]')
    
plt.show()
pfun.end_plot()
    
    
