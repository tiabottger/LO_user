"""
First 1-D Fennel model.  Calculates time-evolution of the 7 state
variables as a function of z, given some initial condition and light.

Like fennel_1.py but with a new treatment of sinking.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from lo_tools import plotting_functions as pfun
import fennel_functions as ff
import shared
from importlib import reload
reload(ff)
reload(shared)

# z-coordinates (bottom to top, positive up)
H = shared.H # max depth [m]
N = shared.N # number of vertical grid cells
Dz = H/N
z_w = np.arange(-H,Dz,Dz)
z_rho = z_w[:-1] + Dz/2

# time
tmax = shared.tmax # max time [days]
dt = .1

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
V = dict()
for vn in vn_list:
    V[vn] = np.nan * np.ones((Ntp, N))
V['E'] = np.nan * np.ones((Ntp, N))
    
# initialize dict of reservoir time series
vnr_list = ['Phy', 'Zoo', 'SDet', 'LDet', 'NO3', 'NH4', 'Lost']
R = dict()
for vn in vnr_list:
    R[vn] = np.nan * np.ones(Ntr)

# environmental conditions
T = 10 * np.ones(N) # temperature [degC] vs. z
swrad0 = 500 # surface swrad [W m-3]

# intial conditions, all [mmol N m-3] except Chl
v = dict()
v['Phy'] = 0.01 * np.ones(N)
v['Chl'] = 2.5 * v['Phy'].copy() # [mg Chl m-3]
v['Zoo'] = 0.1 * v['Phy'].copy()
v['SDet'] = 0 * np.ones(N)
v['LDet'] = 0 * np.ones(N)
v['NO3'] = 20 * np.ones(N)
v['NH4'] = 0 * np.ones(N)

Wsink_dict = {'Phy':ff.w_P, 'Chl':ff.w_P, 'SDet':ff.w_S, 'LDet':ff.w_L}

denitrified = 0
dv = dict() # stores the net change vectors
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
        V['E'][Itp,:] = ff.get_E(swrad0, z_rho, z_w, v['Chl'])
        # report on global conservation
        net_N = 0
        for vn in vn_list:
            if vn == 'Chl':
                pass
            else:
                net_N += np.sum(Dz * v[vn])
        net_N += denitrified
        print(' mean N = %0.3f [mmol N m-3]' % (net_N/H))
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
    
    # Phy: phytoplankton
    # growth
    mu_max = ff.get_mu_max(T)
    E = ff.get_E(swrad0, z_rho, z_w, v['Chl'])
    f = ff.get_f(E, mu_max)
    L_NO3, L_NH4 = ff.get_L(v['NO3'], v['NH4'])
    L = L_NO3 + L_NH4
    mu = mu_max * f * L
    growth_P = mu * v['Phy']
    #  grazing
    Ing = ff.get_Ing(v['Phy'])
    grazing_P = Ing * v['Zoo']
    # mortality
    mortality_P = ff.m_P * v['Phy']
    #  coagulation to large detrius
    coag_P = ff.tau * (v['SDet'] + v['Phy']) * v['Phy']
    # net change
    dv['Phy'] = dt * (growth_P - grazing_P - mortality_P - coag_P)
    
    # Chl: chlorophyll
    # growth
    rho_Chl = ff.rho_Chl(mu, v['Phy'], E, v['Chl'])
    growth_Ch = rho_Chl * mu * v['Chl']
    # grazing
    grazing_Ch = Ing * v['Zoo'] * v['Chl'] / v['Phy']
    #  mortality
    mortality_Ch = ff.m_P * v['Chl']
    # coagulation to large detrius
    coag_Ch = ff.tau * (v['SDet'] + v['Phy']) * v['Chl']
    # net change
    dv['Chl'] = dt * (growth_Ch - grazing_Ch - mortality_Ch - coag_Ch)
    
    # Zoo: zooplankton
    # growth
    growth_Z = Ing * ff.beta * v['Zoo']
    # excretion
    excretion_Z = (ff.l_BM + (ff.l_E * ff.beta * v['Phy']**2 / (ff.k_P + v['Phy']**2))) * v['Zoo']
    # mortality
    mortality_Z = ff.m_Z * v['Zoo']**2
    # net change
    dv['Zoo'] = dt * (growth_Z - excretion_Z - mortality_Z)
    
    # SDet: small detritus
    # sloppy eating
    egestion_S = Ing * (1 - ff.beta) * v['Zoo']
    # coagulation to large detrius
    coag_S = ff.tau * (v['SDet'] + v['Phy']) * v['SDet']
    # remineralization
    remin_S = ff.r_S * v['SDet']
    # net change
    dv['SDet'] = dt * (egestion_S + mortality_P + mortality_Z - coag_S - remin_S)
    
    # LDet: large detritus
    # remineralization
    remin_L = ff.r_L * v['LDet']
    # net change
    dv['LDet'] = dt * (coag_P + coag_S - remin_L)
    
    # NO3: nitrate
    # uptake
    uptake_NO3 = mu_max * f * L_NO3 * v['Phy']
    # nitrification
    n = ff.get_n(E)
    nitrification = n * v['NH4']
    # net_change
    dv['NO3'] = dt * (-uptake_NO3 + nitrification)
    
    # NH4: ammomium
    # uptake
    uptake_NH4 = mu_max * f * L_NH4 * v['Phy']
    # net change
    dv['NH4'] = dt * (-uptake_NH4 - nitrification + excretion_Z + remin_S + remin_L)
    
    # update all variables
    for vn in vn_list:
        v[vn] += dv[vn]
        
    # sinking
    max_denitrification = 0
    for vn in ['Phy', 'Chl', 'SDet', 'LDet']:
        C = v[vn]
        Wsink = Wsink_dict[vn]
        Cnew, Cnet_lost = shared.sink(z_w, z_rho, Dz, N, C, Wsink, dt)
        v[vn] = Cnew
        if vn in ['Phy', 'SDet', 'LDet']:
            max_denitrification += Cnet_lost / Dz
    
    # bottom boundary layer
    denit_fac = 0.25 # fraction of particle flux at bottom that is returned to NH4, 4/16
    v['NH4'][0] += denit_fac * max_denitrification
    
    denitrified += Dz * (1 - denit_fac) * max_denitrification
    
    it += 1
    itp += 1
    itr += 1
        
# plotting
#plt.close('all')
pfun.start_plot(fs=8, figsize=(16,6))

fig, axes = plt.subplots(nrows=1, ncols=len(V.keys()), squeeze=False)
ii = 0
for vn in V.keys():
    ax = axes[0,ii]
    vv = V[vn]
    for tt in range(Ntp):
        ax.plot(vv[tt,:], z_rho, lw=(tt+1)/4)
        if ii == 0:
            ax.set_ylabel('Z [m]')
        if ii > 0:
            ax.set_yticklabels([])
    ax.set_title(vn)
    ii += 1
    
fig = plt.figure(figsize=(16,6))
ax = fig.add_subplot(211)
for vn in vnr_list:
    if vn == 'NO3':
        pass
    else:
        ax.plot(TRvec, R[vn], label=vn, lw=2)
ax.legend()
ax.grid(True)
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
    
    
