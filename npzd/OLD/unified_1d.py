"""
This will integrate a 1-D NPZD model foreward in time.  The integrations
is done using the backward-implicit method from ROMS.

This framework is designed to work for both the Fennel and Banas models.
"""

import numpy as np
import matplotlib.pyplot as plt
from lo_tools import plotting_functions as pfun
import fennel_functions as uf
from importlib import reload
reload(uf)

# set this flag to True to use the Banas model
banas = True

# z-coordinates (bottom to top, positive up)
H = 30 # max depth [m]
N = 30 # number of vertical grid cells
Dz = H/N
z_w = np.arange(-H,Dz,Dz)
z_rho = z_w[:-1] + Dz/2

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
V['E'] = np.nan * np.ones((Ntp, N))
    
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
        # also save PAR profile
        V['E'][Itp,:] = uf.get_E(swrad0, z_rho, z_w, v['Chl'], salt, banas=banas)
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
    
    # In all the processes below we organize the backward-implicit integration
    # around the variable that is being taken from (e.g. NO3 for phytoplankton growth).
    # Hence we always write the "cff" term as: dt * rate factor * variable being taken from.
    # This can sometimes be confusing because it is not how the terms are grouped
    # when the equations are presented in the papers, but it works great for the numerics!
    
    # phytoplankgon growth
    mu_max = uf.get_mu_max(temp, banas=banas)
    E = uf.get_E(swrad0, z_rho, z_w, v['Chl'], salt, banas=banas)
    f = uf.get_f(E, mu_max, banas=banas)
    if banas:
        K3min = 1 / uf.K_NO3_nb
        K3 = K3min + 2*np.sqrt(K3min * v['NO3'])
        cff3 = dt * mu_max * f * (v['Phy'] / (K3 + v['NO3']))
        v['NO3'] = v['NO3'] / (1 + cff3)
        v['Phy'] = v['Phy'] + cff3 * v['NO3']
    else:
        K3 = 1 / uf.K_NO3 # note that these are given as inverse in the dot-in
        K4 = 1 / uf.K_NH4
        cff3 = dt * mu_max * f * (v['Phy'] / (K3 + v['NO3'])) * (K4 / (K4 + v['NH4']))
        cff4 = dt * mu_max * f * (v['Phy'] / (K4 + v['NH4']))
        v['NO3'] = v['NO3'] / (1 + cff3)
        v['NH4'] = v['NH4'] / (1 + cff4)
        v['Phy'] = v['Phy'] + cff3 * v['NO3'] + cff4 * v['NH4']
    
    # chlorophyll growth
    if banas:
        v['Chl'] = 2.5 * v['Phy']
    else:
        mu3 = mu_max * f * (v['NO3'] / (K3 + v['NO3'])) * (K4 / (K4 + v['NH4']))
        mu4 = mu_max * f * (v['NH4'] / (K4 + v['NH4']))
        mu = mu3 + mu4
        Chl2Phy = v['Chl'] / v['Phy']
        rho_Chl = uf.get_rho_Chl(mu, v['Phy'], E, v['Chl'])
        v['Chl'] = v['Chl'] + rho_Chl * Chl2Phy * (cff3 * v['NO3'] + cff4 * v['NH4'])
    
    # grazing by zooplankton
    Ing = uf.get_Ing(v['Phy'], v['Zoo'], banas=banas)
    cff = dt * Ing
    v['Phy'] = v['Phy'] / (1 + cff)
    if banas:
        v['Zoo'] = v['Zoo'] + uf.ZooAE_N_nb * cff * v['Phy']
        v['SDet'] = v['SDet'] + uf.ZooEg_N_nb * (1 - uf.ZooAE_N_nb) * cff * v['Phy']
        v['NO3'] = v['NO3'] + (1 - uf.ZooEg_N_nb) * (1 - uf.ZooAE_N_nb) * cff * v['Phy']
    else:
        v['Chl'] = v['Chl'] / (1 + cff)
        v['Zoo'] = v['Zoo'] + uf.ZooAE_N * cff * v['Phy']
        v['SDet'] = v['SDet'] + (1 - uf.ZooAE_N) * cff * v['Phy']
    
    # zooplankton metabolism
    if banas:
        pass
    else:
        Metab = uf.get_Metab(v['Phy'])
        cff = dt * Metab
        v['Zoo'] = v['Zoo'] / (1 + cff)
        v['NH4'] = v['NH4'] + cff * v['Zoo']
    
    # phytoplankton mortality
    if banas:
        cff = dt * uf.PhyMR_nb
        v['Phy'] = v['Phy'] / (1 + cff)
        v['SDet'] = v['SDet'] + cff * v['Phy']
    else:
        cff = dt * uf.PhyMR
        v['Phy'] = v['Phy'] / (1 + cff)
        v['Chl'] = v['Chl'] / (1 + cff)
        v['SDet'] = v['SDet'] + cff * v['Phy']
    
    # zooplankton mortality
    if banas:
        cff = dt * uf.ZooMR_nb * v['Zoo']
    else:
        cff = dt * uf.ZooMR * v['Zoo']
    v['Zoo'] = v['Zoo'] / (1 + cff)
    v['SDet'] = v['SDet'] + cff * v['Zoo']
    
    # coagulation
    Coag = uf.get_Coag(v['Phy'], v['SDet'], banas=banas)
    if banas:
        cffS = dt * Coag * v['SDet']
        v['SDet'] = v['SDet'] / (1 + cffS)
        v['LDet'] = v['LDet'] + cffS * v['SDet']
    else:
        cffP = dt * Coag * v['Phy']
        v['Phy'] = v['Phy'] / (1 + cffP)
        v['Chl'] = v['Chl'] / (1 + cffP)
        cffS = dt * Coag * v['SDet']
        v['SDet'] = v['SDet'] / (1 + cffS)
        v['LDet'] = v['LDet'] + cffP * v['Phy'] + cffS * v['SDet']
    
    # remineralization
    if banas:
        cffS = dt * uf.SDeRRN_nb
        v['SDet'] = v['SDet'] / (1 + cffS)
        v['NO3'] = v['NO3'] + cffS * v['SDet']
        cffL = dt * uf.LDeRRN_nb
        v['LDet'] = v['LDet'] / (1 + cffL)
        v['NO3'] = v['NO3'] + cffL * v['LDet']
    else:
        cffS = dt * uf.SDeRRN
        v['SDet'] = v['SDet'] / (1 + cffS)
        v['NH4'] = v['NH4'] + cffS * v['SDet']
        cffL = dt * uf.LDeRRN
        v['LDet'] = v['LDet'] / (1 + cffL)
        v['NH4'] = v['NH4'] + cffL * v['LDet']
    
    # nitrification
    if banas:
        pass
    else:
        Nitri = uf.get_Nitri(E)
        cff = dt * Nitri
        v['NH4'] = v['NH4'] / (1 + cff)
        v['NO3'] = v['NO3'] + cff * v['NH4']
    
    # sinking
    max_denitrification = 0
    if banas:
        Wsink_dict = {'SDet':uf.wSDet_nb, 'LDet':uf.wLDet_nb}
        for vn in Wsink_dict.keys():
            C = v[vn].copy()
            Wsink = Wsink_dict[vn]
            h = Wsink * dt
            nn = int(np.floor(h / Dz))
            delta = h - nn * Dz
            Next = nn + 2
            NN = N + Next
            Cext = np.concatenate((C, np.zeros(Next)))
            Cnew = Cext[nn:nn+N]*(Dz - delta)/Dz + Cext[nn+1:nn+N+1]*(delta/Dz)
            Cnet_old = Dz * np.sum(C)
            Cnet_new = Dz * np.sum(Cnew)
            Cnet_lost = Cnet_old - Cnet_new
            v[vn] = Cnew
            if vn == 'Chl':
                pass
            else:
                max_denitrification += Cnet_lost / Dz
        # bottom boundary layer
        # (i) instant remineralization of all sinking particles
        v['NO3'][0] += max_denitrification
        # (ii) some benthic loss
        chi_nb = 1.2 # loss of nitrate to sediments [mmol NO3 m-2 d-1]
        denitrification = np.min((dt*chi_nb/Dz, max_denitrification))
        v['NO3'][0] -= denitrification
        denitrified += Dz * denitrification
    else:
        Wsink_dict = {'Phy':uf.wPhy, 'Chl':uf.wPhy, 'SDet':uf.wSDet, 'LDet':uf.wLDet}
        for vn in Wsink_dict.keys():
            C = v[vn].copy()
            Wsink = Wsink_dict[vn]
            h = Wsink * dt
            nn = int(np.floor(h / Dz))
            delta = h - nn * Dz
            Next = nn + 2
            NN = N + Next
            Cext = np.concatenate((C, np.zeros(Next)))
            Cnew = Cext[nn:nn+N]*(Dz - delta)/Dz + Cext[nn+1:nn+N+1]*(delta/Dz)
            Cnet_old = Dz * np.sum(C)
            Cnet_new = Dz * np.sum(Cnew)
            Cnet_lost = Cnet_old - Cnet_new
            v[vn] = Cnew
            if vn == 'Chl':
                pass
            else:
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
    
    
