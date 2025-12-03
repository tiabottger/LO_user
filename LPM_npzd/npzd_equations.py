import numpy as np
import parameters as p
from importlib import reload
reload(p)

def get_E(Chl, Z, Env, modname):
    """
    Profile of photosynthetically available radiation vs. z
    NOTE: all inputs except swrad0 must be vectors (z)
    NOTE: we assume z_rho, z_w, and Chl are packed bottom-to-top
    NOTE: corrected Chl integral to use mean, as per notes in ROMS Forum:
    https://www.myroms.org/forum/viewtopic.php?p=2444&hilit=AttChl+units#p2444
    
    Input:
    swrad0 = incoming swrad at surface [W m-2] (called I_0 in Fennel)
    z_rho = vertical positions of cell centers, positive up, 0 at surface [m]
    z_w = vertical positions of cell boundaries, positive up, 0 at surface [m]
    Chl = chlorophyll concentration profile at cell centers [mg Chl m-3]
    salt = salinity [psu]
    
    Output:
    E = photosynthetically available radiation [W m-2]
    """
    dz = np.diff(Z['z_w'])
    N = len(Chl)
    mean_Chl = np.zeros(N)
    for ii in range(N):
        this_dz = dz[ii:]
        this_dz[0] *= 0.5
        this_Chl = Chl[ii:]
        mean_Chl[ii] = np.sum(this_dz * this_Chl) / np.sum(this_dz)
    if modname in ['banas', 'mix0']:
        AttSFW_nb = p.AttSW_nb - p.AttFW_nb * (Env['salt'] - 32)
        E = Env['swrad0'] * p.PARfrac * np.exp( Z['z_rho'] * (AttSFW_nb + p.AttChl_nb*mean_Chl))
    elif modname == 'fennel':
        E = Env['swrad0'] * p.PARfrac * np.exp( Z['z_rho'] * (p.AttSW + p.AttChl*mean_Chl))
    return E
    
def sink(vn, C, max_denitrification, Wsink, dt, Z):
    h = Wsink * dt
    nn = int(np.floor(h / Z['Dz']))
    delta = h - nn * Z['Dz']
    Next = nn + 2
    NN = Z['N'] + Next
    Cext = np.concatenate((C, np.zeros(Next)))
    Cnew = Cext[nn:nn+Z['N']]*(Z['Dz'] - delta)/Z['Dz'] + Cext[nn+1:nn+Z['N']+1]*(delta/Z['Dz'])
    Cnet_old = Z['Dz'] * np.sum(C)
    Cnet_new = Z['Dz'] * np.sum(Cnew)
    Cnet_lost = Cnet_old - Cnet_new
    if vn == 'Chl':
        pass
    else:
        max_denitrification += Cnet_lost / Z['Dz']
    return Cnew, max_denitrification

def update_v(v, denitrified, modname, dt, Z, Env):
    
    # In all the processes below we organize the backward-implicit integration
    # around the variable that is being taken from (e.g. NO3 for phytoplankton growth).
    # Hence we always write the "cff" term as: dt * rate factor * variable being taken from.
    # This can sometimes be confusing because it is not how the terms are grouped
    # when the equations are presented in the papers, but it works great for the numerics!
    
    # phytoplankgon growth
    # light profile
    E = get_E(v['Chl'], Z, Env, modname)
    # light response curve
    if modname in ['banas', 'mix0']:
        mu_max = 1.7 # max growth rate [d-1]
        f = p.PhyIS_nb * E / np.sqrt(mu_max**2 + (p.PhyIS_nb * E)**2)
    elif modname == 'fennel':
        mu_0 = 0.59
        mu_max = mu_0 * 1.066**Env['temp']
        f = p.PhyIS * E / np.sqrt(mu_max**2 + (p.PhyIS * E)**2)
    # nutrient limitation curve and phytoplankton growth
    if modname == 'banas':
        K3min = 1 / p.K_NO3_nb
        K3 = K3min + 2*np.sqrt(K3min * v['NO3'])
        cff3 = dt * mu_max * f * (v['Phy'] / (K3 + v['NO3']))
        v['NO3'] = v['NO3'] / (1 + cff3)
        v['Phy'] = v['Phy'] + cff3 * v['NO3']
    elif modname == 'fennel':
        K3 = 1 / p.K_NO3 # note that these are given as inverse in the dot-in
        K4 = 1 / p.K_NH4
        cff3 = dt * mu_max * f * (v['Phy'] / (K3 + v['NO3'])) * (K4 / (K4 + v['NH4']))
        cff4 = dt * mu_max * f * (v['Phy'] / (K4 + v['NH4']))
        v['NO3'] = v['NO3'] / (1 + cff3)
        v['NH4'] = v['NH4'] / (1 + cff4)
        v['Phy'] = v['Phy'] + cff3 * v['NO3'] + cff4 * v['NH4']
    elif modname == 'mix0':
        K3 = 1 / p.K_NO3_nb
        K4 = 1 / p.K_NH4_nb
        cff3 = dt * mu_max * f * (v['Phy'] / (K3 + v['NO3'])) * (K4 / (K4 + v['NH4']))
        cff4 = dt * mu_max * f * (v['Phy'] / (K4 + v['NH4']))
        v['NO3'] = v['NO3'] / (1 + cff3)
        v['NH4'] = v['NH4'] / (1 + cff4)
        v['Phy'] = v['Phy'] + cff3 * v['NO3'] + cff4 * v['NH4']
    # chlorophyll growth
    if modname in ['banas', 'mix0']:
        v['Chl'] = 2.5 * v['Phy']
    elif modname == 'fennel':
        mu3 = mu_max * f * (v['NO3'] / (K3 + v['NO3'])) * (K4 / (K4 + v['NH4']))
        mu4 = mu_max * f * (v['NH4'] / (K4 + v['NH4']))
        mu = mu3 + mu4
        Chl2Phy = v['Chl'] / v['Phy']
        C_AtWt = 12 # Carbon atomic weight [g C / mol C]
        rho_Chl = p.PhyCN * C_AtWt * p.Chl2C_m * mu * v['Phy'] / (p.PhyIS * E * v['Chl'])
        v['Chl'] = v['Chl'] + rho_Chl * Chl2Phy * (cff3 * v['NO3'] + cff4 * v['NH4'])
    
    # grazing by zooplankton
    if modname == 'banas':
        Ing = p.ZooGR_nb * (v['Phy'] * v['Zoo'] / (p.K_Phy_nb + v['Phy']**2))
        cff = dt * Ing
        v['Phy'] = v['Phy'] / (1 + cff)
        v['Zoo'] = v['Zoo'] + p.ZooAE_N_nb * cff * v['Phy']
        v['SDet'] = v['SDet'] + p.ZooEg_N_nb * (1 - p.ZooAE_N_nb) * cff * v['Phy']
        v['NO3'] = v['NO3'] + (1 - p.ZooEg_N_nb) * (1 - p.ZooAE_N_nb) * cff * v['Phy']
    if modname == 'mix0':
        Ing = p.ZooGR_nb * (v['Phy'] * v['Zoo'] / (p.K_Phy_nb + v['Phy']**2))
        cff = dt * Ing
        v['Phy'] = v['Phy'] / (1 + cff)
        v['Zoo'] = v['Zoo'] + p.ZooAE_N_nb * cff * v['Phy']
        v['SDet'] = v['SDet'] + p.ZooEg_N_nb * (1 - p.ZooAE_N_nb) * cff * v['Phy']
        v['NH4'] = v['NH4'] + (1 - p.ZooEg_N_nb) * (1 - p.ZooAE_N_nb) * cff * v['Phy']
    elif modname == 'fennel':
        Ing = p.ZooGR * (v['Phy'] * v['Zoo'] / (p.K_Phy + v['Phy']**2))
        cff = dt * Ing
        v['Phy'] = v['Phy'] / (1 + cff)
        v['Chl'] = v['Chl'] / (1 + cff)
        v['Zoo'] = v['Zoo'] + p.ZooAE_N * cff * v['Phy']
        v['SDet'] = v['SDet'] + (1 - p.ZooAE_N) * cff * v['Phy']
    
    # zooplankton metabolism
    if modname in ['banas', 'mix0']:
        pass
    elif modname == 'fennel':
        Metab = p.ZooBM + p.ZooER * p.ZooAE_N * v['Phy']**2 / (p.K_Phy + v['Phy']**2)
        cff = dt * Metab
        v['Zoo'] = v['Zoo'] / (1 + cff)
        v['NH4'] = v['NH4'] + cff * v['Zoo']
    
    # phytoplankton mortality
    if modname in ['banas', 'mix0']:
        cff = dt * p.PhyMR_nb
        v['Phy'] = v['Phy'] / (1 + cff)
        v['SDet'] = v['SDet'] + cff * v['Phy']
    elif modname == 'fennel':
        cff = dt * p.PhyMR
        v['Phy'] = v['Phy'] / (1 + cff)
        v['Chl'] = v['Chl'] / (1 + cff)
        v['SDet'] = v['SDet'] + cff * v['Phy']
    
    # zooplankton mortality
    if modname in ['banas', 'mix0']:
        cff = dt * p.ZooMR_nb * v['Zoo']
    elif modname == 'fennel':
        cff = dt * p.ZooMR * v['Zoo']
    v['Zoo'] = v['Zoo'] / (1 + cff)
    v['SDet'] = v['SDet'] + cff * v['Zoo']
    
    # coagulation
    if modname in ['banas', 'mix0']:
        Coag = p.CoagR_nb * v['Phy']
        cffS = dt * Coag * v['SDet']
        v['SDet'] = v['SDet'] / (1 + cffS)
        v['LDet'] = v['LDet'] + cffS * v['SDet']
    elif modname == 'fennel':
        Coag = p.CoagR * (v['Phy'] + v['SDet'])
        cffP = dt * Coag * v['Phy']
        v['Phy'] = v['Phy'] / (1 + cffP)
        v['Chl'] = v['Chl'] / (1 + cffP)
        cffS = dt * Coag * v['SDet']
        v['SDet'] = v['SDet'] / (1 + cffS)
        v['LDet'] = v['LDet'] + cffP * v['Phy'] + cffS * v['SDet']
    
    # remineralization
    if modname == 'banas':
        cffS = dt * p.SDeRRN_nb
        v['SDet'] = v['SDet'] / (1 + cffS)
        v['NO3'] = v['NO3'] + cffS * v['SDet']
        cffL = dt * p.LDeRRN_nb
        v['LDet'] = v['LDet'] / (1 + cffL)
        v['NO3'] = v['NO3'] + cffL * v['LDet']
    elif modname == 'fennel':
        cffS = dt * p.SDeRRN
        v['SDet'] = v['SDet'] / (1 + cffS)
        v['NH4'] = v['NH4'] + cffS * v['SDet']
        cffL = dt * p.LDeRRN
        v['LDet'] = v['LDet'] / (1 + cffL)
        v['NH4'] = v['NH4'] + cffL * v['LDet']
    elif modname == 'mix0':
        cffS = dt * p.SDeRRN_nb
        v['SDet'] = v['SDet'] / (1 + cffS)
        v['NH4'] = v['NH4'] + cffS * v['SDet']
        cffL = dt * p.LDeRRN_nb
        v['LDet'] = v['LDet'] / (1 + cffL)
        v['NH4'] = v['NH4'] + cffL * v['LDet']
    
    # nitrification
    if modname == 'banas':
        pass
    elif modname in ['fennel', 'mix0']:
        Nitri = p.NitriR * (1 - np.maximum(0*E, ((E - p.I_thNH4) / (p.D_p5NH4 + E - p.I_thNH4))))
        cff = dt * Nitri
        v['NH4'] = v['NH4'] / (1 + cff)
        v['NO3'] = v['NO3'] + cff * v['NH4']
    
    # sinking
    max_denitrification = 0
    if modname in ['banas', 'mix0']:
        Wsink_dict = {'SDet':p.wSDet_nb, 'LDet':p.wLDet_nb}
        for vn in Wsink_dict.keys():
            C = v[vn].copy()
            Wsink = Wsink_dict[vn]
            Cnew, max_denitrification = sink(vn, C, max_denitrification, Wsink, dt, Z)
            v[vn] = Cnew
        # bottom boundary layer
        # (i) instant remineralization of all sinking particles
        v['NO3'][0] += max_denitrification
        # (ii) some benthic loss
        chi_nb = 1.2 # loss of nitrate to sediments [mmol NO3 m-2 d-1]
        denitrification = np.min((dt*chi_nb/Z['Dz'], max_denitrification))
        v['NO3'][0] -= denitrification
        denitrified += Z['Dz'] * denitrification
    elif modname == 'fennel':
        Wsink_dict = {'Phy':p.wPhy, 'Chl':p.wPhy, 'SDet':p.wSDet, 'LDet':p.wLDet}
        for vn in Wsink_dict.keys():
            C = v[vn].copy()
            Wsink = Wsink_dict[vn]
            Cnew, max_denitrification = sink(vn, C, max_denitrification, Wsink, dt, Z)
            v[vn] = Cnew
        # bottom boundary layer
        denit_fac = 0.25 # fraction of particle flux at bottom that is returned to NH4, 4/16
        v['NH4'][0] += denit_fac * max_denitrification
        denitrified += Z['Dz'] * (1 - denit_fac) * max_denitrification
        
    return v, denitrified