"""
Parameters and functions for the Banas-Daivd-Siedlecki NPZD model.

Coming soon: O and maybe even C.

Units of NPZD are all [uM N] = [mmol N m-3]

I will use the same names for state variables as in the Fennel model,
to facilitate reuse of the 1-D model code (e.g. Phy = phytoplankton)
"""
import numpy as np

# parameters from Davis et al. (2014)

mu_0 = 1.7 # max instantaneous growth rate [d-1]
att_P = 0.03 # light attenuation by phytoplankton [(uM N)-1 m-1]
alpha = 0.07 # initial slope of growth-light curve [(W m-2)-1 d-1]
k_s = 0.1 # min half-saturation for nutrient uptake [dimensionless]
m = 0.1 # non-grazing phytoplankton mortality [d-1]
Chl2N = 2.5 # chlorophyll to nitrogen ratio [mg Chl (mmol N)-1]
Ing_0 = 4.8 # max ingestion rate [d-1]
xi = 2 # zooplankton mortality [(uM N)-1 d-1]
K_s = 3 # half saturation for ingestion [uM N]
epsilon = 0.3 # gross growth efficiency of zooplankton [dimensionless]
f_egest = 0.5 # fraction of losses egested [dimensionless]
r = 0.1 # remineralization rate [d-1]
w_L = 80 # sinking rate for large detritus [m d-1]
w_S = 8 # sinking rate for small detritus [m d-1]
chi = 1.2 # loss of nitrate to sediments [mmol NO3 m-2 d-1]
tau = 0.05 # detrital coagulation rate [(um N)-1 d-1]

def get_att_sw(S):
    """
    This parameter is used in the light calculation.  The result is
    that fresh water has more attenuation.
    
    Input:
    S = Salinity [psu or whatever]
    
    Output:
    att_sw = light attenuation by seawater [m-1]
    """
    att_sw = 0.05 - (0.0065 * (S - 32))
    return att_sw
    
def get_E(swrad0, z_rho, z_w, Phy, S):
    """
    Profile of photosynthetically available radiation vs. z
    NOTE: we assume z_rho, z_w, and Phy are packed bottom-to-top
    
    Input:
    swrad0 = incoming swrad at surface [W m-2], which is converted to:
        E_surface = PAR at surface = 0.43*swrad at surface [W m-2]
    z_rho = vertical positions of cell centers, positive up, 0 at surface [m]
    z_w = vertical positions of cell boundaries, positive up, 0 at surface [m]
    Phy = phytoplankton at cell centers [uM N]
    S = Salinity [psu or whatever]
    
    Output:
    E = photosynthetically available radiation at cell centers [W m-2]
    """
    dz = np.diff(z_w)
    N = len(Phy)
    mean_P = np.zeros(N)
    for ii in range(N):
        this_dz = dz[ii:]
        this_dz[0] *= 0.5
        this_P = Phy[ii:]
        mean_P[ii] = np.sum(this_dz * this_P) / np.sum(this_dz)
    att_sw = get_att_sw(S)
    E_surface = swrad0 * 0.43
    E = E_surface * np.exp( z_rho * (att_sw + att_P*mean_P))
    return E
    
def get_f(E):
    """
    The photosynthesis-light relationship, Evans and Parslow (1985).
    
    Input:
    E = photosynthetically available radiation [W m-2]
    
    Output:
    f = the P-I curve [dimensionless]
    """
    f = alpha * E / np.sqrt(mu_0**2 + (alpha * E)**2)
    return f
    
def get_L(NO3):
    """
    Fundamentally this gives the nutrient concentration that is part
    of the phytoplankton growth rate, but it involves the
    Michaelis-Menten functions for nutrient limitation.

    For zero nutrients L = 0.  As the nutrients become large
    compared to their half-saturations L goes to 1.
    And of course L = 1/2 when the nutrient is at its half-saturation.

    Input:
    NO3 = nitrate concentration [uM N]

    Output:
    L [dimensionless]
    """
    k_s_app = k_s + 2*np.sqrt(k_s * NO3)
    L = NO3 / (k_s_app + NO3)
    return L
    
def get_Ing(Phy):
    """
    Hollings-type s-shaped grazing curve.
    
    Input:
    Phy = phytoplankton [mmol N m-3]
    
    Output:
    I = grazing rate [d-1] "Ingestion"
    """
    Ing = Ing_0 * (Phy**2 / (K_s**2 + Phy**2))
    return Ing
    
    
