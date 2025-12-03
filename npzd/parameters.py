"""
Module of parameters for the Fennel et al. (2006) model,
and the Banas et al. (2009) model.

Parameters are copied and edited for python from bio_Fennel.in

In fennel.h the parameters from bio_Fennel.in appear as, for example, AttSW(ng),
and do not require declaration.  However, note that in bio_Fennel.in they
are defined in this form: AttSW == 0.04d0

Where the functional forms are identical (or close) between Fennel and Banas versions
I also give the Banas parameter as []_nb.

"""

# Light

AttSW = 0.04 # Light attenuation due to seawater [1/m]
AttSW_nb = 0.05
AttFW_nb = 0.0065

AttChl = 0.02486 # Light attenuation by chlorophyll [(mg_Chl m-3)-1 m-1] (fixed typo in units)
AttChl_nb = 0.012 # 0.03/2.5 to get from Att for mean Phy to Att for mean Chl

PARfrac = 0.43 # Fraction of shortwave radiation that is photosynthetically active [nondimensional]

# Phytoplankton

PhyIS = 0.025 # Phytoplankton, initial slope of P-I curve [1/(Watts m-2 day)]
PhyIS_nb = 0.07

K_NO3 = 2.0 # Inverse half-saturation for phytoplankton NO3 uptake [1/(millimole_N m-3)]
K_NO3_nb = 10.0

K_NH4 = 2.0 # Inverse half-saturation for phytoplankton NH4 uptake [1/(millimole_N m-3)]
K_NH4_nb = 10.0

PhyMR = 0.15 # Phytoplankton mortality rate [1/day]
PhyMR_nb = 0.1

PhyMin = 0.001 # Phytoplankton minimum threshold value [millimole_N/m3]
PhyIP = 1.5 # Phytoplankton, NH4 inhibition parameter [1/(millimole_N)]

# Chlorophyll

PhyCN = 6.625 # Phytoplankton Carbon:Nitrogen ratio [mole_C/mole_N]
Chl2C_m = 0.0535 # Maximum chlorophyll to carbon ratio [mg_Chl/mg_C]

# Zooplankton

K_Phy = 2.0 # Zooplankton half-saturation constant (squared) for ingestion [millimole_N m-3]^2
K_Phy_nb = 9.0

ZooGR = 0.6 # Zooplankton maximum growth rate [1/day] (Ingestion)
ZooGR_nb = 4.8

ZooMR = 0.025 # Zooplankton mortality rate [(mmol N m-3)-1 day-1] (fixed typo in units)
ZooMR_nb = 2.0

ZooAE_N = 0.75 # Zooplankton Nitrogen assimilation efficiency [nondimensional]
ZooAE_N_nb = 0.3
ZooEg_N_nb = 0.5 # f_egest [dimensionless]

ZooBM = 0.1 # Zooplankton Basal metabolism [1/day]
ZooER = 0.1 # Zooplankton specific excretion rate [1/day]

# Detritus

CoagR = 0.005 # Coagulation rate: aggregation rate of SDeN + Phy => LDeN [(mmol N m-3)-1 day-1] (fixed typo in units)
CoagR_nb = 0.05 #  just SDeN => LDeN

SDeRRN = 0.03 # Small detritus remineralization rate N-fraction [1/day]
LDeRRN = 0.01 # Large detritus remineralization rate N-fraction [1/day]
SDeRRN_nb = 0.1
LDeRRN_nb = 0.1

wPhy = 0.1 # Vertical sinking velocity for phytoplankton [m/day]
wSDet = 0.1 # Vertical sinking velocity for small detritus [m/day]
wLDet = 1.0 # Vertical sinking velocity for large detritus [m/day]
wSDet_nb = 8.0 # Vertical sinking velocity for small detritus [m/day]
wLDet_nb = 80.0 # Vertical sinking velocity for large detritus [m/day]

# NO3 and NH4 Nitrate and Ammonium (e.g. nitrification)

I_thNH4 = 0.0095 # Radiation threshold for nitrification inhibition [Watts/m2]
D_p5NH4 = 0.1 # Half-saturation radiation for nitrification inhibition [Watts/m2]
NitriR = 0.05 # Nitrification rate: oxidation of NH4 to NO3 [1/day]
