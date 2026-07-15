"""
This script determines the vertical integral of DO and all 
variables in the NPZD module, and saves them in a .nc file.

This script searches for yearly box extractions in LO_output, for the
region "pugetsoundDO"

It also crops out data from the Straits, so as to not bias the results
in Puget Sound. (optional using flag remove_straits)

.nc files are saved in LO_output/intermodel_comparison
"""

# import things
import numpy as np
import xarray as xr
import csv
import pinfo
from lo_tools import Lfun, zrfun

import sys
from pathlib import Path
pth = Path(__file__).absolute().parent.parent.parent/ 'LO' / 'pgrid'
if str(pth) not in sys.path:
    sys.path.append(str(pth))
import gfun

Gr = gfun.gstart()

Ldir = Lfun.Lstart()

##############################################################
##                       USER INPUTS                        ##
##############################################################

regions = ['pugetsoundDO']

years = ['2013']#['2015','2016','2017','2018','2019','2020']

# which  model run to look at?
# gtagex = 'cas7_t1noDIN_x11ab' # 
gtagexes = ['cas7_t1_x11ab']#['cas7_t1y13v2_x11ab']#,'cas7_t1noDIN_x11ab']

# where to put output files
out_dir = Ldir['LOo'] / 'intermodel_comparison'
Lfun.make_dir(out_dir)

##############################################################
##                    HELPER FUNCTIONS                      ##
##############################################################

def start_ds(ocean_time,eta_rho,xi_rho):
    '''
    Initialize dataset to store processed DO data
    ocean_time = ocean time vector
    eta_rho = eta rho vector
    xi_rho = xi_rho vector
    '''
    Ndays = len(ocean_time.values)
    Neta = len(eta_rho.values)
    Nxi = len(xi_rho.values)

    ds = xr.Dataset(data_vars=dict(
        # DO vertical integral
        DO_vert_int             = (['ocean_time','eta_rho','xi_rho'], np.zeros((Ndays,Neta,Nxi))),
        # phytoplankton vertical integral
        phyto_vert_int          = (['ocean_time','eta_rho','xi_rho'], np.zeros((Ndays,Neta,Nxi))),
        # zooplankton vertical integral
        zoop_vert_int           = (['ocean_time','eta_rho','xi_rho'], np.zeros((Ndays,Neta,Nxi))),
        # NO3 vertical integral
        NO3_vert_int            = (['eta_rho','xi_rho'], np.zeros((Neta,Nxi))),
        # NH4 vertical integral
        NH4_vert_int            = (['ocean_time','eta_rho','xi_rho'], np.zeros((Ndays,Neta,Nxi))),
        # SdetritusN vertical integral
        SdetritusN_vert_int    = (['ocean_time','eta_rho','xi_rho'], np.zeros((Ndays,Neta,Nxi))),
        # LdetritusN vertical integral
        LdetritusN_vert_int     = (['ocean_time','eta_rho','xi_rho'], np.zeros((Ndays,Neta,Nxi))),),
    coords=dict(ocean_time=ocean_time, eta_rho=eta_rho, xi_rho=xi_rho,),)
    
    return ds

def add_metadata(ds):
    '''
    Create metadata for processed DO data
    '''

    # DO vertical integral
    ds['DO_vert_int'].attrs['long_name'] = 'vertical integral of DO'
    ds['DO_vert_int'].attrs['units'] = 'mol/m2'

    # phytoplankton vertical integral
    ds['phyto_vert_int'].attrs['long_name'] = 'vertical integral of phytoplankton'
    ds['phyto_vert_int'].attrs['units'] = 'mol/m2'

    # zooplankton vertical integral
    ds['zoop_vert_int'].attrs['long_name'] = 'vertical integral of zooplankton'
    ds['zoop_vert_int'].attrs['units'] = 'mol/m2'

    # NO3 vertical integral
    ds['NO3_vert_int'].attrs['long_name'] = 'vertical integral of nitrate'
    ds['NO3_vert_int'].attrs['units'] = 'mol/m2'

    # NH4 vertical integral
    ds['NH4_vert_int'].attrs['long_name'] = 'vertical integral of ammonium'
    ds['NH4_vert_int'].attrs['units'] = 'mol/m2'

    # SdetritusN vertical integral
    ds['SdetritusN_vert_int'].attrs['long_name'] = 'vertical integral of small detritus'
    ds['SdetritusN_vert_int'].attrs['units'] = 'mol/m2'

    # LdetritusN vertical integral
    ds['LdetritusN_vert_int'].attrs['long_name'] = 'vertical integral of large detritus'
    ds['LdetritusN_vert_int'].attrs['units'] = 'mol/m2'

    return ds


##############################################################
##                      PROCESS DATA                        ##
##############################################################

print('Processing started...\n')

for gtagex in gtagexes:
    for region in regions:
        for year in years:
            print('{}, {}, {}'.format(gtagex,region,year))

            # get data
            fp = Ldir['LOo'] / 'extract' / gtagex / 'box' / (region+'_'+year+'.01.01_'+year+'.12.31.nc')
            ds_raw = xr.open_dataset(fp)

            # original units --------------------------------
                # DO: mmol O2 /m3
                # phyto: mmol N /m3
                # zoop: mmol N /m3
                # NO3: mmol N /m3
                # NH4: mmol N /m3
                # LdetritusN: mmol N /m3
                # SdetritusN mmol N /m3
            # to convert to new units ------------------------
                # divide by 1000 (mmol to mol)
                # multiple by water column thickness (/m3 to /m2)

            # initialize dataset
            ds = start_ds(ds_raw['ocean_time'],
                        ds_raw['eta_rho'],
                        ds_raw['xi_rho'],)
            # add metadata
            ds = add_metadata(ds)

            print('    Calculating vertical thickness of all cells')
            # get thickness of hypoxic layer in watercolumn at ever lat/lon cell (ocean_time: 365, eta_rho: 441, xi_rho: 177)
            # units are in m (thickness of hypoxic layer)
            # get S for the whole grid
            Sfp = Ldir['data'] / 'grids' / 'cas7' / 'S_COORDINATE_INFO.csv'
            reader = csv.DictReader(open(Sfp))
            S_dict = {}
            for row in reader:
                S_dict[row['ITEMS']] = row['VALUES']
            S = zrfun.get_S(S_dict)
            # get cell thickness
            h = ds_raw['h'].values # height of water column
            # z_rho, z_w = zrfun.get_z(h, 0*h, S) 
            # dzr = np.diff(z_w, axis=0) # vertical thickness of all cells [m]  

             # loop over time to get z_rho and z_w:
            zeta = ds_raw['zeta'].values
            Nt = zeta.shape[0]
            Nz = S['N']
            dzr_all = np.empty((Nt, Nz, *h.shape))
            z_w_all = np.empty((Nt, Nz + 1, *h.shape))
            for t in range(Nt):
                # make sure to use zeta as an input to account for SSH variability!!!
                z_rho, z_w = zrfun.get_z(h, zeta[t, :, :], S)
                dzr_all[t, :, :, :] = np.diff(z_w, axis=0) # sigma layer thickness at one time
                z_w_all[t, :, :, :] = z_w # sigma layer boundaries at one time

            print('    Calculating DO vertical integral')
            # Now get oxygen values at every grid cell and convert to mol/m3,
            # and multipy by cell height array
            DO_mol_m2 = dzr_all * ds_raw['oxygen'].values / 1000
            # Sum along z to get thickness of hypoxic layer
            DO_vert_int = np.nansum(DO_mol_m2,axis=1)

            print('    Calculating phyto vertical integral')
            # Now get oxygen values at every grid cell and convert to mol/m3,
            # and multipy by cell height array
            phyto_mol_m2 = dzr_all * ds_raw['phytoplankton'].values / 1000
            # Sum along z to get thickness of hypoxic layer
            phyto_vert_int = np.nansum(phyto_mol_m2,axis=1)

            print('    Calculating zoop vertical integral')
            # Now get oxygen values at every grid cell and convert to mol/m3,
            # and multipy by cell height array
            zoop_mol_m2 = dzr_all * ds_raw['zooplankton'].values / 1000
            # Sum along z to get thickness of hypoxic layer
            zoop_vert_int = np.nansum(zoop_mol_m2,axis=1)

            print('    Calculating NO3 vertical integral')
            # Now get oxygen values at every grid cell and convert to mol/m3,
            # and multipy by cell height array
            NO3_mol_m2 = dzr_all * ds_raw['NO3'].values / 1000
            # Sum along z to get thickness of hypoxic layer
            NO3_vert_int = np.nansum(NO3_mol_m2,axis=1)

            print('    Calculating NH4 vertical integral')
            # Now get oxygen values at every grid cell and convert to mol/m3,
            # and multipy by cell height array
            NH4_mol_m2 = dzr_all * ds_raw['NH4'].values / 1000
            # Sum along z to get thickness of hypoxic layer
            NH4_vert_int = np.nansum(NH4_mol_m2,axis=1)

            print('    Calculating small detritus vertical integral')
            # Now get oxygen values at every grid cell and convert to mol/m3,
            # and multipy by cell height array
            Sdet_mol_m2 = dzr_all * ds_raw['SdetritusN'].values / 1000
            # Sum along z to get thickness of hypoxic layer
            SdetritusN_vert_int = np.nansum(Sdet_mol_m2,axis=1)

            print('    Calculating large detritus vertical integral')
            # Now get oxygen values at every grid cell and convert to mol/m3,
            # and multipy by cell height array
            Ldet_mol_m2 = dzr_all * ds_raw['LdetritusN'].values / 1000
            # Sum along z to get thickness of hypoxic layer
            LdetritusN_vert_int = np.nansum(Ldet_mol_m2,axis=1)


            # add data to ds --------------------------------
            print('    Adding data to dataset')


            ds['DO_vert_int'] = xr.DataArray(DO_vert_int,
                                        coords={'ocean_time': ds_raw['ocean_time'].values,
                                                'eta_rho': ds_raw['eta_rho'].values,
                                                'xi_rho': ds_raw['xi_rho'].values},
                                        dims=['ocean_time','eta_rho', 'xi_rho'])
            
            ds['phyto_vert_int'] = xr.DataArray(phyto_vert_int,
                                        coords={'ocean_time': ds_raw['ocean_time'].values,
                                                'eta_rho': ds_raw['eta_rho'].values,
                                                'xi_rho': ds_raw['xi_rho'].values},
                                        dims=['ocean_time','eta_rho', 'xi_rho'])
            
            ds['zoop_vert_int'] = xr.DataArray(zoop_vert_int,
                                        coords={'ocean_time': ds_raw['ocean_time'].values,
                                                'eta_rho': ds_raw['eta_rho'].values,
                                                'xi_rho': ds_raw['xi_rho'].values},
                                        dims=['ocean_time','eta_rho', 'xi_rho'])
            
            ds['NO3_vert_int'] = xr.DataArray(NO3_vert_int,
                                        coords={'ocean_time': ds_raw['ocean_time'].values,
                                                'eta_rho': ds_raw['eta_rho'].values,
                                                'xi_rho': ds_raw['xi_rho'].values},
                                        dims=['ocean_time','eta_rho', 'xi_rho'])
            
            ds['NH4_vert_int'] = xr.DataArray(NH4_vert_int,
                                        coords={'ocean_time': ds_raw['ocean_time'].values,
                                                'eta_rho': ds_raw['eta_rho'].values,
                                                'xi_rho': ds_raw['xi_rho'].values},
                                        dims=['ocean_time','eta_rho', 'xi_rho'])
            
            ds['SdetritusN_vert_int'] = xr.DataArray(SdetritusN_vert_int,
                                        coords={'ocean_time': ds_raw['ocean_time'].values,
                                                'eta_rho': ds_raw['eta_rho'].values,
                                                'xi_rho': ds_raw['xi_rho'].values},
                                        dims=['ocean_time','eta_rho', 'xi_rho'])
            
            ds['LdetritusN_vert_int'] = xr.DataArray(LdetritusN_vert_int,
                                        coords={'ocean_time': ds_raw['ocean_time'].values,
                                                'eta_rho': ds_raw['eta_rho'].values,
                                                'xi_rho': ds_raw['xi_rho'].values},
                                        dims=['ocean_time','eta_rho', 'xi_rho'])
            

            print('    Saving dataset')
            # save dataset
            # if region == 'pugetsoundDO':
            #     if remove_straits:
            #         straits = 'noStraits'
            #     else:
            #         straits = 'withStraits'
            #     ds.to_netcdf(out_dir / (gtagex + '_' + region + '_' + year + '_NPZD_vert_ints_' + straits + '.nc'))
            # else:
            ds.to_netcdf(out_dir / (gtagex + '_' + region + '_' + year + '_NPZD_vert_ints.nc'))

print('Done')