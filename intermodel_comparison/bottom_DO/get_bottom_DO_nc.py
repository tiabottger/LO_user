"""
This script determines the DO concentration in the lower 14.6% of the watercolumn, 
designed to match the resolution of Salish Sea Model. It creates a new .nc file with 
bottom DO concentration, layer thickness, and depth of water column.

This script has been modified from Aurora's get_bottom_DO.py script

This script searches for yearly box extractions in LO_output, for the
region "pugetsoundDO"

It also crops out data from the Straits, so as to not bias the results
in Puget Sound. (optional using flag remove_straits)

.nc files are saved in LO_output/intermodel_comparison/bottom_DO
"""

# import things
from LO.plotting import pinfo
import numpy as np
import xarray as xr
import csv
from lo_tools import Lfun, zrfun

import sys
from pathlib import Path
pth = Path(__file__).absolute().parent.parent.parent.parent / 'LO' / 'pgrid'
if str(pth) not in sys.path:
    sys.path.append(str(pth))
import gfun

Gr = gfun.gstart()

Ldir = Lfun.Lstart()

##############################################################
##                       USER INPUTS                        ##
##############################################################

regions = ['pugetsoundDO']

years = ['2014'] #['2015','2016','2017','2018','2019','2020']

# which  model run to look at?
gtagexes = ['cas7_t1_x11ab']  

# where to put output files
out_dir = Ldir['LOo'] / 'intermodel_comparison' / 'bottom_DO'
Lfun.make_dir(out_dir)

##############################################################
##                    HELPER FUNCTIONS                      ##
##############################################################

def start_ds(ocean_time,eta_rho,xi_rho):
    '''
    Initialize dataset to store processed DO data
    ocean_time = ocean time vector
    eta_rho = eta_rho vector
    xi_rho = xi_rho vector
    '''
    Ndays = len(ocean_time.values)
    Neta = len(eta_rho.values)
    Nxi = len(xi_rho.values)

    ds = xr.Dataset(data_vars=dict(
        depth_bot   = (['eta_rho','xi_rho'], np.zeros((Neta,Nxi))),
        # DO concentration at bottom
        DO_bot      = (['ocean_time','eta_rho','xi_rho'], np.zeros((Ndays,Neta,Nxi))),
        # DO concentration at bottom 14.6% of water column
        DO_bot146   = (['ocean_time','eta_rho','xi_rho'], np.zeros((Ndays,Neta,Nxi))),
        # thickness of bottom layer
        thick_bot   = (['ocean_time','eta_rho','xi_rho'], np.zeros((Ndays,Neta,Nxi))),
        # thickness of bottom 14.6% layer
        thick_bot146   = (['ocean_time','eta_rho','xi_rho'], np.zeros((Ndays,Neta,Nxi))),
        # thickness of hypoxic layer
        hyp_thick   = (['ocean_time','eta_rho','xi_rho'], np.zeros((Ndays,Neta,Nxi))),
        # thickness of 1mg/L layer
        one_mgL_thick   = (['ocean_time','eta_rho','xi_rho'], np.zeros((Ndays,Neta,Nxi))),
        # thickness of 3mg/L layer
        three_mgL_thick   = (['ocean_time','eta_rho','xi_rho'], np.zeros((Ndays,Neta,Nxi))),),
    coords=dict(ocean_time=ocean_time, eta_rho=eta_rho, xi_rho=xi_rho,),)

    
    return ds

def add_metadata(ds):
    '''
    Create metadata for processed DO data
    '''

    ds['depth_bot'].attrs['long_name'] = 'watercolumn depth'
    ds['depth_bot'].attrs['units'] = 'm'

    ds['DO_bot'].attrs['long_name'] = 'DO concentration at bottom'
    ds['DO_bot'].attrs['units'] = 'mg/L'
    
    ds['DO_bot146'].attrs['long_name'] = 'DO concentration at bottom 14.6% of water column'
    ds['DO_bot146'].attrs['units'] = 'mg/L'

    ds['thick_bot'].attrs['long_name'] = 'thickness of bottom layer'
    ds['thick_bot'].attrs['units'] = 'm'

    ds['thick_bot146'].attrs['long_name'] = 'thickness of bottom 14.6% layer'
    ds['thick_bot146'].attrs['units'] = 'm'

    ds['hyp_thick'].attrs['long_name'] = 'thickness of hypoxic layer'
    ds['hyp_thick'].attrs['units'] = 'm'

    ds['one_mgL_thick'].attrs['long_name'] = 'thickness of 1 mg/L layer (or lower concentration)'
    ds['one_mgL_thick'].attrs['units'] = 'm'

    ds['three_mgL_thick'].attrs['long_name'] = 'thickness of 3 mg/L layer (or lower concentration)'
    ds['three_mgL_thick'].attrs['units'] = 'm'

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

            # initialize dataset
            ds = start_ds(ds_raw['ocean_time'],
                        ds_raw['eta_rho'],
                        ds_raw['xi_rho'],)
            # add metadata
            ds = add_metadata(ds)
            
            print('    Calculating hypoxic thickness')
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

            zeta = ds_raw['zeta'].values # sea surface height
            Nt = zeta.shape[0]
            Nz = S['N']
            dzr_all = np.empty((Nt, Nz, *h.shape))
            z_w_all = np.empty((Nt, Nz + 1, *h.shape)) # create empty array to store z_w values for all time steps, vertical layer, and horizontal grid cell
                                                       # h.shape unpacks (eta_rho, xi_rho)
                                                       # Nz+1 because there is one more vertical interface than layers
            
            # loop over time to get z_rho and z_w:
            for t in range(Nt):
                # make sure to use zeta as an input to account for SSH variability!!!
                z_rho, z_w = zrfun.get_z(h, zeta[t, :, :], S) # convert from coordinates to physical depths
                dzr_all[t, :, :, :] = np.diff(z_w, axis=0) # sigma layer thickness at one time
                z_w_all[t, :, :, :] = z_w # sigma layer boundaries at one time
            # z_rho, z_w = zrfun.get_z(h, ds_raw.zeta.to_numpy(), S) # make sure to use zeta as an input to account for SSH variability!!!
            # dzr = np.diff(z_w, axis=0) # vertical thickness of all cells [m]  

            # Now get oxygen values at every grid cell and convert to mg/L
            oxy_mgL = pinfo.fac_dict['oxygen'] * ds_raw['oxygen'].values
            # shape: (ocean_time, s_rho, eta_rho, xi_rho)
            
            # remove all non-hypoxic values (greater than 2 mg/L)
            hypoxic = np.where(oxy_mgL <= 2, 1, np.nan) # array of nans and ones. one means hypoxic, nan means nonhypoxic
            one_mgL = np.where(oxy_mgL <= 1, 1, np.nan) # array of nans and ones. one means hypoxic, nan means nonhypoxic
            three_mgL = np.where(oxy_mgL <= 3, 1, np.nan) # array of nans and ones. one means hypoxic, nan means nonhypoxic
            # Multiple cell height array by hypoxic array boolean array
            hyp_cell_thick = dzr_all * hypoxic
            one_mgL_cell_thick = dzr_all * one_mgL
            three_mgL_cell_thick = dzr_all * three_mgL
            # Sum along z to get thickness of hypoxic layer
            hyp_thick = np.nansum(hyp_cell_thick,axis=1)
            one_mgL_thick = np.nansum(one_mgL_cell_thick,axis=1)
            three_mgL_thick = np.nansum(three_mgL_cell_thick,axis=1)
            
            print('    Calculating bottom 14.6% DO concentration')
                
            # bottom LiveOcean cell thickness
            thick_bot = z_w_all[:, 1, :, :] - z_w_all[:, 0, :, :]
        
            # bottom 14.6% cutoff depth
            z_bot = -h[None, :,:]
            z_146 = -0.8538 * h[None, :,:]
            
            # cell interfaces
            z_lower = z_w_all[:, :-1, :, :]
            z_upper = z_w_all[:, 1:, :, :]
            
            overlap = np.maximum(
                0,
                np.minimum(z_upper, z_146[:, None, :, :])
                - np.maximum(z_lower, z_bot[:, None, :, :])
            )
            
            # thickness-weighted bottom DO
            overlap_sum = np.sum(overlap, axis=1)
            
            thick_bot146 = overlap_sum

            DO_bot146 = np.sum(oxy_mgL * overlap, axis=1) / overlap_sum
            
            # avoid divide-by-zero issues on land/masked cells
            DO_bot146 = np.where(overlap_sum > 0, DO_bot146, np.nan)
            
            
            print('    Calculating bottom DO concentration')
            # get bottom DO concentration
            DO_bot = pinfo.fac_dict['oxygen'] * ds_raw['oxygen'][:,0,:,:].values

            # add data to ds
            print('    Adding data to dataset')
            ds['depth_bot'] = xr.DataArray(h, 
                                        coords={'eta_rho': ds_raw['eta_rho'].values, # h is 2D
                                                'xi_rho': ds_raw['xi_rho'].values},
                                        dims=['eta_rho', 'xi_rho'])
            # DO concentration at bottom
            ds['DO_bot'] = xr.DataArray(DO_bot,
                                        coords={'ocean_time': ds_raw['ocean_time'].values,
                                                'eta_rho': ds_raw['eta_rho'].values,
                                                'xi_rho': ds_raw['xi_rho'].values},
                                        dims=['ocean_time','eta_rho', 'xi_rho'])
            # DO concentration at bottom 14.6% of water column
            ds['DO_bot146'] = xr.DataArray(DO_bot146,
                                        coords={'ocean_time': ds_raw['ocean_time'].values,
                                                'eta_rho': ds_raw['eta_rho'].values,
                                                'xi_rho': ds_raw['xi_rho'].values},
                                        dims=['ocean_time','eta_rho', 'xi_rho'])
            # bottom layer thickness
            ds['thick_bot'] = xr.DataArray(thick_bot,
                                        coords={'ocean_time': ds_raw['ocean_time'].values,
                                                'eta_rho': ds_raw['eta_rho'].values,
                                                'xi_rho': ds_raw['xi_rho'].values},
                                        dims=['ocean_time','eta_rho', 'xi_rho'])
            # bottom 14.6% layer thickness
            ds['thick_bot146'] = xr.DataArray(thick_bot146,
                                        coords={'ocean_time': ds_raw['ocean_time'].values,
                                                'eta_rho': ds_raw['eta_rho'].values,
                                                'xi_rho': ds_raw['xi_rho'].values},
                                        dims=['ocean_time','eta_rho', 'xi_rho'])
            # hypoxic layer thickness
            ds['hyp_thick'] = xr.DataArray(hyp_thick,
                                        coords={'ocean_time': ds_raw['ocean_time'].values,
                                                'eta_rho': ds_raw['eta_rho'].values,
                                                'xi_rho': ds_raw['xi_rho'].values},
                                        dims=['ocean_time','eta_rho', 'xi_rho'])
            
            # < 1 mg/L layer thickness
            ds['one_mgL_thick'] = xr.DataArray(one_mgL_thick,
                                        coords={'ocean_time': ds_raw['ocean_time'].values,
                                                'eta_rho': ds_raw['eta_rho'].values,
                                                'xi_rho': ds_raw['xi_rho'].values},
                                        dims=['ocean_time','eta_rho', 'xi_rho'])
            
            # < 1 mg/L layer thickness
            ds['three_mgL_thick'] = xr.DataArray(three_mgL_thick,
                                        coords={'ocean_time': ds_raw['ocean_time'].values,
                                                'eta_rho': ds_raw['eta_rho'].values,
                                                'xi_rho': ds_raw['xi_rho'].values},
                                        dims=['ocean_time','eta_rho', 'xi_rho'])

            print('    Saving dataset')
            ds.to_netcdf(out_dir / (gtagex + '_' + region + '_' + year + '_bottom_DO_info.nc'))

print('Done')