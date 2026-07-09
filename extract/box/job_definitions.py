"""
Module of functions to create job definitions for a box extraction.
"""

def get_box(job, Lon, Lat):
    vn_list = 'h,f,pm,pn,mask_rho,salt,temp,rho,zeta,u,v,ubar,vbar' # default list
    # specific jobs
    if job == 'prelimDO':
        aa = [-123.2, -122.1, 46.95, 48.45]
        vn_list = 'h,pm,pn,mask_rho,salt,temp,oxygen'
    elif job == 'pugetsound':
        aa = [-123.2, -122.1, 46.95, 48.45]
        vn_list =  ('h,pm,pn,mask_rho,salt,temp,zeta,NO3,NH4,phytoplankton,'
                + 'zooplankton,SdetritusN,LdetritusN,oxygen')
    elif job == 'pugetsoundDO':
        aa = [-123.29, -122.1, 46.95, 48.93]
        vn_list =  ('h,pm,pn,mask_rho,salt,temp,zeta,NO3,NH4,phytoplankton,'
                + 'zooplankton,SdetritusN,LdetritusN,oxygen,ubar,vbar,AKv,AKs')
        
    return aa, vn_list
