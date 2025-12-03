"""
A module of parameters shared across the npzd models.
"""
import numpy as np

def sink(z_w, z_rho, Dz, N, C, Wsink, dt):

    Next = 10
    NN = N + Next

    Cext = np.concatenate((C, np.zeros(Next)))
    Zwext = np.concatenate((z_w, np.arange(Dz,Next*Dz + Dz,Dz)))
    Zext = Zwext[:-1] + Dz/2
    for ii in range(NN):
        zbot = Zwext[ii]
        ztop = Zwext[ii+1]
        NB = 1000
        dz = Dz/NB
        zb = np.arange(zbot,ztop,dz) + dz/2
        cb = np.ones(NB) * Cext[ii]
        if ii == 0:
            Zb = zb
            Cb = cb
        else:
            Zb = np.concatenate((Zb,zb))
            Cb = np.concatenate((Cb,cb))
        
    Zb_new = Zb - Wsink*dt

    ind = np.digitize(Zb_new, z_w, right=True)
    Cnew = np.zeros(N)
    for ii in range(N+1):
        mask = ind==ii
        if len(mask) > 0:
            Cnew[ii-1] = Cb[mask].mean()
        else:
            Cnew[ii-1] = 0
        
    Cnet_old = Dz * np.sum(C)
    Cnet_new = Dz * np.sum(Cnew)
    Cnet_lost = Cnet_old - Cnet_new
        
    return Cnew, Cnet_lost
