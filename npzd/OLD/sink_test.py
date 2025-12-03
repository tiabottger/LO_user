"""
Code to develop and test the sinking algorithm.
"""
import numpy as np
import matplotlib.pyplot as plt

# z-coordinates (bottom to top, positive up)
H = 50 # max depth [m]
N = 25 # number of vertical grid cells
C = np.exp(-np.linspace(-2,2,N)**2)
Wsink = .1#80 # m per day
dt = .1 # time step in days

Dz = H/N
z_w = np.arange(-H,Dz,Dz)
z_rho = z_w[:-1] + Dz/2

# new algorithm
h = Wsink * dt
nn = int(np.floor(h / Dz))
delta = h - nn * Dz

Next = nn + 2
NN = N + Next
Cext = np.concatenate((C, np.zeros(Next)))
Zwext = np.concatenate((z_w, np.arange(Dz,Next*Dz + Dz,Dz)))
Zext = Zwext[:-1] + Dz/2

Cnew = np.zeros(N)
for ii in range(N):
    Cnew[ii] = Cext[ii + nn]*(Dz - delta)/Dz + Cext[ii + nn + 1]*(delta/Dz)
    
Cnet_old = Dz * np.sum(C)
Cnet_new = Dz * np.sum(Cnew)
Cnet_lost = Cnet_old - Cnet_new
    
print('Cnet_old = %0.7f' % (Cnet_old))
print('Cnet_new = %0.7f' % (Cnet_new))
print('Cnet_lost = %0.7f' % (Cnet_lost))

plt.close('all')
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
ax.plot(Cext,Zext,'-og')
# ax.plot(Cb,Zb,'-b')
# ax.plot(Cb,Zb_new,'--b')
ax.plot(Cnew,z_rho,'-or')
plt.show()
