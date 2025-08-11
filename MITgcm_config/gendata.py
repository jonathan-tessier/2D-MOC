import numpy as np
from MITgcmutils.utils import writebin, readbin
import matplotlib.pyplot as plt
from scipy import interpolate
nx=30;
ny=70;
nz=40;

SOyind = 12

ydeg = np.arange(-69,70,2)
xdeg = np.arange(1,60,2)

print(ydeg)
print(xdeg)

xx,yy = np.meshgrid(xdeg,ydeg) # indexing [Ny,Nx]

######## topo ##########

topo = -4000.*np.ones_like(yy)
topo[ 0,:] = 0. # Southern wall 
topo[-1,:] = 0. # Northern wall
#topo[:, 0] = 0 # Western wall - closed basin
#topo[:,-1] = 0 # Eastern wall - closed basin
topo[SOyind:, 0] = 0. # ACC channel
topo[SOyind:,-1] = 0. # ACC channel

mask        = np.zeros([nz,ny,nx])
mask[0,:,:] = (topo!=0).astype('float')

writebin('topog.bin',topo)
writebin('rbcs_mask.bin',mask)

######### SST ##########
offset = 1.0
SSTmax = 30

SST = offset+(SSTmax-offset)*np.cos(np.pi*yy/(yy.max()-yy.min()))**2;
SST[:35,:] = (SSTmax*np.cos(np.pi*yy/(yy.max()-yy.min()))**2)[:35,:];

#SST[35:] = 30. # for abyssal cell

#deltab = 15.
#SST = deltab*((yy-yy.min())/(yy[SOyind,0]-yy.min()))
#SST[SOyind:,:] = deltab

SST = np.repeat(SST[np.newaxis,:,:],nz,axis=0)

writebin('rbcs_T.bin',SST) # [nz,ny,nx]

######## wind ##########

# just channel wind
wind = 0.1*np.sin(6*np.pi*yy/(yy.max()-yy.min()))**2; wind[SOyind:] = 0;

writebin('wind_x.bin',wind)

######## kappa ##########

zF = np.array([    0.,   -37.,   -77.,  -121.,  -169.,  -221.,  -277.,  -337.,\
        -400.,  -466.,  -535.,  -607.,  -682.,  -760.,  -841.,  -925.,\
       -1012., -1102., -1195., -1291., -1390., -1492., -1597., -1705.,\
       -1816., -1930., -2047., -2167., -2290., -2416., -2545., -2677.,\
       -2812., -2950., -3091., -3235., -3382., -3532., -3685., -3841.,\
       -4000.])
zC = np.array([  -18.5,   -57. ,   -99. ,  -145. ,  -195. ,  -249. ,  -307. ,\
        -368.5,  -433. ,  -500.5,  -571. ,  -644.5,  -721. ,  -800.5,\
        -883. ,  -968.5, -1057. , -1148.5, -1243. , -1340.5, -1441. ,\
       -1544.5, -1651. , -1760.5, -1873. , -1988.5, -2107. , -2228.5,\
       -2353. , -2480.5, -2611. , -2744.5, -2881. , -3020.5, -3163. ,\
       -3308.5, -3457. , -3608.5, -3763. , -3920.5])

lk       = 1250
kappa_0  = 2e-4

kappa = kappa_0*(np.exp(-(zC-zF.min())/lk)); 

print("lk = "+str(lk)+", kb = "+str(kappa_0)+"\n"+repr(kappa))
