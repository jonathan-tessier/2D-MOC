#
# Basic solver for a 2D zonally averaged Atlantic Meridional Overturning Circulation
# 
# zonal mean buoyancy:  b_t = + Psi_z*b_y - Psi_y*b_z + d/dz(kappa_z*b_z)
# Adv. streamfunction:  Psi = Psi/Lx, where Psi = PsiTW + PsiGM
# Thermal wind comp:    PsiTW_zzy = -1/f b_y, solved by integration, with lin.corr
# Baroclinic eddies:    PsiGM = -kappaGM*(b_y/b_z) where isopyc. slope tapered to Smax 
# Velocitites (v,w):    v = -Psi_z, w = Psi_y
#
# By: Jonathan Tessier, PhD in Oceanography, UQAR-ISMER (2022-2026)

# library imports
import h5py, sys, time
import numpy as np
import copy as cp
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from scipy import interpolate
from mitgcmtools import * # This file is made by me, includes post-processing utils for MITgcm simulations 
from MITgcmutils.utils import writebin, readbin # reads/writes binary input for MITgcm (eg. wind_x.bin)

# start timer to record computation time
t_start = time.time()

# Useful Constants
days2secs = 24*60*60        # converts days to seconds
degs2metr = 111177.4765625  # converts degrees to meters (lat)

# Simulation Parameters
#############################################################################

# Pull data from MITgcm runs, to do comparisons
#datalabel = 'new-abyssal-fixedkappa'; bgamma=1/6 # label of run in MITgcm-output dir.
#datalabel = 'new-middepth-weakwind'; bgamma = 1/6# turn off z diffusion for this
datalabel = 'new-amoc-fixedkappa'; bgamma= 1/6 # label of run in MITgcm-output dir.

#datalabel = 'new-abyssal-10degSO'; bgamma= 1/2
#datalabel = 'basin-nowind'; bgamma=1/8
#fields = get_fields(datalabel=datalabel,snap=50000000)

fields    = get_fields(datalabel=datalabel,snap=12522600) # extract fields at 20k yrs
grid      = load_grid(datalabel=datalabel) # extract grid information
forc      = get_forcing(datalabel=datalabel) # extract forcing fields (sst and wind stress)
avgfields = zonalavg_fields(fields,grid) # compute zonal averages of fields

# Temporal parameters
t0 = 0e3*365*days2secs    # init. time
tf = 1e4*365*days2secs  # final time (s)
dt = 5e-3*365*days2secs # timestep (s)
tplot = 2e3*dt          # plotting/storing frequency

# Constant physical parameters
omega    = 7.2921e-5    # Earth Rotation frequency (1/s)
f0       = 2*omega      # f-plane coriolis frequency (1/s)
rho0     = 1035         # typical density (kg/m^3) 
tAlpha   = 2e-4         # thermal expansion coeff (for T -> B transformation)
g0       = 9.81         # gravity (for T -> B transformation))
offset   = 0.1          # ###### SST (deg C) different between S.O. and northern edge 
SSTmax   = 30           # max SST (deg C) at equator (cos^2 profile)

# physics: convection
convectb = True         # include convection param in evolution
gamma    = 1e-5         # target lapse rate.. dT/dz diag. from MITgcm runs... (1e-5)

# physics: diffusion
vertdiff = True         # include vertical diffusion
kapzprof = True         # use non-uniform vertical diffusion, requires vertdiff=True
kappa_z  = 1e-4         # uniform vertical buoyancy diffusivity (m^2/s) if kapzprof = False, typical 1e-5
kbg      = 0            # background diffusivity (m^2/s) if kapzprof = True
kb       = 2e-4         # ###### bottom diffusivity value (m^2/s)
zk       = 2000         # diffusivity e-folding scale (m) kappa_z = kbg + (kb - kbg)*np.exp(-(zF-zF.min())/zk)
horzdiff = False        # include uniform horizontal diffusion
kappa_y  = 1e2          # horizontal buoyancy diffusivity (m^2/s) (typical 1e2)

# physics: advection (components are cumulative)
usePsiGM = True         # use PsiGM in advection (Gent McWilliams barocolinic eddies) 
usePsiTW = True         # use Thermal Wind in advection (Psi_zzy = -1/f b_y)
use_wind = False        # include basin winds, compoensate Ekman transport with integration constant
use_Psi0 = False        # override Psi and use a predetermined Psi0, to be defined below

# physics: Southern Channel (includes barotropic Psi_tau, no thermal wind in channel)
channel  = True         # include southern channel, Psi_tau replaces Psi_TW in channel latitudes
SOyind   = 12           # latitude grid index of channel boundary (12 orig.) ~ -46 degrees
matchSO  = True         # lin. correction to basin Psi_TW to match Psi_Res at channel interface, otherwise, goes to 0.
free_TW  = True        # don't apply linear correction to basin, let Psi_TW integrate to whatever. overides matchSO 

# physics: GM slope clipping (if all false, full slope returns)
kappaGM  = 1e3          # Gent-McWilliams eddy diff. (m^2/s) (typical 1e3)
Smax     = 1e-2         # max allowed slope by/bz in PsiGM taper (typical 1e-2)
clipping = True         # tapering scheme 1: Cox slope clipping, clips to Smax. 
tapergkw = False        # tapering scheme 2: GKW91 large slopes go to 0
taperdmw = False        # tapering scheme 3: DMW95 large slopes go to 0

# Plotting Options
just_ICs = True        # Show initial conditions, stats, and exit
pauselast= False        # run plt.show() on last time_frame (debugging)
pcolor   = False        # plot on pcolor instead of contourf for pixel-peeping
lat_lims = [-68,68]     # lat. limits in deg for plotting [-70,70] (F-grid), [-68,68] to exclude walls
dep_lims = [-4,  0]     # depth limits in km for plotting [-4,0] (F-grid)
Psi_lims = [-8,8]     # when plotting Psi, contour limits to be fixed, in Sv
fig_size = [15.1,8.5]   # figure size, for when working on various screens
mk_movie = True         # save plots as pngs and make ffmpeg movie at 12 fps
moviename= 'movie.mp4'  # movie name for above if mk_movie=True
fancy    = False        # plot contours of b on top of Psi for prettier figures, instead of Psi components
cmap     = 'seismic'    # colormap for contourf and pcolor ('seismic', 'bwr', 'etc')
save2h5  = True         # save data to HDF5 file, named output.h5, for pickups and later post-processing
h5freq   = tf/4         # data save frequency for pickups, only save 4 times to save space.
rm_topo  = True         # account for MITgcm providing fields with topo at southern/northern boundaries
pltmitgcm= True         # generate equivalent figure of MITgcm run to compare, while running.
plt_ICs  = False         # plt topo, kappa, surf temp
pltslice = False

# Grid Initialization
#############################################################################

# Grid domain params (C grid size. F grid +1 in both directions)
N     = [grid.Ny,grid.Nz]         # number of C-grid points (tracer) on C-grid

# F grid vectors
yF    = np.linspace(-70*degs2metr, 70*degs2metr, N[0]+1)
dyF   = np.diff(yF)
zF    = grid.rF

#zF    = np.linspace(0, -4000, N[1]+1) # uniform grid
dzF   = np.diff(zF)
LF    = [np.max(yF)-np.min(yF),np.min(zF)-np.max(zF)]

# C grid vectors
yC    = np.linspace(-70*degs2metr + dyF[0]/2,70*degs2metr - dyF[-1]/2,N[0])
#print(yC/degs2metr); sys.exit()
dyC   = np.diff(yC)
zC    = grid.rC 
#zC    = np.linspace(0+dzF[0]/2, -4000-dzF[-1]/2, N[1]) # uniform grid
dzC   = np.diff(zC)
LC    = [np.max(yC)-np.min(yC),np.min(zC)-np.max(zC)]

# Set SST on original grid like MITgcm 
SST = offset+(30.-offset)*np.cos(np.pi*yC/(yC.max()-yC.min()))**2;
SST[:35] = (30.*np.cos(np.pi*yC/(yC.max()-yC.min()))**2)[:35];

# zonal domain width (meters) on C-grid ...
#LxC    = 30*degs2metr*np.ones([N[0]  ,N[1]+1]) # no curvature on y C-grid
LxC    = grid.LxC2D # from 20 deg near poles to 60 degs at equator
LxC    = np.concatenate([LxC,LxC[:,1].reshape([N[0],1])],axis=1) # [Ny+1,Nz+1] for y,z F-grid

# ... and on F grid.
#LxF    = 30*degs2metr*np.ones([N[0]+1,N[1]+1]) # no curvature on y F-grid
#LxF = grid.LxF[SOyind]*np.ones([N[0]+1,N[1]+1])
#LxF    = np.nanmean(grid.LxF)*np.ones([N[0]+1,N[1]+1])
LxF    = grid.LxF2D # from 20 deg near poles to 60 degs at equator
LxF    = np.concatenate([LxF,LxF[0,:].reshape([1,N[1]])],axis=0) # fix missiing boundary value
LxF    = np.concatenate([LxF,LxF[:,-1].reshape([N[0]+1,1])],axis=1) # [Ny+1,Nz+1] for y,z F-grid

#plt.plot(yF,LxF); plt.show(); sys.exit()

# print stats
print("")
print("F-Grid Latitude Domain (deg): [{:.2f}, {:.2f}]".format(yF.min()/degs2metr,yF.max()/degs2metr))
print("C-Grid Latitude Domain (deg): [{:.2f}, {:.2f}]".format(yC.min()/degs2metr,yC.max()/degs2metr))
print("F-Grid    Depth Domain  (m):  [{:.2f}, {:.2f}]".format(zF.min(),zF.max()))
print("C-Grid    Depth Domain  (m):  [{:.2f}, {:.2f}]".format(zC.min(),zC.max()))
print("")

# if pulling fields from MITgcm output, southern and northern boundaries are walls, contain nans
if rm_topo: # remove these and recude domain to -68 to 68 degrees lat 
    yC  = yC[1:-1];    yF  = yF[1:-1];
    dyC = dyC[1:-1];   dyF = dyF[1:-1]
    LxC = LxC[1:-1,:]; LxF = LxF[1:-1,:]
    N[0]-= 2;       SOyind = SOyind-1

# domain grids: C grid center, F grid faces 
yyC ,  zzC = np.meshgrid(yC,zC,indexing='ij') 
dyyC, dzzC = np.meshgrid(dyC,dzC,indexing='ij') 
yyF ,  zzF = np.meshgrid(yF,zF,indexing='ij')
dyyF, dzzF = np.meshgrid(dyF,dzF,indexing='ij') 

# v/w points
yyV ,  zzV = np.meshgrid(yF,zC,indexing='ij')         # [Ny+1,Nz]
dyyV, dzzV = np.diff(yyV,axis=0), np.diff(zzV,axis=1) # [Ny,Nz] and [Ny+1,Nz-1]
yyW ,  zzW = np.meshgrid(yC,zF,indexing='ij')         # [Ny, Nz+1]
dyyW, dzzW = np.diff(yyW,axis=0), np.diff(zzW,axis=1) # [Ny-1,Nz+1] and [Ny,Nz]

# need to add a column in y to match v dimensions: [Ny+1,Nz] dzV
dzzFF = np.concatenate([dzzF,dzzF[-1,:].reshape([1,dzzF.shape[1]])],axis=0)
# need to add a column in z to match w dimensions: [Ny,Nz+1] dyW
dyyFF = np.concatenate([dyyF,dyyF[:,-1].reshape([dyyF.shape[0],1])],axis=1)
# need to add a column in y to match b dimensions: [Ny+1,Nz]
dzzCC = np.concatenate([dzzC,dzzC[-1,:].reshape([1,dzzC.shape[1]])],axis=0)
# need to add a column in z to match w dimensions: [Ny,Nz+1]
dyyCC = np.concatenate([dyyC,dyyC[:,-1].reshape([dyyC.shape[0],1])],axis=1)

# Ghost layer 
dyyCG = np.pad(dyyCC,pad_width=1,mode='edge')[:,1:-1] # [Ny+1,Nz]
dzzCG = np.pad(dzzCC,pad_width=1,mode='edge')[1:-1,:] # [Ny,Nz+1]

# pritn stats
print(" yC,  zC shape: "+str(yyC.shape))
print("dyC, dzC shape: "+str(dyyC.shape))
print(" yF,  zF shape: "+str(yyF.shape))
print("dyF, dzF shape: "+str(dyyF.shape))
print("")

# print channel location
if channel:
    print("Channel interface at y = {:.2f} degs lat".format(yyF[SOyind,0]/degs2metr))
    print("")

# Coriolis, Kappa, and Wind Initialization
#############################################################################

# Coriolis Frequency
f = 2*omega*np.sin(np.deg2rad(yyF/degs2metr))
# define non-singular inverse coriolis parameter (with taper about origin)
#equatordiv = np.exp(-1.5e3*(abs(yyF)/LF[0]-1/2)**8)
finv1 = np.where(np.abs(f) == 0, 0.0, 1/f)
eps = 1e-5; finv = f/(f**2 +eps**2)
#plt.figure(); plt.plot(yyF[:,0]/degs2metr,finv1[:,0],'-r'); plt.plot(yyF[:,0]/degs2metr,finv[:,0],'-b'); 
#plt.grid(True); plt.show(); sys.exit()

if kapzprof: # non-uniform vertical diffusion, if to be used
    kappa_z = kbg + (kb - kbg)*np.exp(-(zF-zF.min())/zk)
    #print(kappa_z[0]); print(kappa_z[-1]) 
    #plt.semilogx(kappa_z,zF); plt.grid(True); plt.show(); sys.exit()

#tau = np.transpose(np.tile(readbin('wind_old.bin',[70,30])[:,0],(N[1]+1,1))) # (N/m^2)
#tau = np.transpose(np.tile(forc.wind[0],(N[1]+1,1))) # uses MITgcm wind_x.bin file to extract surface forcing.
#tau = np.concatenate([tau,np.zeros([1,N[1]+1])],axis=0) # make 2D and close at bottom 
#if rm_topo: tau = cp.copy(tau[1:-1,:]) # remove topo points at North and South boundaries

tau = np.interp(yF,yC,forc.wind[0][1:-1])
tau = np.transpose(np.tile(tau,(N[1]+1,1)))

#plt.figure(); 
#plt.plot(yyF[:,0]/degs2metr,tau[:,0],'-b');
#plt.plot(yyC[:,0]/degs2metr,forc.wind[0][1:-1],'-k')
#plt.show(); sys.exit()

# channel Psi Tau (barotropic)
Psi_tau = -LxF*finv*tau/rho0
# impose no flow bcs for prescribed flow
Psi_tau = np.pad(Psi_tau[1:-1,1:-1],pad_width=1,mode='constant') # whole halo of zeros

# Initial Condition for stratification (b)
#############################################################################

# if pulling from pickup file:

#file = h5py.File("output-amoc-gamma-1em5.h5", mode="r")
#file = h5py.File("output-abyssal-gamma-1em5.h5", mode="r")
#file = h5py.File("output-middepth-gamma-1em5.h5", mode="r")
#file = h5py.File("output-ABL-comp.h5",mode="r")

#file = h5py.File("output-10deg-comp.h5",mode="r")
#file = h5py.File("output-ref.h5",mode="r")
#b0 = cp.copy(file["b"][:][-1,:,:])

# if pulling from MITgcm zonally avg fields
b0 = cp.copy(avgfields.b); 
if rm_topo: b0 = b0[1:-1,:] # remove walls from MITgcm
#b0[SOyind:,0] = b0[SOyind,0]
#print(b0[:,0]); sys.exit()
#dbdz_0 = np.diff(b0[ -1,:])/dzC
#dbdz_0 = np.diff(np.mean(b0[:-2,:],axis=0))/dzC
#print("SO dbdz = "+str(dbdz_0))
#dbdz_c = np.interp(-zC,-zF[1:-1],dbdz_0) # + 1e-6
#gamma = np.max(dbdz_0)/(g0*tAlpha) 
#gamma = dbdz_c/(g0*tAlpha) 
#gamma = 1e-5*np.exp(-zC/LC[1]) 
#print("gamma = "+str(gamma))
#plt.figure(); plt.semilogx(gamma,zC); plt.show()
#T2theta = gamma*zC; T2theta = T2theta - np.sum(T2theta*dzF)/LF[1]
#print(np.sum(T2theta*dzF))
#sys.exit()

#offset = 0; SSTmax = 30
#SST = offset+(30.-offset)*np.cos(np.pi*yC/(yC.max()-yC.min()))**2;
#SST = (30.*np.cos(np.pi*yC/(yC.max()-yC.min()))**2)[34:];
b0[:,0] = g0*tAlpha*SST[1:-1]

# if warm start with actual MITgcm SST forcing (relaxation..)
#b0      = 5*g0*tAlpha*((yyC-yyC.min())/(yyC[SOyind,0]-yyC.min()) - zzC/zzC.min()) # init whole basin at 10 deg C.
#b0[:,0] = np.transpose(forc.sst[0])[1:-1]*g0*tAlpha
#b0[:SOyind,0] = b0[:SOyind,0]*0.5
#b0      = np.tile(b0[:,0][:,np.newaxis],[1,N[1]])

# if applying some other user-defined profile for b.

#b0 = 0.05*np.exp(-zzC/(zzC.min()/8))*np.sin(1*np.pi*((yyC-yyC.min())/(yyC.max()-yyC.min()))); #all db/dy in SO
#b0[np.where(b0<0)] = 0
#b0[SOyind:,:] = np.tile(b0[SOyind,:],(N[0]-SOyind,1))
#b0[35:,:] = np.tile(b0[34,:],(N[0]-35,1))
#b0[:,1:] = 2*g0*tAlpha #np.nanmin(b0)

#plt.figure(); 
#plt.plot(yC/degs2metr,b0[:,0]/(g0*tAlpha),'-k');
#plt.plot(yC/degs2metr,avgfields.b[1:-1,0]/(g0*tAlpha),'-r')
#plt.grid(); plt.show(); sys.exit()

# Initial Condition for flow, if using prescribed Psi (Psi0)
#############################################################################

# prescribed flow (if use_Psi0 = True)
#Psi0 = -0.5*np.sin(np.pi*yyF/yyF.max())*np.sin(np.pi*zzF/zzF.min())
#Psi0 = np.pad(Psi0[1:-1,1:-1],pad_width=1,mode='constant')

# Initialize plotting and pickup file stuff
#############################################################################

# plotting/time parameters
t = np.arange(t0,tf+dt,dt)    # time vector
npt  = int(tplot/dt)          # number of skipped frames for plotting
tplt = np.arange(t0,tf+tplot,tplot) # time vector for plotting

# pickup time vector stuff
t_h5 = np.arange(t0,tf+h5freq,h5freq) # pickup time vector
n_h5 = int(h5freq/dt) # number of skipped frames for pickup

# Output parameters for pickup files in HDF5
if save2h5:
    filename = "output.h5"                        # output filename
    file = h5py.File(filename, mode="w")          # output file
    file.create_dataset("LC",     data = LC)      # save domain lengths
    file.create_dataset("size",   data = N)       # save domain resolution
    file.create_dataset("t_h5",   data = t_h5)    # save pickup temporal vector
    file.create_dataset("f",      data = f)       # save coriolis   
    file.create_dataset("tau",    data = tau)     # save wind-stress 
    file.create_dataset("kappa_z",data = kappa_z) # save buoyancy diffusivitiy 

    # Output Fields
    output_b      = file.create_dataset('b',   (len(t_h5),N[0],N[1])) # init b output
    output_Psi    = file.create_dataset('Psi', (len(t_h5),N[0]+1,N[1]+1)) # init Psi Residual output
    output_PsiTW  = file.create_dataset('Psi_TW', (len(t_h5),N[0]+1,N[1]+1)) # init Psi TW output
    output_PsiLC  = file.create_dataset('Psi_LC', (len(t_h5),N[0]+1,N[1]+1)) # init Psi Lin. Corr. output
    output_PsiGM  = file.create_dataset('Psi_GM', (len(t_h5),N[0]+1,N[1]+1)) # init Psi GM output

    print("File "+filename+" opened")

# Helper Functions for convection, integration, Psi computation, etc...
#############################################################################

def plotICs():

    plt.figure(3,figsize=(15.1,8.8))
    plt.subplot(221)
    plt.plot(yC/degs2metr,forc.sst[0][1:-1]); plt.grid(True); plt.title(r"$SST(y)$"); plt.xlabel('y (deg)'); 
    plt.subplot(222)
    plt.plot(yC/degs2metr,forc.wind[0][1:-1]); plt.grid(True); plt.title(r"$\tau(y)$"); plt.xlabel('y (deg)'); 
    plt.subplot(223)
    plt.semilogx(kappa_z,zF); plt.grid(True); plt.title(r"$\kappa(z)$"); plt.ylabel('z (m)')
    plt.subplot(224)
    plt.pcolormesh(grid.yC,grid.xC,grid.depth); plt.title("topo(x,y)"); plt.xlabel('y (deg)'); plt.gca().invert_yaxis(); 
    plt.tight_layout(); plt.draw(); plt.savefig('frame_ICs.png', dpi=200)
if plt_ICs: plotICs()

def plotslice(b):

    plt.clf()
    plt.figure(4,figsize=(6,5))
    plt.semilogx(np.nanmean(np.diff(avgfields.b[1:-1])/dzzCC,axis=0),zF[1:-1],label='MIT')
    plt.xlabel(r"$\overline{\partial_z b}$"); plt.ylabel("z (m)"); plt.grid(True)
    plt.semilogx(np.nanmean(np.diff(b)/dzzCC,axis=0),zF[1:-1],label='Model'); plt.legend()
    plt.tight_layout(); plt.draw(); plt.savefig('frame_slice.png', dpi=200)


def level_vector():
    # find contours to plot, when plotting stratification, b
    
    tmp = np.linspace(0,1,75); stretch_power =10
    #tmp = np.linspace(0,1,75); stretch_power = 10
    stretch_vector = (np.exp(tmp)**stretch_power - 1)/(np.exp(1)**stretch_power-1)
    #levels = np.nanmin(b0) + stretch_vector*(np.nanmax(b0) - np.nanmin(b0))
    levels = np.nanmin(fields.B) + stretch_vector*(np.nanmax(fields.B) - np.nanmin(fields.B))  
    #levels = np.sort(np.unique(np.hstack([fields.B[1,1,:],avgfields.b[35,:]])))   
    return levels

def applybcs(b):
    # enforce boundary conditions on b: [Ny,Nz]
    # takes in b and modifies boudary elements 
    # returns modified b field, same size

    # Dirichlet boundary condition
    #b[-1,:] = cp.copy(b0[-1,:])  # NORTH 
    #b[:,-1] = cp.copy(b0[:,-1])  # BOTTOM   
    #b[0, :] = cp.copy(b0[0 ,:])  # SOUTH
    b[ :, 0] = cp.copy(b0[ :,0])  # SURFACE --
    # IMPOSING DIRICHLET AT THE SURFACE PREVENTS TRACER CONSERVATION

    #if np.nanmin(b) < np.nanmin(b0[:,0]):
        #b[np.where(b<np.nanmin(b0[:,0]))] = np.nanmin(b0[:,0])
        #print("Warning: denser water in domain compared to surface")
        #raise(ValueError("Denser Water in domain compared to surface min b"))

    return b

def convect(b):
    # --- column-wise convection parameterization ---
    # conserative scheme of Akmaev (1991, MWR)
    
    #gamma = 1e-9
    theta_arr = cp.copy(b)/(g0*tAlpha) # from current startification, get temperature [Ny,Nz].
    #print(np.sum(dyyF*dzzF*theta_arr)/(LC[0]*LC[1]))
    #theta_adj = np.zeros_like(theta_arr)
    #plt.figure(); plt.contourf(yyC,zzC,theta_arr,levels=51,cmap='seismic',norm=colors.PowerNorm(gamma=1/2))
    #plt.colorbar(); plt.tight_layout(); plt.show(); sys.exit()
    
    #print(q); sys.exit()
    for yind in range(N[0]):
        cycle = 0
        theta = theta_arr[yind,:]
        #print('yi = '+str(yind))
        while np.min(np.diff(theta)/dzC)<0: 
            theta = akmaev_column((theta))
            cycle += 1
            if cycle > 10000:
                plt.figure(9)
                plt.plot(theta,zC/1e3,'-k'); plt.grid(True)
                plt.xlabel("T (deg. C)"); plt.ylabel("z (km)")
                plt.title("y = "+str(yC[yind]/degs2metr)+" deg. lat.")
                plt.tight_layout(); plt.show()
                raise(RuntimeError("Convection didn't converge after 1000 iters"))
            #print("Theta = "+str(theta))
            #print("dT/dz = "+str(np.diff(theta)/dzC))
        theta_arr[yind,:] = theta

    #plt.figure(); plt.contourf(yyC,zzC,theta_adj,levels=51,cmap='seismic',norm=colors.PowerNorm(gamma=1/2))
    #plt.colorbar(); plt.tight_layout(); plt.show(); sys.exit()

    return theta_arr*g0*tAlpha

def akmaev_column(theta=None):

    #theta_adj = np.empty(N[1])
    #print(theta)
    q = abs(dzF) #4.2e3/g0* # heat capacity of the layers. [Nz]

    #print("q = "+str(q))
    #print("theta = "+str(theta))
    #print("dT/dz = "+str(np.diff(theta)/dzC))

    theta_k=np.empty(N[1]) # theta_k is a temperature vector where convective layers have been joined
    n_k    =np.empty(N[1],dtype='int') # number of convectively joined layers in each k-layer
    s_k    =np.empty(N[1]) # total effective heat capactity in each k-layer
    t_k    =np.empty(N[1]) # total energy in each k-layer
 
    k = 1
    n_k[k-1] = 1
    theta_k[k-1] = theta[k-1]
    l = 2
 
    while True: # This is effectively a loop over l (doing convective adjustment for all layers below l)
        # Akmaev step 2 
        n = 1
        thistheta = theta[l-1]
        while True:
            # This is effectively a loop over k (check all layers below, starting from previous k and do convective
            # adjustment if needed - notice that previous k points either at l-1 if no convection happened at last level or
            # the bottom of the convective layer below l - which is by construction homogeneous in theta )
            # Akmaev step 3
            #print("theta_k   = "+str(theta_k[k-1]))
            #print("thistheta = "+str(thistheta))

            if theta_k[k-1] >= thistheta:
                #print(" - STABLE")
                # stable - nothing (more) to do for this level
                # Akmaev step 6
                k += 1
                break  # to step 7 - move on to next l-level (unless we are done)
            else:
                #print(" - UNSTABLE")
                # unstable, need to do convective adjustment:
                if n <= 1: # first convective step - create neutral layer
                    #print("n<=1, create neutral layer")
                    s = q[l-1]
                    t = s*thistheta
                # Akmaev step 4
                if n_k[k-1] <= 1:
                # lower adjacent level is not an earlier-formed neutral layer
                    #print("n_k[k-1]<=1, lower adj lvl is not already neutral layer ")
                    s_k[k-1] = q[l-n-1]
                    t_k[k-1] = s_k[k-1] * theta_k[k-1]
                # Akmaev step 5
                #  join current and underlying layers
                n += n_k[k-1]
                s += s_k[k-1]
                t += t_k[k-1] # Notice this is really the total energy in the neutral layer
                s_k[k-1] = s
                t_k[k-1] = t
                thistheta = t/s # By dividing pseudo-temp by sum of all heat cpapcities we get the actual mean temp
                # -------- gamma t
                if k==1:
                    #  joint neutral layer is the first one
                    break  # to step 7 - move on to next l-level (unless we are done)
                k -= 1
                # back to step 3
        # Akmaev step 7
        if l == N[1]:  # the scan is over
            #print("scan is over")
            break  # to step 8
        l += 1
        n_k[k-1] = n #print(type(n_k[k-1]))
        theta_k[k-1] = thistheta
        # back to step 2

    # update the potential temperatures
    while True: # This is effectively a loop over k (i.e. the unified layers)
        # Akmaev step 8
        while n>1:
            # find center of mass of unstable layer...
            # at bottom of water column, climbing upwards to find cm of unstable

            # mixed layer boundaries
            nl_bottom = l #+ 1
            nl_top    = l - n 
            # initial background stratification
            T2theta_in = (gamma*zC)[nl_top:nl_bottom];

            # remove dz weighted mean, such that T2theta conserves centre of mass 
            T2theta_in = T2theta_in - np.sum(T2theta_in*dzF[nl_top:nl_bottom])/np.sum(dzF[nl_top:nl_bottom])
            #print(T2theta.shape)
            #print(np.sum(T2theta*dzF[nl_top:nl_bottom])/np.sum(dzF[nl_top:nl_bottom]))

            # Set all potential temperatures in this convective layer to thistheta
            while True:
                #  Akmaev step 9
                # # add background start to well-mixed layer

                theta[l-1] = thistheta + T2theta_in[n-1]  

                if n==1:
                    break
                l -= 1
                n -= 1
        # Akmaev step 11
        if k==1:
            break
        k -= 1
        l -= 1
        n = n_k[k-1]
        thistheta = theta_k[k-1]  
        # back to step 8
    Energy_final = np.sum(theta*dzF)
    #print("Energy final: "+str(Energy_final))     
    return theta 

def PsiGM(b):
    # take in strat b with size: [Ny,Nz] on [C,C]-grid
    # returns PsiGM = kappaGM*s(b) with size: [Ny+1,Nz+1] on [F,F]-grid
    # PsiGM boundary conditions set to zero all around. Interior uses 
    # average of c-grid points to project onto F grid for differencing

    # compute gradient components, falling on velocity points on grid
    b_y = np.diff(b,axis=0)/dyyCC # v-pts [Ny-1,Nz] missing north/south
    b_z = np.diff(b,axis=1)/dzzCC # w-pts [Ny,Nz-1] missing top/bottom
    # avg dbdy in z and dbdz in y for it to fall on Psi point [using 4 points in b for the Psi in the middle]   
    dbdy = (b_y[:,:-1]*dzzF[:-1,:-1] + b_y[:,1:]*dzzF[1:,1:]) / (dzzF[:-1,:-1] + dzzF[1:,1:]) # [Ny-1,Nz-1]
    dbdz = (b_z[:-1,:]*dyyF[:-1,:-1] + b_z[1:,:]*dyyF[1:,1:]) / (dyyF[:-1,:-1] + dyyF[1:,1:]) # [Ny-1,Nz-1]

    # define slope at the ratio of the derivatives
    S = dbdy/dbdz # [Ny-1,Nz-1] on [F,F] grid, missing all boundary conditions
   
    # choose slope limiting scheme, if none chosen, returns unlimited slope.
    if clipping: # Cox clipping
        S = np.nan_to_num(S, neginf = -Smax, posinf = Smax)
        S = np.where(abs(S) > Smax, np.sign(S)*Smax, S)
    elif tapergkw: # GKW91 tapering
        S = np.nan_to_num(S, neginf = 0, posinf = 0)
        S = np.minimum(1,(Smax/S)**2)*S
    elif taperdmw: # DMW95 tapering
        Sd = 0.001
        S = np.nan_to_num(S, neginf = 0, posinf = 0)
        S = 0.5*(1 + np.tanh((Smax - abs(S))/Sd))*S

    # then define Psi GM with tappered slope 
    Psi_GM = -kappaGM*S # [Ny-1,Nz-1]
    # pad with zeros to close circulation at all boundaries
    Psi_GM = LxF*np.pad(Psi_GM[:,:],pad_width=1,mode='constant') # now [Ny+1,Nz+1], [F,F] grid

    if save2h5 and cnt % n_h5 == 0:
        cnt_h5             = int(cnt/n_h5)
        output_PsiGM[cnt_h5] = Psi_GM

    return Psi_GM

def intdzCF(field,dzz,reverse=True): # C-grid to F-grid
    # take in C-grid field, say [Ny,Nz] and integrate in z 
    # using dzzF matrix, returns cumulative integral on F-grid
    # with shape [Ny,Nz+1], first row being zeros.
    if reverse: # bottom to top
        int_field = -np.flip(np.nancumsum(np.flip(field*dzz,axis=1),axis=1),axis=1)
        return np.concatenate([int_field,np.zeros([field.shape[0],1])],axis=1) 
    else: # top to bottom
        int_field = np.nancumsum(field*dzz,axis=1)
        return np.concatenate([np.zeros([field.shape[0],1]),int_field],axis=1)

def intdyCF(field,dyy,reverse=True): # C-grid to F-grid
    # take in C-grid field, say [Ny,Nz] and integrate in y 
    # using dyyF matrix, returns cumulative integral on F-grid
    # with shape [Ny+1,Nz], first col being zeros.
    if reverse: # North to South
        int_field = -np.flip(np.nancumsum(np.flip(field*dyy,axis=0),axis=0),axis=0)
        return np.concatenate([int_field,np.zeros([1,field.shape[1]])],axis=0)
    else: # South to North
        int_field = np.nancumsum(field*dyy,axis=0)
        return np.concatenate([np.zeros([1,field.shape[1]]),int_field],axis=0)

def zF2zC(field,y):
    # interpolate F-grid field to C-grid in z (vertical)
    # retains y coordinates, no bcs required, C grid inside F grid
    interp = interpolate.RegularGridInterpolator((y, zF), field)
    yy, zz = np.meshgrid(y,zC,indexing='ij')
    return interp((yy,zz))

def yF2yC(field,z):
    # interpolate F-grid field to C-grid in y (horizontal)
    # retains z coordinates, no bcs required, C grid inside F grid
    interp = interpolate.RegularGridInterpolator((yF, z), field)
    yy, zz = np.meshgrid(yC,z,indexing='ij')
    return interp((yy,zz))

def zC2zF(field,y,top=None,bottom=None):
    # interpolate C-grid field to F-grid in z (vertical)
    # retains y coordinates, needs top/bottom defined
    interp = interpolate.RegularGridInterpolator((y, zC), field)
    yy, zz = np.meshgrid(y,zF[1:-1],indexing='ij')
    intfield = interp((yy,zz))
    if np.any(top): # if top bc spec.
        intfield = np.concatenate([top.reshape([field.shape[0],1]),intfield],axis=1)
    else: # otherwise put zeros
        intfield = np.concatenate([np.zeros([field.shape[0],1]),intfield],axis=1)
    if np.any(bottom): # if bottom bc spec.
        intfield = np.concatenate([intfield,bottom.reshape([field.shape[0],1])],axis=1)
    else: # otherwise put zeros
        intfield = np.concatenate([intfield,np.zeros([field.shape[0],1])],axis=1)
    return intfield

def yC2yF(field,z,south=None,north=None):
    # interpolate C-grid field to F-grid in y (horizontal)
    # retains z coordinates, needs north/south defined
    if len(z)>1: # If field is in fact 2D in [y,z]
        interp = interpolate.RegularGridInterpolator((yC, z), field)
        yy, zz = np.meshgrid(yF[1:-1],z,indexing='ij')
        intfield = interp((yy,zz))
        if np.any(south): # if south bc spec
            intfield = np.concatenate([south.reshape([1,field.shape[1]]),intfield],axis=0)
        else: # otherwise put zeros
            intfield = np.concatenate([np.zeros([1,field.shape[1]]),intfield],axis=0)
        if np.any(north): # if north bc spec.
            intfield = np.concatenate([intfield,north.reshape([1,field.shape[1]])],axis=0)
        else: # otherwise put zeros
            intfield = np.concatenate([intfield,np.zeros([1,field.shape[1]])],axis=0)
    else: # Otherwise, interp single slice (eg. surface) in [y,z=z0], deal in same way
        intfield = np.interp(yF[1:-1],yC,field)
        if np.any(south): 
            intfield = np.concatenate([south,intfield])
        else:
            intfield = np.concatenate([[0],intfield])
        if np.any(north):
            intfield = np.concatenate([intfield,north])
        else:
            intfield = np.concatenate([intfield,[0]])

    return intfield

def ddykappaby(b):
    # horizontal component of diffusion laplacian, returns d/dy(kappa_y*b_y)
    # dbdy is on v points, missing edge values, with size [Ny-1,Nz]
    bp = np.pad(b,pad_width=1,mode='edge')[:,1:-1] # pad with ghost layer in y
    dbdy = np.diff(bp,axis=0)/dyyCG # [Ny+1,Nz] south/north values zero
    d2bdy2 = np.diff(kappa_y*dbdy,axis=0)/dyyF # [Ny,Nz] on b grid
    return d2bdy2

def ddzkappabz(b):
    # vertical component of diffusion laplacian, returns d/dz(kappa_z*b_z)
    # dbdz is on w points, missing edge values, with size [Ny,Nz-1]
    bp = np.pad(b,pad_width=1,mode='edge')[1:-1,:] # pad with ghost layer in z
    dbdz = np.diff(bp,axis=1)/dzzCG # [Ny,Nz+1] surf/bott values are zero
    d2bdz2 = np.diff(kappa_z*dbdz,axis=1)/dzzF # [Ny,Nz] on b grid
    #if np.max(abs(d2bdz2[:,-1]))!=0: print("WARNING DIFFUSION THROUGH BOTTOM")
    return d2bdz2

def PsiTW(b):
    # from the stratification (b), calculate the local streamfunction defined by:
    # Thermal Wind -> Psi_zzy = -1/f b_y. Returns the advective streamfunction,
    # where v = -d/dz Psi, w = d/dy Psi

    # calculate db/dy, need ghost points to deal with boundary conditions
    bp = np.pad(b,pad_width=1,mode='edge')[:,1:-1] # [Ny+2,Nz] on [yC,zC] filled with ghost
    dbdy = np.diff(bp,axis=0)/dyyCG # [Ny+1,Nz] on [yF,zC] grid, N/S are zero
    # calculate thermal wind u = -1/f int_-H^z db/dy dz
    u_therm = -finv*intdzCF(dbdy,dzzFF) # [Ny+1, Nz+1] on [yF,zF] grid, int from zero
    u_therm = zF2zC(u_therm,yF) # interp u onto [Ny+1,Nz] on [yF,zC]
    # now integrate u_thermal from F-grid to F-grid...
    Cz = -1/LF[1]*intdzCF(u_therm,dzzFF,reverse=False)[:,-1]
    Cz  = np.repeat((Cz)[:,np.newaxis],N[1],axis=1) # [Ny+1,Nz]

    u   = u_therm + Cz # full u solution # [Ny+1, Nz+1] on [yF,zF] grid
    u = yF2yC(u,zC) # interp u onto [Ny,Nz] on [yC,zC]
    Psi_thermal = intdyCF(intdzCF(u,dzzF),dyyFF) # in m3/s

    if free_TW: # don't correct 
        if channel: 
            Psi_thermal[:SOyind,:] = (Psi_tau)[:SOyind,:] # if channel, swap channel Psi_thermal with Psi_Ekman
        return Psi_thermal 

    if channel: # if include southern channel
        if matchSO:
            lincorrboundary = np.repeat((Psi_tau-Psi_thermal)[SOyind,:][np.newaxis,:],N[0]+1,axis=0)  
        else:
            lincorrboundary = np.repeat((-Psi_thermal)[SOyind,:][np.newaxis,:],N[0]+1,axis=0) # just close thermal

        thetaF = np.deg2rad(yyF/degs2metr) # actual latitude
        linfunc = (np.sin(thetaF) - np.sin(thetaF.max()))/(np.sin(thetaF[SOyind,0]) - np.sin(thetaF.max())) # linear corr to Basin...
        linfunc[:SOyind,:] = 1
        #linfunc = (yyF - yyF.max())/(yyF[SOyind,0] - yyF.max()) # linear corr to Basin...

        lincorr = lincorrboundary*linfunc

        if save2h5 and cnt % n_h5 == 0:
            cnt_h5             = int(cnt/n_h5)
            output_PsiLC[cnt_h5] = lincorr

        Psi_thermal = Psi_thermal + lincorr  

        Psi_thermal[:SOyind,:] = (Psi_tau)[:SOyind,:]; # remove thermal wind in channel, replace with ekman

        #plt.figure(figsize=fig_size); 
        #plt.subplot(231); plt.plot(yF/degs2metr,linfunc[:,10]); plt.title("lin func"); 
        #plt.subplot(232); plt.plot(yF/degs2metr,lincorr[:,10]); plt.title("lin corr")
        #plt.subplot(233); plt.plot(yF/degs2metr,Psi_thermal[:,10]/1e6); plt.title("")
        #plt.tight_layout(); plt.show()
        #print("Psi Thermal = "+str(1e-6*Psi_thermal[SOyind,:])); sys.exit()
        #Psi_thermal[SOyind,:] = 0

    else: # otherwise closed basin
        lincorrboundary = np.repeat((-Psi_thermal)[0,:][np.newaxis,:],N[0]+1,axis=0)      
        thetaF = np.deg2rad(yyF/degs2metr) # actual latitude
        linfunc = (np.sin(thetaF) - np.sin(thetaF.max()))/(np.sin(thetaF[SOyind,0]) - np.sin(thetaF.max())) # linear corr to Basin...
        linfunc[:SOyind,:] = 1
        lincorr = lincorrboundary*linfunc
        Psi_thermal = Psi_thermal + lincorr

    if save2h5 and cnt % n_h5 == 0:
        cnt_h5             = int(cnt/n_h5)
        output_PsiTW[cnt_h5] = Psi_thermal

    return Psi_thermal

def vels(Psi):
    # takes in streamfunction Psi: [Ny+1,Nz+1]
    # and returns the merid. (v) and vert. (w) velocities
    #  Psi: [N[0]+1,N[1]+1], outer all zeroes 
    #    v: [N[0]+1,N[1]  ], North/South zeroes
    #    w: [N[0]  ,N[1]+1], top/bott zeroes
    # v = -d/dz Psi, w = d/dy Psi
    v = -1/LxF[:,1:]*np.diff(Psi,axis=1)/dzzFF
    #v[:,0] = 0
    w = +1/LxC*np.diff(Psi,axis=0)/dyyFF
    #print(w[:,-1])
    if np.max(abs(w[:,-1])) > 1e-15:
        print("WARNING FLOW THROUGH BOTTOM")
    else:
        w[:,-1] = 0 # accounting for numerical error w ~ 1e-25 ----------------!!!
    if np.max(abs(w[:,0])) > 1e-15:
        print("WARNING FLOW THROUGH SURFACE")
    else:
        w[:,0] = 0 # accounting for numerical error w ~ 1e-25 ----------------!!!     
    return v,w

def upwind(Psi,b):
    # FIRST ORDER upwind scheme for J(Psi,b)
    # return: Psi_y*b_z - Psi_z*b_y

    # velocity components
    v,w = vels(Psi) # v = [Ny+1,Nz], w = [Ny,Nz+1]

    # upwind velocities 
    vp = np.maximum(v,0)[:-1,:]; # v-vel northward [Ny,Nz] ignoring north wall.
    vm = np.minimum(v,0)[ 1:,:]; # v-vel southward [Ny,Nz] ignoring south wall.
    wp = np.maximum(w,0)[:, 1:]; # w-vel upwards   [Ny,Nz] ignoring bottom
    wm = np.minimum(w,0)[:,:-1]; # w-vel downwards [Ny,Nz] ignoring surface

    bp = np.pad(b,pad_width=1,mode='edge') # now [Ny+2,Nz+2] on ghosted C-grid,  

    dbdy = np.diff(bp[:,1:-1],axis=0)/dyyCG # on y F-grid, [Ny+1,Nz] without top/bottom ghost
    dbdz = np.diff(bp[1:-1,:],axis=1)/dzzCG # on z F-grid, [Ny,Nz+1] without south/north ghost

    b_ym = dbdy[:-1,:]; b_yp = dbdy[1:,:]; b_zm = dbdz[:,:-1]; b_zp = dbdz[:,1:]

    return vp*b_ym + vm*b_yp + wm*b_zm + wp*b_zp # w positive. points in neg indices

def update_Psi(b):
    # Wrapper function to select what flows are included for advection
    # Init Psi on [F,F] grid, [Ny+1,Nz+1]
    Psi = np.zeros_like(yyF)

    if use_Psi0: # set fixed streamfunction, defined by user in Initial Conditions
        return Psi0
    else: # otherwise, using physics
        if usePsiGM: # include baroclinic eddies (GMredi param, advective form)
            Psi_GM = PsiGM(b) # compute using strat
            Psi += Psi_GM # include PsiGM in advection 
        if usePsiTW: # include Thermal wind (eulerian circulation)
            Psi_TW = PsiTW(b) # compute using start
            Psi += Psi_TW # include Thermal Wind in advection 
            # (wind effects even Psi_tau in S.O. are in PsiTW)
        return Psi

def bflux(b,Psi):
    # Compute the numerical flux to the b equation 
    # takes in b: [Ny,Nz] and Psi: [Ny+1,Nz+1], 
    # imposes bcs to b using ICs, advects b field 
    # and introduces vert/horz diffusion (if asked by user in flags)
    # flux returned same size as b: [Ny,Nz]
    # returns: -Jac(Psi,b) + d/dz(kappa_z*b_z) + d/dy(kappa_y*b_y)

    # impose boundary conditions
    b = applybcs(b)
 
    # Check numerical stability 
    check_CFL(Psi)

    # compute advection term
    flux = -upwind(Psi,b)

    if vertdiff: # explicit vertical diffusion
        flux = flux + ddzkappabz(b)
    if horzdiff: # explicit horizontal diffusion
        flux = flux + ddykappaby(b)

    #print("Total b flux = {:.2e}".format(total(flux)*(LF[0]*LF[1])))
    return flux

def calculate_error(b2,b1):
    # simple norm difference between b1 and b2 for measuring change in the solution.
    # will eventually set the number of itirations needed to find the steady soluton.
    #print("total b2: {}, total b1: {}".format(total(b2),total(b1)))
    return (total(b2) - total(b1))/total(b2)

def total(b): # first order riemann sum for domain integrated b
    # normalized by domain area so technically a domain avg
    return np.sum(np.sum(dyyF*dzzF*b,axis=0))/(LF[0]*LF[1])

def check_CFL(Psi):
    # Debug outputs for CFL conditions
    v,w = vels(Psi) # calculate velocitites
    CFLadvy = np.max(abs(v))*dt/abs(dyyF).min() # horizontal CFL condition
    CFLadvz = np.max(abs(w))*dt/abs(dzzF).min() # vertical CFL condition
    CFL = np.max([CFLadvy,CFLadvz]) # compute the absolute CFL as the max of the two.
    if CFL > 1: # should really be kept of 0.8 for AB3 timestepping
        print("WARNING: ADVC CFL number ({:.2f}) > 1 (unstable). Reduce timestep.".format(CFL))
    return CFL

def plotmit():
    # Basic plotting of the MITgcm solution equivalent of this simulation
    # plots Psi_residual, b and Psi_euler and Psi_GM in realtime computation
    # plots line at channel-basin interface if desired. (at y=SOyind=12)
    # constant in time, saves plot in 1 frame: frame_mitgcm.png in CWD
    lvls = np.linspace(Psi_lims[0],Psi_lims[1],51) # uses colorbar limits defined by user for Psi
    plt.figure(24,figsize=fig_size) #(15.1,8.8))
    #plt.figure(24,figsize=(10,6))

    plt.clf()
    plt.subplot(2,2,1)
    if True:
        if pcolor:
            plt.pcolormesh(yyF/degs2metr, zzF/1e3, avgfields.PsiRes[1:-1,:]/1e6, cmap = cmap, \
            norm=colors.CenteredNorm())
        else:
            plt.contourf(yyF/degs2metr, zzF/1e3, avgfields.PsiRes[1:-1,:]/1e6, cmap = cmap, \
            levels = lvls, norm=colors.CenteredNorm()) 
        plt.title(r'MITgcm $\Psi_{Resid}$ $(Sv)$')
    plt.xlim(lat_lims); plt.ylim(dep_lims); plt.ylabel("z (km)"); plt.colorbar()
    plt.contour(yyC/degs2metr,zzC/1e3,avgfields.ConvAdj[1:-1,:],'-w',levels=2)
    if channel: plt.plot(yyF[SOyind,0]*np.ones(N[1]+1)/degs2metr,zzF[SOyind,:]/1e3,'-k')

    plt.subplot(2,2,2)
    if pcolor:
        plt.pcolormesh(yyC/degs2metr, zzC/1e3, avgfields.b[1:-1,:], cmap = cmap, \
            norm=colors.PowerNorm(gamma=bgamma)) 
    else:
        plt.contourf(yyC/degs2metr, zzC/1e3, avgfields.b[1:-1,:], cmap = cmap, \
            levels = level_vector(), norm=colors.PowerNorm(gamma=bgamma))
    plt.title(r'MITgcm $\overline{b}$' + r' $(m/s^2)$')
    plt.xlim(lat_lims); plt.ylim(dep_lims); plt.colorbar(format=ticker.FuncFormatter(fmt_cb))
    plt.contour(yyC/degs2metr,zzC/1e3,avgfields.ConvAdj[1:-1,:],'-w',levels=2)
    if channel: plt.plot(yyF[SOyind,0]*np.ones(N[1]+1)/degs2metr,zzF[SOyind,:]/1e3,'-k')

    plt.subplot(2,2,3)
    if True:
        if pcolor:
            plt.pcolormesh(yyF/degs2metr, zzF/1e3, avgfields.PsiEuler[1:-1,:]/1e6, cmap = cmap, \
            norm=colors.CenteredNorm())
        else:
            plt.contourf(yyF/degs2metr, zzF/1e3, avgfields.PsiEuler[1:-1,:]/1e6, cmap = cmap, \
            levels = lvls, norm=colors.CenteredNorm()) 
        plt.title(r'MITgcm $\Psi_{Euler}$ $(Sv)$')
    plt.xlim(lat_lims); plt.ylim(dep_lims); plt.ylabel("z (km)"); plt.xlabel("y (deg)"); plt.colorbar()
    plt.contour(yyC/degs2metr,zzC/1e3,avgfields.ConvAdj[1:-1,:],'-w',levels=2)
    if channel: plt.plot(yyF[SOyind,0]*np.ones(N[1]+1)/degs2metr,zzF[SOyind,:]/1e3,'-k')

    plt.subplot(2,2,4)
    if True:
        if pcolor:
            plt.pcolormesh(yyF/degs2metr, zzF/1e3, avgfields.PsiEddie[1:-1,:]/1e6, cmap = cmap, \
            norm=colors.CenteredNorm())
        else:
            plt.contourf(yyF/degs2metr, zzF/1e3, avgfields.PsiEddie[1:-1,:]/1e6, cmap = cmap, \
            levels = lvls, norm=colors.CenteredNorm()) 
        plt.title(r'MITgcm $\Psi_{GM}$ $(Sv)$')
    plt.xlim(lat_lims); plt.ylim(dep_lims); plt.ylabel("z (km)"); plt.xlabel("y (deg)"); plt.colorbar()
    plt.contour(yyC/degs2metr,zzC/1e3,avgfields.ConvAdj[1:-1,:],'-w',levels=2)
    if channel: plt.plot(yyF[SOyind,0]*np.ones(N[1]+1)/degs2metr,zzF[SOyind,:]/1e3,'-k')
    plt.tight_layout(); plt.draw()
    plt.savefig('frame_mitgcm.png', dpi=200)

def plotcomp(b,Psi,t):
    # Basic plotting function for this simulation
    # plots Psi_residual, b and Psi_euler and Psi_GM in realtime 
    # plots line at channel-basin interface if desired. (at y=SOyind=12)
    # also saves plots to frame_XXXX.png in CWD, and creates mp4 animation if desired.
    lvls = np.linspace(Psi_lims[0],Psi_lims[1],51) # uses colorbar limits defined by user for Psi

    plt.figure(1)
    plt.clf()
    plt.subplot(2,2,1)
    if True:
        if pcolor:
            plt.pcolormesh(yyF/degs2metr, zzF/1e3, Psi/1e6, cmap = cmap, \
            norm=colors.CenteredNorm())
        else:
            plt.contourf(yyF/degs2metr, zzF/1e3, Psi/1e6, cmap = cmap, \
            levels = lvls, norm=colors.CenteredNorm()) 
        plt.title(r'$\Psi$ $(Sv)$ at t = {:.2f} years'.format(t/(365*days2secs)))
    plt.xlim(lat_lims); plt.ylim(dep_lims); plt.ylabel("z (km)"); plt.colorbar()
    if channel: plt.plot(yyF[SOyind,0]*np.ones(N[1]+1)/degs2metr,zzF[SOyind,:]/1e3,'-k')

    plt.subplot(2,2,2)
    if pcolor:
        plt.pcolormesh(yyC/degs2metr, zzC/1e3, b, cmap = cmap, \
            norm=colors.PowerNorm(gamma=bgamma)) 
    else:
        plt.contourf(yyC/degs2metr, zzC/1e3, b, cmap = cmap, \
            levels = level_vector(), norm=colors.PowerNorm(gamma=bgamma))
    plt.title(r'$\overline{b}$' + r' $(m/s^2)$ at t = {:.2f} years'.format(t/(365*days2secs)))
    plt.xlim(lat_lims); plt.ylim(dep_lims); plt.colorbar(format=ticker.FuncFormatter(fmt_cb))
    if channel: plt.plot(yyF[SOyind,0]*np.ones(N[1]+1)/degs2metr,zzF[SOyind,:]/1e3,'-k')

    plt.subplot(2,2,3)
    if True:
        if pcolor:
            plt.pcolormesh(yyF/degs2metr, zzF/1e3, PsiTW(b)/1e6, cmap = cmap, \
            norm=colors.CenteredNorm())
        else:
            plt.contourf(yyF/degs2metr, zzF/1e3, PsiTW(b)/1e6, cmap = cmap, \
            levels = lvls, norm=colors.CenteredNorm()) 
        plt.title(r'$\Psi_{TW}$'+r' $(Sv)$ at t = {:.2f} years'.format(t/(365*days2secs)))
    plt.xlim(lat_lims); plt.ylim(dep_lims); plt.ylabel("z (km)"); plt.xlabel("y (deg)"); plt.colorbar()
    if channel: plt.plot(yyF[SOyind,0]*np.ones(N[1]+1)/degs2metr,zzF[SOyind,:]/1e3,'-k')

    plt.subplot(2,2,4)
    if True:
        if pcolor:
            plt.pcolormesh(yyF/degs2metr, zzF/1e3, PsiGM(b)/1e6, cmap = cmap, \
            norm=colors.CenteredNorm())
        else:
            plt.contourf(yyF/degs2metr, zzF/1e3, PsiGM(b)/1e6, cmap = cmap, \
            levels = lvls, norm=colors.CenteredNorm()) 
        plt.title(r'$\Psi_{GM}$'+r' $(Sv)$ at t = {:.2f} years'.format(t/(365*days2secs)))
    plt.xlim(lat_lims); plt.ylim(dep_lims); plt.ylabel("z (km)"); plt.xlabel("y (deg)"); plt.colorbar() 
    if channel: plt.plot(yyF[SOyind,0]*np.ones(N[1]+1)/degs2metr,zzF[SOyind,:]/1e3,'-k')

    plt.tight_layout()
    plt.draw()
    #plt.pause(0.01)

    if mk_movie:
        plt.savefig('frame_{0:04d}.png'.format(int(cnt/npt), dpi=200))

def fmt_cb(x, pos):
    a, b = '{:.3e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def fmt(x):
    # formatting option for pltfancy function below. 
    # allows writing isopycnal values along contours  
    s = f"{x:.1e}"
    if s.endswith("0"):
        s = f"{x:.0e}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

def pltfancy(b,Psi,t):
    # Fancy plotting option for realtime computation or final frame.
    # Basic plotting of both Psi and b to use in realtime computation
    lvls = np.linspace(Psi_lims[0],Psi_lims[1],51) #100;
    plt.figure(1)

    plt.clf()
    if pcolor:
        plt.pcolormesh(yyF/degs2metr, zzF/1e3, Psi/1e6, cmap = cmap, \
            norm=colors.CenteredNorm())
    else:
        plt.contourf(yyF/degs2metr, zzF/1e3, Psi/1e6, cmap = cmap, \
            levels = lvls, norm=colors.CenteredNorm()) 
    plt.title(r'$\Psi$ $(Sv)$ at t = {:.2f} years'.format(t/(365*days2secs)))
    plt.xlim(lat_lims); plt.ylim(dep_lims); plt.ylabel("z (km)"); plt.xlabel("y (deg)");
    plt.colorbar() 
    CS = plt.contour(yyC/degs2metr, zzC/1e3, b, colors='black', levels=level_vector())
    plt.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
    if channel: plt.plot(yyF[SOyind,0]*np.ones(N[1]+1)/degs2metr,zzF[SOyind,:]/1e3,'-k')
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    if mk_movie:
        plt.savefig('frame_{0:04d}.png'.format(int(cnt/npt), dpi=200))

# BEGIN MAIN TIMESTEPPING CODE
#############################################################################
# Given a Psi, evolve b through adv-diff equation:
# ->  bflux = Psi_z*b_y - Psi_y*b_z + d/dz(kappa_z*b_z)
# Given new b, compute new Psi adv. from thermal wind and Gent-McWilliams treatment of baroclinic eddies
#  - first step Euler: b[1] = b[0] + dt*bflux
#  - 2nd step AB2: b[2] = b[1] + 0.5*dt*(3.*bflux[1]-bflux[0])
#  - 3rd+ step AB3: b[cnt] = b[cnt-1] + dt/12.*(23*bflux[cnt-1] - 16.*bflux[cnt-2] + 5.*bflux[cnt-3])

# initialize main figure for realtime plotting
plt.figure(1,figsize=fig_size) #(15.1,8.8))
#plt.figure(1,figsize=(10,6))

# Init timestep counter
cnt = 0

if pltmitgcm: plotmit() # plot the corresponding simulation from MITgcm, only once. saves to frame_mitgcm.png
# Initialize variables for time-stepping, apply bcs to b-field
b_nm3 = applybcs(b0)

# calculate advective Psi.
Psi_nm3 = update_Psi(b_nm3)

# print diagnostics
print("t = {:.3f} yrs, dt = {:.2f} days, CFL = {:.3f}, <b> = {:.3e}".format\
        (t[cnt]/(365*days2secs),dt/days2secs,check_CFL(Psi_nm3),total(b_nm3)))

# save to hdf5 output, just b and Psi.
if save2h5:
    output_b[0]   = b_nm3
    output_Psi[0] = Psi_nm3 

# plot ICs
if fancy: # plotting isopycnal contours on top of Psi contourf
    pltfancy(b_nm3,Psi_nm3,t[cnt])
else: # plotting strat alone, along with Psi components
    plotcomp(b_nm3,Psi_nm3,t[cnt])

# If only plotting Initial Condtions, print stats (for debugging) + exit
if just_ICs:
    print("Bot:   b[ :,-1] min/max = {:.2e} / {:.2e} (m/s^2)".format(np.min(b0[:,-1]),np.max(b0[:,-1])))
    print("Top:   b[ :, 0] min/max = {:.2e} / {:.2e} (m/s^2)".format(np.min(b0[:,0]),np.max(b0[:,0])))
    print("Nor:   b[-1, :] min/max = {:.2e} / {:.2e} (m/s^2)".format(np.min(b0[-1,:]),np.max(b0[-1,:])))
    print("Sou:   b[ 0, :] min/max = {:.2e} / {:.2e} (m/s^2)".format(np.min(b0[0,:]),np.max(b0[0,:])))
    print("Bot: Psi[ :,-1] min/max = {:.2e} / {:.2e} (Sv)".format(np.min((Psi_nm3*1e-6)[:,-1]),np.max((Psi_nm3*1e-6)[:,-1])))
    print("Top: Psi[ :, 0] min/max = {:.2e} / {:.2e} (Sv)".format(np.min((Psi_nm3*1e-6)[:,0]),np.max((Psi_nm3*1e-6)[:,0])))
    print("Nor: Psi[-1, :] min/max = {:.2e} / {:.2e} (Sv)".format(np.min((Psi_nm3*1e-6)[-1,:]),np.max((Psi_nm3*1e-6)[-1,:])))
    print("Sou: Psi[ 0, :] min/max = {:.2e} / {:.2e} (Sv)".format(np.min((Psi_nm3*1e-6)[0,:]),np.max((Psi_nm3*1e-6)[0,:])))
    plt.tight_layout(); plt.show(); sys.exit()

# First Step in time is Euler:
cnt   = 1
b_nm2 = b_nm3 + dt*bflux(b_nm3,Psi_nm3)
b_nm2 = applybcs(b_nm2) # surf Dirichlet enforced

if convectb: # if we choose to convect b à la Malta
    b_nm2 = convect(b_nm2) # convection also imposes bcs

Psi_nm2 = update_Psi(b_nm2)

if t[cnt]/(365*days2secs) % 10 == 0: 
    print("t = {:.3f} yrs, dt = {:.2f} days, CFL = {:.3f}, <b> = {:.3e}".format\
        (t[cnt]/(365*days2secs),dt/days2secs,check_CFL(Psi),total(b)))

# plot Euler step
if cnt % npt == 0:
    if fancy:
        pltfancy(b_nm2,Psi_nm2,t[cnt])
    else:
        plotcomp(b_nm2,Psi_nm2,t[cnt])

if save2h5 and cnt % n_h5 == 0:
    cnt_h5             = int(cnt/n_h5)
    output_b[cnt_h5]   = b_nm2
    output_Psi[cnt_h5] = Psi_nm2

# Second step in time is AB2:
cnt   = 2
b_nm1 = b_nm2 + dt*bflux(b_nm2,Psi_nm2) # Euler is desired
#b_nm1 = b_nm2 + 0.5*dt*(3.*bflux(b_nm2,Psi_nm2)-bflux(b_nm3,Psi_nm3)) # AB2 step
b_nm1 = applybcs(b_nm1)

if convectb: # if we choose to convect b à la Malta
    b_nm1 = convect(b_nm1) # convection also imposes bcs

Psi_nm1 = update_Psi(b_nm1)

if t[cnt]/(365*days2secs) % 10 == 0: 
    print("t = {:.3f} yrs, dt = {:.2f} days, CFL = {:.3f}, <b> = {:.3e}".format\
        (t[cnt]/(365*days2secs),dt/days2secs,check_CFL(Psi),total(b)))

# plot AB2 step
if cnt % npt == 0:
    if fancy:
        pltfancy(b_nm1,Psi_nm1,t[cnt])
    else:
        plotcomp(b_nm1,Psi_nm1,t[cnt])

if save2h5 and cnt % n_h5 == 0:
    cnt_h5             = int(cnt/n_h5)
    output_b[cnt_h5]   = b_nm1
    output_Psi[cnt_h5] = Psi_nm1


# main loop AB3 using nm3 nm2 nm1 variables (i.e at step n-3, n-2 and n-1 to find n)
for cnt in range(3,len(t)):

    # Other Options: Euler
    b = b_nm1 + dt*bflux(b_nm1,Psi_nm1)
    # Other Options: AB2
    #b = b_nm1 + 0.5*dt*(3.*bflux(b_nm1,Psi_nm1)-bflux(b_nm2,Psi_nm2))
    # Currently using AB3:
    #b = b_nm1 + dt/12.*(23*bflux(b_nm1,Psi_nm1) - 16.*bflux(b_nm2,Psi_nm2) + 5.*bflux(b_nm3,Psi_nm3))

    b = applybcs(b) # impose bcs

    if convectb: # if we choose to convect b
        b = convect(b)

    Psi = update_Psi(b) # update Psi
            
    if t[cnt]/(365*days2secs) % 10 == 0: 
        print("t = {:.3f} yrs, dt = {:.2f} days, CFL = {:.3f}, <b> = {:.3e}".format\
            (t[cnt]/(365*days2secs),dt/days2secs,check_CFL(Psi),total(b)))
    
    # plot and store AB3 step
    if cnt % npt == 0:
        if fancy:
            pltfancy(b,Psi,t[cnt])
        else:
            plotcomp(b,Psi,t[cnt])
    if save2h5 and cnt % n_h5 == 0:
        cnt_h5             = int(cnt/n_h5)
        output_b[cnt_h5]   = b
        output_Psi[cnt_h5] = Psi

    # Check for numerical instability, beyond just CFL condition which throws a warning
    if np.max(np.abs(b))>1e10 or np.isnan(b).any(): 
        raise(OverflowError("Solution Diverged."))
        sys.exit()

    # cycle solutions for next time-step, both in b and Psi
    b_nm3   = b_nm2
    b_nm2   = b_nm1
    b_nm1   = b
    Psi_nm3 = Psi_nm2
    Psi_nm2 = Psi_nm1
    Psi_nm1 = Psi

# Outside main loop, all solutions were stored to file. Close file and exit.
if save2h5:
    file.close()
    print("File closed")

# if you want to see last frame for debug/analysis, pause there.
if pauselast:
    plt.show()

# generate movie from all stored frame_XXXX.png files, in CWD
if mk_movie and not(just_ICs):
   makemovie(moviename)

if pltslice: plotslice(b);

# finish timer.
t_end = time.time()

# Print Simulation compute time.
print("Simulation took {:.3f} seconds (or {:.3f} hours)".format(t_end-t_start,(t_end-t_start)/(60*60)))
print("Done.")

