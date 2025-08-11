########################################################
# Tool-Set to deal with MITgcm output and post-process #
# By: Jonathan Tessier (for PhD at UQAR/ISMER 2022-26) #
########################################################

import numpy as np
import sys, h5py, subprocess
from MITgcmutils import mds
import matplotlib as mpl    
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import integrate,interpolate
from MITgcmutils.utils import writebin, readbin
np.set_printoptions(threshold=sys.maxsize)

# Change this depending on where you store your outputs
pathtocwd='/Users/jonathantessier/Workspace/sandbox/MITgcm-out/'

# some constants for your runs
days2secs = 24*60*60
degs2metr = 111177.475
tracerdt = 50400
tAlpha = 2e-4
sBeta  = 0
rho0   = 1035
tRef   = 10
g0     = 9.81

# Get grid from output
class load_grid:
    def __init__(self,datalabel):
        self.xC  = np.transpose(mds.rdmds(pathtocwd+datalabel+'/XC')) # C grid longitude [Nx,Ny] (deg)
        self.yC  = np.transpose(mds.rdmds(pathtocwd+datalabel+'/YC')) # C grid latitude  [Nx,Ny] (deg)
        self.xG  = np.transpose(mds.rdmds(pathtocwd+datalabel+'/XG')) # F grid longitude [Nx,Ny] (deg)
        self.yG  = np.transpose(mds.rdmds(pathtocwd+datalabel+'/YG')) # F grid latitude  [Nx,Ny] (deg)
        self.dxC = np.transpose(mds.rdmds(pathtocwd+datalabel+'/DXC')) # C grid zonal spacing [Nx,Ny] (m)
        self.dyC = np.transpose(mds.rdmds(pathtocwd+datalabel+'/DYC')) # C grid merid spacing [Nx,Ny] (m)
        self.dxG = np.transpose(mds.rdmds(pathtocwd+datalabel+'/DXG')) # F grid zonal spacing [Nx,Ny] (m) 
        self.dyG = np.transpose(mds.rdmds(pathtocwd+datalabel+'/DYG')) # F grid merid spacing [Nx,Ny] (m)
        self.dxF = np.transpose(mds.rdmds(pathtocwd+datalabel+'/DXF')) # F grid zonal spacing [Nx,Ny] (m) 
        self.dyF = np.transpose(mds.rdmds(pathtocwd+datalabel+'/DYF')) # F grid merid spacing [Nx,Ny] (m)
        self.dxV = np.transpose(mds.rdmds(pathtocwd+datalabel+'/DXV')) # F grid zonal spacing [Nx,Ny] (m) 
        self.dyU = np.transpose(mds.rdmds(pathtocwd+datalabel+'/DYU')) # F grid merid spacing [Nx,Ny] (m)
        self.rC  =   np.squeeze(mds.rdmds(pathtocwd+datalabel+'/RC')) # C grid depth [Nz] (m)
        self.rF  =   np.squeeze(mds.rdmds(pathtocwd+datalabel+'/RF')) # F grid depth [Nz+1] (m)
        self.drC =   np.squeeze(mds.rdmds(pathtocwd+datalabel+'/DRC')) # C grid vertical spacing [Nz+1] (m) sums to 4km
        self.drF =   np.squeeze(mds.rdmds(pathtocwd+datalabel+'/DRF')) # F grid vertical spacing [Nz] (m) sums to 4km
        self.rAC = np.transpose(mds.rdmds(pathtocwd+datalabel+'/RAC')) # Tracer(T) Horizontal Cell Area [Nx,Ny] (m^2)
        self.rAW = np.transpose(mds.rdmds(pathtocwd+datalabel+'/RAW')) # Uvelocity Horizontal Cell Area [Nx,Ny] (m^2) = RAC FOR CONST DY
        self.rAS = np.transpose(mds.rdmds(pathtocwd+datalabel+'/RAS')) # Vvelocity Horizontal Cell Area [Nx,Ny] (m^2) = RAZ FOR CONST DY
        self.rAZ = np.transpose(mds.rdmds(pathtocwd+datalabel+'/RAZ')) # Vorticity Horizontal Cell Area [Nx,Ny] (m^2)
        self.hFacC=np.transpose(mds.rdmds(pathtocwd+datalabel+'/hFacC'))# [0,1] vertical fraction of the depth cell used in Tracer    grid [Nx,Ny,Nz]
        self.hFacW=np.transpose(mds.rdmds(pathtocwd+datalabel+'/hFacW'))# [0,1] vertical fraction of the depth cell used in Uvelocity grid [Nx,Ny,Nz] 
        self.hFacS=np.transpose(mds.rdmds(pathtocwd+datalabel+'/hFacS'))# [0,1] vertical fraction of the depth cell used in Vvelocity grid [Nx,Ny,Nz] 
        self.depth=np.transpose(mds.rdmds(pathtocwd+datalabel+'/Depth'))# Depth of the domain...
        self.Nx, self.Ny, self.Nz = np.shape(self.hFacC)

        self.LxC = np.nansum(self.dxC[1:-1],axis=0) # accounting for walls at E/W, then ignoring x=0,60 deg long 
        self.LxF = np.nansum(self.dxG[1:-1],axis=0) # accounting for walls at E/W, then ignoring x=0,60 deg long
        
        self.LxC2D = np.repeat(self.LxC[:,np.newaxis],self.Nz,axis=1) # {Ny,Nz} tile 
        self.LxF2D = np.repeat(self.LxF[:,np.newaxis],self.Nz,axis=1) # (Ny,Nz) tile

        self.dxC3D = np.repeat(self.dxC[:,:,np.newaxis],self.Nz,axis=2)
        self.dyC3D = np.repeat(self.dyC[:,:,np.newaxis],self.Nz,axis=2)
        self.dxF3D = np.repeat(self.dxG[:,:,np.newaxis],self.Nz,axis=2)
        self.dyF3D = np.repeat(self.dyG[:,:,np.newaxis],self.Nz,axis=2)

        self.dzC3D = np.repeat(np.repeat(self.drC[np.newaxis,:],self.Ny,axis=0)[np.newaxis,:,:],self.Nx,axis=0)
        self.dzF3D = np.repeat(np.repeat(self.drF[np.newaxis,:],self.Ny,axis=0)[np.newaxis,:,:],self.Nx,axis=0)
        
        self.dyy,  self.dzz  = self.dyC3D[0,:,:], -self.dzC3D[0,:,:]
        self.dyyF, self.dzzF = self.dyF3D[0,:,:], -self.dzF3D[0,:,:]
        self.yy,   self.zz   = np.meshgrid(self.yC[0,:],self.rC,indexing='ij')
        self.yyF,  self.zzF  = np.meshgrid(self.yG[0,:],self.rF,indexing='ij')

#grid = load_grid(datalabel='basin-nowind')

class get_forcing:
    def __init__(self,datalabel):
        grid = load_grid(datalabel=datalabel)
        self.wind   = np.transpose(readbin(pathtocwd+datalabel+'/wind_x.bin',(grid.Ny,grid.Nx)))
        self.sst    = np.transpose(readbin(pathtocwd+datalabel+'/rbcs_T.bin',(grid.Nz,grid.Ny,grid.Nx))[0,:,:])
        self.ssb    = g0*tAlpha*(self.sst-np.nanmin(self.sst)) 

# Get fields from MITgcm
class get_fields:
    def __init__(self,datalabel,snap):
        print("")
        print("Fields taken at {:.2f} years".format(tracerdt*snap/(365*24*60*60)))
        print("")
        data = mds.rdmds(pathtocwd+datalabel+'/Diag4',snap)
        self.U       = np.transpose(data[0,:,:,:]) # Zonal Mass-Weighted Comp of Velocity (m/s)
        self.V       = np.transpose(data[1,:,:,:]) # Meridional Mass-Weighted Comp of Velocity (m/s)
        self.W       = np.transpose(data[2,:,:,:]) # Vertical Component of Velocity (r_units/s)
        self.T       = np.transpose(data[3,:,:,:]) # Potential Temperature (deg C)
        self.B       = g0*tAlpha*(self.T-np.nanmin(self.T)); 
        self.STRATIF = np.transpose(data[4,:,:,:]) # dsig/dz (kg/m3/z)
        self.GM_PsiX = np.transpose(data[6,:,:,:]) # GM Bolus transport stream-function : U component
        self.GM_PsiY = np.transpose(data[7,:,:,:]) # GM Bolus transport stream-function : V component
        self.CONVADJ = np.transpose(data[8,:,:,:]) # CONVECTIVE adjustment index.. 
        self.Nx, self.Ny, self.Nz = np.shape(self.U) # domain shape

# put nans in topo
def clean_data(fields,grid):
   fields.U[np.where(grid.hFacW==0)]       = np.nan
   fields.V[np.where(grid.hFacS==0)]       = np.nan
   fields.W[np.where(grid.hFacC==0)]       = np.nan
   fields.GM_PsiY[np.where(grid.hFacS==0)] = np.nan
   fields.STRATIF[np.where(grid.hFacC==0)] = np.nan
   fields.CONVADJ[np.where(grid.hFacC==0)] = np.nan
   fields.T[np.where(grid.hFacC==0)]       = np.nan

# Differential operator d2/dy2
def FD2y(Ny,y,dy): # build Dyy
    dy2 = dy**2 # square of dy
    Dyy = np.diag(1/dy2*np.ones(Ny),-1)   \
        + np.diag(-2/np.hstack([dy2[0],dy2])*np.ones(Ny+1),0) \
        + np.diag(1/dy2*np.ones(Ny),1) 
    Dyy[0,0:3] = np.array([1, -2, 1]/dy2[0])
    Dyy[Ny,Ny-2:Ny+1] = np.array([1, -2, 1]/dy2[-1])
    return Dyy

# Differential operator d2/dz2
def FD2z(Nz,z,dz): # built Dzz
    dz2 = dz**2 # square of dz
    Dzz = np.diag(1/dz2*np.ones(Nz),-1)   \
        + np.diag(-2/np.hstack([dz2[0],dz2])*np.ones(Nz+1),0) \
        + np.diag(1/dz2*np.ones(Nz),1) 
    Dzz[0,0:3] = np.array([1, -2, 1]/dz[0])
    Dzz[Nz,Nz-2:Nz+1] = np.array([1, -2, 1]/dz2[-1])
    return Dzz

def zC2zF(field,y,top=None,bottom=None,grid=None):
    # interpolate C-grid field to F-grid in z (vertical)
    # retains y coordinates, needs top/bottom defined
    interp = interpolate.RegularGridInterpolator((y, grid.rC), field)
    yy, zz = np.meshgrid(y,grid.rF[1:-1],indexing='ij')
    intfield = interp((yy,zz))
    return np.concatenate([np.zeros([field.shape[0],1]),intfield,np.zeros([field.shape[0],1])],axis=1)

############################################################3

# make zonal avg fields to use in model
class zonalavg_fields:
    def __init__(self,fields,grid): 
        T        = fields.T; T[np.where(grid.hFacC==0)]    = np.nan
        B        = g0*tAlpha*(fields.T);
        V_EU     = fields.V; V_EU[np.where(grid.hFacS==0)] = np.nan;
        V_EU     = np.concatenate([V_EU,np.nan*np.ones([grid.Nx,1,grid.Nz])],axis=1)
        U_EU     = fields.U; U_EU[np.where(grid.hFacW==0)] = np.nan;
        W_EU     = fields.W; W_EU[np.where(grid.hFacC==0)] = np.nan; 
        PsiGM    = fields.GM_PsiY; 
        PsiGM    = np.concatenate([PsiGM,np.zeros([grid.Nx,1,grid.Nz])],axis=1)
        PsiV     = np.zeros(np.shape(V_EU)) # init 3D quantity

        # to get zonally avg B gradient
        
        dBdy = np.diff(B[1:-1,1:-1,:],axis=1)/grid.dyC3D[1:-1,2:-1,:]
        self.by = np.nansum(dBdy*grid.dxC3D[1:-1,2:-1,:],axis=0)#/grid.LxF2D

        V_GM = np.zeros(np.shape(V_EU))
        for kk in range(grid.Nz-1):
            V_GM[:,:,kk] = -(PsiGM[:,:,kk]-PsiGM[:,:,kk+1])/grid.drF[kk]
        V_GM[:,:,-1] = -PsiGM[:,:,-1]/grid.drF[-1]
        V_GM[np.where(grid.hFacS==0)] = np.nan

        V_res = V_EU + V_GM
        V_res[np.where(grid.hFacS==0)] = np.nan

        dA = np.zeros(np.shape(fields.V)); dx = grid.dxG
        for kk in range(grid.Nz): # calculate areas for merid flux
            dA[:,:,kk] = dx*grid.drF[kk]

        #plt.figure(9); plt.contourf(grid.dxC3D[:,:,0]); plt.colorbar(); plt.draw()

        Psi3D = np.zeros(np.shape(V_EU)) # init 3D quantity = -int v dz
        PsiEuler3D = np.zeros(np.shape(V_EU)) # init 3D quantity = -int v dz
        PsiGM3D = np.zeros(np.shape(V_EU)) # init 3D quantity = -int v dz

        # should be all on [yF,zF] grid...
        for jj in range(grid.Ny): # all latitudes (y)
            for ii in range(grid.Nx): # all longitudes (x)
                Psi3D[ii,jj,:] = np.nancumsum(V_res[ii,jj,:]*dA[ii,jj,:]) 
                PsiEuler3D[ii,jj,:] = np.nancumsum(V_EU[ii,jj,:]*dA[ii,jj,:])
                PsiGM3D[ii,jj,:] = np.nancumsum(V_GM[ii,jj,:]*dA[ii,jj,:])

        y = np.append(grid.yG[0],-grid.yG[0,0])
        self.ConvAdj  = np.nansum(fields.CONVADJ,axis=0)
        self.PsiRes   = np.nansum(Psi3D,axis=0); # add row of zeroes above... instead of interpolating...!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.PsiRes   = np.concatenate([np.zeros([grid.Ny+1,1]),self.PsiRes],axis=1); self.PsiRes[:,-1] = 0
        self.PsiEuler = np.nansum(PsiEuler3D,axis=0)
        self.PsiEuler = np.concatenate([np.zeros([grid.Ny+1,1]),self.PsiEuler],axis=1); self.PsiEuler[:,-1] = 0
        self.PsiEddie = np.nansum(PsiGM3D,axis=0)
        self.PsiEddie = np.concatenate([np.zeros([grid.Ny+1,1]),self.PsiEddie],axis=1); self.PsiEddie[:,-1] = 0

        self.b        = np.nansum((B*grid.dxC3D)[1:-1,:,:],axis=0)/grid.LxC2D; 
        self.b[0,:] = np.nan; self.b[-1,:] = np.nan # ignored x=0,60 in channel for uniform Lx(y), also N/S edges
        self.t        = np.nansum((T*grid.dxC3D)[1:-1,:,:],axis=0)/grid.LxC2D; 
        self.t[0,:] = np.nan; self.t[-1,:] = np.nan # ignored x=0,60 in channel for uniform Lx(y), also N/S edges
        self.u        = np.nansum((U_EU*grid.dxC3D)[1:-1,:,:],axis=0)/grid.LxC2D
        self.v        = np.nansum((V_EU[:,:-1,:]*grid.dxF3D)[1:-1,:,:],axis=0)/grid.LxF2D
        self.w        = np.nansum((W_EU*grid.dxC3D)[1:-1,:,:],axis=0)/grid.LxC2D

        #self.PsiEuler[0,:] = 0;  self.PsiEuler[-1,:] = 0;  self.PsiEuler[:,0] = 0;  self.PsiEuler[:,-1] = 0;
        #self.PsiEddie[0,:] = 0;  self.PsiEddie[-1,:] = 0;  self.PsiEddie[:,0] = 0;  self.PsiEddie[:,-1] = 0;

def makemovie(moviename=None):
    if moviename is None:
        moviename = 'movie.mp4'
    f_log = open("ffmpeg.log", "w")
    f_err = open("ffmpeg.err", "w")
    cmd_gen = ['ffmpeg', '-framerate', str(12), '-i', 'frame_%04d.png', '-y', 
            '-q', '1', '-threads', '0', '-pix_fmt', 'yuv420p', moviename]
    subprocess.call(cmd_gen, stdout=subprocess.PIPE)

# EOF
