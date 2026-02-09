# grid
from MITgcmutils import mds
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Set up grid parameters        
class grid_parameters(object):

    def __init__(
            self,
            channel   = True, # channel in topography? or closed basin (False)
            mitgrid   = True, # use grid from MITgcm simulation
            rm_topo   = True, # differ between masked and unmasked grids remove N/S walls
            datalabel = 'path/to/3D/simulation', # label of 3D simulation
            dtheta    = 2, # default horizontal resolution (latitude, deg)
            lat_lims  = [-70,70], # latitude limits of domain (F-grid)
            dlambda   = 2, # default horizontal resolution (longitude, deg)
            Nlambda   = 30, # num points in longitude (Llamb = dlamb*Nlamb)
            R         = 6370e3, # Earth's radius (m) 
            #SOyind    = 12, # index of latitude vector for channel interface
            chan_int  = -46, # latitude of channel interface
            delR      = np.array([ 37.,  40.,  44.,  48.,  52.,  56.,  60.,  63.,  66.,  69.,  72.,\
                                   75.,  78.,  81.,  84.,  87.,  90.,  93.,  96.,  99., 102., 105.,\
                                  108., 111., 114., 117., 120., 123., 126., 129., 132., 135., 138.,\
                                  141., 144., 147., 150., 153., 156., 159.]) # layer thicknesses (m) 
    ):

        """
        Grid Parameters
        """
        self.mitgrid   = mitgrid
        self.rm_topo   = rm_topo
        self.channel   = channel
        self.datalabel = datalabel

        if mitgrid: # typical C-grid stagger, non-uniform in depth
            print('\nUsing extracted MITgcm grid')
            # from 3D MIT resolution, calculate latitude -> meter conversion in spherical
            self.degs2metr = mds.rdmds(datalabel+'/DYC')[0,0]/dtheta
            self.yF = np.transpose(mds.rdmds(datalabel+'/YG'))[0]*self.degs2metr
            self.yF = np.hstack([self.yF,-self.yF[0]])
            self.yC = np.transpose(mds.rdmds(datalabel+'/YC'))[0]*self.degs2metr
            self.zF = np.squeeze(mds.rdmds(datalabel+'/RF'))
            self.zC = np.squeeze(mds.rdmds(datalabel+'/RC'))
            # calculate zonal width of basin
            dxC = np.transpose(mds.rdmds(datalabel+'/DXC')) # [Nx,Ny]
            self.LxC = np.nansum(dxC[1:-1],axis=0) # accounting for walls at E/W,
            self.LxC = np.repeat(self.LxC[:,np.newaxis],len(self.zC),axis=1)
            dxF = np.transpose(mds.rdmds(datalabel+'/DXG')) # [Nx,Ny]
            self.LxF = np.nansum(dxF[1:-1],axis=0) # accounting for walls at E/W, 
            self.LxF = np.repeat(self.LxF[:,np.newaxis],len(self.zC),axis=1)
            self.LxC    = np.concatenate([self.LxC,self.LxC[:,1].reshape([len(self.yC),1])],axis=1) # [Ny+1,Nz+1] for y,z C-grid
            self.LxF    = np.concatenate([self.LxF,self.LxF[0,:].reshape([1,len(self.zC)])],axis=0) # fix missiing boundary value
            self.LxF    = np.concatenate([self.LxF,self.LxF[:,-1].reshape([len(self.yC)+1,1])],axis=1) # [Ny+1,Nz+1] for y,z F-grid
        else:
            print('Using user-defined grid')
            self.degs2metr = (R*np.pi/180); #print(self.degs2metr)
            self.yF = np.arange(lat_lims[0], lat_lims[1]+dtheta, dtheta)*self.degs2metr
            self.yC = np.arange(lat_lims[0] + dtheta/2, lat_lims[1]+dtheta/2, dtheta)*self.degs2metr

            Nr = len(delR);
            # Compute interface depths (zF)
            self.zF = np.zeros(Nr + 1)
            for k in range(1, Nr + 1):
                self.zF[k] = self.zF[k - 1] - delR[k - 1]
            # Compute cell-center depths (zC)
            self.zC = 0.5 * (self.zF[:-1] + self.zF[1:])

            # zonal domain width (meters) on C/F-grid ...
            self.dxC = dlambda*np.pi/180*R*np.cos(np.pi*(self.yC/self.degs2metr)/180)
            self.dxF = dlambda*np.pi/180*R*np.cos(np.pi*(self.yF/self.degs2metr)/180)

            self.LxC = (Nlambda-2)*self.dxC # accounting for walls at E/W,
            self.LxC = np.repeat(self.LxC[:,np.newaxis],len(self.zF),axis=1)

            self.LxF = (Nlambda-2)*self.dxF # accounting for walls at E/W,  
            self.LxF = np.repeat(self.LxF[:,np.newaxis],len(self.zF),axis=1)

        self.SOyind = np.where(self.yF/self.degs2metr > chan_int)[0][0]; #print(self.SOyind)

        self.N  = [len(self.yC),len(self.zC)]  
        self.dyF = np.diff(self.yF)
        self.dyC = np.diff(self.yC)
        self.dzF = np.diff(self.zF)
        self.dzC = np.diff(self.zC)
        self.LF  = [np.max(self.yF)-np.min(self.yF),np.min(self.zF)-np.max(self.zF)]
        self.LC  = [np.max(self.yC)-np.min(self.yC),np.min(self.zC)-np.max(self.zC)]

        # print stats
        print("")
        print("Points in C-grid (lat/z)  N = [{},{}]".format(self.N[0],self.N[1]))
        print("F-Grid Latitude Domain (deg): [{:.2f}, {:.2f}]".format(self.yF.min()/self.degs2metr,self.yF.max()/self.degs2metr))
        print("C-Grid Latitude Domain (deg): [{:.2f}, {:.2f}]".format(self.yC.min()/self.degs2metr,self.yC.max()/self.degs2metr))
        print("F-Grid    Depth Domain  (m):  [{:.2f}, {:.2f}]".format(self.zF.min(),self.zF.max()))
        print("C-Grid    Depth Domain  (m):  [{:.2f}, {:.2f}]".format(self.zC.min(),self.zC.max()))
        print("")
        
        # if pulling fields from MITgcm output, southern and northern boundaries are walls, contain nans
        if rm_topo: # remove north and south walls 
            self.yC  = self.yC[1:-1];    self.yF  = self.yF[1:-1];
            self.dyC = self.dyC[1:-1];   self.dyF = self.dyF[1:-1]
            self.LxC = self.LxC[1:-1,:]; self.LxF = self.LxF[1:-1,:]
            self.N[0] -= 2;      self.SOyind = self.SOyind-1
        
        # domain grids: C grid center, F grid faces 
        self.yyC ,  self.zzC = np.meshgrid(self.yC,self.zC,indexing='ij') 
        self.dyyC, self.dzzC = np.meshgrid(self.dyC,self.dzC,indexing='ij') 
        self.yyF ,  self.zzF = np.meshgrid(self.yF,self.zF,indexing='ij')
        self.dyyF, self.dzzF = np.meshgrid(self.dyF,self.dzF,indexing='ij') 
        
        # v/w points
        self.yyV ,  self.zzV = np.meshgrid(self.yF,self.zC,indexing='ij')         # [Ny+1,Nz]
        self.dyyV, self.dzzV = np.diff(self.yyV,axis=0), np.diff(self.zzV,axis=1) # [Ny,Nz] and [Ny+1,Nz-1]
        self.yyW ,  self.zzW = np.meshgrid(self.yC,self.zF,indexing='ij')         # [Ny, Nz+1]
        self.dyyW, self.dzzW = np.diff(self.yyW,axis=0), np.diff(self.zzW,axis=1) # [Ny-1,Nz+1] and [Ny,Nz]
        
        # need to add a column in y to match v dimensions: [Ny+1,Nz] dzV
        self.dzzFF = np.concatenate([self.dzzF,self.dzzF[-1,:].reshape([1,self.dzzF.shape[1]])],axis=0)
        # need to add a column in z to match w dimensions: [Ny,Nz+1] dyW
        self.dyyFF = np.concatenate([self.dyyF,self.dyyF[:,-1].reshape([self.dyyF.shape[0],1])],axis=1)
        # need to add a column in y to match b dimensions: [Ny+1,Nz]
        self.dzzCC = np.concatenate([self.dzzC,self.dzzC[-1,:].reshape([1,self.dzzC.shape[1]])],axis=0)
        # need to add a column in z to match w dimensions: [Ny,Nz+1]
        self.dyyCC = np.concatenate([self.dyyC,self.dyyC[:,-1].reshape([self.dyyC.shape[0],1])],axis=1)
        
        # Ghost layer 
        self.dyyCG = np.pad(self.dyyCC,pad_width=1,mode='edge')[:,1:-1] # [Ny+1,Nz]
        self.dzzCG = np.pad(self.dzzCC,pad_width=1,mode='edge')[1:-1,:] # [Ny,Nz+1]
        
        # pritn stats
        print("and with topography cropped out")
        print(" yC,  zC shape: "+str(self.yyC.shape))
        print("dyC, dzC shape: "+str(self.dyyC.shape))
        print(" yF,  zF shape: "+str(self.yyF.shape))
        print("dyF, dzF shape: "+str(self.dyyF.shape))
        print("")

        # print channel location
        if channel:
            print("Channel interface at y = {:.2f} degs lat (index = {})".format(self.yyF[self.SOyind,0]/self.degs2metr,self.SOyind))
            print("")
        else:
            print("Closed basin, no channel")

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

def zF2zC(field,y,grid):
    # interpolate F-grid field to C-grid in z (vertical)
    # retains y coordinates, no bcs required, C grid inside F grid
    interp = interpolate.RegularGridInterpolator((y, grid.zF), field)
    yy, zz = np.meshgrid(y,grid.zC,indexing='ij')
    return interp((yy,zz))

def yF2yC(field,z,grid):
    # interpolate F-grid field to C-grid in y (horizontal)
    # retains z coordinates, no bcs required, C grid inside F grid
    interp = interpolate.RegularGridInterpolator((grid.yF, z), field)
    yy, zz = np.meshgrid(grid.yC,z,indexing='ij')
    return interp((yy,zz))

def zC2zF(field,y,grid,top=None,bottom=None):
    # interpolate C-grid field to F-grid in z (vertical)
    # retains y coordinates, needs top/bottom defined
    interp = interpolate.RegularGridInterpolator((y, grid.zC), field)
    yy, zz = np.meshgrid(y,grid.zF[1:-1],indexing='ij')
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

def yC2yF(field,z,grid,south=None,north=None):
    # interpolate C-grid field to F-grid in y (horizontal)
    # retains z coordinates, needs north/south defined
    if len(z)>1: # If field is in fact 2D in [y,z]
        interp = interpolate.RegularGridInterpolator((grid.yC, z), field)
        yy, zz = np.meshgrid(grid.yF[1:-1],z,indexing='ij')
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
        intfield = np.interp(grid.yF[1:-1],grid.yC,field)
        if np.any(south): 
            intfield = np.concatenate([south,intfield])
        else:
            intfield = np.concatenate([[0],intfield])
        if np.any(north):
            intfield = np.concatenate([intfield,north])
        else:
            intfield = np.concatenate([intfield,[0]])

    return intfield

