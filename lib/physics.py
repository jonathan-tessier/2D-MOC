# physics
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import copy as cp
from scipy import interpolate
from lib.grid import intdzCF, intdyCF, zF2zC, yF2yC
np.seterr(divide='ignore', invalid='ignore')

# Set up parameters
class physical_parameters(object):

    def __init__(
            self,
            grid, 
            # Default parameters (can be editted via driver script: 

            # Useful Constants
            days2secs = 24*60*60,    # converts days to seconds

            # Constant physical parameters
            omega    = 7.2921e-5,    # Earth Rotation frequency (1/s)
            rho0     = 1035,         # typical density (kg/m^3) 
            tAlpha   = 2e-4,         # thermal expansion coeff (for T -> B transformation)
            g0       = 9.81,         # gravity (for T -> B transformation))

            # wind forcing over channel
            tau0     = 0.1,          # strength of wind in channel (N^2/m) sin^2(theta) shape
            mit_tau  = False,        # use wind-stress from generated binary for MITgcm run, wind_x.bin?

            # SST parameters
            restoring = True,        # surface restoring (Imposed profile - Dirichlet - if False)
            t_rest    = 30*86400.,   # restoring timescale for surface.. 1 day = 86400
            sstref   = 0.1,          # SST at southern edge (T_S), (mostly b > 0 with convection scheme)
            offset   = 0.1,          # SST (deg C) at northern edge (T_N), Southern edge is sstref.
            SSTmax   = 30,           # max SST (deg C) at equator (cos^2 profile)
            warm_SST = 0,            # deg sudden surface warming (set to 0 if no sudden warming)
            warm_dt  = 1e2,          # years before sudden warming (to allow equil)
            mit_SST  = False,        # use SST from generated binary for MITgcm run, rbcs_T.bin?

            # physics: convection
            convectb = True,         # include convection param in evolution
            gamma    = 1e-8,         # target lapse rate.. dT/dz diag. from MITgcm runs... (1e-5)
            geps     = 1e-15,        # error factor for convection convergence (convects if dT/dz < gamma - geps, imposed gamma   
            
            # physics: diffusion
            vertdiff = True,         # include vertical diffusion
            kapzprof = True,         # use non-uniform vertical diffusion, requires vertdiff=True
            kbg      = 0   ,         # background diffusivity (m^2/s) if kapzprof = True
            kb       = 2e-4,         # ###### bottom diffusivity value (m^2/s)
            zk       = 1250,         # diffusivity e-folding scale (m) kappa_z = kbg + (kb - kbg)*np.exp(-(zF-zF.min())/zk)
            horzdiff = False,        # include uniform horizontal diffusion
            kappa_y  = 1e2,          # horizontal buoyancy diffusivity (m^2/s) (typical 1e2)
            
            # physics: advection (components are cumulative)
            usePsiGM = True,         # use PsiGM in advection (Gent McWilliams barocolinic eddies) 
            usePsiTW = True,         # use Thermal Wind in advection (Psi_zzy = -1/f b_y)
            
            # physics: Southern Channel (includes barotropic Psi_tau, no thermal wind in channel)
            matchSO  = True ,        # lin. correction to basin Psi_TW to match Psi_Res at channel interface, otherwise, goes to 0.
            free_TW  = False,        # don't apply linear correction to basin, let Psi_TW integrate to whatever. overides matchSO 
            
            # physics: GM slope clipping (if all false, full slope returns)
            kappaGM  = 1e3   ,       # Gent-McWilliams eddy diff. (m^2/s) (typical 1e3)
            Smax     = 1e-2  ,       # max allowed slope by/bz in PsiGM taper (typical 1e-2)

            # coriolis - equator boudary layer eps etc
            eps = 1e-6,              # 1/f = f/(f^2+eps^2) to avoid 1/f singularity at equator...

    ):
        
        """
        Physical Parameters
        """
 
        # Useful Constants
        self.days2secs = days2secs
            
        # Constant physical parameters
        self.omega    = omega        
        self.rho0     = rho0        
        self.tAlpha   = tAlpha         
        self.g0       = g0      
        self.tau0     = tau0

        # SST parameters
        self.restoring= restoring
        self.t_rest   = t_rest
        self.sstref   = sstref
        self.offset   = offset           
        self.SSTmax   = SSTmax        
        self.warm_SST = warm_SST           
        self.warm_dt  = warm_dt  

        # physics: convection
        self.convectb = convectb      
        self.gamma    = gamma 
        self.geps     = geps

        # physics: diffusion
        self.vertdiff = vertdiff 
        self.kapzprof = kapzprof
        self.kbg      = kbg
        self.kb       = kb  
        self.zk       = zk    
        
        if self.kapzprof: # non-uniform vertical diffusion, if to be used
            self.kappa_z = self.kbg + (self.kb - self.kbg)*np.exp(-(grid.zF-grid.zF.min())/self.zk)
        else: # constant diff profile with kbg
            self.kappa_z = self.kbg

        self.horzdiff = horzdiff  
        self.kappa_y  = kappa_y
        
        # physics: advection (components are cumulative)
        self.usePsiGM = usePsiGM
        self.usePsiTW = usePsiTW
        
        # physics: Southern Channel (includes barotropic Psi_tau, no thermal wind in channel)
        self.channel  = grid.channel
        self.SOyind   = grid.SOyind
        self.matchSO  = matchSO 
        self.free_TW  = free_TW 
        
        # physics: GM slope clipping (if all false, full slope returns)
        self.kappaGM  = kappaGM
        self.Smax     = Smax

        # coriolis
        self.f = 2*omega*np.sin(np.deg2rad(grid.yyF/grid.degs2metr))
        self.fC = 2*omega*np.sin(np.deg2rad(grid.yC/grid.degs2metr))
        self.finv = self.f/(self.f**2+eps**2)
    
        # use mitcgm winds or not
        self.mit_tau = mit_tau
        self.mit_SST = mit_SST

def applybcs(soln,phys,t,dt):
    # surface boundary condition
    
    if t<phys.warm_dt*365*phys.days2secs:
        b_s = cp.copy(soln.b0[ :,0])
    else: # sudden surface warming (e.g. +3 deg C after 1000 yrs)
        b_s = cp.copy(soln.b0[ :,0]) + phys.g0*phys.tAlpha*phys.warm_SST

    if phys.restoring: # if nudging to profile with spec timescale
        soln.b[:,0] += dt/phys.t_rest*(soln.b0[:,0] - soln.b[:,0]) 
    else: # otherwise, dirichlet...
        soln.b[:,0]  = b_s

def convect(b,phys,grid):
    # --- column-wise convection parameterization ---
    # conserative scheme of Akmaev (1991, MWR)
    theta_arr = cp.copy(b)/(phys.g0*phys.tAlpha) # from current startification, get temperature [Ny,Nz].
    for yind in range(grid.N[0]):
        cycle = 0
        theta = theta_arr[yind,:]
        #print('yi = '+str(yind))
        while np.min(np.diff(theta)/grid.dzC)<phys.gamma-phys.geps: 
            theta = akmaev_column(theta,phys,grid)
            cycle += 1
            if cycle > 10000:
                plt.figure(9)
                plt.plot(theta,grid.zC/1e3,'-k'); plt.grid(True)
                plt.xlabel("T (deg. C)"); plt.ylabel("z (km)")
                plt.title("y = "+str(grid.yC[yind]/grid.degs2metr)+" deg. lat.")
                plt.tight_layout(); plt.show()
                raise(RuntimeError("Convection didn't converge after 10000 iters"))
        theta_arr[yind,:] = theta
    return theta_arr*phys.g0*phys.tAlpha

def akmaev_column(T,phys,grid):
    # in single column, convect conserving b and imposing unif background strat gamma.

    theta = T - phys.gamma*grid.zC
    #Energy_init = np.sum(theta*grid.dzF)
    q = abs(grid.dzF) #/beta #4.2e3/g0* # heat capacity of the layers. [Nz]
    theta_k=np.empty(grid.N[1]) # theta_k is a temperature vector where convective layers have been joined
    n_k    =np.empty(grid.N[1],dtype='int') # number of convectively joined layers in each k-layer
    s_k    =np.empty(grid.N[1]) # total effective heat capactity in each k-layer
    t_k    =np.empty(grid.N[1]) # total energy in each k-layer
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
        if l == grid.N[1]:  # the scan is over
            break  # to step 8
        l += 1
        n_k[k-1] = n #print(type(n_k[k-1]))
        theta_k[k-1] = thistheta
        # back to step 2

    # update the potential temperatures
    while True: # This is effectively a loop over k (i.e. the unified layers)
        # Akmaev step 8
        while n>1:
            # Set all potential temperatures in this convective layer to thistheta
            while True:
                #  Akmaev step 9
                theta[l-1] = thistheta  
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
    #Energy_final = np.sum(theta*grid.dzF)
    #print("T*dz Mass loss: "+str(Energy_init - Energy_final))     
    return theta + phys.gamma*grid.zC 

def PsiGM(soln,phys,grid):
    # take in strat b with size: [Ny,Nz] on [C,C]-grid
    # returns PsiGM = kappaGM*s(b) with size: [Ny+1,Nz+1] on [F,F]-grid
    # PsiGM boundary conditions set to zero all around. Interior uses 
    # average of c-grid points to project onto F grid for differencing

    # compute gradient components, falling on velocity points on grid
    b_y = np.diff(soln.b,axis=0)/grid.dyyCC # v-pts [Ny-1,Nz] missing north/south
    b_z = np.diff(soln.b,axis=1)/grid.dzzCC # w-pts [Ny,Nz-1] missing top/bottom
    # avg dbdy in z and dbdz in y for it to fall on Psi point [using 4 points in b for the Psi in the middle]   
    dbdy = (b_y[:,:-1]*grid.dzzF[:-1,:-1] + b_y[:,1:]*grid.dzzF[1:,1:]) / (grid.dzzF[:-1,:-1] + grid.dzzF[1:,1:]) # [Ny-1,Nz-1]
    dbdz = (b_z[:-1,:]*grid.dyyF[:-1,:-1] + b_z[1:,:]*grid.dyyF[1:,1:]) / (grid.dyyF[:-1,:-1] + grid.dyyF[1:,1:]) # [Ny-1,Nz-1]

    # define slope at the ratio of the derivatives
    S = dbdy/dbdz # [Ny-1,Nz-1] on [F,F] grid, missing all boundary conditions
   
    # slope limiting scheme, slope clipping.
    S = np.nan_to_num(S, neginf = -phys.Smax, posinf = phys.Smax)
    S = np.where(abs(S) > phys.Smax, np.sign(S)*phys.Smax, S)

    # then define Psi GM with tappered slope 
    Psi_GM = -phys.kappaGM*S # [Ny-1,Nz-1]
    # pad with zeros to close circulation at all boundaries
    Psi_GM = grid.LxF*np.pad(Psi_GM[:,:],pad_width=1,mode='constant') # now [Ny+1,Nz+1], [F,F] grid

    return Psi_GM

def ddykappaby(b,phys,grid):
    # horizontal component of diffusion laplacian, returns d/dy(kappa_y*b_y)
    # dbdy is on v points, missing edge values, with size [Ny-1,Nz]
    bp = np.pad(b,pad_width=1,mode='edge')[:,1:-1] # pad with ghost layer in y
    dbdy = np.diff(bp,axis=0)/grid.dyyCG # [Ny+1,Nz] south/north values zero
    d2bdy2 = np.diff(phys.kappa_y*dbdy,axis=0)/grid.dyyF # [Ny,Nz] on b grid
    return d2bdy2

def ddzkappabz(b,phys,grid):
    # vertical component of diffusion laplacian, returns d/dz(kappa_z*b_z)
    # dbdz is on w points, missing edge values, with size [Ny,Nz-1]
    bp = np.pad(b,pad_width=1,mode='edge')[1:-1,:] # pad with ghost layer in z
    dbdz = np.diff(bp,axis=1)/grid.dzzCG # [Ny,Nz+1] surf/bott values are zero
    d2bdz2 = np.diff(phys.kappa_z*dbdz,axis=1)/grid.dzzF # [Ny,Nz] on b grid
    return d2bdz2

def PsiTW(soln,phys,grid):
    # from the stratification (b), calculate the local streamfunction defined by:
    # Thermal Wind -> Psi_zzy = -1/f b_y. Returns the streamfunction, Psi

    # calculate db/dy, need ghost points to deal with boundary conditions
    bp = np.pad(soln.b,pad_width=1,mode='edge')[:,1:-1] # [Ny+2,Nz] on [yC,zC] filled with ghost
    dbdy = np.diff(bp,axis=0)/grid.dyyCG # [Ny+1,Nz] on [yF,zC] grid, N/S are zero
    # calculate thermal wind u = -1/f int_-H^z db/dy dz
    u_therm = -phys.finv*intdzCF(dbdy,grid.dzzFF) # [Ny+1, Nz+1] on [yF,zF] grid, int from zero
    u_therm = zF2zC(u_therm,grid.yF,grid) # interp u onto [Ny+1,Nz] on [yF,zC]

    # now integrate u_thermal from F-grid to F-grid...
    Cz = -1/grid.LF[1]*intdzCF(u_therm,grid.dzzFF,reverse=False)[:,-1]
    Cz  = np.repeat((Cz)[:,np.newaxis],grid.N[1],axis=1) # [Ny+1,Nz]

    u   = u_therm + Cz # full u solution # [Ny+1, Nz+1] on [yF,zF] grid
    u = yF2yC(u,grid.zC,grid) # interp u onto [Ny,Nz] on [yC,zC]

    Psi_thermal = intdyCF(intdzCF(u,grid.dzzF),grid.dyyFF) # in m3/s

    if phys.free_TW: # don't correct 
        if phys.channel: 
            Psi_thermal[:phys.SOyind,:] = (soln.PsiSO)[:phys.SOyind,:] # if channel, swap channel Psi_thermal with Psi_Ekman
        return Psi_thermal 

    if phys.channel: # if include southern channel

        if phys.matchSO: # construct Psi0
            lincorrboundary = np.repeat((soln.PsiSO-Psi_thermal)[phys.SOyind,:][np.newaxis,:],grid.N[0]+1,axis=0) 
        else:
            lincorrboundary = np.repeat((-Psi_thermal)[phys.SOyind,:][np.newaxis,:],grid.N[0]+1,axis=0) # just close thermal
        thetaF = np.deg2rad(grid.yyF/grid.degs2metr) # actual latitude
        linfunc = (np.sin(thetaF) - np.sin(thetaF.max()))/(np.sin(thetaF[phys.SOyind,0]) - np.sin(thetaF.max())) # linear corr to Basin...
        linfunc[:phys.SOyind,:] = 1
        lincorr = lincorrboundary*linfunc

        # Apply Psi0
        Psi_thermal = Psi_thermal + lincorr 
        # Apply wind in channel
        Psi_thermal[:phys.SOyind,:] = (soln.PsiSO)[:phys.SOyind,:]; # remove thermal wind in channel, replace with ekman

    else: # otherwise closed basin
        lincorrboundary = np.repeat((-Psi_thermal)[0,:][np.newaxis,:],grid.N[0]+1,axis=0)      
        thetaF = np.deg2rad(grid.yyF/grid.degs2metr) # actual latitude
        linfunc = (np.sin(thetaF) - np.sin(thetaF.max()))/(np.sin(thetaF[0,0]) - np.sin(thetaF.max())) # linear corr to Basin...
        lincorr = lincorrboundary*linfunc
        Psi_thermal = Psi_thermal + lincorr

    return Psi_thermal

