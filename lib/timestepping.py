# time stepping
import sys
import numpy as np
from scipy import interpolate
import copy as cp
import matplotlib.pyplot as plt
from lib.physics import PsiGM, PsiTW, ddykappaby, ddzkappabz, convect, applybcs 
from lib.plotting import plotmit, plotcomp, plotfancy, makemovie 
from scipy.sparse.linalg import spsolve

days2secs = 24*60*60

class temporal_parameters(object):

    def __init__(
            self,
            
            t0    = 0 * 365*days2secs, 
            tf    = 1e4 * 365*days2secs, 
            dt    = 5e-3* 365*days2secs, 
    ):
        # Useful Constants
        self.t0 = t0
        self.tf = tf
        self.dt = dt
        self.t = np.arange(self.t0,self.tf+self.dt,self.dt) 

def update_Psi(soln,phys,grid):
    # Wrapper function to select what flows are included for advection
    # Init Psi on [F,F] grid, [Ny+1,Nz+1]

    soln.Psi = np.zeros_like(soln.Psi)
    if phys.usePsiGM: # include baroclinic eddies (GMredi param, advective form)
        soln.PsiGM = PsiGM(soln,phys,grid) # compute using strat
        soln.Psi += soln.PsiGM # include PsiGM in advection 
    if phys.usePsiTW: # include Thermal wind (eulerian circulation)
        soln.PsiTW = PsiTW(soln,phys,grid) # compute using start
        soln.Psi += soln.PsiTW # include Thermal Wind in advection 
            # (wind effects even Psi_tau in S.O. are in PsiTW)
    return soln.Psi

def bflux(soln,phys,grid,t):
    # Compute the numerical flux to the b equation 
    # takes in b: [Ny,Nz] and Psi: [Ny+1,Nz+1], 
    # imposes bcs to b using ICs, advects b field 
    # and introduces vert/horz diffusion (if asked by user in flags)
    # flux returned same size as b: [Ny,Nz]
    # returns: -Jac(Psi,b) + d/dz(kappa_z*b_z) + d/dy(kappa_y*b_y)

    # compute advection term
    flux = -upwind(soln,grid)

    if phys.vertdiff: # explicit vertical diffusion
        flux = flux + ddzkappabz(soln.b,phys,grid)
    if phys.horzdiff: # explicit horizontal diffusion
        flux = flux + ddykappaby(soln.b,phys,grid)

    return flux

def vels(soln,grid):
    # takes in streamfunction Psi: [Ny+1,Nz+1]
    # and returns the merid. (v) and vert. (w) velocities
    #  Psi: [N[0]+1,N[1]+1], outer all zeroes 
    #    v: [N[0]+1,N[1]  ], North/South zeroes
    #    w: [N[0]  ,N[1]+1], top/bott zeroes
    # v = -d/dz Psi, w = d/dy Psi
    v = -1/grid.LxF[:,1:]*np.diff(soln.Psi,axis=1)/grid.dzzFF
    w = +1/grid.LxC*np.diff(soln.Psi,axis=0)/grid.dyyFF
    if np.max(abs(w[:,-1])) > 1e-15:
        print("WARNING FLOW THROUGH BOTTOM")
    #if np.max(abs(w[:,0])) > 1e-15:
    #    print("WARNING FLOW THROUGH SURFACE")
    if np.max(abs(v[-1,:])) > 1e-15:
        print("WARNING FLOW THROUGH NORTH WALL")
    if np.max(abs(v[0,:])) > 1e-15:
        print("WARNING FLOW THROUGH SOUTH WALL")

    return v,w

def upwind(soln,grid):
    # FIRST ORDER upwind scheme for J(Psi,b)
    # return: Psi_y*b_z - Psi_z*b_y

    # velocity components
    v,w = vels(soln,grid) # v = [Ny+1,Nz], w = [Ny,Nz+1]

    # upwind velocities 
    vp = np.maximum(v,0)[:-1,:]; # v-vel northward [Ny,Nz] ignoring north wall.
    vm = np.minimum(v,0)[ 1:,:]; # v-vel southward [Ny,Nz] ignoring south wall.
    wp = np.maximum(w,0)[:, 1:]; # w-vel upwards   [Ny,Nz] ignoring bottom
    wm = np.minimum(w,0)[:,:-1]; # w-vel downwards [Ny,Nz] ignoring surface

    bp = np.pad(soln.b,pad_width=1,mode='edge') # now [Ny+2,Nz+2] on ghosted C-grid,  

    dbdy = np.diff(bp[:,1:-1],axis=0)/grid.dyyCG # on y F-grid, [Ny+1,Nz] without top/bottom ghost
    dbdz = np.diff(bp[1:-1,:],axis=1)/grid.dzzCG # on z F-grid, [Ny,Nz+1] without south/north ghost

    b_ym = dbdy[:-1,:]; b_yp = dbdy[1:,:]; b_zm = dbdz[:,:-1]; b_zp = dbdz[:,1:]

    return vp*b_ym + vm*b_yp + wm*b_zm + wp*b_zp # w positive. points in neg indices

def calculate_error(b2,b1,grid):
    # simple norm difference between b1 and b2 for measuring change in the solution.
    # will eventually set the number of itirations needed to find the steady soluton.
    #print("total b2: {}, total b1: {}".format(total(b2),total(b1)))
    return (total(b2,grid) - total(b1,grid))/total(b2,grid)

def total(b,grid): # first order riemann sum for domain integrated b
    # normalized by domain area so technically a domain avg
    return np.sum(np.sum(grid.dyyF*grid.dzzF*b,axis=0))/(grid.LF[0]*grid.LF[1])

def check_CFL(soln,times,grid):
    # Debug outputs for CFL conditions
    v,w = vels(soln,grid) # calculate velocitites
    CFLadvy = np.max(abs(v))*times.dt/abs(grid.dyyF).min() # horizontal CFL condition
    CFLadvz = np.max(abs(w))*times.dt/abs(grid.dzzF).min() # vertical CFL condition
    CFL = np.max([CFLadvy,CFLadvz]) # compute the absolute CFL as the max of the two.
    if CFL > 1: # should really be kept of 0.8 for AB3 timestepping
        print("WARNING: ADVC CFL number ({:.2f}) > 1 (unstable). Reduce timestep.".format(CFL))
    return CFL

def solve_model(grid,phys,times,soln,plotting,output):

    # BEGIN MAIN TIMESTEPPING CODE
    #############################################################################
    # Given a Psi, evolve b through adv-diff equation:
    # ->  bflux = Psi_z*b_y - Psi_y*b_z + d/dz(kappa_z*b_z)
    # Given new b, compute new Psi adv. from thermal wind and Gent-McWilliams treatment of baroclinic eddies
    # initialize main figure for realtime plotting
    plt.figure(1,figsize=plotting.fig_size)
    
    # Init timestep counter
    cnt = 0 
    if plotting.pltmitgcm: plotmit(grid,plotting,soln.datalabel) # plot the corresponding simulation from MITgcm, only once. saves to frame_mitgcm.png

    # calculate advective Psi.
    update_Psi(soln,phys,grid)

    if output.save2h5 and cnt % output.n_h5 == 0:
        cnt_h5             = int(cnt/output.n_h5)
        output.output_PsiGM[cnt_h5] = soln.PsiGM
        output.output_PsiTW[cnt_h5] = soln.PsiTW
        output.output_Psi[cnt_h5]   = soln.Psi
        output.output_b[cnt_h5]     = soln.b0
    
    # print diagnostics
    print("t = {:.3f} yrs, dt = {:.2f} days, CFL = {:.3f}, <b> = {:.3e}".format\
            (times.t[cnt]/(365*days2secs),times.dt/days2secs,check_CFL(soln,times,grid),total(soln.b,grid)))
     
    # plot ICs
    if plotting.fancy: # plotting isopycnal contours on top of Psi contourf
        plotfancy(soln,cnt,times,grid,plotting)
    else: # plotting strat alone, along with Psi components
        plotcomp(soln,cnt,times,grid,plotting)
    
    # If only plotting Initial Condtions, print stats (for debugging) + exit
    if plotting.just_ICs:
        print("Bot:   b[ :,-1] min/max = {:.2e} / {:.2e} (m/s^2)".format(np.min(soln.b[:,-1]),np.max(soln.b[:,-1])))
        print("Top:   b[ :, 0] min/max = {:.2e} / {:.2e} (m/s^2)".format(np.min(soln.b[:,0]),np.max(soln.b[:,0])))
        print("Nor:   b[-1, :] min/max = {:.2e} / {:.2e} (m/s^2)".format(np.min(soln.b[-1,:]),np.max(soln.b[-1,:])))
        print("Sou:   b[ 0, :] min/max = {:.2e} / {:.2e} (m/s^2)".format(np.min(soln.b[0,:]),np.max(soln.b[0,:])))
        print("Bot: Psi[ :,-1] min/max = {:.2e} / {:.2e} (Sv)".format(np.min((soln.Psi*1e-6)[:,-1]),np.max((soln.Psi*1e-6)[:,-1])))
        print("Top: Psi[ :, 0] min/max = {:.2e} / {:.2e} (Sv)".format(np.min((soln.Psi*1e-6)[:,0]),np.max((soln.Psi*1e-6)[:,0])))
        print("Nor: Psi[-1, :] min/max = {:.2e} / {:.2e} (Sv)".format(np.min((soln.Psi*1e-6)[-1,:]),np.max((soln.Psi*1e-6)[-1,:])))
        print("Sou: Psi[ 0, :] min/max = {:.2e} / {:.2e} (Sv)".format(np.min((soln.Psi*1e-6)[0,:]),np.max((soln.Psi*1e-6)[0,:])))
        plt.tight_layout(); plt.show(); sys.exit()    
    
    # main loop AB3 using nm3 nm2 nm1 variables (i.e at step n-3, n-2 and n-1 to find n)
    for cnt in range(1,len(times.t)):
    
        # Other Options: Euler
        flux = bflux(soln,phys,grid,times.t[cnt]) 
        soln.b = soln.b + times.dt*flux 

        applybcs(soln,phys,times.t[cnt],times.dt) # impose bcs

        if phys.convectb: # if we choose to convect b
            soln.b = convect(soln.b,phys,grid)
    
        update_Psi(soln,phys,grid) # update Psi
        
        if times.t[cnt]/(365*days2secs) % 10 == 0: 
            print("t = {:.3f} yrs, dt = {:.2f} days, CFL = {:.3f}, <b> = {:.3e}".format\
                (times.t[cnt]/(365*days2secs),times.dt/days2secs,check_CFL(soln,times,grid),total(soln.b,grid)))
     
        # plot and store step
        if cnt % int(plotting.tplot/times.dt) == 0:
            if plotting.fancy: # plotting isopycnal contours on top of Psi contourf
                plotfancy(soln,cnt,times,grid,plotting)
            else: # plotting strat alone, along with Psi components
                plotcomp(soln,cnt,times,grid,plotting)

        if output.save2h5 and cnt % int(plotting.tplot/times.dt) == 0:
            cnt_h5             = int(cnt/int(plotting.tplot/times.dt))
            output.output_PsiGM[cnt_h5] = soln.PsiGM
            output.output_PsiTW[cnt_h5] = soln.PsiTW
            output.output_Psi[cnt_h5]   = soln.Psi 
            output.output_b[cnt_h5]     = soln.b
    
        # Check for numerical instability, beyond just CFL condition which throws a warning
        if np.max(np.abs(soln.b))>1e10 or np.isnan(soln.b).any(): 
            raise(OverflowError("Solution Diverged."))
            sys.exit()
     
    # Outside main loop, all solutions were stored to file. Close file and exit.
    if output.save2h5:
        output.file.close()
        print("\nFile closed")
    
    # generate movie from all stored frame_XXXX.png files, in CWD
    if plotting.mk_movie and not(plotting.just_ICs):
       makemovie(plotting.moviename)


