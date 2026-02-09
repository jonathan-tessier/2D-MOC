
#######################################################################
# Solver for a 2D zonally averaged Meridional Overturning Circulation #
#                                                                     #
# Advection:    b_t = + v*b_y + w*b_z + d/dz(kappa_z*b_z)             #
# Flow:       Basin = PsiTW + PsiGM + Psi0,  S.O. = PsiSO + PsiGM     #
# Thermal wind comp:    PsiTW_zzy = -1/f b_y, solved by integration   #
# Baroclinic eddies:    PsiGM = -kappaGM*(b_y/b_z), tapered to Smax   #
# Velocitites (v,w):    v = -1/Lx*Psi_z, w = 1/Lx*Psi_y               #
# Dom. Width (lon.):    Lx = R*del_lamb*cos(theta), y = R*theta,      #
#                                                                     #
# By: Jonathan Tessier, PhD in Oceanography, UQAR-ISMER (2022-2026)   #
#######################################################################
# Usage: Modify below parameters for your needs. Then run model using #
# $ python3 driver.py                                                 #
#                                                                     #
# Notes:                                                              #
# - data is saved to .h5 file at user-specified freqency h5freq       #
# - animation will be created with default name: 'movie.mp4' (ffmpeg) #
# - all frames to create the animation will be saved in the cwd       #
# - diagnostic plots can be created onthefly in lib/data_output.py    #
# - code accomodates copying grid/forcing from a MITgcm simulation to #
#   keep identical parameters (for comparison)                        #
#                                                                     #
# Requires packages:                                                  #
# h5py, numpy, copy, matplotlib, MITgcmutils, scipy, sys, subprocess  #
# Also requires 'ffmpeg' for animations... (https://www.ffmpeg.org/)  #
#######################################################################

# library imports
import h5py, sys, time
import numpy as np
import copy as cp
from scipy import interpolate

from lib.grid          import grid_parameters # grid parameters
from lib.initialize    import init_solution   # init solution
from lib.physics       import physical_parameters # thermal wind, diffusion, eddies, etc.
from lib.plotting      import plotting_parameters # plotting 
from lib.timestepping  import temporal_parameters, solve_model # time-stepping scheme, runtime diags
from lib.data_output   import output_parameters, generate_diagnostics # deals with output to .h5

def main():
    
    # Set grid parameters (...) can pull grid directly from MITgcm output if required
    grid = grid_parameters(mitgrid   = False,    # use grid from MITgcm simulation (requires correct datalabel/diagnostics below)
                           channel   = True,     # channel in topography? (True) or closed basin (False -> no wind) 
                           lat_lims  = [-70,70], # latitude limits of domain (F-grid) including walls
                           dtheta    = 2,        # default horizontal resolution (latitude, deg)
                           dlambda   = 2,        # default horizontal resolution (longitude, deg)
                           Nlambda   = 30,       # points in longitude (calculate Lx = R*dlamb*Nlamb*cos(theta))
                           chan_int  = -46,      # latitude of channel interface (degrees) (sets index based on F-grid)
                           delR      = np.array([ 37.,  40.,  44.,  48.,  52.,  56.,  60.,  63.,  66.,  69.,  72.,\
                                                  75.,  78.,  81.,  84.,  87.,  90.,  93.,  96.,  99., 102., 105.,\
                                                 108., 111., 114., 117., 120., 123., 126., 129., 132., 135., 138.,\
                                                 141., 144., 147., 150., 153., 156., 159.]), # layer thicknesses (m)
                           # Change this depending on where (if) you store your MITgcm outputs (/path/to/MITgcm/out)
                           datalabel = '/path/to/MITgcm/run'
                           # expected to contain grid: XC,YC,XG,YG,DXC,DYC,DXG,DYG,RC,RF,DRC,DRF,hFacC,hFacW,hFacS,Depth (data/meta)
                           # and diagnostic fields: in diag4.XXXXXXXXXX.(data/meta) order: U,V,W,T,dsig/dz,GM_PsiX,GM_PsiY,CONVADJ
                           # and forcing in wind_x.bin (wind stress) and rbcs_T.bin (SST). No salinity advection considered here.
                           )

    # Set physical parameters
    phys = physical_parameters(grid,
                               # Surface boundary conditions
                               restoring = False,    # surface restoring/nudging (if False, Dirichlet with imposed profile)
                               t_rest    = 30*86400, # restoring timescale (s) (1 day = 86400 s)
                               # wind forcing over channel
                               tau0      = 0.1,      # strength of wind in channel (N^2/m) sin^2(theta) shape
                               mit_tau   = False,    # use wind-stress from generated binary for MITgcm run, wind_x.bin? (grid.datalabel)
                               # SST surface profile (sin^2 shape with T_south = 0, T_north = offset, or pull from MITgcm binary)
                               sstref    = 0.06,     # SST (deg C) at the southern edge of the basin (mostly to ensure b>0 with convection)
                               offset    = 0.1,      # SST (deg C) difference between S.O. and northern SST (= T_north) 
                               SSTmax    = 30,       # max SST (deg C) at equator (cos^2 profile) 
                               mit_SST   = False,    # use SST from generated binary for MITgcm run, rbcs_T.bin? (look in grid.datalabel)
                               # physics: convection ()
                               convectb  = True,     # include convection param in evolution
                               gamma     = 1e-8,     # target background strat, dT/dz. (will be mult. *g0*tAlpha for b)
                               # physics: diffusion
                               vertdiff  = True,     # include vertical diffusion
                               kapzprof  = True,     # use non-uniform vertical diffusion, requires vertdiff=True 
                               kbg       = 0   ,     # background diffusivity (m^2/s), if kapzprof=False, kappa=kbg
                               kb        = 2e-4,     # bottom diffusivity value (m^2/s)
                               zk        = 1250,     # diffusivity e-folding scale (m), kappa = kbg + (kb-kbg)exp(z/zk)
                               # physics: advection (components are cumulative)
                               usePsiGM  = True,     # use PsiGM in advection (Gent McWilliams barocolinic eddies) 
                               usePsiTW  = True,     # use Thermal Wind in advection (Psi_zzy = -1/f b_y)
                               # physics: GM slope clipping (if all false, full slope returns)
                               kappaGM   = 1e3,      # Gent-McWilliams eddy diff. (m^2/s) (typical 1e3)
                               Smax      = 1e-2,     # max allowed slope by/bz in PsiGM taper (typical 1e-2) 
                               )

    # Set timestepping parameters 
    times = temporal_parameters(t0 = 0*365*24*60*60,    # init. time (sec)  
                                tf = 1e4*365*24*60*60,  # final time (sec) 
                                dt = 5e-3*365*24*60*60, # time-step  (sec)
                                )

    plotting = plotting_parameters(# plotting params
                                   tplot = 10*365*24*60*60, # plotting frequency for movie etc  
                                   pltmitgcm= False,        # generate figure of MITgcm run to compare
                                   just_ICs = False,        # Show initial conditions, stats, and exit
                                   onthefly = False,        # show images as they are generated
                                   lat_lims = [-68,68],     # lat. limits for plotting (F-grid), [-68,68] to exclude walls
                                   dep_lims = [-4,  0],     # depth limits in km for plotting [-4,0] (F-grid)
                                   Psi_lims = [-15,  15],   # when plotting Psi, contour limits to be fixed, in Sv
                                   fig_size = [15.1,8.5],   # figure size, for when working on various screens
                                   fancy    = False,        # plot contours of b on top of Psi for prettier figures,
                                   cmap     = 'seismic',    # colormap for contourf and pcolor ('seismic', 'bwr', 'etc')
                                   )

    output = output_parameters(times, grid, phys,
                               mk_movie = True,             # save plots as pngs and make ffmpeg movie at 12 fps
                               moviename= 'movie.mp4',      # movie name for above if mk_movie=True
                               save2h5  = True,             # save data to HDF5 file, for pickups and later post-processing
                               h5freq   = 10*365*24*60*60,  # data save frequency for pickups/post-processing
                               filename = "output.h5",      # save h5 to filename
                               )

    # Initialize solution (pickup looks for file: pickupfilename = "output-ref.h5") 
    soln = init_solution(grid, phys, times, method='warm', datalabel = grid.datalabel) 

    # Simulation Timer (real time)
    time_initial = time.time()

    # Start Simulation
    solve_model(grid, phys, times, soln, plotting, output)

    # Simulation Timer (real time)
    print("Done. Simulated {:.1f} years in {:.1f} minutes".format(\
            (times.tf - times.t0)/(365*24*60*60),(time.time() - time_initial)/(60)))

    # Generate diagnostics from saved .h5 output
    generate_diagnostics(output.filename,grid,plotting)
    # look up this function as a reference for how to use the saved .h5 in postprocessing

main()


