import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import subprocess
import numpy as np
from lib.mitgcmtools import *
days2secs = 24*60*60

class plotting_parameters(object):

    def __init__(
            self,
            # Plotting Options
            onthefly = False,        # show figures as they are generated
            just_ICs = False,        # Show initial conditions, stats, and exit
            # fix plotting boundaries
            lat_lims = [-68,68],     # lat. limits in deg for plotting [-70,70] (F-grid), [-68,68] to exclude walls
            dep_lims = [-4,  0],     # depth limits in km for plotting [-4,0] (F-grid)
            Psi_lims = [-8,8],       # when plotting Psi, contour limits to be fixed, in Sv
            b_lims   = [0,0.0585],   # buoyancy plotting limits..
            fig_size = [15.1,8.5],   # figure size, for when working on various screens
            mk_movie = True       ,  # save plots as pngs and make ffmpeg movie at 12 fps
            moviename= 'movie.mp4',  # movie name for above if mk_movie=True
            fancy    = False      ,  # plot contours of b on top of Psi for prettier figures, instead of Psi components
            cmap     = 'seismic'  ,  # colormap for contourf and pcolor ('seismic', 'bwr', 'etc')
            pltmitgcm= False      ,  # generate equivalent figure of MITgcm run to compare, while running.
            plt_ICs  = False      ,  # plt topo, kappa, surf temp
            bgamma   = 1/6        ,  # controls color contrast in strat plots
            tplot = 10*365*24*60*60, # freqnecy to plot frames 

            ):
        
        self.onthefly = onthefly
        self.just_ICs = just_ICs
        self.lat_lims = lat_lims
        self.dep_lims = dep_lims
        self.Psi_lims = Psi_lims
        self.fig_size = fig_size
        self.mk_movie = mk_movie
        self.moviename = moviename
        self.fancy     = fancy
        self.cmap      = cmap
        self.pltmitgcm = pltmitgcm
        self.bgamma    = bgamma
        self.tplot     = tplot
        # plotting levels for Psi
        dpsi = 1
        self.lvls = np.arange(Psi_lims[0],Psi_lims[1]+dpsi,dpsi) # for psi

        # plotting levels for b
        stretch_power = 10
        stretch_vector = (np.exp(np.linspace(0,1,75))**stretch_power - 1)/(np.exp(1)**stretch_power-1)
        self.level_vector = b_lims[0] + stretch_vector*(b_lims[1] - b_lims[0])


def plotmit(grid,plotting,datalabel):
    # Basic plotting of the MITgcm solution equivalent of this simulation
    # plots Psi_residual, b and Psi_euler and Psi_GM in realtime computation
    # plots line at channel-basin interface if desired. (at y=SOyind=12)
    # constant in time, saves plot in 1 frame: frame_mitgcm.png in CWD
    plt.figure(2,figsize=plotting.fig_size); plt.clf()

    mit_grid = load_grid(datalabel=datalabel)
    try:
        fields    = get_fields(datalabel=datalabel,snap=12522600) # extract fields at 20k yrs
    except:
        fields    = get_fields(datalabel=datalabel,snap=10018080)
    avgfields = zonalavg_fields(fields,mit_grid)

    if grid.rm_topo: 
        avgfields.PsiRes   = avgfields.PsiRes[1:-1]
        avgfields.b        = avgfields.b[1:-1]
        avgfields.PsiEuler = avgfields.PsiEuler[1:-1]
        avgfields.PsiEddy  = avgfields.PsiEddy[1:-1]

    plt.subplot(2,2,1)
    plt.contourf(grid.yyF/grid.degs2metr, grid.zzF/1e3, avgfields.PsiRes/1e6, cmap = plotting.cmap, \
        levels = plotting.lvls, norm=colors.CenteredNorm()) 
    plt.title(r'MITgcm $\Psi$ $(Sv)$')
    plt.xlim(plotting.lat_lims); plt.ylim(plotting.dep_lims); plt.ylabel("z (km)"); plt.xlabel("lat (deg)"); plt.colorbar()
    if grid.channel: plt.plot(grid.yyF[grid.SOyind,0]*np.ones(grid.N[1]+1)/grid.degs2metr,grid.zzF[grid.SOyind,:]/1e3,'-k')

    plt.subplot(2,2,2)
    plt.contourf(grid.yyC/grid.degs2metr, grid.zzC/1e3, avgfields.b, cmap = plotting.cmap, \
        levels = plotting.level_vector, norm=colors.PowerNorm(gamma=plotting.bgamma))
    plt.title(r'MITgcm $\overline{b}$ $(m/s^2)$')
    plt.xlim(plotting.lat_lims); plt.ylim(plotting.dep_lims); plt.ylabel("z (km)"); plt.xlabel("lat (deg)"); plt.colorbar(format=ticker.FuncFormatter(fmt_cb))
    if grid.channel: plt.plot(grid.yyF[grid.SOyind,0]*np.ones(grid.N[1]+1)/grid.degs2metr,grid.zzF[grid.SOyind,:]/1e3,'-k')

    plt.subplot(2,2,3)
    plt.contourf(grid.yyF/grid.degs2metr, grid.zzF/1e3, avgfields.PsiEuler/1e6, cmap = plotting.cmap, \
        levels = plotting.lvls, norm=colors.CenteredNorm()) 
    plt.title(r'MITgcm $\Psi_{EU}$ $(Sv)$')
    plt.xlim(plotting.lat_lims); plt.ylim(plotting.dep_lims); plt.ylabel("z (km)"); plt.xlabel("lat (deg)"); plt.colorbar()
    if grid.channel: plt.plot(grid.yyF[grid.SOyind,0]*np.ones(grid.N[1]+1)/grid.degs2metr,grid.zzF[grid.SOyind,:]/1e3,'-k')

    plt.subplot(2,2,4)
    plt.contourf(grid.yyF/grid.degs2metr, grid.zzF/1e3, avgfields.PsiEddy/1e6, cmap = plotting.cmap, \
        levels = plotting.lvls, norm=colors.CenteredNorm()) 
    plt.title(r'MITgcm $\Psi_{GM}$ $(Sv)$')
    plt.xlim(plotting.lat_lims); plt.ylim(plotting.dep_lims); plt.ylabel("z (km)"); plt.xlabel("lat (deg)"); plt.colorbar()
    if grid.channel: plt.plot(grid.yyF[grid.SOyind,0]*np.ones(grid.N[1]+1)/grid.degs2metr,grid.zzF[grid.SOyind,:]/1e3,'-k')

    plt.tight_layout()
    plt.draw()
    plt.savefig('mitgcm_frame.png', dpi=200)

def plotcomp(soln,cnt,times,grid,plotting):
    # Basic plotting function for this simulation
    plt.figure(1,figsize=plotting.fig_size); plt.clf()

    plt.subplot(2,2,1)
    plt.contourf(grid.yyF/grid.degs2metr, grid.zzF/1e3, soln.Psi/1e6, cmap = plotting.cmap, \
        levels = plotting.lvls, norm=colors.CenteredNorm()) 
    plt.title(r'$\Psi$ $(Sv)$ at t = {:.2f} years'.format(times.t[cnt]/(365*days2secs)))
    plt.xlim(plotting.lat_lims); plt.ylim(plotting.dep_lims); plt.ylabel("z (km)"); plt.xlabel("lat (deg)"); plt.colorbar()
    if grid.channel: plt.plot(grid.yyF[grid.SOyind,0]*np.ones(grid.N[1]+1)/grid.degs2metr,grid.zzF[grid.SOyind,:]/1e3,'-k')

    plt.subplot(2,2,2)
    plt.contourf(grid.yyC/grid.degs2metr, grid.zzC/1e3, soln.b, cmap = plotting.cmap, \
        levels = plotting.level_vector, norm=colors.PowerNorm(gamma=plotting.bgamma))
    plt.title(r'$\overline{b}$' + r' $(m/s^2)$ at t = {:.2f} years'.format(times.t[cnt]/(365*days2secs)))
    plt.xlim(plotting.lat_lims); plt.ylim(plotting.dep_lims); plt.ylabel("z (km)"); plt.xlabel("lat (deg)"); plt.colorbar(format=ticker.FuncFormatter(fmt_cb))
    if grid.channel: plt.plot(grid.yyF[grid.SOyind,0]*np.ones(grid.N[1]+1)/grid.degs2metr,grid.zzF[grid.SOyind,:]/1e3,'-k')

    plt.subplot(2,2,3)
    plt.contourf(grid.yyF/grid.degs2metr, grid.zzF/1e3, soln.PsiTW/1e6, cmap = plotting.cmap, \
        levels = plotting.lvls, norm=colors.CenteredNorm()) 
    plt.title(r'$\Psi_{TW}$'+' $(Sv)$ at t = {:.2f} years'.format(times.t[cnt]/(365*days2secs)))
    plt.xlim(plotting.lat_lims); plt.ylim(plotting.dep_lims); plt.ylabel("z (km)"); plt.xlabel("lat (deg)"); plt.colorbar()
    if grid.channel: plt.plot(grid.yyF[grid.SOyind,0]*np.ones(grid.N[1]+1)/grid.degs2metr,grid.zzF[grid.SOyind,:]/1e3,'-k')

    plt.subplot(2,2,4)
    plt.contourf(grid.yyF/grid.degs2metr, grid.zzF/1e3, soln.PsiGM/1e6, cmap = plotting.cmap, \
        levels = plotting.lvls, norm=colors.CenteredNorm()) 
    plt.title(r'$\Psi_{GM}$'+' $(Sv)$ at t = {:.2f} years'.format(times.t[cnt]/(365*days2secs)))
    plt.xlim(plotting.lat_lims); plt.ylim(plotting.dep_lims); plt.ylabel("z (km)"); plt.xlabel("lat (deg)"); plt.colorbar()
    if grid.channel: plt.plot(grid.yyF[grid.SOyind,0]*np.ones(grid.N[1]+1)/grid.degs2metr,grid.zzF[grid.SOyind,:]/1e3,'-k')

    plt.tight_layout()
    plt.draw()
    if plotting.onthefly:
        plt.pause(0.001)

    if plotting.mk_movie:
        plt.savefig('frame_{0:04d}.png'.format(int(cnt/int(plotting.tplot/times.dt)), dpi=200))

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

def plotfancy(soln,cnt,times,grid,plotting):
    # Fancy plotting option for realtime computation or final frame.
    # Basic plotting of both Psi and b to use in realtime computation

    plt.figure(1,figsize=plotting.fig_size); plt.clf()

    plt.contourf(grid.yyF/grid.degs2metr, grid.zzF/1e3, soln.Psi/1e6, cmap = plotting.cmap, \
        levels = plotting.lvls, norm=colors.CenteredNorm())
    plt.title(r'$\Psi$ $(Sv)$ at t = {:.2f} years'.format(times.t[cnt]/(365*days2secs)))
    plt.xlim(plotting.lat_lims); plt.ylim(plotting.dep_lims); plt.ylabel("z (km)"); plt.colorbar()
    CS = plt.contour(grid.yyC/grid.degs2metr, grid.zzC/1e3, soln.b, colors='black', levels=plotting.level_vector[::5])
    plt.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
    if grid.channel: plt.plot(grid.yyF[grid.SOyind,0]*np.ones(grid.N[1]+1)/grid.degs2metr,grid.zzF[grid.SOyind,:]/1e3,'-k')
    plt.tight_layout()
    plt.draw()
    if plotting.mk_movie:
        plt.savefig('frame_{0:04d}.png'.format(int(cnt/int(plotting.tplot/times.dt)), dpi=200))

def makemovie(moviename=None):
    if moviename is None:
        moviename = 'movie.mp4'
    f_log = open("ffmpeg.log", "w")
    f_err = open("ffmpeg.err", "w")
    cmd_gen = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-framerate', str(12), '-i', 'frame_%04d.png', '-y', 
            '-q', '1', '-threads', '0', '-pix_fmt', 'yuv420p', moviename]
    subprocess.call(cmd_gen, stdout=subprocess.PIPE)
    print("Generated animation: "+moviename)

