# output
import h5py
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from lib.mitgcmtools   import * #MITgcm post-processing 
from MITgcmutils.utils import writebin, readbin # reads/writes binary for MITgcm

class output_parameters(object):

    def __init__(
            self,
            times,
            grid,
            phys,
            mk_movie = True,         # save plots as pngs and make ffmpeg movie at 12 fps
            moviename= 'movie.mp4',  # movie name for above if mk_movie=True
            save2h5  = True,         # save data to HDF5 file, named output.h5, for pickups and later post-processing
            h5freq   = 10*365*24*60*60, # data save frequency for pickups, only save 4 times to save space.
            filename = "output.h5"
            ):

        # pickup time vector stuff
        self.t_h5 = np.arange(times.t0,times.tf+h5freq,h5freq) # pickup time vector
        self.n_h5 = int(h5freq/times.dt) # number of skipped frames for pickup
        self.save2h5 = save2h5
        self.filename = filename

        if save2h5:
            self.file = h5py.File(self.filename, mode="w")          # output file
            self.file.create_dataset("LC",     data = grid.LC)      # save domain lengths
            self.file.create_dataset("size",   data = grid.N)       # save domain resolution
            self.file.create_dataset("t_h5",   data = self.t_h5)    # save pickup temporal vector
            #file.create_dataset("tau",    data = phys.tau)     # save wind-stress 
            self.file.create_dataset("kappa_z",data = phys.kappa_z) # save buoyancy diffusivitiy 

            # Output Fields
            self.output_b      = self.file.create_dataset('b',   (len(self.t_h5),grid.N[0],grid.N[1])) # init b output
            self.output_Psi    = self.file.create_dataset('Psi', (len(self.t_h5),grid.N[0]+1,grid.N[1]+1)) # init Psi Residual output
            self.output_PsiTW  = self.file.create_dataset('Psi_TW', (len(self.t_h5),grid.N[0]+1,grid.N[1]+1)) # init Psi TW output
            self.output_PsiGM  = self.file.create_dataset('Psi_GM', (len(self.t_h5),grid.N[0]+1,grid.N[1]+1)) # init Psi GM output
            
            print("File "+filename+" opened, saving state every {} years\n".format(h5freq/(365*24*60*60)))


def generate_diagnostics(filename, grid, plotting):

    file = h5py.File(filename,mode="r") 
    b    = cp.copy(file["b"][:][:,:,:]) # Nt, Ny, Nz
    Psi  = cp.copy(file["Psi"][:][:,:,:]) # Nt, Ny+1, Nz+1
    t_h5 = cp.copy(file["t_h5"][:])
    Nt = len(t_h5)

    # init diagnostics...
    b_mean  = np.zeros(Nt) # save domain-wide mean buoyancy
    p_slice = np.zeros([Nt,grid.N[1]+1]); yind = 59 # yind=59 -> at 50 deg N (grid pts are 2 deg apart)

    ttz, zzt = np.meshgrid(t_h5,grid.zF,indexing='ij')

    for cnt in range(Nt):
        b_mean[cnt] = np.sum(np.sum(grid.dyyF*grid.dzzF*b[cnt,:,:],axis=0))/(grid.LF[0]*grid.LF[1])
        p_slice[cnt,:] = Psi[cnt,yind,:]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10,6), layout='constrained')

    plt.subplot(211)
    axes[0].plot(t_h5/(365*24*60*60),b_mean); plt.ylabel(r'$\bar b$'); plt.grid()
    plt.xlim([0,t_h5[-1]/(365*24*60*60)])

    plt.subplot(212)
    s = axes[1].contourf(ttz/(365*24*60*60),zzt/1e3,p_slice/1e6,cmap='seismic',levels=plotting.lvls,norm=colors.CenteredNorm()); 
    plt.xlabel('t (yrs)'); plt.ylabel(r'$z$ (km)'); 
    plt.title(r"$\Psi$ (Sv) at $\theta = $ {:.0f}$^\circ$ lat".format(grid.yF[yind]/grid.degs2metr)); 

    fig.colorbar(s, ax=axes[1], pad=0); #plt.tight_layout();
    plt.savefig('diag.png',dpi=300)

    file.close()
