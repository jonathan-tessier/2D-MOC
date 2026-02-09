# init solution
from lib.mitgcmtools import * # MITgcm post-processing
import numpy as np
import copy as cp
from scipy import interpolate
import matplotlib.pyplot as plt

class init_solution(object):

    def __init__(
            self, 
            grid,
            phys,
            times,
            method    = 'mitgcm', # init method (mitgcm,warm,pickup) 
            datalabel = "/path/to/3D/simulation", # label of directory for MITgcm output to pull from 
            snap      = 12522600, # e.g. 20k yrs, label of output file from MITgcm
            pickupfilename = "output-ref.h5",
            warmtemp  = 5 # uniform temperature to initialize basin is method=='warm'
    ):

        N = grid.N
        self.datalabel = datalabel

        self.b0     = np.zeros(N) # initial strat..
        self.b      = np.zeros(N) # current strat
        self.PsiTW  = np.zeros([N[0]+1,N[1]+1]) # eulerian circulation
        self.PsiGM  = np.zeros([N[0]+1,N[1]+1]) # eddy circulation
        self.PsiSO  = np.zeros([N[0]+1,N[1]+1]) # deacon cell in channel
        self.Psi    = np.zeros([N[0]+1,N[1]+1]) # total circulation

        if method=='mitgcm' or phys.mit_tau or phys.mit_SST: # if you need to pull from mitgcm files
            mit_grid  = load_grid(datalabel=datalabel)
            try:
                fields    = get_fields(datalabel=datalabel,snap=snap) # extract fields at 20k yrs
            except:
                fields    = get_fields(datalabel=datalabel,snap=10018080) # change this to your lowest output snapshot
            forc      = get_forcing(datalabel=datalabel) # extract forcing fields (sst and wind stress)
            avgfields = zonalavg_fields(fields,mit_grid) # compute zonal averages of fields

        if phys.mit_tau: # use MITgcm wind-stress
            print("Using MITgcm wind-stress")
            self.tauC = forc.wind[0] # wind stress on C grid
            if grid.rm_topo: self.tauC = self.tauC[1:-1]
            self.tau = np.interp(grid.yF,grid.yC,self.tauC) # -> F grid.
            self.tau = np.transpose(np.tile(self.tau,(grid.N[1]+1,1))) # project in depth
        else: # use user defined wind-stress
            print("Using user-defined wind-stress")
            self.tau = phys.tau0*np.sin(np.pi*(grid.yyF[grid.SOyind] - grid.yyF)/(grid.yyF[grid.SOyind]-grid.yyF.min()))**2; 
        self.tau[0,:] = 0; self.tau[grid.SOyind:] = 0; # no wind over basin in this version

        # generate Psi in the channel
        PsiSO = -grid.LxF*phys.finv*self.tau/phys.rho0
        self.PsiSO = np.pad(PsiSO[1:-1,1:-1],pad_width=1,mode='constant') # halo of zeros ensure no flux

        if phys.mit_SST: # use MITgcm SST
            print("Using MITgcm SST")
            SST = forc.sst[0,:]
            if grid.rm_topo: SST = SST[1:-1]
        else: # use user-defined SST
            print("Using user-defined SST")
            SST                      = phys.offset+(phys.SSTmax-phys.offset-phys.sstref)*np.cos(np.pi*grid.yC/(grid.yC.max()-grid.yC.min()))**2;
            SST[:int(grid.N[0]/2)+1] = ((phys.SSTmax-phys.sstref)*np.cos(np.pi*grid.yC/(grid.yC.max()-grid.yC.min()))**2)[:int(grid.N[0]/2)+1];
            SST += phys.sstref # use this offset to control colorbar...
            #print(SST); sys.exit()
        if method=='mitgcm':
            self.b0 = cp.copy(avgfields.b); 
            if grid.rm_topo: self.b0 = self.b0[1:-1,:] # remove walls from MITgcm 
        elif method=='pickup':
            pickupfile = h5py.File(pickupfilename,mode="r")
            self.b0 = cp.copy(pickupfile["b"][:][-1,:,:]) # last saved
        elif method=='warm':
            self.b0  = phys.g0*phys.tAlpha*warmtemp*np.ones(grid.N) 
        else:
            raise(ValueError('initialization method invalid, (choose: mitgcm, pickup, warm)'))
        self.b0[:,0] = phys.g0*phys.tAlpha*SST
        #self.b0 = np.repeat(self.b0[:,0][:,np.newaxis],grid.N[1],axis=1)
        self.b = cp.copy(self.b0)

        print('Northern SST = {:.2f} deg C'.format(self.b0[-1,0]/(phys.g0*phys.tAlpha)))
        print('Equator  SST = {:.2f} deg C'.format(self.b0[int(grid.N[0]/2),0]/(phys.g0*phys.tAlpha)))
        print('Southern SST = {:.2f} deg C'.format(self.b0[0,0]/(phys.g0*phys.tAlpha)))
        print('')

        plt.figure(3,figsize=(10,6))
        plt.subplot(221)
        plt.plot(grid.yF/grid.degs2metr,self.tau[:,0]); plt.title(r'$\tau(\theta)$ $(N/m^2)$'); plt.grid();
        plt.xlabel(r'latitude $\theta$'); plt.ylabel(r'$\tau$')
        #plt.plot(grid.yF[grid.SOyind]/grid.degs2metr,[0],'*')
        plt.subplot(222)
        plt.semilogx(phys.kappa_z,grid.zF); plt.grid(); plt.title(r"$\kappa(z)$ ($m^2/s$)"); 
        plt.ylabel('z (m)'); 
        plt.subplot(223)
        plt.plot(grid.yC/grid.degs2metr,SST); plt.title(r'$SST(\theta)$ (deg C)'); plt.grid(); 
        #plt.plot(grid.yF[grid.SOyind]/grid.degs2metr,SST[grid.SOyind],'*')
        plt.xlabel(r'latitude $\theta$'); plt.tight_layout(); plt.savefig('forcing.png',dpi=200)
        plt.subplot(224)
        plt.plot()
        #plt.show(); sys.exit()
 
