#mock HI observations of EAGLE centrals in z=0 snapshot
#import matplotlib
from numpy import *
import h5py                  as h5
#import matplotlib.pyplot     as plt
from misc import *
#import matplotlib.lines as pltline
#from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os
from scipy.constants import *
from martini.sources import SPHSource#SOSource
from martini import DataCube, Martini
from martini.beams import GaussianBeam
from martini.noise import GaussianNoise
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import GaussianKernel, WendlandC2Kernel, DiracDeltaKernel
import astropy.units as U
from astropy.coordinates import CartesianRepresentation,\
    CartesianDifferential, ICRS
from astropy.coordinates.matrix_utilities import rotation_matrix
import astropy.constants as C
from hydromass import HI_H2_masses as Hmass
from scipy.stats import binned_statistic as binit
import multiprocessing as mp
from scipy.special import erf
from astropy.convolution import convolve, Box1DKernel
import pickle
from astropy.cosmology import Planck15, z_at_value

def gaussint(m,s,a,b):#integral of gaussian
 I = 0.5*(erf((b-m)/(sqrt(2)*s))-erf((a-m)/(sqrt(2)*s)))
 return I

def bsens(x,fwhm):#gaussian beam sensitivity (for a single dish observation)
 sens = exp(-4*log(2)*(x/fwhm)**2)#eq. 3.115 at https://www.cv.nrao.edu/~sransom/web/Ch3.html 
 return sens

Base    = '/group/pawsey0119/'
DList   = ['amanuwal/']
Gc      = 43.0091 # Newton's gravitational constant in Gadget Units
files=open('simfiles','r')
lines=files.readlines()
s=len(lines)

#os.chdir(Base+str(DList[0]))
martini='n'#if you want to use MARTINI to mimic inteferometric observation
mdist='y'#if you want to create H I line profiles based on the mass profiles
make='y'#if you want to create any profile
sigdish='n'#if you want to mimic an unresolved, single dish observation 
direct = 'lineprofs'
#os.system('rm -r /scratch/pawsey0119/amanuwal/'+direct)
os.system('mkdir /scratch/pawsey0119/amanuwal/'+direct)
for idir,DirList in enumerate(DList):
  for i in range(s):
    line=lines[i].split('\n')[0].split('_')
    Num=line[0]
    fend='_'+line[1]
    exts         = Num.zfill(3)
    fn = Base+DirList+'HYDRO_'+exts+fend+'_100Mpc_halodat.hdf5'
    fn1 = Base+DirList+'HYDRO_'+exts+fend+'_100Mpc_Gas.hdf5'
    fn2 = Base+DirList+'100Mpc_HImassfunc_BR06_tforce.hdf5'
    if (os.path.exists(fn)==True):
     print('\n Reading Header from file:',fn)
     f  = h5.File(fn,"r")
     f1 = h5.File(fn1,"r")
     f2 = h5.File(fn2,"r")
        
     hpar = f['Header/h'].value    
     GroupPos = f['HaloData/GroupPos'].value/hpar
     Om = f['Header/Omega'].value
     R_200 = f['HaloData/Group_R_Crit200'].value/hpar
     Mass_DM = f['Header/PartMassDM'].value*1e+10/hpar
     BS = f['Header/BoxSize'].value/hpar
     z = f['Header/Redshift'].value
     Pos = f1['PartData/PosGas'].value/hpar
     Vel = f1['PartData/VelGas'].value
     GNGas = abs(f1['PartData/GrpNum_Gas'].value)
     SNGas = abs(f1['PartData/SubNum_Gas'].value)
    # Velc = f['HaloData/Vbulk'].value
     f.close()

     T = f1['PartData/TempGas'].value
     HSM = f1['PartData/HSML_Gas'].value/hpar
     SFR = f1['PartData/SFR'].value
     EOS = f1['PartData/EOS'].value
     fH = f1['PartData/fHSall'].value
     TMass = f1['PartData/MassGas'].value*1e+10/hpar#total gas particle mass including all species
     rho = f1['PartData/DensGas'].value*hpar**2*1e+10/(1e+6)**3
     Z = f1['PartData/GasSZ'].value
     f1.close()
     
     GNs = f2['GN'].value
     Mstar = f2['Mstar'].value
     Ngas = f2['Ngasa'].value
     nhi50 = f2['nhi50a'].value
     MHI = f2['MHIa'].value
     f2.close()

     #Select galaxies above the cuts
     inds = where((Mstar>1e+9) & (MHI/Mstar>0.02) & (MHI>=1e+8) & (nhi50>=1000))[0]#xGASS selection + HI-rich particle number cut
     Grps = GNs[inds]

     local=False
     #Generating angles for random orientations
     rinc = degrees(arccos(random.uniform(-1,1,len(Grps)))) #random inclination with cos uniform distribution
     rorient = random.uniform(0,360,len(Grps))
     
     #Loading the KDTree for gas particles (expedites the particle search)   
     f=open('/group/pawsey0119/amanuwal/gastree.p','rb')
     Tree=pickle.load(f)

     def martini_hi(l):
      for j in [7]:
        GN = int(Grps[l])
        print('Group No.:',GN)
        r200 = float(R_200[GN-1])        
        indices = Tree.query_ball_point(x=GroupPos[GN-1],r=r200)#Searching particle within an aperture large enough to encompass all bound gas
        #Generic conditions to check if there are at least 3 gas particles (not relevant if the galaxies are already selected by particle number)
        #if str(indices)=='None':
         # continue
        #if len(indices)<3:
         #continue
        gns = GNGas[indices]
        sgns = SNGas[indices] 

        d = sqrt(sum(GroupPos[GN-1]**2))
        zgal = 0#galaxy redshift taken as zero for simplicity for a local Universe (replace this with the redshift corresponding to a luminosity distance of 'd' for the adopted cosmology) 
        pos = Pos[indices,:]
        cen = GroupPos[GN-1]
        pos = do_wrap(pos-cen,BS)+cen
        P1 = pos
        TM1 = TMass[indices]
        T1 = T[indices]
        EOS1 = EOS[indices]
        hsm1 = HSM[indices]
        V1 = Vel[indices,:]
        SFR1 = SFR[indices]
        fH1 = fH[indices]
        rho1 = rho[indices]
        Z1 = Z[indices]
       
        indices = where((gns==GN) & (sgns==0))[0]#Selecting all the bound gas
        if len(indices)>3:#MARTINI needs at least 3 gas particles to work
         P1 = P1[indices]
         TM1 = TM1[indices]#total gas particle mass
         T1 = T1[indices]
         EOS1 = EOS1[indices]
         hsm1 = hsm1[indices]
         V1 = V1[indices]
         rho1 = rho1[indices]
         fH1 = fH1[indices]
         Z1 = Z1[indices]
         SFR1 = SFR1[indices]

         T1[where((EOS1>0) & (SFR1>0))[0]] = 1e+4#Star forming gas particles on the temperature floor

         #radiative transfer approximation for fHI (Stevens et al. 2019)
         HM = Hmass(method=j, radius=None, pos=None, mass=TM1,fneutral=None,SFR=SFR1,X=fH1,Z=Z1,rho=rho1,temp=T1,redshift=z,UVB='HM12',local=local)
         M1 = HM[0]
         M2 = HM[1]

         if j==2:
          pr='GK11'
         if j==4:
          pr='K13'
         if j==5:
          pr='GD14'
         if j==6:
          pr='L08'
         if j==7:
          pr='BR06'
         if j==8:
          pr='allatomic'

         #Adding Hubble Flow
         a=1.0/(1+z)#scale factor
         H=100*hpar
         Hz=H*sqrt(Om[0]*(1+z)**3+Om[1])#Hubble constant in units of h            
         V1 = V1*sqrt(a)+Hz*P1

         #HI mass weighted centering
         PCM = sum(multiply(M1[:,newaxis],P1),axis=0)/sum(M1)
         VCM = sum(multiply(M1[:,newaxis],V1),axis=0)/sum(M1)
         ctype = 'HICOM'

         r = sqrt(sum((P1-cen)**2,axis=1))#particle radii with respect to the potential centre

         P = (P1 - PCM)
         V = (V1 - VCM)
         xyz_g = P.T
         vxyz_g = V.T

         r1 = sqrt(sum(P**2,axis=1))
         inds = where(r1<=percentile(r1,70))[0]#indices of inner-most 70% particles

         ascale=60
         for d in [78]:#galaxy distance in Mpc
           for gnoise in [0]:#Gaussian instrumental noise in mJy   
            inc = round(rinc[l],2)
            orient = round(rorient[l],2)
            if martini=='y':
             for vres in [1.4]:#Effective velocity resolution
              source = SPHSource(
               distance=d*U.Mpc,
               rotation={'L_coords': (inc * U.deg, orient * U.deg)},
               ra=0. * U.deg,
               dec=0. * U.deg,
               h = hpar,
               T_g = T1*U.K,
               mHI_g = M1*U.Msun,
               xyz_g = xyz_g*U.Mpc, 
               vxyz_g = vxyz_g*U.km/U.s,
               hsm_g = hsm1*U.Mpc
              )
        
              datacube = DataCube(
               n_px_x=128,
               n_px_y=128,
               n_channels=500,
               px_size=ascale * U.arcsec,
               channel_width=vres * U.km * U.s ** -1,
               velocity_centre=source.vsys
              )
        
              beam = GaussianBeam(
               bmaj=3.5*ascale* U.arcsec,
               bmin=3.5*ascale* U.arcsec,
               bpa=0. * U.deg,
               truncate=4.
              )

              noise = GaussianNoise(
               rms = gnoise*1e-3/(pi*(3.5*60)**2) * U.Jy * U.arcsec ** -2
              ) 

              spectral_model = GaussianSpectrum(
               sigma='thermal'
              )    

              sph_kernel = DiracDeltaKernel()#GaussianKernel.mimic(WendlandC2Kernel)

              M = Martini(
               source=source,
               datacube=datacube,
               beam=beam,
               noise=noise,
               spectral_model=spectral_model,
               sph_kernel=sph_kernel
              )

              M.insert_source_in_cube(skip_validation=True)
              M.add_noise()
              M.convolve_beam()
 
              hdf='/scratch/pawsey0119/amanuwal/'+direct+'/100MpcGMWC2_HI_cenG'+str(GN)+'i'+str(inc)+'o'+str(orient)+'cntr'+ctype+'_'+pr+'_vres'+str(vres)+'_rms'+str(gnoise)+'_d'+str(d)+'.hdf5'
              f = M.write_hdf5(hdf, channels='velocity', memmap = True, compact = False)
              #print(hdf)
              flux = f['FluxCube'].value
              flux = sum(flux,axis=1)
              flux = sum(flux,axis=0)
              vel = unique(f['channel_mids'].value)*1e-3 - 100*hpar*d
              f.close()

              f = h5.File(hdf,'w')
              dset = f.create_dataset('Flux', data = flux)
              dset = f.create_dataset('Velocity', data = vel)
              f.close()

            #Mass Distribution
            if mdist=='y':
             do_rot = eye(3)

             #aligning L
             frac=0.3

             Ps = P[inds]
             Vs = V[inds]
             xyz = Ps.T
             vxyz = Vs.T

             #print('shape(vxyz):',shape(vxyz))
 
             m = M1[inds]

             rsort = argsort(sum(power(xyz, 2), axis=0), kind='quicksort')
             p = m[newaxis] * vxyz
             L = cross(xyz, p, axis=0)
             p = p[:, rsort]
             L = L[:, rsort]
             m = m[rsort]
             mcumul = cumsum(m) / sum(m)
             Nfrac = argmin(abs(mcumul - frac))
             Nfrac = max([Nfrac, 100])  # use a minimum of 100 particles
             Nfrac = min([Nfrac, len(m)])  # unless this exceeds particle count
             p = p[:, :Nfrac]
             L = L[:, :Nfrac]
             Ltot = sqrt(sum(power(sum(L, axis=1), 2)))
             Lhat = sum(L, axis=1) / Ltot
             zhat = Lhat / sqrt(sum(power(Lhat, 2)))  # normalized
             xaxis = array([1., 1., 1.])  # default unlikely Laxis
             xhat = xaxis - xaxis.dot(zhat) * zhat
             xhat = xhat / sqrt(sum(power(xhat, 2)))  # normalized
             yhat = cross(zhat, xhat)  # guarantees right-handedness

             rotmat = vstack((xhat, yhat, zhat))  # units will be dropped (desired)
             rotmat = roll(rotmat, 1, axis=0)

             do_rot = rotmat.dot(do_rot)

             #do_rot = rotation_matrix(orient, axis='x').dot(do_rot)
             do_rot = rotation_matrix(inc, axis='y').dot(do_rot)
             do_rot = rotation_matrix(orient, axis='x').dot(do_rot)

             v = matmul(do_rot,vxyz_g)
             p = matmul(do_rot,xyz_g)
       
             if singdish=='y':
              v = matmul(do_rot,vxyz_g)
              p = matmul(do_rot,xyz_g)

              vx = v[0]

              #Add Thermal Broadening
              k_b = 1.380648528e-23
              m_p = 1.6726219e-27
              sig = sqrt(k_b*T1/m_p)*1e-3/sqrt(3)#km/s; for los velocity      

              #vres = 1.4
              for vres in [1.4]:#effective velocity resolution
               dvlos = 1.4
               vbins = arange(-840-dvlos/2,840+dvlos,dvlos)
               vl = vbins[:-1]
               vu = vbins[1:]
               vcens = 0.5*(vl+vu)

               vxs = tile(vx,(len(vcens),1))#each row for each velocity bin edge
               sigs = tile(sig,(len(vcens),1))

               vls = multiply(vl[:,newaxis],ones((len(vl),len(vx))))
               vus = multiply(vu[:,newaxis],ones((len(vu),len(vx))))

               weights = gaussint(vxs,sigs,vls,vus)
               ws = sum(weights,axis=0)
               ws[ws==0] = 1
               weights = weights/ws#for conserving mass

               #mimicing the effect of a Gaussian beam
               theta = (sqrt(p[1]**2+p[2]**2)/d)*(180*60/pi)#angle subtended by the particle onto the beam (in arcmin)
               M1s = bsens(theta,3.5)*M1#3.5 arcmin is the FWHM of Arecibo beam

               M1s = weights*M1s#Mass in each bin for each particle
               #pool = mp.Pool(mp.cpu_count())
               #pool.map(gaussianm,[i for i in range(len(vx))])#range(0,90+15,15)])
               #pool.close()       
               Nsm = int(vres/1.4)
               dm = sum(M1s,axis=1)#Total Mass in each velocity bin
               gnoise1 = gnoise*sqrt(Nsm)
               flux = dm*(1+zgal)/((2.356e+5)*d**2*dvlos) + random.normal(scale=gnoise1*1e-3,size=len(dm))
               flux = convolve(flux, Box1DKernel(Nsm))#Box car smoothing
                
               fn = '/scratch/pawsey0119/amanuwal/'+direct+'/100MpcGMWC2_HIparts_cenG'+str(GN)+'i'+str(inc)+'o'+str(orient)+'cntr'+ctype+'_'+pr+'_vres'+str(vres)+'_rms'+str(gnoise)+'_d'+str(d)+'.hdf5'
               f1 = h5.File(fn, "w")
               f1.create_dataset('p', data = p)
               f1.create_dataset('vlos', data = vx)
               f1.create_dataset('v', data = v)
               #f1.create_dataset('sff', data = sff)
               f1.create_dataset('Flux', data = flux)
               f1.create_dataset('MHI', data = M1)
               f1.create_dataset('SFR', data = SFR1)
               f1.create_dataset('MH2', data= M2)
               f1.create_dataset('Mgas', data = TM1)
               f1.create_dataset('dm', data = dm)
               f1.create_dataset('Velocity', data = vcens)
               #f.create_dataset('m_r', data = m_r)
               #f.create_dataset('r', data = p[1])
               f1.close()

             sel = where(r<=0.07)[0]#applying a 70 kpc aperture (omit if you want to use all the bound H I)
             vxyz_g1 = vxyz_g[:,sel]
             xyz_g1 = xyz_g[:,sel]
             M11 = M1[sel]
             T11 = T1[sel]
             SFR11 = SFR1[sel]
             M21 = M2[sel]
             TM11 = TM1[sel]

             v = matmul(do_rot,vxyz_g1)
             p = matmul(do_rot,xyz_g1)

             vx = v[0]
             #print('shape(vx):',shape(vx))
 
             #Add Thermal Broadening
             k_b = 1.380648528e-23
             m_p = 1.6726219e-27

             sig = sqrt(k_b*T11/m_p)*1e-3/sqrt(3)#contribution from thermal disperson in km/s (for line-of-sight velocity) 
  
             for vres in [1.4]:#effective velocity resolution  
              dvlos = 1.4 
              vbins = arange(-840-dvlos/2,840+dvlos,dvlos)
              vl = vbins[:-1]
              vu = vbins[1:]
              vcens = 0.5*(vl+vu)

              vxs = tile(vx,(len(vcens),1))#each row for each velocity bin edge
              sigs = tile(sig,(len(vcens),1))
  
              vls = multiply(vl[:,newaxis],ones((len(vl),len(vx))))
              vus = multiply(vu[:,newaxis],ones((len(vu),len(vx))))            
 
              weights = gaussint(vxs,sigs,vls,vus)
              ws = sum(weights,axis=0)
              ws[ws==0] = 1
              weights = weights/ws#for conserving mass
             
              M1s = weights*M11#Mass in each bin for each particle
              #pool = mp.Pool(mp.cpu_count())
              #pool.map(gaussianm,[i for i in range(len(vx))])#range(0,90+15,15)])
              #pool.close()      
              fn = '/scratch/pawsey0119/amanuwal/'+direct+'/100MpcGMWC2_HIparts_cenG'+str(GN)+'i'+str(inc)+'o'+str(orient)+'cntr'+ctype+'_'+pr+'_vres'+str(vres)+'_rms'+str(gnoise)+'_d'+str(d)+'.hdf5'        
              Nsm = int(vres/1.4)
              dm = sum(M1s,axis=1)#Total Mass in each velocity bin
              gnoise1 = gnoise*sqrt(Nsm)
              zgal = z_at_value(Planck15.luminosity_distance,d*U.Mpc)
              flux = dm*(1+zgal)/((2.356e+5)*d**2*dvlos) + random.normal(scale=gnoise1*1e-3,size=len(dm))
              flux = convolve(flux, Box1DKernel(Nsm))#Box car smoothing
              #dm = binit(vxs,M1s,statistic='sum',bins=vbins)[0]                                 
              #fn = '/scratch/pawsey0119/amanuwal/'+direct+'/100MpcGMWC2_HIparts_cenG'+str(GN)+'i'+str(inc)+'o'+str(orient)+'cntr'+ctype+'_'+pr+'_vres'+str(vres)+'_rms'+str(gnoise)+'_d'+str(d)+'.hdf5'
              f1 = h5.File(fn, "w")
              dset = f1.create_dataset('p', data = p)
              dset = f1.create_dataset('vlos', data = vx)
              dset = f1.create_dataset('v', data = v)
              #dset = f1.create_dataset('sff', data = sff)
              dset = f1.create_dataset('Flux', data = flux)
              dset = f1.create_dataset('MHI', data = M11)
              dset = f1.create_dataset('SFR', data = SFR11)
              dset = f1.create_dataset('MH2', data= M21)
              dset = f1.create_dataset('Mgas', data = TM11)
              dset = f1.create_dataset('dm', data = dm)
              dset = f1.create_dataset('Velocity', data = vcens)
              #dset = f.create_dataset('m_r', data = m_r)
              #dset = f.create_dataset('r', data = p[1])
              f1.close()

     #Parallelising the generation of H I line profiles for each galaxy (can be done as these are independent tasks. Do not use task-based parallelisation for storing values in an array)           
     pool = mp.Pool(mp.cpu_count())
     pool.map(martini_hi,[l for l in range(len(Grps))])#range(0,90+15,15)])
     pool.close()  
