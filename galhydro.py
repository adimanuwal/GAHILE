#Atomic and molecular hydrogen mass of EAGLE galaxies for a redshift z snapshot
import matplotlib
from   numpy                 import *
import h5py                  as h5
import matplotlib.pyplot     as plt
from misc import *
import matplotlib.lines as pltline
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os
from scipy.constants import *
#from prody.kdtree.kdtree import KDTree
from astropy.coordinates import CartesianRepresentation,\
    CartesianDifferential, ICRS
from astropy.coordinates.matrix_utilities import rotation_matrix
from hydromass import HI_H2_masses as Hmass
import multiprocessing as mp
from scipy.stats import binned_statistic as binit
#from pyfof import friends_of_friends as fof

#os.chdir('/home/adiman/Desktop/PhD/EAGLE')
Base    = '/group/pawsey0119/'
DList   = ['amanuwal/']
Gc      = 43.0091 # Newton's gravitational constant in Gadget Units
files=open('simfiles','r')
lines=files.readlines()
size=len(lines)
sdens=1
for idir,DirList in enumerate(DList):
    line=lines[0].split('\n')[0].split('_')
    Num=line[0]
    fend='_'+line[1]
    exts         = Num.zfill(3)
    fn = Base+DirList+'HYDRO_'+exts+fend+'_100Mpc_halodat.hdf5'
    fn1 = Base+DirList+'HYDRO_'+exts+fend+'_100Mpc_Gas.hdf5'
    fn2 = Base+DirList+'HYDRO_'+exts+fend+'_100Mpc_MProfs_Star.hdf5'
    if (os.path.exists(fn)==True):
     print('\n Reading Header from file:',fn)
     fh = h5.File(fn,"r")
     fg = h5.File(fn1,'r')
     fs = h5.File(fn2,'r')

     hpar = fh['Header/h'].value       
     z = fh['Header/Redshift'].value
     a = 1.0/(1+z)
     Om = fh['Header/Omega'].value 
     GroupPos = fh['HaloData/GroupPos'].value*a/hpar
     FirstSub = fh['HaloData/FirstSub'].value
     M_200 = fh['HaloData/Group_M_Crit200'].value*1e+10/hpar 
     BS = fh['Header/BoxSize'].value*a/hpar
     Vbulk = fh['HaloData/Vbulk'].value 
     Pos = fg['PartData/PosGas'].value*a/hpar
     Vel = fg['PartData/VelGas'].value
     T = fg['PartData/TempGas'].value
     SFR = fg['PartData/SFR'].value
     EOS = fg['PartData/EOS'].value
    # HSM = f['PartData/HSML_Gas'].value/hpar
    # VR = f['HaloData/VR'].value
    # fsub = f['HaloData/fsub'].value
    # s = f['HaloData/s'].value
     #s = f['HaloData/s_star30'].value
     fH = fg['PartData/fHSall'].value
     TMass = fg['PartData/MassGas'].value*1e+10/hpar#total gas particle mass including all species
     rho = fg['PartData/DensGas'].value*hpar**2*1e+10/(a*1e+6)**3
     Z = fg['PartData/GasSZ'].value
     GNGas = abs(fg['PartData/GrpNum_Gas'].value)
     SNGas = abs(fg['PartData/SubNum_Gas'].value)

     #print(min(nH),max(nH))

     Mr_star = fs['HaloData/Mr_star'].value*1e+10/hpar#Stellar Particle Mass in solar masses
     r_cen = fs['HaloData/r_cen'].value*1e+3#radius from central centre in kpc
     fh.close()
     fs.close()
     fg.close()

     Mr_star = Mr_star[:,unique(where(r_cen <= 30))]#taking values till 30 kpc
     Ms_30 = sum(Mr_star, axis=1)
     indices = where(Ms_30>=1e+9)[0]
     Grps = indices+1 #Selecting well resolved galaxies

     #For H I mass profile and surface density
     rbins = 10**(arange(-2,log10(70)+1.5*0.1,0.1))*1e-3
     r1 = 10**((log10(rbins[1:])+log10(rbins[:-1]))/2.0)
     areas = pi*(rbins[1:]**2-rbins[:-1]**2)*(1e+6)**2#pc^2
    
  #   Tree = KDTree(unitcell=asarray([BS,BS,BS]), coords = Pos, bucketsize=10)  

     f=open('/group/pawsey0119/amanuwal/gastree.p','rb')
     Tree=pickle.load(f)
     f.close()
  
     def masses(j):
     #for j in [2,4,5,6,7]:
      size = len(Grps)
      MHI = zeros(size)
      MH2 = zeros(size)
      Mgas = zeros(size)
      Ngas = zeros(size)
      nhi50 = zeros(size)
      nhi40 = zeros(size)
      nhi30 = zeros(size)
      MHIa = zeros(size)
      MH2a = zeros(size)
      Mgasa = zeros(size)
      Ngasa = zeros(size)
      nhi50a = zeros(size)
      nhi40a = zeros(size)
      nhi30a = zeros(size)
      Dist = zeros(size)
      Mstar = zeros(size)
      M200 = zeros(size)
      r50_h = zeros(size)
      r90_h = zeros(size)
      r1_h = zeros(size)
      Mr = zeros((size,len(r1)))
      Mrp = zeros((size,len(r1)))
      Sigr = zeros((size,len(r1)))

      print('No. of galaxies:',size)
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
      if j=='8':
        pr='allatomic'

      fname = Base+DirList+'100Mpc_HImassfunc_'+pr+'_tforce.hdf5'
      f1 = h5.File(fname,"w")

      local=False
      for i in range(size):
       #def masses(i):
       GN = Grps[i]
       print('Group No.:',Grps[i])
       #Tree.search(center=reshape(GroupPos[GN-1],(3,)),radius=0.1)
       #indices = Tree.getIndices()
       indices = Tree.query_ball_point(x=GroupPos[GN-1],r=0.1)
       if str(indices)=='None':
        continue
       if len(indices)<=3:
        continue
       gns = GNGas[indices]
       sgns = SNGas[indices]

       TM1 = TMass[indices]#total gas particle mass
       T1 = T[indices]
       EOS1 = EOS[indices]
       rho1 = rho[indices]
       fH1 = fH[indices]
       Z1 = Z[indices]
       SFR1 = SFR[indices]

       p_gas = Pos[indices,:] - GroupPos[GN-1]#central frame
       V = Vel[indices,:]

       indices = where((gns==GN) & (sgns==0))[0]
       if len(indices)>3:
        TM1 = TM1[indices]#total gas particle mass
        T1 = T1[indices]
        EOS1 = EOS1[indices]
        rho1 = rho1[indices]
        fH1 = fH1[indices]
        Z1 = Z1[indices]
        SFR1 = SFR1[indices]

        T1[(EOS1>0) & (SFR1>0)] = 1e+4
        
        p_gas = p_gas[indices,:]
        p_gas = do_wrap(p_gas,BS)
        dr_mod = sqrt(sum(p_gas**2,axis=1))
        inds = argsort(dr_mod)
        y = cumsum(TM1[inds])
        y = y/y[-1]
        x = dr_mod[inds]
        r50g = interp(0.5,y,x)#gas half-mass radius

        #radiative transfer approximation for fHI 
        NH = Hmass(method=j,pos=p_gas,radius=r50g,mass=TM1,fneutral=None,SFR=SFR1,X=fH1,Z=Z1,rho=rho1,temp=T1,redshift=z,UVB='HM12',local=local)
        M1 = NH[0]
        MHI[i] = sum(M1)
        MH2[i] = sum(NH[1])
        Mgas[i] = sum(TM1)
        M200[i] = M_200[GN-1]
        Mstar[i] = Ms_30[GN-1]
        Ngas[i] = len(indices)
        Dist[i] = sqrt(sum(GroupPos[GN-1]**2))
        fhi = M1/TM1
        nhi50[i] = len(fhi[fhi>0.5])
        nhi40[i] = len(fhi[fhi>0.4])
        nhi30[i] = len(fhi[fhi>0.3])

        dr = sqrt(p_gas[:,0]**2+p_gas[:,1]**2)

        inds = where(dr_mod<=70*1e-3)[0]
        if len(inds)>3:
         Ngasa[i] = len(inds)
         MHIa[i] = sum(M1[inds])
         MH2a[i] = sum(NH[1][inds])
         Mgasa[i] = sum(TM1[inds])
         fhi = M1[inds]/TM1[inds]
         nhi50a[i] = len(fhi[fhi>0.5])
         nhi40a[i] = len(fhi[fhi>0.4])
         nhi30a[i] = len(fhi[fhi>0.3])

         r2 = 10**(arange(-2,log10(70)+0.001,0.001))*1e-3

         Mr[i,:] = binit(dr_mod,M1,statistic='sum',bins=rbins)[0]
         Mrp[i,:] = binit(dr,M1,statistic='sum',bins=rbins)[0]

         intM = interp(r2,r1,cumsum(Mr[i,:]))

         r50_h[i] = max(r2[intM<=0.5*max(intM)])*1e+3
         r90_h[i] = max(r2[intM<=0.9*max(intM)])*1e+3

         #Surface Density
         if MHIa[i]>0:
           #Adding Hubble Flow
           P = p_gas[inds] + GroupPos[GN-1]
           a=1.0/(1+z)
           H=100*hpar
           Hz=H*sqrt(Om[0]*(1+z)**3+Om[1])#Hubble constant in units of h            
           V = V[indices,:]
           V = V[inds]
           V = V*sqrt(a)+Hz*P/a#positions should be in co-moving units here
           vcen = Vbulk[FirstSub[GN-1],:]+Hz*GroupPos[GN-1]/a
           V = V - vcen
           P = P - GroupPos[GN-1]
 
           xyz_g = P.T
           vxyz_g = V.T

           #Clump removal through FoF
           sel = where(fhi>0.5)[0] 
           if len(sel)>0:
            #pos = P[sel]
            #M1 = M1[inds]
            #mass = M1[sel]
            #vel = V[sel]

            #p_one = ones((3,len(sel)))
            #dr = sqrt(sum((multiply(pos[:,:,newaxis],p_one) - transpose(pos))**2,axis=1))#inter-particle spacing 
            #dr = dr[triu_indices(len(sel),k=1)]

            #Clump identification using FOF
            #link = 0.15*mean(dr)#linking length
            #groups = fof(pos,link)
            #size = shape(groups)[0]
            #cmass = zeros(size)
            #for k in range(size):
            # cmass[k] = sum(mass[groups[k]])
            #sel = int(where(cmass==max(cmass))[0])#selecting the most massive clump
            #inds = groups[sel]

            r1 = sqrt(sum(P**2,axis=1))
            inds = where(r1<=percentile(r1,70))[0]

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

            do_rot = rotation_matrix(90, axis='y').dot(do_rot)

            v = matmul(do_rot,vxyz_g)
            p = matmul(do_rot,xyz_g)
            r = sqrt(p[0]**2+p[1]**2)
           
            mprof = binit(r,M1,statistic='sum',bins=rbins)[0]
            Sigr[i,:] = mprof/areas
            Sigr1 = interp(r2,r1,Sigr[i,:])
            if len(r2[Sigr1>=1])!=0:
             r1_h[i] = max(r2[Sigr1>=1])*1e+3

      #pool = mp.Pool(mp.cpu_count())
      #pool.map(masses,[i for i in range(size)])#range(0,90+15,15)])
      #pool.close()      

      f1.create_dataset('GN',data=Grps)
      f1.create_dataset('MHI',data=MHI)
      f1.create_dataset('MH2',data=MH2)
      f1.create_dataset('Mgas',data=Mgas)
      f1.create_dataset('Ngas',data=Ngas)
      f1.create_dataset('nhi50',data=nhi50)
      f1.create_dataset('nhi40',data=nhi40)
      f1.create_dataset('nhi30',data=nhi30)
      f1.create_dataset('MHIa',data=MHIa)
      f1.create_dataset('MH2a',data=MH2a)
      f1.create_dataset('Mgasa',data=Mgasa)
      f1.create_dataset('Ngasa',data=Ngasa)
      f1.create_dataset('nhi50a',data=nhi50a)
      f1.create_dataset('nhi40a',data=nhi40a)
      f1.create_dataset('nhi30a',data=nhi30a)
      f1.create_dataset('Dist',data=Dist)
      f1.create_dataset('Mstar',data=Mstar)
      f1.create_dataset('M200',data=M200)
      f1.create_dataset('r_cen',data=r1)
      f1.create_dataset('Mr_h',data=Mr)
      f1.create_dataset('Mrp_h',data=Mrp)
      f1.create_dataset('Sigr_h',data=Sigr)
      f1.create_dataset('r50_h',data=r50_h)
      f1.create_dataset('r90_h',data=r90_h)
      f1.create_dataset('r1_h',data=r1_h)

      f1.close()
      print('Output:',fname)

     pool = mp.Pool(mp.cpu_count())
     pool.map(masses,[j for j in [2,4,5,6,7]])#range(0,90+15,15)])
     pool.close()
