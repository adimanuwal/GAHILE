from   scipy                 import spatial
from   scipy.integrate import *
from   numpy                 import *
from   misc import *
import h5py                  as h5
import warnings
import time
import sys
import os
from scipy.stats import binned_statistic as binit
from prody.kdtree.kdtree import KDTree

Ncrit=1000
files=open('simfiles','r')
lines=files.readlines()
line=lines[0].split('\n')[0].split('_')
Num=line[0]
fend='_'+line[1]
exts         = Num.zfill(3)

#os.chdir('/mnt/su3ctm/amanuwal/')
ch='y'
fh  = h5.File('HYDRO_'+exts+fend+'_100Mpc_halodat.hdf5','r')
fStar = h5.File('HYDRO_'+exts+fend+'_100Mpc_Star.hdf5','r')
h = fh['Header/h'].value
BS = fh['Header/BoxSize'].value
Ngrps = fh['Header/Ngrps'].value
M_200 = fh['HaloData/Group_M_Crit200'].value
#R_200 = fh['HaloData/Group_R_Crit200'].value
GroupPos = fh['HaloData/GroupPos'].value
MassType = fh['HaloData/MassType'].value
FirstSub = fh['HaloData/FirstSub'].value
PartMassDM = fh['Header/PartMassDM'].value
MassStar = fStar['PartData/MassStar'].value
PosStar = fStar['PartData/PosStar'].value
GNStar = abs(fStar['PartData/GrpNum_Star'].value)
SNStar = abs(fStar['PartData/SubNum_Star'].value)

fh.close()
#fDM.close()
fStar.close()

#print('Building Tree ...')
#Tree = KDTree(unitcell=asarray([BS,BS,BS]), coords = PosStar, bucketsize=10)

print('\nCreating the hdf5 file for the simulation box...')
fn ='HYDRO_'+exts+fend+'_100Mpc_MProfs_Star.hdf5'
output  = h5.File(fn, "w")
grp1 = output.create_group('HaloData')
if(ch=='y'):
        #Mass profiles of centrals
        d=0.1
        r_bins=10**(arange(-5.0-d/2,0+d/2+d,d))#bins till 1 Mpc
        r_cen = 10**((log10(r_bins[1:])+log10(r_bins[:-1]))/2.0)
        nbins = len(r_cen)
        M_star = zeros((Ngrps,nbins))
        Mp_star = zeros((Ngrps,nbins))
        rho_star = zeros((Ngrps,nbins))

        for GrNr in range(Ngrps):
         #lgrp_Gas = where((GrpNum_Gas == GrNr+1))[0]
         if M_200[GrNr]/PartMassDM > Ncrit and MassType[FirstSub[GrNr],4] != 0:
            #lcen_Gas = where((GrpNum_Gas == GrNr+1) & (SubNum_Gas == 0))[0]
            #Tree.search(center=reshape(GroupPos[GrNr],(3,)),radius=1*h)
            #lcen_Star = Tree.getIndices()
            lcen_Star = where((GNStar==GrNr+1) & (SNStar==0))[0]
            pc_star = PosStar[lcen_Star,:] - GroupPos[GrNr,:]#central frame
            mc_star = MassStar[lcen_Star]
            pc_star = do_wrap(pc_star,BS)
            rc_star = sqrt(pc_star[:,0]**2 + pc_star[:,1]**2 + pc_star[:,2]**2)
            rcp_star = sqrt(pc_star[:,0]**2 + pc_star[:,1]**2)
            
            M_star[GrNr,:] = binit(rc_star/h,mc_star,statistic='sum',bins=r_bins)[0]
            Mp_star[GrNr,:] = binit(rcp_star/h,mc_star,statistic='sum',bins=r_bins)[0]
            dr = diff(r_bins)
            rho_star[GrNr,:] = M_star[GrNr,:]/(4*pi*r_cen**2*dr)
             
         lgrph = where(GNStar == GrNr+1)[0]
         #GrpNum = delete(GrpNum, lgrph, 0)
         #SubGrpNum = delete(SubGrpNum, lgrph, 0)
         #MassType = delete(MassType, lgrph, 0)
         #SubPos = delete(SubPos, lgrph, 0)
         GNStar = delete(GNStar, lgrph, 0)
         SNStar = delete(SNStar, lgrph, 0)
         PosStar = delete(PosStar, lgrph, 0)
         MassStar = delete(MassStar, lgrph, 0)

        dset    = grp1.create_dataset('r_cen', data = r_cen)
        dset    = grp1.create_dataset('Mr_star', data = M_star)
       # dset    = grp1.create_dataset('Mr_gas', data = M_gas) 
       # dset    = grp1.create_dataset('Mr_star', data = M_star)
        dset    = grp1.create_dataset('Mrp_star', data = Mp_star)
       # dset    = grp1.create_dataset('Mrp_gas', data = Mp_gas) 
       # dset    = grp1.create_dataset('Mrp_star', data = Mp_star)
        dset    = grp1.create_dataset('rhor_star', data = rho_star)
       # dset    = grp1.create_dataset('rhor_gas', data = rho_gas) 
       # dset    = grp1.create_dataset('rhor_star', data = rho_star)

        output.close()
        print('\nOutput:',fn)
