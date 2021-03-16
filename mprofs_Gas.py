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
#fDM = h5.File('HYDRO_'+exts+fend+'_100Mpc_DM.hdf5','r')
#fStar = h5.File('HYDRO_'+exts+fend+'_100Mpc_Star.hdf5','r')
fGas = h5.File('HYDRO_'+exts+fend+'_100Mpc_Gas.hdf5','r')
h = fh['Header/h'].value
BS = fh['Header/BoxSize'].value
Ngrps = fh['Header/Ngrps'].value
M_200 = fh['HaloData/Group_M_Crit200'].value
#R_200 = fh['HaloData/Group_R_Crit200'].value
GroupPos = fh['HaloData/GroupPos'].value
MassType = fh['HaloData/MassType'].value
#GrpNum = abs(fh['HaloData/GrpNum'].value)
#SubGrpNum = abs(fh['HaloData/SubGrpNum'].value)
FirstSub = fh['HaloData/FirstSub'].value
PartMassDM = fh['Header/PartMassDM'].value
#PosDM = fDM['PartData/PosDM'].value
#GrpNum_DM = abs(fDM['PartData/GrpNum_DM'].value)
#SubNum_DM = abs(fDM['PartData/SubNum_DM'].value)
#MassStar = fStar['PartData/MassStar'].value
#PosStar = fStar['PartData/PosStar'].value
#GrpNum_Star = abs(fStar['PartData/GrpNum_Star'].value)
#SubNum_Star = abs(fStar['PartData/SubNum_Star'].value)
MassGas = fGas['PartData/MassGas'].value
PosGas = fGas['PartData/PosGas'].value
GNGas = abs(fGas['PartData/GrpNum_Gas'].value)
SNGas = abs(fGas['PartData/SubNum_Gas'].value)

fh.close()
#fDM.close()
#fStar.close()
fGas.close()

#print('Building Tree ...')
#Tree = KDTree(unitcell=asarray([BS,BS,BS]), coords = PosGas, bucketsize=10)

print('\nCreating the hdf5 file for the simulation box...')
fn ='HYDRO_'+exts+fend+'_100Mpc_MProfs_Gas.hdf5'
output  = h5.File(fn, "w")
grp1 = output.create_group('HaloData')
if(ch=='y'):
        #Mass profiles of centrals
        d=0.1
        r_bins=10**(arange(-5.0-d/2,0+d/2+d,d))#bins till 1 Mpc
        r_cen=10**(arange(-5.0,0+d,d))
        nbins = len(r_cen)
        M_gas = zeros((Ngrps,nbins))
        Mp_gas = zeros((Ngrps,nbins))
        rho_gas = zeros((Ngrps,nbins))

        for GrNr in range(Ngrps):
         if M_200[GrNr]/PartMassDM > Ncrit and MassType[FirstSub[GrNr],0] != 0:
            lcen_Gas = where((GNGas == GrNr+1) & (SNGas == 0))[0]
            #Tree.search(center=reshape(GroupPos[GrNr],(3,)),radius=1*h)
            #lcen_Gas = Tree.getIndices()            

            pc_gas = PosGas[lcen_Gas,:] - GroupPos[GrNr,:]#central frame
            mc_gas = MassGas[lcen_Gas]
            pc_gas = do_wrap(pc_gas,BS)
            rc_gas = sqrt(pc_gas[:,0]**2 + pc_gas[:,1]**2 + pc_gas[:,2]**2)
            rcp_gas = sqrt(pc_gas[:,0]**2 + pc_gas[:,1]**2)
            
            M_gas[GrNr,:] = binit(rc_gas/h,mc_gas,statistic='sum',bins=r_bins)[0]
            Mp_gas[GrNr,:] = binit(rcp_gas/h,mc_gas,statistic='sum',bins=r_bins)[0]
            dr = diff(r_bins)
            rho_gas[GrNr,:] = M_gas[GrNr,:]/(4*pi*r_cen**2*dr)
         
         lgrp_Gas = where(GNGas == GrNr+1)[0]    
         GNGas = delete(GNGas, lgrp_Gas, 0)
         SNGas = delete(SNGas, lgrp_Gas, 0)
         PosGas = delete(PosGas, lgrp_Gas, 0)
         MassGas = delete(MassGas, lgrp_Gas, 0)

        dset    = grp1.create_dataset('r_cen', data = r_cen)
        dset    = grp1.create_dataset('Mr_gas', data = M_gas)
       # dset    = grp1.create_dataset('Mr_gas', data = M_gas) 
       # dset    = grp1.create_dataset('Mr_star', data = M_star)
        dset    = grp1.create_dataset('Mrp_gas', data = Mp_gas)
       # dset    = grp1.create_dataset('Mrp_gas', data = Mp_gas) 
       # dset    = grp1.create_dataset('Mrp_star', data = Mp_star)
        dset    = grp1.create_dataset('rhor_gas', data = rho_gas)
       # dset    = grp1.create_dataset('rhor_gas', data = rho_gas) 
       # dset    = grp1.create_dataset('rhor_star', data = rho_star)

        output.close()
        print('\nOutput:',fn)
