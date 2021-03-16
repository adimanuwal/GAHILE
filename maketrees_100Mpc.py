#Make and store KDTrees for positions of DM, stars, gas and subhalos
from   numpy                 import *
import h5py                  as h5
import time
import os
from misc import *
import pickle
from scipy.spatial import cKDTree as KDTree
import multiprocessing as mp

Ncrit = 1000
Gc      = 43.0091 # Newton's gravitational constant in Gadget Units
files=open('simfiles','r')
lines=files.readlines()
line=lines[0].split('\n')[0].split('_')
Num=line[0]
fend='_'+line[1]
exts         = Num.zfill(3)

#os.chdir('/mnt/su3ctm/amanuwal/') 
fh  = h5.File('HYDRO_'+exts+fend+'_100Mpc_halodat.hdf5','r')
fDM = h5.File('HYDRO_'+exts+fend+'_100Mpc_DM.hdf5','r')
fStar = h5.File('HYDRO_'+exts+fend+'_100Mpc_Star.hdf5','r')
fGas = h5.File('HYDRO_'+exts+fend+'_100Mpc_Gas.hdf5','r')

h = fh['Header/h'].value
BoxSize = fh['Header/BoxSize'].value/h
#SubPos = fh['HaloData/SubPos'].value/h
PosDM = fDM['PartData/PosDM'].value/h
PosStar = fStar['PartData/PosStar'].value/h
PosGas = fGas['PartData/PosGas'].value/h

fh.close()
fDM.close()
fStar.close()
fGas.close()

def makeit(i):
 if i==0:
  Tree = KDTree(PosGas,leafsize=10,boxsize=BoxSize)
  f=open('/group/pawsey0119/amanuwal/gastree.p','wb')
  pickle.dump(Tree,f,protocol=4)
  f.close()
 if i==1:
  Tree1 = KDTree(PosStar,leafsize=10,boxsize=BoxSize)
  f1=open('/group/pawsey0119/amanuwal/startree.p','wb')
  pickle.dump(Tree1,f1,protocol=4)
  f1.close()
 if i==2:
  Tree2 = KDTree(PosDM,leafsize=10,boxsize=BoxSize)
  f2=open('/group/pawsey0119/amanuwal/dmtree.p','wb')
  pickle.dump(Tree2,f2,protocol=4)
  f2.close()
 #if i==3:
  #Tree3 = KDTree(SubPos,leafsize=10,boxsize=BoxSize)
  #f3=open('/group/pawsey0119/amanuwal/subpostree.p','wb')
  #pickle.dump(Tree3,f3)
  #f3.close()

pool = mp.Pool(mp.cpu_count())
pool.map(makeit,[i for i in range(3)])
pool.close()
                      




