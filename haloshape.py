#Halo Shape
from   numpy                 import *
import h5py                  as h5
import time
import os
from misc import *
import multiprocessing as mp
import pickle

def Mcomp(mdm,p,rp):
 M = zeros((3,3))
 for i in range(3):
  for j in range(3):
   M[i,j] = sum((mdm/rp**2)*p[:,i]*p[:,j])/sum(mdm/rp**2)#Eq.4 of Thob et al. 2019
 return M

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

h = fDM['Header/h'].value
Om = fh['Header/Omega'].value
z = fh['Header/Redshift'].value
BoxSize = fh['Header/BoxSize'].value/h
Ngrps = fh['Header/Ngrps'].value
GroupPos = fh['HaloData/GroupPos'].value/h
#SubPos = fh['HaloData/SubPos'].value
mdm = fh['Header/PartMassDM'].value*1e+10/h
M_200 = fh['HaloData/Group_M_Crit200'].value*1e+10/h
R_200 = fh['HaloData/Group_R_Crit200'].value/h
MassType = fh['HaloData/MassType'].value*1e+10/h
FirstSub = fh['HaloData/FirstSub'].value
fh.close()

PosDM = fDM['PartData/PosDM'].value/h
fDM.close()

fn = 'cendmhisym1.hdf5'
f = h5.File(fn,'r')
GNe = f['GN'].value#group numbers to process
f.close()

Ngrps = len(GNe)
print('Loading the KDTree ...')
f=open('dmtree_'+red+'.p','rb')
Tree=pickle.load(f)
f.close()

d = 0.1
rbyr200 = 10**(arange(-1.0,0+d,d))
size = len(rbyr200)+1
a = zeros((Ngrps,size))
b = zeros((Ngrps,size))
c = zeros((Ngrps,size))
epsi = zeros((Ngrps,size))
T = zeros((Ngrps,size))
conv = zeros((Ngrps,size))
msteps = zeros((Ngrps,size))
lparts = zeros((Ngrps,size))
#d = 0.1
#rbyr200 = 10**(arange(-1.0,0+d,d))

print('\nCreating the hdf5 file for the simulation box...')
fn ='100Mpc_haloshapes.hdf5'
out  = h5.File(fn, 'w')
for GrNr in range(Ngrps):
      #print('Group Nr.',GrNr+1)
      #if M_200[GrNr]/mdm > Ncrit and MassType[FirstSub[GrNr],4] != 0:
       for i in range(size):
         if i!=size-1:
          rsearch = rbyr200[i]*float(R_200[GrNr])
         else:
          rsearch = 0.07
         cen = GroupPos[GNe[GrNr]-1]
         lcen_DM = Tree.query_ball_point(x=cen,r=rsearch)

         if str(lcen_DM)!='None':
          p_DM = PosDM[lcen_DM,:] - cen#group frame     
          p = do_wrap(p_DM,BoxSize)
 
          rp = sqrt(p[:,0]**2 + p[:,1]**2 + p[:,2]**2)
          M = Mcomp(mdm,p,rp)#Quadrupole moment tensor   
          vals,vecs = linalg.eig(M)#eigenvalues and eigenvectors
          axes = sqrt(vals)
          sortind = argsort(axes)
          c,b,a = axes[sortind]
          vecs = vecs[sortind]
          q2 = b/a
          s2 = c/a

          #Iterative convergence
          t1 = 1
          t2 = 1

          steps=1

          while (t1>0.01) and (t2>0.01):
           #Previous iteration
           q1 = q2
           s1 = s2

           R = (a**2/(b*c))**(1/3.0)*rsearch

           #Current iteration
           r1 = dot(p,vecs[2])/sqrt(sum(vecs[2]**2))
           r2 = dot(p,vecs[1])/sqrt(sum(vecs[1]**2))
           r3 = dot(p,vecs[0])/sqrt(sum(vecs[0]**2))

           rp = sqrt(r1**2 + (r2/q1)**2 + (r3/s1)**2)
           sel = where(rp**2<=R**2)[0]
           p = p[sel]
           rp = rp[sel]

           if (steps>100) or (shape(p)[0]<10):
            conv[GrNr] = 0
            if steps>100:
             msteps[GrNr]=1
            if shape(p)[0]<10:
             lparts[GrNr]=1
            print('Failed to converge!')
            break

           steps+=1
           
           M = Mcomp(mdm,p,rp)#Quadrupole moment tensor   
           vals,vecs = linalg.eig(M)#eigenvalues and eigenvectors
           axes = sqrt(vals)
           sortind = argsort(axes)
           c,b,a = axes[sortind]
           vecs = vecs[sortind]

           q2 = b/a
           s2 = c/a

           t1 = 1-s2/s1
           t2 = 1-q2/q1

          epsi[GrNr,i] = 1-c/a
          T[GrNr,i] = (a**2-b**2)/(a**2-c**2)
          #print('Flatenning:',epsi[GrNr])
          #print('Triaxiality:',T[GrNr])

dset = out.create_dataset('GN', data=GNe)
dset = out.create_dataset('conv', data=conv)
dset = out.create_dataset('msteps', data=msteps)
dset = out.create_dataset('lparts', data=lparts)
dset = out.create_dataset('epsi', data=epsi)
dset = out.create_dataset('T', data=T)
out.close()
            
          
          
          
                  
          
  

           
          
