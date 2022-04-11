#For computing stripping forces on the galaxies
from numpy import *
import h5py                  as h5
#import matplotlib.pyplot     as plt
from misc import *
import os
#from prody.kdtree.kdtree import KDTree
from galcalc1 import HI_H2_masses as Hmass
from astropy.coordinates.matrix_utilities import rotation_matrix
import pickle

Base    = '/group/pawsey0119/'
DList   = ['amanuwal/']
Gc      = 43.0091 # Newton's gravitational constant in Gadget Units
files=open('simfiles','r')
lines=files.readlines()
s=len(lines)

def Mcomp(mdm,p,rp):
 M = zeros((3,3))
 for i in range(3):
  for j in range(3):
   M[i,j] = sum((mdm/rp**2)*p[:,i]*p[:,j])/sum(mdm/rp**2)#Eq.4 of Thob et al. 2019
 return M

for idir,DirList in enumerate(DList):
  for i in range(s):
    line=lines[i].split('\n')[0].split('_')
    Num=line[0]
    fend='_'+line[1]
    exts         = Num.zfill(3)
    fn = Base+DirList+'HYDRO_'+exts+fend+'_100Mpc_halodat.hdf5'
    fn1 = Base+DirList+'HYDRO_'+exts+fend+'_100Mpc_Gas.hdf5'
    fn2 = Base+DirList+'100Mpc_HImassfunc_satBR06_tforce.hdf5'    
    fn3 = Base+DirList+'HYDRO_'+exts+fend+'_100Mpc_Star.hdf5'
    fn4 = Base+DirList+'HYDRO_'+exts+fend+'_100Mpc_DM.hdf5'

    if (os.path.exists(fn)==True):
     print('\n Reading Header from file:',fn)
     f  = h5.File(fn,"r")
     f1 = h5.File(fn1,"r")
     f2 = h5.File(fn2,"r")
     f3 = h5.File(fn3,"r")
     f4 = h5.File(fn4,"r") 

     hpar = f['Header/h'].value
     Om = f['Header/Omega'].value
     SubPos = f['HaloData/SubPos'].value/hpar
     FirstSub = f['HaloData/FirstSub'].value
     R_200 = f['HaloData/Group_R_Crit200'].value/hpar
     R_max = f['HaloData/Rmax'].value/hpar
     Mass_DM = f['Header/PartMassDM'].value*1e+10/hpar
     SubMass = f['HaloData/SubMass'].value*1e+10/hpar
     BS = f['Header/BoxSize'].value/hpar
     z = f['Header/Redshift'].value
     Velc = f['HaloData/Vbulk'].value
     Etot = f['HaloData/Etot'].value/hpar
     Nsubs = f['Header/Nsubs'].value
     f.close()

     Pos = f1['PartData/PosGas'].value/hpar
     Vel = f1['PartData/VelGas'].value
     EOS = f1['PartData/EOS'].value
     T = f1['PartData/TempGas'].value
     HSM = f1['PartData/HSML_Gas'].value/hpar
     SFR = f1['PartData/SFR'].value
     fH = f1['PartData/fHSall'].value
     Z = f1['PartData/GasSZ'].value
     TMass = f1['PartData/MassGas'].value*1e+10/hpar#total gas particle mass including all species
     rho = f1['PartData/DensGas'].value*hpar**2*1e+10/(1e+6)**3
     #print(min(nH),max(nH))
     GNGas = abs(f1['PartData/GrpNum_Gas'].value)
     SNGas = abs(f1['PartData/SubNum_Gas'].value)
     f1.close()

     #print(max(SNGas))

     GNStar = abs(f3['PartData/GrpNum_Star'].value)
     SNStar = abs(f3['PartData/SubNum_Star'].value)
     PosStar = f3['PartData/PosStar'].value/hpar
     MassStar = f3['PartData/MassStar'].value*1e+10/hpar  
     VelStar = f3['PartData/VelStar'].value
     f3.close()

     PosDM = f4['PartData/PosDM'].value/hpar
     GNDM = abs(f4['PartData/GrpNum_DM'].value)
     SNDM = abs(f4['PartData/SubNum_DM'].value)
     f4.close()

     GNs = f2['GN'].value
     SGNs = f2['SGN'].value
     Mstar = f2['Mstar'].value
     #Ngas = f2['Ngas'].value
     nhi50 = f2['nhi50a'].value
     rhi = f2['r90_h'].value
     MHI = f2['MHIa'].value
     Sigr = f2['Sigr_h'].value
     r_cen = f2['r_cen'].value
     f2.close()

     #Select galaxies above the cuts
     inds = where((Mstar>1e+9) & (MHI/Mstar>0.02) & (MHI>=1e+8) & (nhi50>=1000))[0]#xGASS cut
     Grps = GNs[inds]
     Sgrps = SGNs[inds]
     rhi = rhi[inds]
     Sigr = Sigr[inds,:]

     size = len(inds)
     ram = zeros(size)
     tidal = zeros(size)
     enc = zeros(size)
     sighi = zeros(size)

     print('For forces on '+str(len(inds))+' galaxies ...')

     print('Loading Trees ...')
     #Tree = KDTree(unitcell=asarray([BS,BS,BS]), coords = Pos, bucketsize=10)
     #Tree1 = KDTree(unitcell=asarray([BS,BS,BS]), coords = PosStar, bucketsize=10)
     #Tree2 = KDTree(unitcell=asarray([BS,BS,BS]), coords = PosDM, bucketsize=10)
     #Tree3 = KDTree(unitcell=asarray([BS,BS,BS]), coords = SubPos, bucketsize=10)

     f=open('/group/pawsey0119/amanuwal/gastree.p','rb')
     Tree=pickle.load(f)
     f=open('/group/pawsey0119/amanuwal/startree.p','rb')
     Tree1=pickle.load(f)
     f=open('/group/pawsey0119/amanuwal/dmtree.p','rb')
     Tree2=pickle.load(f)
     #f=open('/group/pawsey0119/amanuwal/subpostree.p','rb')
     #Tree3=pickle.load(f)

     for l in range(size):
       if rhi[l]==0:
        continue
       GN = int(Grps[l])
       SGN = int(Sgrps[l])

       #print('Ram pressure...')
       #Ram Pressure Stripping (Gunn & Gott 1972)
       r200 = float(R_200[GN-1])

       indices = Tree.query_ball_point(x=SubPos[FirstSub[GN-1]+SGN],r=r200)
       gns = GNGas[indices]
       sgns = SNGas[indices]

       igm = where((sgns==0) | (sgns>1e+9))[0]
 
       m = TMass[indices]#total gas particle mass
       rho1 = rho[indices]
       p = Pos[indices] - SubPos[FirstSub[GN-1]+SGN]#satellite's frame
       p = do_wrap(p,BS) + SubPos[FirstSub[GN-1]+SGN]
       v = Vel[indices]

       m = m[igm]
       rho1 = rho1[igm]
       p = p[igm]
       v = v[igm]

       #Adding Hubble Flow
       a=1.0/(1+z)
       H=100*hpar
       Hz=H*sqrt(Om[0]*(1+z)**3+Om[1])            
       v = v*sqrt(a)+Hz*p
       p = p - SubPos[FirstSub[GN-1]+SGN]

       r = sqrt(sum(p**2,axis=1))
       #IGM parameters
       sortind = argsort(r)
       r = r[sortind]
       v = v[sortind]
       rho1 = rho1[sortind]
       m = m[sortind]

       #print(len(sortind))

       sel = arange(500)
       vigm = sum(m[sel,newaxis]*v[sel],axis=0)/sum(m[sel])

       indices1 = arange(FirstSub[GN-1]+1,FirstSub[GN-1]+Nsubs[GN-1])
       #Tree3.search(center=reshape(GroupPos[GrNr],(3,)),radius=float(R_200[GrNr]))
       #indices1 = Tree3.getIndices()
       #indices1 = indices1[indices1!=FirstSub[GrNr]]
       if len(indices1)!=0:
        subpos = SubPos[indices1]-SubPos[FirstSub[GN-1]]
        subpos = do_wrap(subpos,BS)+SubPos[FirstSub[GN-1]]
        subind = where(indices1==FirstSub[GN-1]+SGN)[0]
        subpos1 = subpos[subind]
        submass = SubMass[indices1]
        dfcen = sqrt(sum((subpos-SubPos[FirstSub[GN-1]])**2,axis=1))
        centosub = subpos1 - SubPos[FirstSub[GN-1]]
        rsearch = sqrt(sum(centosub**2))

        G = 6.674081e-11
        msol = 1.989e+30
        pctom = 3.086e+16
        if rsearch<2*r200:
         Vbulk = Velc[FirstSub[GN-1]+SGN]
         Vbulk = Vbulk*sqrt(a)+Hz*subpos1
         vigm = vigm - Vbulk
         vigm_mod = sqrt(sum(vigm**2))
         rhoigm =sum(m[sel]*rho1[sel])/sum(m[sel])

         pram = rhoigm*vigm_mod**2*(msol*1e+6/pctom**3)#ram pressure
         #Mid-plane acceleration
         epsi = 2.66*1e-3#Plummer-equivalent softening
         sighi[l] = 1#interp(rhi[l]*1e-3,r_cen,Sigr[l])
         sigism = sighi[l]/0.752*msol/pctom**2

         #Stars
         stars=1
         if stars==1:
          indices = Tree1.query_ball_point(x=SubPos[FirstSub[GN-1]+SGN],r=r200)

          gns = GNStar[indices]
          sgns = SNStar[indices]
          
          gal = where((gns==GN) & (sgns==SGN))[0]
          ms = MassStar[indices]#total gas particle mass
          ps = PosStar[indices]-SubPos[FirstSub[GN-1]+SGN]#satellite's frame
          ps = do_wrap(ps,BS)
 
          ms = ms[gal]
          ps = ps[gal]
 
          #finding the mid-plane using intertia tensor for stars within R_HI
          pa = ps
          ma = ms
          rp = sqrt(sum(ps**2,axis=1))
          Rap = rhi[l]*1e-3
          sel = where(rp<Rap)[0]
          p = ps[sel]
          m = ms[sel]
          rp = rp[sel]

          M = Mcomp(m,p,rp)#Quadrupole moment tensor   

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

           R = (a**2/(b*c))**(1/3.0)*Rap

           #Current iteration
           r1 = dot(p,vecs[2])/sqrt(sum(vecs[2]**2))
           r2 = dot(p,vecs[1])/sqrt(sum(vecs[1]**2))
           r3 = dot(p,vecs[0])/sqrt(sum(vecs[0]**2))

           rp = sqrt(r1**2 + (r2/q1)**2 + (r3/s1)**2)
           sel = where(rp**2<=R**2)[0]
           p = p[sel]
           rp = rp[sel]
           m = m[sel]

           if (steps>100) or (shape(p)[0]<10):
            conv[GrNr] = 0
            if steps>100:
             msteps[GrNr]=1
            if shape(p)[0]<10:
             lparts[GrNr]=1
            print('Failed to converge!')
            break

           steps+=1

           M = Mcomp(m,p,rp)#Quadrupole moment tensor   
           vals,vecs = linalg.eig(M)#eigenvalues and eigenvectors
           axes = sqrt(vals)
           sortind = argsort(axes)
           c,b,a = axes[sortind]
           vecs = vecs[sortind]

           q2 = b/a
           s2 = c/a

           t1 = 1-s2/s1
           t2 = 1-q2/q1

          majv = vecs[2]#vector along major axis
          minv = vecs[1]#vector along minor axis 

          #Making galaxies edge-on based on the derived mid-plane 
          vec = cross(majv,minv)
          Lhat = vec / sqrt(sum(vec**2))
          zhat = Lhat / sqrt(sum(power(Lhat, 2)))  # normalized
          xaxis = array([1., 1., 1.])  # default unlikely Laxis
          xhat = xaxis - xaxis.dot(zhat) * zhat
          xhat = xhat / sqrt(sum(power(xhat, 2)))  # normalized
          yhat = cross(zhat, xhat)  # guarantees right-handedness
          rotmat = vstack((xhat, yhat, zhat))  # units will be dropped (desired)
          rotmat = roll(rotmat, 1, axis=0)
          do_rot = eye(3)
          do_rot = rotmat.dot(do_rot)
          do_rot = rotation_matrix(90, axis='y').dot(do_rot)
     
          ma = append(ma,ms)
          pa = append(pa,ps,axis=0)

         #DM
         DM=1
         if DM==1:
          indices = Tree2.query_ball_point(x=SubPos[FirstSub[GN-1]+SGN],r=r200)

          gns = GNDM[indices]
          sgns = SNDM[indices]

          gal = where((gns==GN) & (sgns==SGN))[0]
          ps = PosDM[indices]-SubPos[FirstSub[GN-1]+SGN]#satellite's frame
          ps = do_wrap(ps,BS)
          ps = ps[gal]
 
          ma = append(ma,Mass_DM*ones(len(gal)))
          pa = append(pa,ps,axis=0)

         #Gas
         indices = Tree.query_ball_point(x=SubPos[FirstSub[GN-1]+SGN],r=r200)

         gns = GNGas[indices]
         sgns = SNGas[indices]

         gal = where((gns==GN) & (sgns==SGN))[0]
         ms = TMass[indices]#total gas particle mass
         ps = Pos[indices]-SubPos[FirstSub[GN-1]+SGN]#satellite's frame
         ps = do_wrap(ps,BS)

         ma = append(ma,ms[gal])
         pa = append(pa,ps[gal],axis=0)

         pa = matmul(do_rot,pa.T)#rotate the position vectors
         pa = pa.T

        #Potentials
        #sl=0.7e-3*hpar#Softening length
        #p_samp=p
        #p_one = ones((3,int(shape(p)[0])))
        #dr = sqrt(sum((multiply(p_samp[:,:,newaxis],p_one) - transpose(p_samp))**2,axis=1))#distance matrices
        #dr[where(dr<sl)] = sl
        
        #dr_inv = 1.0/dr
        #phi = G*m*(sum(dr_inv,axis=1)-1.0/sl)*msol/(pctom*1e+6)

        #sel = where((abs(p[:,1])>=(rhi[l]-1)*1e-3) & (abs(p[:,1])<=(rhi[l]+1)*1e-3) & (abs(p[:,2])>=0) & (abs(p[:,2])<=2*epsi))[0]
        #phis = phi[sel]
        #zs = abs(p[sel,2])

        #phi1 = interp(0,zs,phis)
        #phi2 = interp(2*epsi,zs,phis)

        #pgrav = abs(phi2 - phi1)/(2*epsi)*sigism#gravitational pressure
        #x = linspace(-rhi[l]*1e-3,rhi[l]*1e-3,100)
        #y = sqrt((rhi[l]*1e-3)**2-x**2)

         thetas = arange(0,360,10)
         nt = len(thetas)
         prest = zeros(nt)
         x = rhi[l]*1e-3*cos(thetas*pi/180.0)
         y = rhi[l]*1e-3*sin(thetas*pi/180.0)

         pgrav = zeros(nt)
         for k in range(nt):
          point = array([x[k],y[k],2*epsi])
          dr = pa - point
          drsq = sum(dr**2,axis=1)
          perp = array([0,0,-2*epsi])
          costheta = dot(dr,perp)/(sqrt(drsq)*2*epsi)
          a1 = sum(G*ma*msol/(drsq*(1e+6*pctom)**2)*costheta)#acceleration at 2*epsilon above the plane

          point = array([x[k],y[k],0])
          dr = pa - point
          drsq = sum(dr**2,axis=1)*(1e+6*pctom)**2
          perp = array([0,0,-2*epsi])
          costheta = dot(dr,perp)/(sqrt(drsq)*2*epsi)
          a2 = sum(G*ma*msol/(drsq*(1e+6*pctom)**2)*costheta)

          pgrav[k] = (a1-a2)*sigism
 
         ram[l] = log10(pram/abs(mean(pgrav)))#pressure ratio

        #print('Tidal stripping...')
         #Tidal Stripping (Binney & Tremaine 2008)
         #masses within r     
         indices = Tree.query_ball_point(x=SubPos[FirstSub[GN-1]],r=rsearch)
         Mgas = sum(TMass[indices])
 
         indices = Tree1.query_ball_point(x=SubPos[FirstSub[GN-1]],r=rsearch)
         Mstar = sum(MassStar[indices])
    
         indices = Tree2.query_ball_point(x=SubPos[FirstSub[GN-1]],r=rsearch)
         Mdm = Mass_DM*len(indices)

         Mr = Mgas+Mstar+Mdm
 
         d = 0.05
         r_bins=10**(arange(-1-d/2,0+1.5*d,d))*r200#r/R200 bins till R200
         r_cen=(r_bins[1:]+r_bins[:-1])/2.0
         r = sqrt(sum(pa**2,axis=1))

         mprof = binit(r,ma,statistic='sum',bins=r_bins)[0]
         sel = where(mprof!=0)[0]
         print(len(where(mprof==0)[0]))
         ms = diff(log10(mprof))/diff(log10(r_cen))
         x = r_cen[:-1]
         m = interp(rsearch,x,ms)
         rtidaln[l] = (SubMass[FirstSub[GN-1]+SGN]/((2-m)*Mr))**(1/3.0)*rsearch*1e+3#tidal radius (kpc)
         
         tidal[l] = log10(rhi[l]/rtidal)

         #print(tidal[l])
         #print('Satellite encounters...')
        #Satellite Encounters (Binney & Tremaine 2008)   
        #if len(indices1)!=0:
         vsub = Velc[indices1]
         vsub = vsub*sqrt(a)+Hz*subpos
         vrel = vsub - Vbulk
         vrel_mod = sqrt(sum(vrel**2,axis=1))
         prel = subpos - subpos1
         prel_mod = sqrt(sum(prel**2,axis=1))

         #print(len(vrel_mod))
         sel = where((prel_mod!=0) & (dfcen<2*R_200[GN-1]))[0]#exclude the subhalo under consideration and false sats
         #print(len(sel))
         if len(sel)!=0:
          vrel = vrel[sel]
          prel = prel[sel]
          submass = submass[sel]
          prel_mod = prel_mod[sel]
          vrel_mod = vrel_mod[sel]
 
          #sintheta = sqrt(1-(dot(prel,vrel)/(prel_mod*vrel_mod))**2)
          b = prel_mod*1e+6*pctom#*sintheta
          rsq = sum(ma*sum(pa**2,axis=1))/sum(ma)*(1e+6*pctom)**2
 
          Es = max(4/3.0*G**2*SubMass[FirstSub[GN-1]+SGN]*(submass/vrel_mod)**2*rsq/b**4*(msol**3/1e+6))#Injected Energy
          print('Es:',Es)
          print('submass:',SubMass[FirstSub[GN-1]+SGN])
          print('min b',min(b))
          print('min vrel_mod',min(vrel_mod)) 
          
          enc[l] = log10(Es/abs(Etot[FirstSub[GN-1]+SGN]*1e+10*msol*1e+6))#Encounter energy ratio
          print(enc[l])

fn ='satforces_psig_n.hdf5'
out  = h5.File(fn, 'w')     
dset = out.create_dataset('GN', data=Grps)
dset = out.create_dataset('SGN', data=Sgrps)
dset = out.create_dataset('ram', data=ram)
dset = out.create_dataset('tidal', data=tidal)
dset = out.create_dataset('enc', data=enc)
dset = out.create_dataset('Sig_hi', data=sighi)
out.close()
print('Completed!')
        


