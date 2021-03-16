#HI line asymmetry measurements
from   scipy                 import spatial
from   scipy.integrate import *
from scipy.signal import argrelextrema
from   numpy                 import *
import h5py                  as h5
import time
import sys
import os
import multiprocessing as mp
from lmfit import Minimizer, Parameters
import matplotlib.pyplot as plt
import matplotlib.lines as pltline

def lop(par,flux,vel):
 v = par['v']
 fleft = sum(flux[vel<v])
 fright = sum(flux[vel>v])
 return abs(fleft-fright)/abs(fleft+fright)

os.chdir('/scratch/pawsey0119/amanuwal/cenprofs/')
os.system('ls > /group/pawsey0119/amanuwal/hifiles')
os.chdir('/group/pawsey0119/amanuwal/')

files=open('hifiles','r')
lines=files.readlines()
size=len(lines)
m=1#method(fixed)
fn = 'cendmhisym1.hdf5'
out = h5.File(fn,'w')

direct = 'cenprofs'

fn1 = 'HYDRO_028_z000p000_100Mpc_halodat.hdf5'
f1 = h5.File(fn1,'r')
hpar1 = f1['Header/h'].value
FSub1 = f1['HaloData/FirstSub'].value
Vmax1 = f1['HaloData/Vmax'].value
f1.close()

fn1 = 'HYDRO_028_z000p000_25Mpc_halodat.hdf5'
f1 = h5.File(fn1,'r')
hpar2 = f1['Header/h'].value
FSub2 = f1['HaloData/FirstSub'].value
Vmax2 = f1['HaloData/Vmax'].value
f1.close()

box = ['' for x in range(size)]
GN = zeros(size,dtype=int)
inc = zeros(size,dtype=float)
orient = zeros(size,dtype=float)
presc = ['' for x in range(size)]
fratio = zeros(size)
k = zeros(size)
pratio = zeros(size)
vp1 = zeros(size)
vp2 = zeros(size)
flipres = zeros(size)
voff = zeros(size)
veq = zeros(size)
w951 = zeros(size)
w952 = zeros(size)
w95 = zeros(size)
w201 = zeros(size)
w202 = zeros(size)
w20 = zeros(size)
w501 = zeros(size)
w502 = zeros(size)
w50 = zeros(size)
vmax = zeros(size)
nhi30 = zeros(size)
nhi40 = zeros(size)
nhi50 = zeros(size)
ngas = zeros(size)
minal = zeros(size)
minal1 = zeros(size)
vmean = zeros(size)
nchan = zeros(size)

for i in range(len(lines)):
  fname = '/scratch/pawsey0119/amanuwal/'+direct+'/'+lines[i].split('\n')[0]
  spname = fname.split('.hdf5')[0].split('_')
  spname2 = spname[2].split('cntr')[0]
  GN[i] = int(spname2.split('i')[0].split('cenG')[1])
  inc[i] = float(spname2.split('i')[1].split('o')[0])
  orient[i] = float(spname2.split('i')[1].split('o')[1])
  presc[i] = 'BR06'
  box[i] = spname[0].split('/')[-1].split('G')[0]
  if box[i]=='100Mpc':
   vmax[i] = Vmax1[FSub1[GN[i]-1]]
  if box[i]=='25Mpc':
   vmax[i] = Vmax2[FSub2[GN[i]-1]]

  fname = '/scratch/pawsey0119/amanuwal/'+direct+'/'+box[i]+'GMWC2_HIparts_cenG'+str(GN[i])+'i'+str(inc[i])+'o'+str(orient[i])+'cntrHICOM'+'_'+presc[i]+'.hdf5'
  f = h5.File(fname,'r')
  #flux = f['Flux'].value
  dvlos = 1.4#km/s
  flux = f['dm'].value/dvlos
  vel = f['Velocity'].value
  mhi = f['MHI'].value
  mgas = f['Mgas'].value
  f.close()

  fhi = mhi/mgas
  nhi30[i] = len(fhi[fhi>0.3])
  nhi40[i] = len(fhi[fhi>0.4])
  nhi50[i] = len(fhi[fhi>0.5])
  ngas[i] = len(mgas)
 
  #Velocity Widths
  pinds = argrelextrema(flux, greater)[0]
  peaks = flux[pinds]
  vpeaks = vel[pinds]#sorted by velocities 
  
  #if len(peaks)==None:
  print('Gal',i+1)
  #if len(pinds)==0:
   #print flux
  reject = where(peaks>0.2*max(peaks))[0]#reject false peaks

  peaks = peaks[reject]
  vpeaks = vpeaks[reject]
  
  #print 'peaks:',peaks
  fmax = max(peaks)

  if len(peaks)>=2:
   if len(peaks[peaks==fmax])>1:#if two major peaks have same level
     fpeak1 = fmax
     fpeak2 = fpeak1
     vpeaks = vpeaks[peaks==fmax]
     vpeak1 = min(vpeaks)
     vpeak2 = max(vpeaks)
   else:#get the highest and second highest peaks
     fsmax = max(peaks[peaks!=fmax])#second highest peak
     vfmax = vpeaks[peaks==fmax][0]
     vfsmax = vpeaks[peaks==fsmax][0]
     peaks = array([fmax,fsmax])
     vpeaks = array([vfmax,vfsmax])
     vpeak1 = min(vpeaks)
     vpeak2 = max(vpeaks)
     #print 'vpeaks:',vpeaks
     #print 'fpeaks:',fpeaks
     fpeak1 = peaks[vpeaks==vpeak1]
     fpeak2 = peaks[vpeaks==vpeak2]
  else:#just one peak
    fpeak1 = max(peaks)
    fpeak2 = max(peaks)
    vpeak1 = max(vpeaks)
    vpeak2 = max(vpeaks)

  vp1[i] = vpeak1
  vp2[i] = vpeak2

  #W20 and W50
  vp = vpeaks[peaks==fmax]
  fpeak = fmax
  left = where(vel<=vp1[i])[0]
  right = where(vel>=vp2[i])[0]
  vleft = vel[left]
  vright = vel[right]
  fleft = flux[left]
  fright = flux[right]
  if len(vleft[fleft<=0.2*fpeak])!=0 and len(vright[fright<=0.2*fpeak])!=0:
   w201[i] = max(vleft[fleft<=0.2*fpeak])
   w202[i] = min(vright[fright<=0.2*fpeak])
   w20[i] = w202[i] - w201[i]
  if len(vleft[fleft<=0.5*fpeak])!=0 and len(vright[fright<=0.5*fpeak])!=0:
   w501[i] = max(vleft[fleft<=0.5*fpeak])
   w502[i] = min(vright[fright<=0.5*fpeak])
   w50[i] = w502[i] - w501[i]

  #print 'vpeak1,vpeak2:'+str(vpeak1)+','+str(vpeak2)
  vp1[i] = vpeak1
  vp2[i] = vpeak2

  vels = vel[flux>0.5*max(flux)]
  midv = 0.5*(vels[0]+vels[-1])

  p = 0.95
  #cut = 0.03
  vleft = vel[vel<midv]
  vright = vel[vel>midv]
  fleft = flux[vel<midv]
  #vl = vleft[fleft<cut*max(flux)][-1]
  #fleft = fleft[vleft>=vl] 
  #vleft = vleft[vleft>=vl]
  fleft = cumsum(fleft[::-1]*dvlos)
  fintl = fleft[-1]
  vleft = vleft[::-1]
  w951[i] = vleft[fleft>=p*fintl][0]
  fright = flux[vel>midv]
  #vr = vright[fright<cut*max(flux)][0]
  #fright = fright[vright<=vr]
  #vright = vright[vright<=vr]
  fright = cumsum(fright*dvlos)
  fintr = fright[-1]
  w952[i] = vright[fright>=p*fintr][0]
   
  w95[i] = w952[i]-w951[i]#p% flux limits

  if GN[i]<0:
   fig = plt.figure()
   fig.set_size_inches(6, 4, forward=True)
   pos  = [0.07, 0.06, 0.85, 0.85] ; ax  = fig.add_axes(pos)
 
   ax.plot(vel,flux/max(flux))
   ylims = ax.get_ylim()
   y1 = float(ylims[0])
   y2 = float(ylims[1])
 
   line4=pltline.Line2D([w951[i],w951[i]],[y1,y2],linewidth=1.5,linestyle='--',color='purple',label=r'W$_\mathrm{95}$')
   line5=pltline.Line2D([w952[i],w952[i]],[y1,y2],linewidth=1.5,linestyle='--',color='purple',label=r'W$_\mathrm{95}$')
 
   ax.add_line(line4)
   ax.add_line(line5)
   ax.set_xlim(-500,500)
   #print(cut*max(flux)/1e+7)
   #print(flux/1e+7)
   plt.show()

  if w95[i]!=0:
   midv = w951[i]+0.5*w95[i]

   #Normalized residual
   lch = len(flux[where((vel>=w951[i]) & (vel<midv))[0]])#number of left channels
   rch = len(flux[where((vel>midv) & (vel<=w952[i]))[0]])#number of right channels
   le = w951[i]
   re = w952[i]
   indl = where(vel==w951[i])[0]
   indr = where(vel==w952[i])[0]
   #print('GN,vres,rms:',GN[i],vres[i],rms[i])
   #print(lch,rch)
   if lch>rch:
     le = w951[i]
     #re = w952[i]+(lch-rch)*dvlos
     re = vel[indr+int(lch-rch)]
   if lch<rch:
     #le = w201[i]+(lch-rch)*dvlos
     le = vel[indl+int(lch-rch)]
     re = w952[i]

   f1 = flux[where((vel>=le) & (vel<midv))[0]]
   f2 = flux[where((vel>midv) & (vel<=re))[0]]

   f2 = f2[::-1]
 
   I = sum(abs(f1-f2))/sum(f1+f2)

   f1 = flux[where((vel<le))[0]]
   f2 = flux[where((vel>re))[0]]
   
   #print(len(f1),len(f2))
   if len(f1)>len(f2):
    f1 = f1[0:len(f2)]
   if len(f1)<len(f2):
    f2 = f2[0:len(f1)]

   f2 = f2[::-1]

   O = 0#sum(abs(f1-f2))/sum(abs(f1+f2))

   flipres[i] = I-O

   #Lopsidedness
   regsel = where((vel>=w951[i]) & (vel<=w952[i]))[0]
   flux = flux[regsel]
   vel = vel[regsel]
   nchan[i] = len(vel)

   f1 = sum(flux[vel<midv])
   f2 = sum(flux[vel>midv])
   fratio[i] = abs(f1-f2)/(f1+f2)

   #kurtosis
   vm = sum(vel*flux)/sum(flux) #flux weighted mean velocity
   mu2 = sum(flux*(vel - vm)**2)/sum(flux)
   mu4 = sum(flux*(vel - vm)**4)/sum(flux)
   k[i] = mu4/mu2**2 - 3

   #Velocity offset
   par = Parameters()
   par.add('v',value=vm)
   minner = Minimizer(lop,par,fcn_args=(flux,vel))
   result = minner.minimize(method='Nelder')
   veq[i] = float(result.params['v'])
   voff[i] = abs(midv-veq[i])/(0.5*w95[i])
   minal[i] = float(result.residual)

   f1 = sum(flux[vel<vm])
   f2 = sum(flux[vel>vm])
   minal1[i] = abs(f1-f2)/(f1+f2)
   vmean[i] = vm

   #Peak asymmetry
   pratio[i] = abs(fpeak1 - fpeak2)/(fpeak1+fpeak2)
   #print 'pratio:',pratio[i]

#pool = mp.Pool(mp.cpu_count())
#pool.map(asym,[j for j in range(len(lines))])#range(0,90+15,15)])
#pool.close()                  

out.create_dataset('GN', data = GN)
#out.create_dataset('presc',dtype='S10',data = presc)
out.create_dataset('vp1', data = vp1)
out.create_dataset('vp2', data = vp2)
out.create_dataset('w951', data = w951)
out.create_dataset('w952', data = w952)
out.create_dataset('w95', data = w95)
out.create_dataset('w201', data = w201)
out.create_dataset('w202', data = w202)
out.create_dataset('w20', data = w20)
out.create_dataset('w501', data = w501)
out.create_dataset('w502', data = w502)
out.create_dataset('w50', data = w50)
#out.create_dataset('box', dtype='S10',data = box)
out.create_dataset('inc', data = inc)
out.create_dataset('orient', data = orient)
#out.create_dataset('w50f', data = w50f)
#out.create_dataset('w20f', data = w20f)
out.create_dataset('nhi30', data = nhi30)
out.create_dataset('nhi40', data = nhi40)
out.create_dataset('nhi50', data = nhi50)
out.create_dataset('ngas', data = ngas)
out.create_dataset('vmax', data = vmax)
out.create_dataset('veq', data = veq)
out.create_dataset('fratio', data = fratio)
out.create_dataset('kurtosis', data = k)
out.create_dataset('pratio', data = pratio)
out.create_dataset('flipres',data = flipres)
out.create_dataset('voff', data = voff)
out.create_dataset('minal', data = minal)
out.create_dataset('vmean', data = vmean)
out.create_dataset('minal1', data = minal1)
out.create_dataset('nchan', data = nchan)
out.close()
print('Completed Bro!')
