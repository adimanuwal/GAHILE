import h5py                  as h5
import os
from   numpy                 import *

def ReadEagleSubfind_halo(Base,DirList,fend,exts):
   fn            = Base+DirList+'/groups_'+exts+fend+'/'+'eagle_subfind_tab_'+exts+fend+'.0.hdf5'
   print ' __' ; print ' Directory:',Base+DirList+'/groups_'+exts+fend+'/'+'eagle_subfind_tab_'+exts+fend+' ...'
   fs               = h5.File(fn,"r")
   Header           = fs['Header'].attrs
   Ntask            = Header['NTask']
   TotNgroups       = Header['TotNgroups']
   TotNsubgroups    = Header['TotNsubgroups']

   fs.close()
   
   # --- Define Group Arrays
   if TotNgroups > 0:
      Group_M_Crit200  = empty(TotNgroups,    dtype=float)
      Group_R_Crit200  = empty(TotNgroups,    dtype=float)
      #Group_R_Mean200  = empty(TotNgroups,    dtype=float)
      #GroupMass        = empty(TotNgroups,    dtype=float)
      GroupPos         = empty((TotNgroups,3), dtype=float)
      #GroupLen         = empty(TotNgroups,    dtype=int)
      FirstSub         = empty(TotNgroups,    dtype=int)
      NumOfSubhalos    = empty(TotNgroups,    dtype=int)
   # --- Define Subhalo Arrays
      #HalfMassRad      = empty((TotNsubgroups,6), dtype=float)
      #HalfMassProjRad  = empty((TotNsubgroups,6), dtype=float)
      MassType         = empty((TotNsubgroups,6), dtype=float)
      Pcom           = empty((TotNsubgroups,3), dtype=float)
      SubPos           = empty((TotNsubgroups,3), dtype=float)
      #SubLenType       = empty((TotNsubgroups,6), dtype=int)
      Vbulk            = empty((TotNsubgroups,3), dtype=float)
      Vmax             = empty(TotNsubgroups,     dtype=float)
      VmaxRadius       = empty(TotNsubgroups,     dtype=float)
      SFRate           = empty(TotNsubgroups,     dtype=float)
      #BHMass           = empty(TotNsubgroups,     dtype=float)
      #StellarMass      = empty(TotNsubgroups,     dtype=float)
      SubMass          = empty(TotNsubgroups,     dtype=float)
      SubLen           = empty(TotNsubgroups,     dtype=int)
      #SubOffset        = empty(TotNsubgroups,     dtype=int)
      SubGroupNumber   = empty(TotNsubgroups,     dtype=int)
      GroupNumber      = empty(TotNsubgroups,     dtype=int)
      #MostBoundID      = empty(TotNsubgroups,     dtype=int)
      Etot      = empty(TotNsubgroups,     dtype=int)
      Ekin      = empty(TotNsubgroups,     dtype=int)

#      Mass_001kpc      = empty((TotNsubgroups,6), dtype=float)
#      Mass_003kpc      = empty((TotNsubgroups,6), dtype=float)
#      Mass_005kpc      = empty((TotNsubgroups,6), dtype=float)
#      Mass_010kpc      = empty((TotNsubgroups,6), dtype=float)
#      Mass_020kpc      = empty((TotNsubgroups,6), dtype=float)
      Mass_030kpc      = empty((TotNsubgroups,6), dtype=float)
#      Mass_040kpc      = empty((TotNsubgroups,6), dtype=float)
      #Mass_050kpc      = empty((TotNsubgroups,6), dtype=float)
#      Mass_070kpc      = empty((TotNsubgroups,6), dtype=float)
#      Mass_100kpc      = empty((TotNsubgroups,6), dtype=float)

#      SFR_001kpc       = empty(TotNsubgroups,     dtype=float)
#      SFR_003kpc       = empty(TotNsubgroups,     dtype=float)
#      SFR_005kpc       = empty(TotNsubgroups,     dtype=float)
#      SFR_010kpc       = empty(TotNsubgroups,     dtype=float)
#      SFR_020kpc       = empty(TotNsubgroups,     dtype=float)
      SFR_030kpc       = empty(TotNsubgroups,     dtype=float)
#      SFR_040kpc       = empty(TotNsubgroups,     dtype=float)
#      SFR_050kpc       = empty(TotNsubgroups,     dtype=float)
      SFR_070kpc       = empty(TotNsubgroups,     dtype=float)
#      SFR_100kpc       = empty(TotNsubgroups,     dtype=float)

      #subtab_index     = empty(TotNsubgroups,     dtype=int)
      #sub_location     = empty(TotNsubgroups,     dtype=int)

      NGrp_c           = 0
      NSub_c           = 0

      for ifile in range(0,Ntask):
         if os.path.isdir(Base+DirList+'data/'):
            fn         = Base+DirList+'data/groups_'+exts+fend+'/'+'eagle_subfind_tab_'+exts+fend+'.'+str(ifile)+'.hdf5'
         else:
            fn         = Base+DirList+'/groups_'+exts+fend+'/'+'eagle_subfind_tab_'+exts+fend+'.'+str(ifile)+'.hdf5'
         fs            = h5.File(fn,"r")      
         Header        = fs['Header'].attrs
         HubbleParam   = Header['HubbleParam']
         #PartMassDM    = Header['PartMassDM']
         Redshift      = Header['Redshift']
         Om            = [Header['Omega0'],Header['OmegaLambda'],Header['OmegaBaryon']]
         Ngroups       = Header['Ngroups']
         BoxSize       = Header['BoxSize']
         #TotNgroups    = Header['TotNgroups']
         Nsubgroups    = Header['Nsubgroups']
         #TotNsubgroups = Header['TotNsubgroups']

         if Ngroups > 0:
            Group_M_Crit200[NGrp_c:NGrp_c+Ngroups]     = fs["FOF/Group_M_Crit200"].value
            Group_R_Crit200[NGrp_c:NGrp_c+Ngroups]     = fs["FOF/Group_R_Crit200"].value
            #Group_R_Mean200[NGrp_c:NGrp_c+Ngroups]     = fs["FOF/Group_R_Mean200"].value
            #GroupLen[NGrp_c:NGrp_c+Ngroups]            = fs["FOF/GroupLength"].value
            #GroupMass[NGrp_c:NGrp_c+Ngroups]           = fs["FOF/GroupMass"].value
            GroupPos[NGrp_c:NGrp_c+Ngroups]            = fs["FOF/GroupCentreOfPotential"].value
            FirstSub[NGrp_c:NGrp_c+Ngroups]            = fs["FOF/FirstSubhaloID"].value
            NumOfSubhalos[NGrp_c:NGrp_c+Ngroups]       = fs["FOF/NumOfSubhalos"].value

            NGrp_c += Ngroups

         if Nsubgroups > 0:
            #HalfMassRad[NSub_c:NSub_c+Nsubgroups,:]    = fs["Subhalo/HalfMassRad"].value
            #HalfMassProjRad[NSub_c:NSub_c+Nsubgroups,:]= fs["Subhalo/HalfMassProjRad"].value
            #SubLenType[NSub_c:NSub_c+Nsubgroups,:]     = fs["Subhalo/SubLengthType"].value
            MassType[NSub_c:NSub_c+Nsubgroups,:]       = fs["Subhalo/MassType"].value
            SubPos[NSub_c:NSub_c+Nsubgroups,:]         = fs["Subhalo/CentreOfPotential"].value
            Pcom[NSub_c:NSub_c+Nsubgroups,:]         = fs["Subhalo/CentreOfMass"].value
            Vbulk[NSub_c:NSub_c+Nsubgroups]            = fs["Subhalo/Velocity"].value
            Vmax[NSub_c:NSub_c+Nsubgroups]             = fs["Subhalo/Vmax"].value
            VmaxRadius[NSub_c:NSub_c+Nsubgroups]       = fs["Subhalo/VmaxRadius"].value
            SubLen[NSub_c:NSub_c+Nsubgroups]           = fs["Subhalo/SubLength"].value
            SFRate[NSub_c:NSub_c+Nsubgroups]           = fs["Subhalo/StarFormationRate"].value
            #BHMass[NSub_c:NSub_c+Nsubgroups]           = fs["Subhalo/BlackHoleMass"].value
            #StellarMass[NSub_c:NSub_c+Nsubgroups]      = fs["Subhalo/Stars/Mass"].value
            SubMass[NSub_c:NSub_c+Nsubgroups]          = fs["Subhalo/Mass"].value
            #SubOffset[NSub_c:NSub_c+Nsubgroups]        = fs["Subhalo/SubOffset"].value
            SubGroupNumber[NSub_c:NSub_c+Nsubgroups]   = fs["Subhalo/SubGroupNumber"].value
            GroupNumber[NSub_c:NSub_c+Nsubgroups]      = fs["Subhalo/GroupNumber"].value
            #MostBoundID[NSub_c:NSub_c+Nsubgroups]      = fs["Subhalo/IDMostBound"].value
      
#            Mass_001kpc[NSub_c:NSub_c+Nsubgroups,:]    = fs["Subhalo/ApertureMeasurements/Mass/001kpc"].value
#            Mass_003kpc[NSub_c:NSub_c+Nsubgroups,:]    = fs["Subhalo/ApertureMeasurements/Mass/003kpc"].value
#            Mass_005kpc[NSub_c:NSub_c+Nsubgroups,:]    = fs["Subhalo/ApertureMeasurements/Mass/005kpc"].value
#            Mass_010kpc[NSub_c:NSub_c+Nsubgroups,:]    = fs["Subhalo/ApertureMeasurements/Mass/010kpc"].value
#            Mass_020kpc[NSub_c:NSub_c+Nsubgroups,:]    = fs["Subhalo/ApertureMeasurements/Mass/020kpc"].value
            Mass_030kpc[NSub_c:NSub_c+Nsubgroups,:]    = fs["Subhalo/ApertureMeasurements/Mass/030kpc"].value
#            Mass_040kpc[NSub_c:NSub_c+Nsubgroups,:]    = fs["Subhalo/ApertureMeasurements/Mass/040kpc"].value
            #Mass_050kpc[NSub_c:NSub_c+Nsubgroups,:]    = fs["Subhalo/ApertureMeasurements/Mass/050kpc"].value
#            Mass_070kpc[NSub_c:NSub_c+Nsubgroups,:]    = fs["Subhalo/ApertureMeasurements/Mass/070kpc"].value
#            Mass_100kpc[NSub_c:NSub_c+Nsubgroups,:]    = fs["Subhalo/ApertureMeasurements/Mass/100kpc"].value

#            SFR_001kpc[NSub_c:NSub_c+Nsubgroups]       = fs["Subhalo/ApertureMeasurements/SFR/001kpc"].value
#            SFR_003kpc[NSub_c:NSub_c+Nsubgroups]       = fs["Subhalo/ApertureMeasurements/SFR/003kpc"].value
#            SFR_005kpc[NSub_c:NSub_c+Nsubgroups]       = fs["Subhalo/ApertureMeasurements/SFR/005kpc"].value
#            SFR_010kpc[NSub_c:NSub_c+Nsubgroups]       = fs["Subhalo/ApertureMeasurements/SFR/010kpc"].value
#            SFR_020kpc[NSub_c:NSub_c+Nsubgroups]       = fs["Subhalo/ApertureMeasurements/SFR/020kpc"].value
            SFR_030kpc[NSub_c:NSub_c+Nsubgroups]       = fs["Subhalo/ApertureMeasurements/SFR/030kpc"].value
#            SFR_040kpc[NSub_c:NSub_c+Nsubgroups]       = fs["Subhalo/ApertureMeasurements/SFR/040kpc"].value
#            SFR_050kpc[NSub_c:NSub_c+Nsubgroups]       = fs["Subhalo/ApertureMeasurements/SFR/050kpc"].value
            SFR_070kpc[NSub_c:NSub_c+Nsubgroups]       = fs["Subhalo/ApertureMeasurements/SFR/070kpc"].value
#            SFR_100kpc[NSub_c:NSub_c+Nsubgroups]       = fs["Subhalo/ApertureMeasurements/SFR/100kpc"].value

            Etot[NSub_c:NSub_c+Nsubgroups]       = fs["Subhalo/TotalEnergy"].value
            Ekin[NSub_c:NSub_c+Nsubgroups]       = fs["Subhalo/KineticEnergy"].value
            #subtab_index[NSub_c:NSub_c+Nsubgroups].fill(ifile)
            #sub_location[NSub_c:NSub_c+Nsubgroups]     = arange(0,Nsubgroups)

            NSub_c += Nsubgroups
         
         fs.close()

      return {'Group_M_Crit200':Group_M_Crit200, 'Group_R_Crit200':Group_R_Crit200, 'GroupPos':GroupPos, 
              #'Group_R_Mean200':Group_R_Mean200, 'GroupMass':GroupMass,            #'GroupLen':GroupLen,                
              'FirstSub':FirstSub,               #'subtab_index':subtab_index,       'sub_location':sub_location,
              'NumOfSubhalos':NumOfSubhalos,    'Vmax':Vmax, 'Vbulk':Vbulk, 'Rmax':VmaxRadius,       
              'SubPos':SubPos,  'Pcom':Pcom,              'SubLen':SubLen,                   
              'SubGroupNumber':SubGroupNumber,   #'SubOffset':SubOffset,           'HalfMassProjRad':HalfMassProjRad,
              'GroupNumber':GroupNumber,         #'Mass_001kpc':Mass_001kpc,         'Mass_003kpc':Mass_001kpc,
#              'Mass_003kpc':Mass_003kpc,         'Mass_005kpc':Mass_005kpc,         'Mass_010kpc':Mass_010kpc,
              #'Mass_020kpc':Mass_020kpc,         'Mass_070kpc':Mass_070kpc,         'Mass_040kpc':Mass_040kpc,
              'Mass_030kpc':Mass_030kpc,         'SFR_030kpc':SFR_030kpc,            #'Mass_050kpc':Mass_050kpc,         
#              'SFR_001kpc':SFR_001kpc,           'SFR_003kpc':SFR_001kpc,          'Mass_100kpc':Mass_100kpc,
#              'SFR_003kpc':SFR_003kpc,           'SFR_005kpc':SFR_005kpc,           'SFR_010kpc':SFR_010kpc,
              'SFR_030kpc':SFR_030kpc,            'SFR_070kpc':SFR_070kpc,           #'SFR_040kpc':SFR_040kpc,
#              'SFR_050kpc':SFR_050kpc,           'SFR_020kpc':SFR_020kpc,           'SFR_100kpc':SFR_100kpc,
              'TotNgroups':TotNgroups,           'TotNsubgroups':TotNsubgroups,     
              'Redshift':Redshift,               'BoxSize':BoxSize, 'HubbleParam':HubbleParam,#'PartMassDM':PartMassDM,               # 'StellarMass':StellarMass,         
              'Omega':Om, 'SFRate':SFRate,       'MassType':MassType,               
              'SubMass':SubMass, 'Etot':Etot, 'Ekin':Ekin                # 'MostBoundID':MostBoundID
              }
   else:
      return {'Omega':Om,  'BoxSize':BoxSize, 'Redshift':Redshift, 'HubbleParam':HubbleParam}# 'PartMassDM':PartMassDM}
   
