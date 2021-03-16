import h5py  as h5
import os
from   numpy import *
from scipy.integrate import quad

def ReadEagleParticles_Star(Base,DirList,fend,exts):
    fn        = Base+DirList+'/particledata_'+exts+fend+'/'+'eagle_subfind_particles_'+exts+fend+'.0.hdf5'
    print ' __' ; print ' Particles:',Base+DirList+'/particledata_'+exts+fend+'/'+'eagle_subfind_particles_'+exts+fend,' ...'
    fs            = h5.File(fn,"r")
    RuntimePars   = fs['RuntimePars'].attrs
    Header        = fs['Header'].attrs
    #Units         = fs['Units'].attrs
    #Constants     = fs['Constants'].attrs
    FNumPerSnap   = RuntimePars['NumFilesPerSnapshot']
    NumParts      = Header['NumPart_ThisFile']
    NumPartTot    = Header['NumPart_Total']
    h   = Header['HubbleParam']
    om            = [Header['Omega0'],Header['OmegaLambda'],Header['OmegaBaryon']]
    fs.close()

    if NumPartTot[4]>0:
        PosStar        = empty((NumPartTot[4],3),dtype=float)
        VelStar        = empty((NumPartTot[4],3),dtype=float)
        MassStar       = empty(NumPartTot[4],    dtype=float)
        #BirthDensity   = empty(NumPartTot[4],    dtype=float)
        Metallicity    = empty(NumPartTot[4],    dtype=float)
        Star_aform     = empty(NumPartTot[4],    dtype=float)
        Star_tform     = empty(NumPartTot[4],    dtype=float)
        BindingEnergy  = empty(NumPartTot[4],    dtype=float)
        HSML_Star      = empty(NumPartTot[4],    dtype=float)
        GrpNum_Star    = empty(NumPartTot[4],    dtype=int)
        SubNum_Star    = empty(NumPartTot[4],    dtype=int)
        MaxT           = empty(NumPartTot[0],    dtype=float)
        MaxTa          = empty(NumPartTot[0],    dtype=float) 
        MaxTt          = empty(NumPartTot[0],    dtype=float)

    NStar_c            = 0

    for ifile in range(0,FNumPerSnap):
        if os.path.isdir(Base+DirList+'data/'):
            fn                 = Base+DirList+'data/particledata_'+exts+fend+'/'+'eagle_subfind_particles_'+exts+fend+'.'+str(ifile)+'.hdf5'
        else:
            fn                 = Base+DirList+'/particledata_'+exts+fend+'/'+'eagle_subfind_particles_'+exts+fend+'.'+str(ifile)+'.hdf5'
        fs                     = h5.File(fn,"r")
        Header                 = fs['Header'].attrs
        NumParts               = Header['NumPart_ThisFile']

        def lbt(a,omm=om[0],oml=om[1]):
         z = 1.0/a-1
         t = zeros(len(z))
         for i in range(len(z)):
          t[i] = 1e+3*3.086e+16/(3.154e+7*1e+9)*(1.0/(100*h))*quad(lambda z: 1/((1+z)*sqrt(omm*(1+z)**3+oml)),0,z[i])[0]#in billion years
         return(t)

        if NumParts[4]>0:
            PosStar[NStar_c:NStar_c+NumParts[4],:]    = fs["PartType4/Coordinates"].value
            VelStar[NStar_c:NStar_c+NumParts[4],:]    = fs["PartType4/Velocity"].value#Check this
            MassStar[NStar_c:NStar_c+NumParts[4]]     = fs["PartType4/Mass"].value#Check this
            #BirthDensity[NStar_c:NStar_c+NumParts[4]] = fs["PartType4/BirthDensity"].value
            Metallicity[NStar_c:NStar_c+NumParts[4]]  = fs["PartType4/SmoothedMetallicity"].value
            Star_aform[NStar_c:NStar_c+NumParts[4]]   = fs["PartType4/StellarFormationTime"].value
            Star_tform[NStar_c:NStar_c+NumParts[4]]   = lbt(Star_aform[NStar_c:NStar_c+NumParts[4]])
            BindingEnergy[NStar_c:NStar_c+NumParts[4]]= fs["PartType4/ParticleBindingEnergy"].value
            HSML_Star[NStar_c:NStar_c+NumParts[4]]    = fs["PartType4/SmoothingLength"].value
            GrpNum_Star[NStar_c:NStar_c+NumParts[4]]  = fs["PartType4/GroupNumber"].value
            SubNum_Star[NStar_c:NStar_c+NumParts[4]]  = fs["PartType4/SubGroupNumber"].value
            MaxT[NStar_c:NStar_c+NumParts[0]]   = fs["PartType4/MaximumTemperature"].value
            MaxTa[NStar_c:NStar_c+NumParts[0]]   = fs["PartType4/AExpMaximumTemperature"].value
            MaxTt[NStar_c:NStar_c+NumParts[4]]   = lbt(MaxTt[NStar_c:NStar_c+NumParts[4]])
        NStar_c            += NumParts[4]

        fs.close()

    if (NumPartTot[4]>0):
        return {'PosStar':PosStar, 'VelStar':VelStar, 'MassStar':MassStar, 'BEStar':BindingEnergy, 'HSML_Star':HSML_Star,
                'GrpNum_Star':GrpNum_Star, 'SubNum_Star':SubNum_Star, 'SFa':Star_aform, 'SFt':Star_tform, 'Z':Metallicity
               }
