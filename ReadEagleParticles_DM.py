import h5py  as h5
import os
from   numpy import *

def ReadEagleParticles_DM(Base,DirList,fend,exts):
    fn        = Base+DirList+'/particledata_'+exts+fend+'/'+'eagle_subfind_particles_'+exts+fend+'.0.hdf5'
    print ' __' ; print ' Particles:',Base+DirList+'/particledata_'+exts+fend+'/'+'eagle_subfind_particles_'+exts+fend,' ...'
    fs            = h5.File(fn,"r")
    RuntimePars   = fs['RuntimePars'].attrs
    Header        = fs['Header'].attrs
    #Units         = fs['Units'].attrs
    #Constants     = fs['Constants'].attrs
    HubbleParam = Header['HubbleParam']
    PartMassDM = Header['MassTable'][1]
    FNumPerSnap   = RuntimePars['NumFilesPerSnapshot']
    NumParts      = Header['NumPart_ThisFile']
    NumPartTot    = Header['NumPart_Total']

    fs.close()

    PosDM              = empty((NumPartTot[1],3),dtype=float)
    VelDM              = empty((NumPartTot[1],3),dtype=float)
    IDsDM              = empty(NumPartTot[1],    dtype=float)
    GrpNum_DM          = empty(NumPartTot[1],    dtype=int)
    SubNum_DM          = empty(NumPartTot[1],    dtype=int)

    NDM_c              = 0
    for ifile in range(0,FNumPerSnap):
        if os.path.isdir(Base+DirList+'data/'):
            fn                 = Base+DirList+'data/particledata_'+exts+fend+'/'+'eagle_subfind_particles_'+exts+fend+'.'+str(ifile)+'.hdf5'
        else:
            fn                 = Base+DirList+'/particledata_'+exts+fend+'/'+'eagle_subfind_particles_'+exts+fend+'.'+str(ifile)+'.hdf5'
        fs                     = h5.File(fn,"r")
        Header                 = fs['Header'].attrs
        NumParts               = Header['NumPart_ThisFile']

        PosDM[NDM_c:NDM_c+NumParts[1],:]              = fs["PartType1/Coordinates"].value
        VelDM[NDM_c:NDM_c+NumParts[1],:]              = fs["PartType1/Velocity"].value
        IDsDM[NDM_c:NDM_c+NumParts[1]]                = fs["PartType1/ParticleIDs"].value
        GrpNum_DM[NDM_c:NDM_c+NumParts[1]]            = fs["PartType1/GroupNumber"].value
        SubNum_DM[NDM_c:NDM_c+NumParts[1]]            = fs["PartType1/SubGroupNumber"].value

        NDM_c              += NumParts[1]

        fs.close()

    if (NumPartTot[0]>0):
        return {'PartMassDM':PartMassDM, 'HubbleParam':HubbleParam,
                'IDsDM':IDsDM,     
                'PosDM':PosDM, 'VelDM':VelDM,'GrpNum_DM':GrpNum_DM, 'SubNum_DM':SubNum_DM          
               }
    else:
        return {'PartMassDM':PartMassDM, 'HubbleParam':HubbleParam,
                'IDsDM':IDsDM, 
                'GrpNum_DM':GrpNum_DM,
                'PosDM':PosDM,'VelDM':VelDM,
                'SubNum_DM':SubNum_DM
                #'NDM':NDM_c, 'NGas':NGas_c, 'NStar':NStar_c
               }

