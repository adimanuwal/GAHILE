import h5py  as h5
import os
from   numpy import *
from scipy.integrate import quad

def ReadEagleParticles_Gas(Base,DirList,fend,exts):
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

    if NumPartTot[0]>0:
        PosGas         = empty((NumPartTot[0],3),dtype=float)
        VelGas         = empty((NumPartTot[0],3),dtype=float)
        DensGas        = empty(NumPartTot[0],    dtype=float)
        HSML_Gas       = empty(NumPartTot[0],    dtype=float)
        #fHAll          = empty(NumPartTot[0],    dtype=float)
        TempGas        = empty(NumPartTot[0],    dtype=float)
        MaxT           = empty(NumPartTot[0],    dtype=float)
        MaxTa          = empty(NumPartTot[0],    dtype=float)         
        SFR            = empty(NumPartTot[0],    dtype=float)
        MassGas        = empty(NumPartTot[0],    dtype=float)
        BindingEnergy  = empty(NumPartTot[0],    dtype=float)
        InternalEnergy  = empty(NumPartTot[0],    dtype=float)
        #fHall    = empty(NumPartTot[0],    dtype=float)
        fHSall    = empty(NumPartTot[0],    dtype=float)
        #GasZ = empty(NumPartTot[0],    dtype=float)
        GasSZ = empty(NumPartTot[0],    dtype=float)
        EOS = empty(NumPartTot[0],    dtype=float)
        GrpNum_Gas     = empty(NumPartTot[0],    dtype=int)
        SubNum_Gas     = empty(NumPartTot[0],    dtype=int)

    NGas_c             = 0

    for ifile in range(0,FNumPerSnap):
        if os.path.isdir(Base+DirList+'data/'):
            fn                 = Base+DirList+'data/particledata_'+exts+fend+'/'+'eagle_subfind_particles_'+exts+fend+'.'+str(ifile)+'.hdf5'
        else:
            fn                 = Base+DirList+'/particledata_'+exts+fend+'/'+'eagle_subfind_particles_'+exts+fend+'.'+str(ifile)+'.hdf5'
        fs                     = h5.File(fn,"r")
        Header                 = fs['Header'].attrs
        NumParts               = Header['NumPart_ThisFile']

        if NumParts[0]>0:
            PosGas[NGas_c:NGas_c+NumParts[0],:]       = fs["PartType0/Coordinates"].value
            VelGas[NGas_c:NGas_c+NumParts[0],:]       = fs["PartType0/Velocity"].value
            DensGas[NGas_c:NGas_c+NumParts[0]]        = fs["PartType0/Density"].value
            #GasZ[NGas_c:NGas_c+NumParts[0]] = fs["PartType0/Metallicity"].value
            GasSZ[NGas_c:NGas_c+NumParts[0]] = fs["PartType0/SmoothedMetallicity"].value
            HSML_Gas[NGas_c:NGas_c+NumParts[0]]       = fs["PartType0/SmoothingLength"].value
            SFR[NGas_c:NGas_c+NumParts[0]]            = fs["PartType0/StarFormationRate"].value
            TempGas[NGas_c:NGas_c+NumParts[0]]        = fs["PartType0/Temperature"].value
            EOS[NGas_c:NGas_c+NumParts[0]]            = fs["PartType0/OnEquationOfState"].value
            MassGas[NGas_c:NGas_c+NumParts[0]]        = fs["PartType0/Mass"].value
            BindingEnergy[NGas_c:NGas_c+NumParts[0]]  = fs["PartType0/ParticleBindingEnergy"].value
            InternalEnergy[NGas_c:NGas_c+NumParts[0]]  = fs["PartType0/InternalEnergy"].value
            #fHall[NGas_c:NGas_c+NumParts[0]]    = fs["PartType0/ElementAbundance/Hydrogen"]
            fHSall[NGas_c:NGas_c+NumParts[0]]  = fs["PartType0/SmoothedElementAbundance/Hydrogen"]
            GrpNum_Gas[NGas_c:NGas_c+NumParts[0]]     = fs["PartType0/GroupNumber"].value
            SubNum_Gas[NGas_c:NGas_c+NumParts[0]]     = fs["PartType0/SubGroupNumber"].value
            MaxT[NGas_c:NGas_c+NumParts[0]]   = fs["PartType0/MaximumTemperature"].value
            MaxTa[NGas_c:NGas_c+NumParts[0]]   = fs["PartType0/AExpMaximumTemperature"].value

        NGas_c             += NumParts[0]

        fs.close()

    if(NumPartTot[0]>0): 
        return {'PosGas':PosGas, 'HSML_Gas':HSML_Gas, 
                'DensGas':DensGas, 'SFR':SFR,
                'MassGas':MassGas, 'fHSall':fHSall,
                'VelGas':VelGas,#'Star_aform':Star_aform,              
                'TempGas':TempGas, 'EOS':EOS,'BEGas':BindingEnergy,
                'GrpNum_Gas':GrpNum_Gas, 'SubNum_Gas':SubNum_Gas, 'GasSZ':GasSZ,'IEGas':InternalEnergy,
                'MaxT':MaxT, 'MaxTa':MaxTa
               }
