#For determining total neutral and atomic/molecular hydrogen in gas particles. The functions are slightly modified versions of the original code written by Adam Stevens and can be found in the 'galcalc.py' file at 'https://github.com/arhstevens/Dirty-AstroPy/blob/master/galprops/galcalc.py'.

import numpy as np
import math
import eib as ib
from localuv import *
from scipy import optimize as op
from scipy import signal as ss
from scipy import interpolate
from time import time

def piecewise_linear_parabola(x, xbreak, l0, l1, p0, p1):
    y = l0*x + l1
    p2 = l0*xbreak + l1 - p0*xbreak**2 - p1*xbreak
    fbreak = (x>xbreak)
    y[fbreak] = p0*x[fbreak]**2 + p1*x[fbreak] + p2
    return y

def u2temp(u, gamma=5./3, mu=1.0):
	# Convert gas energy per unit mass to temperature.  Assumes input of J/kg = (m/s)^2
	M_H = 1.673e-27 # Mass of hydrogen in kg
	k_B = 1.3806488e-23 # Boltzmann constant (J/K)
	temp = u * M_H * mu * (gamma-1.) / k_B
	return temp

def BaryMP(x,y,eps=0.01,grad=1):
        """
        Find the radius for a galaxy from the BaryMP method
        x = r/r_200
        y = cumulative baryonic mass profile
        eps = epsilon, if data 
        """
        dydx = np.diff(y)/np.diff(x)

        maxarg = np.argwhere(dydx==np.max(dydx))[0][0] # Find where the gradient peaks
        xind = np.argwhere(dydx[maxarg:]<=grad)[0][0] + maxarg # The index where the gradient reaches 1

        x2fit_new, y2fit_new = x[xind:], y[xind:] # Should read as, e.g., "x to fit".
        x2fit, y2fit = np.array([]), np.array([]) # Gets the while-loop going

        while len(y2fit)!=len(y2fit_new):
                x2fit, y2fit = np.array(x2fit_new), np.array(y2fit_new)
                p = np.polyfit(x2fit, y2fit, 1)
                yfit = p[0]*x2fit + p[1]
                chi = abs(yfit-y2fit) # Separation in the y-direction for the fit from the data
                chif = (chi<eps) # Filter for what chi-values are acceptable
                x2fit_new, y2fit_new = x2fit[chif], y2fit[chif]

        r_bmp = x2fit[0] # Radius from the baryonic-mass-profile technique, returned as a fraction of the virial radius!
        Nfit = len(x2fit) # Number of points on the profile fitted to in the end

        return r_bmp, Nfit


def rahmati2013_neutral_frac(redshift, nH, T, onlyA1=True,noCol = False,onlyCol = False,extrapolate=False,local=False, UVB='HM12'):
    
#; --------------------------------------------------------------------------------------------
#;+
#; NAME:
#;       rahmati2013_neutral_frac
#;
#; PURPOSE:
#;       Computes particle neutral fractions based on the fitting functions of
#;       Rahmati et al. (2013a). By default, it uses the parameters of Table A1
#;       (based on small cosmological volumes) for z > 1, and of Table A2 (based
#;       on a 50Mpc volume) for z < 1, to better account for the effects of
#;       collisional ionisation on the self-shielding density.
#;
#; CATEGORY:
#;       I/O, HDF5, EAGLE, HI
#;
#; CALLING SEQUENCE:
#;       NeutralFraction = rahmati2013_neutral_frac(redshift,nH,Temperature_Type0)
#;       To compute neutral (HI+H_2) mass of particle, multiply NeutralFraction by
#;       xH and ParticleMass
#;
#; INPUTS:
#;       redshift:        Redshift of snapshot.
#;       nH:  hydrogen number density of the gas
#;       T:     temperature of the gas
#;
#; KEYWORD PARAMETERS:
#;       onlyA1:          routine will use Table A1 parameters for z < 0.5
#;       noCol:           the contribution of collisional ionisation to
#;       the overall ionisation rate is neglected
#;       onlyCol:         the contribution of photoionisation to
#;       the overall ionisation rate is neglected
#;       SSH_Thresh:      all particles above this density are assumed
#;       to be fully shielded (f_neutral=1)
#;
#; OUTPUTS:
#;       Array containing neutral fractions
#;
#; DEVELOPMENT:
#;       Written by Rob Crain, Leiden, March 2014, with input from Ali
#;       Rahmati. Based on Rahmati et al. (2013).
#;       Converted to python by Michelle Furlong, Dec 2014.
#;      Further edits by Adam Stevens, Jan 2018
#; --------------------------------------------------------------------------------------------
    if redshift>5:
        print('Using the z=5 relation for rahmati2013_neutral_frac when really z=',redshift)
        redshift = 5.0
    if redshift < 1.0:
        dlogz = (np.log10(1+redshift) - 0.0)/np.log10(2.)
        if onlyA1:
            lg_n0_lo     = -2.94
            gamma_uvb_lo =  8.34e-14 if UVB=='HM01' else 3.99e-14
            SSH_Thresh_lo = 1.1e-3 if UVB=='HM01' else 7.7e-4
            if UVB=='HM12':
             gamma_uvb_lo = 2.27e-14
             SSH_Thresh_lo = 5.1e-4
            alpha1_lo    = -3.98
            alpha2_lo    = -1.09
            beta_lo      =  1.29
            f_lo         =  0.01
            
            
        else:
            lg_n0_lo     = -2.56
            gamma_uvb_lo =  8.34e-14 if UVB=='HM01' else 3.99e-14
            SSH_Thresh_lo = 1.1e-3 if UVB=='HM01' else 7.7e-4
            if UVB=='HM12':
             gamma_uvb_lo = 2.27e-14
             SSH_Thresh_lo = 5.1e-4
            alpha1_lo    = -1.86
            alpha2_lo    = -0.51
            beta_lo      =  2.83
            f_lo         =  0.01
        lg_n0_hi     = -2.29
        gamma_uvb_hi =  7.39e-13 if UVB=='HM01' else 3.03e-13
        SSH_Thresh_hi = 5.1e-3 if UVB=='HM01' else 3.1e-3
        if UVB=='HM12':
          gamma_uvb_hi = 3.42e-13
          SSH_Thresh_hi = 3.3e-3
        alpha1_hi    = -2.94
        alpha2_hi    = -0.90
        beta_hi      =  1.21
        f_hi         =  0.03

    elif (redshift >= 1.0 and redshift < 2.0):
        dlogz = (np.log10(1+redshift) - np.log10(2.))/(np.log10(3.)-np.log10(2.))
        lg_n0_lo     = -2.29
        gamma_uvb_lo =  7.39e-13 if UVB=='HM01' else 3.03e-13
        SSH_Thresh_lo = 5.1e-3 if UVB=='HM01' else 3.1e-3
        if UVB=='HM12':
          gamma_uvb_hi = 3.42e-13
          SSH_Thresh_hi = 3.3e-3
        alpha1_lo    = -2.94
        alpha2_lo    = -0.90
        beta_lo      =  1.21
        f_lo         =  0.03
        
        lg_n0_hi     = -2.06
        gamma_uvb_hi =  1.50e-12 if UVB=='HM01' else 6.00e-13
        SSH_Thresh_hi = 8.7e-3 if UVB=='HM01' else 5.1e-3
        if UVB=='HM12':
          gamma_uvb_hi = 8.98e-13
          SSH_Thresh_hi = 6.1e-3
        alpha1_hi    = -2.22
        alpha2_hi    = -1.09
        beta_hi      =  1.75
        f_hi         =  0.03

    elif (redshift >= 2.0 and redshift < 3.0):
        dlogz = (np.log10(1+redshift) - np.log10(3.))/(np.log10(4.)-np.log10(3.))
        lg_n0_lo     = -2.06
        gamma_uvb_lo =  1.50e-12 if UVB=='HM01' else 6.00e-13
        SSH_Thresh_lo = 8.7e-3 if UVB=='HM01' else 5.1e-3
        if UVB=='HM12':
          gamma_uvb_lo = 8.98e-13
          SSH_Thresh_lo = 6.1e-3
        alpha1_lo    = -2.22
        alpha2_lo    = -1.09
        beta_lo      =  1.75
        f_lo         =  0.03
        
        lg_n0_hi     = -2.13
        gamma_uvb_hi =  1.16e-12 if UVB=='HM01' else 5.53e-13
        SSH_Thresh_hi = 7.4e-3 if UVB=='HM01' else 5.0e-3
        if UVB=='HM12':
          gamma_uvb_hi = 8.74e-13
          SSH_Thresh_hi = 6.0e-3
        alpha1_hi    = -1.99
        alpha2_hi    = -0.88
        beta_hi      =  1.72
        f_hi         =  0.04

    elif (redshift >= 3.0 and redshift < 4.0):
        dlogz = (np.log10(1+redshift) - np.log10(4.))/(np.log10(5.)-np.log10(4.))
        lg_n0_lo     = -2.13
        gamma_uvb_lo =  1.16e-12 if UVB=='HM01' else 5.53e-13
        SSH_Thresh_lo = 7.4e-3 if UVB=='HM01' else 5.0e-3
        if UVB=='HM12':
          gamma_uvb_hi = 8.74e-13
          SSH_Thresh_hi = 6.0e-3
        alpha1_lo    = -1.99
        alpha2_lo    = -0.88
        beta_lo      =  1.72
        f_lo         =  0.04
        
        lg_n0_hi     = -2.23
        gamma_uvb_hi =  7.92e-13 if UVB=='HM01' else 4.31e-13
        SSH_Thresh_hi = 5.8e-3 if UVB=='HM01' else 4.4e-3
        if UVB=='HM12':
          gamma_uvb_hi = 6.14e-13
          SSH_Thresh_hi = 4.7e-3
        alpha1_hi    = -2.05
        alpha2_hi    = -0.75
        beta_hi      =  1.93
        f_hi         =  0.02

    elif (redshift >= 4.0 and redshift <= 5.0):
        dlogz = (np.log10(1+redshift) - np.log10(5.))/(np.log10(6.)-np.log10(5.))
        lg_n0_lo     = -2.23
        gamma_uvb_lo =  7.92e-13 if UVB=='HM01' else 4.31e-13
        SSH_Thresh_lo = 5.8e-3 if UVB=='HM01' else 4.4e-3
        if UVB=='HM12':
          gamma_uvb_hi = 6.14e-13
          SSH_Thresh_hi = 4.7e-3
        alpha1_lo    = -2.05
        alpha2_lo    = -0.75
        beta_lo      =  1.93
        f_lo         =  0.02
        
        lg_n0_hi     = -2.35
        gamma_uvb_hi =  5.43e-13 if UVB=='HM01' else 3.52e-13
        SSH_Thresh_hi = 4.5e-3 if UVB=='HM01' else 4.0e-3
        if UVB=='HM12':
          gamma_uvb_hi = 4.57e-13
          SSH_Thresh_hi = 3.9e-3
        alpha1_hi    = -2.63
        alpha2_hi    = -0.57
        beta_hi      =  1.77
        f_hi         =  0.01

    else:
        print('[rahmati2013_neutral_frac] ERROR: parameters only valid for z < 5, you asked for z = ', redshift)
        exit()

    # [Adam] All of this code could be massively reduced by just putting the hi/low values into a table and using np.interp....
    lg_n0     = lg_n0_lo     + dlogz*(lg_n0_hi     - lg_n0_lo)
    n0        = 10.**lg_n0
    lg_gamma_uvb_lo, lg_gamma_uvb_hi = np.log10(gamma_uvb_lo), np.log10(gamma_uvb_lo)
    gamma_uvb = 10**(lg_gamma_uvb_lo + dlogz*(lg_gamma_uvb_hi - lg_gamma_uvb_lo))
    lg_SSH_Thresh_lo, lg_SSH_Thresh_hi = np.log10(SSH_Thresh_lo), np.log10(SSH_Thresh_hi)
    SSH_Thresh = 10**(lg_SSH_Thresh_lo + dlogz*(lg_SSH_Thresh_hi - lg_SSH_Thresh_lo))
    alpha1    = alpha1_lo    + dlogz*(alpha1_hi    - alpha1_lo)
    alpha2    = alpha2_lo    + dlogz*(alpha2_hi    - alpha2_lo)
    beta      = beta_lo      + dlogz*(beta_hi      - beta_lo)
    f         = f_lo         + dlogz*(f_hi         - f_lo)
    
#    if onlyA1:
#        print '[rahmati2013_neutral_frac] using Table A1 parameters for all redshifts'
#    else:
#        print '[rahmati2013_neutral_frac] using Table A1 parameters for z > 1 and Table A2 parameters for z < 1'
#    if noCol:
#        print '[rahmati2013_neutral_frac] neglecting collisional ionisation'
#    if onlyCol:
#        print '[rahmati2013_neutral_frac] neglecting photoionisation'
#    print '[rahmati2013_neutral_frac] adopting SSH_Thresh/cm^-3 = ',SSH_Thresh
#    print '[rahmati2013_neutral_frac] using Rahmati et al. 2013 parameters for z = ',redshift
#    print ' lg_n0/cm^-3    = ',lg_n0
#    print ' gamma_uvb/s^-1 = ',gamma_uvb
#    print ' alpha1         = ',alpha1
#    print ' alpha2         = ',alpha2
#    print ' beta           = ',beta
#    print ' f              = ',f

    # Use fitting function as per Rahmati, Pawlik, Raicevic & Schaye 2013
#    print ' nH range       = ', min(nH), max(nH), np.median(nH)
    
    gamma_ratio = (1.-f) * (1. + (nH / n0)**beta)**alpha1 + f*(1. + (nH / n0))**alpha2
    gamma_phot  = gamma_uvb * gamma_ratio
    
    # if "local" is set, we include an estimate of the local
    # photoionisation rate from local sources, as per Ali's paper
    if local:
        # from Rahmati et al. (2013), equation (7)
        # assuming unity gas fraction and P_tot = P_thermal
        gamma_local = 1.3e-13 * nH**0.2 * (T/1.0e4)**0.2
        gamma_phot += gamma_local

    lambda_T  = 315614.0 / T
    AlphaA    = 1.269e-13 * (lambda_T)**(1.503) / ((1. + (lambda_T / 0.522)**0.470)**1.923)
    LambdaT   = 1.17e-10*(np.sqrt(T)*np.exp(-157809.0/T)/(1.0 + np.sqrt(T/1.0e5)))
    
    if noCol: LambdaT    = 0.0
    if onlyCol: gamma_phot = 0.0
    
    A = AlphaA + LambdaT
    B = 2.0*AlphaA + (gamma_phot/nH) + LambdaT
    sqrt_term = np.array([np.sqrt(B[i]*B[i] - 4.0*A[i]*AlphaA[i]) if (B[i]*B[i] - 4.0*A[i]*AlphaA[i])>0 else 0.0 for i in range(len(B))])
    f_neutral = (B - sqrt_term) / (2.0*A)
    f_neutral[f_neutral <= 0] = 1e-30 #negative values seem to arise from rounding errors - AlphaA and A are both positive, so B-sqrt_term should be positive!
    
    #if SSH_Thresh:
        #print '[rahmati2013_neutral_frac] setting the eta = 1 for densities higher than: ', SSH_Thresh
       # ind = np.where(nH > SSH_Thresh)[0]
       # if(len(ind) > 0): f_neutral[ind] = 1.0

    return f_neutral

def HI_H2_masses(mass, pos, radius, SFR, Z, rho, temp, fneutral, redshift, local=False, method=4, mode='T', UVB='FG09-Dec11', U_MW_z0=None, rho_sd=0.01, col=2, gamma_fixed=None, mu_fixed=None, S_Jeans=True, T_CNMmax=243., Pth_Lagos=False, Jeans_cold=False, Sigma_SFR0=1e-9, UV_MW=None, X=None, UV_pos=None, f_esc=0.15, f_ISM=None):
    """
        This is my own version of calculating the atomic- and molecular-hydrogen masses of gas particles/cells from simulations.  This was originally adapted from the Python scripts written by Claudia Lagos and Michelle Furlong, and followed the basis of Appendix A of Lagos et al (2015b).  This has since been vastly modified and is still being developed further.  This has been developed in tandem with Benedikt Diemer's code for Illustris-TNG, and has been tested to produce the same results.  Please cite Stevens et al. (2019, MNRAS, 483, 5334) if you use this function for a publication (also note the erratum to that paper: 2019, MNRAS, 484, 5499).
        Expects each non-default input as an array, except for reshift.  Input definitions and units are as follows:
        mass = total mass of gas particles/cells [M_sun]
	pos = position of gas particles/cells [Mpc]
	radius = half-mass radius of gas [Mpc]
        SFR = star formation rate of particles/cells [M_sun/yr]
        Z = ratio of metallic mass to total particle/cell mass
        rho = density of particles/cells [M_sun/pc^3]
        temp = temperature of particles [K] OR specific thermal energy [(m/s)^2] (see mode)
        fneutral = fraction of particle/cell mass that is not ionized.  If given as None, it will automatically be calculated using the Rahmati+13 prescription.
        redshift = single float for the redshift being considered
        method = 0 - Return results for methods 2, 3, and 4
                 1 - Gnedin & Kravtsov (2011) eq 6
                 2 - Gnedin & Kravtsov (2011) eq 10
                 3 - Gnedin & Draine (2014) eq 6
                 4 - Krumholz (2013) eq 10
                 5 - Gnedin & Draine (2014) eq 8
                 6 - Leroy (2008) eq 32
                 7 - Blitz & Rosolowsky (2006)
                 8 - All neutral hydrogen as atomic
          all of these also use Schaye and Dalla Vecchia (2008) for the Jeans length
        mode = 'T' - temp is actually temperature
               'u' - have fed in internal energy per unit mass instead of temperature
        UVB = 'HM12' - Haardt & Madau (2012)
              'FG09' - Faucher-Giguere et al. (2009)
              'FG09-Dec11' - Updated FG09 table.  Uses pre-built values normalised by the Draine (1978) field, tabulated by Benedikt Diemer
        U_MW_z0 = strength of UV background at z=0 in units of the Milky Way's interstellar radiation field.  Has a default value if set to None.
        rho_sd = local density of dark matter and stars. Used in method 4. [Msun/pc^3]
        col = only used for UVB='FG09-Dec11', decides on column to use in table
        gamma_fixed = set gamma to be a fixed value if not None, even if that breaks self-consistency (there for testing)
        mu_fixed = as above but for mu
        S_Jeans = use Jeans scale for S variable in GD14, else use the cube root of cell volume (former makes more sense for SPH, but not cells)
        T_CNMmax = Maximum temperature of cold neutral medium for K13
        Pth_Lagos = calculate P_th for K13 as in eq.A15 of Lagos+15b rather than eq.6 of K13.  Not advised for science.  Here for comparison purposes.
        Jeans_cold = use cold clouds of SF gas cells only for calculating Jeans length [NOT YET IMPLEMENTED]
        Sigma_SFR0 = local star formation rate surface density of the solar neighbourhood [Msun/yr/pc^2]
        UV_MW = pre-computed UV fluxes (normed by Milky Way) for each cell
        X = pre-computed hydrogen fractions for particles/cells (or a chosen constant)
        UV_pos = positions of particles/cells, used to approximate the UV field of non-SF particles/cells based on nearby SF particles/cells.  Using this will definitely slow the code but should return more realistic HI/H2 fractions for non-star-forming cells/particles.  Is redundant when UV_MW is provided. [pc]
        f_esc = fudge factor for escape fraction in the approximate calculation for UV
        f_ISM = boolean array for particles/cells stating whether they should be considered 'ISM' for the sake of the K13 prescription.  Those that are not will not have the nCNMhydro floor applied.  This will also inform the mean Sigma_SFR for the UV calculation.
    """
    
    
    kg_per_Msun = 1.989e30
    m_per_pc = 3.0857e16
    s_per_yr = 60*60*24*365.24

    # Convert physical constants to internal units
    m_p = 1.6726219e-27 / kg_per_Msun
    G = 6.67408e-11 * kg_per_Msun * s_per_yr**2 / m_per_pc**3
    k_B = 1.38064852e-23 * s_per_yr**2 / m_per_pc**2 / kg_per_Msun
    const_ratio = k_B / (m_p * G)
    f_th = 1.0 # Assuming all gas is thermal

    Z[Z<1e-5] = 1e-5 # Floor on metallicity (BBN means there must be a tiny bit)
    if X is None: # Approximate hydrogen fraction from pre-determined fitting function (very simplistic)
        p = [0.0166, -1.3452, 0.75167, 31.193, -2.38233]
        X = piecewise_parabola_linear(Z, *p)
        X[X>0.76] = 0.76 # safety
    Y = 1. - X - Z
    
    # Protons per cm^3 in all forms of hydrogen
    denom = m_p * (m_per_pc*100)**3
    n_H = X * rho / denom
    
    
    if gamma_fixed is not None:
        gamma = 1.0*gamma_fixed
    else:
        gamma = 5./3. # initialise

    if mode=='u':
        u = 1.0*temp
        temp = u2temp(u, gamma, 0.59) # initialise

    if fneutral is None:
        calc_fneutral = True
    else:
        calc_fneutral = False

    # Calculate (initialise in the case of mode='u') neutral fraction if it wasn't already provided
    if calc_fneutral:
        fneutral = rahmati2013_neutral_frac(redshift, n_H, temp, UVB=UVB, local=local)

    fzero = (fneutral <= 0)
    fneutral[fzero] = 1e-6 # Floor on neutral fraction.  Prevents division by zero below

    if mu_fixed is not None:
        mu = 1.0*mu_fixed
    else:
        mu = (X + 4*Y)/((2-fneutral)*(X+Y)) # Initialise mean molecular weight


    # Initialise lists if all methods wanted
    mHI_list, mH2_list = [], []

    # Set floor of interstellar radiation field from UV background, in units of Milky Way field
    if UVB not in ['HM12', 'FG09', 'FG09-Dec11']:
        print('Could not interpret input for UVB.  UVB should be set to either HM12, FG09, or FG09-Dec11.  Defaulting to FG09-Dec11.')
        UVB = 'FG09-Dec11'
    
    if UVB=='FG09-Dec11':
        if col not in [1,2,3]: col = 3
        data = ib.U_MW_FG09_Dec11()
        redshift_UVB = data[:,0]
        UVbackground = data[:,col]
    else:
        if UVB=='HM12':
            data = ib.HaardtMadau12()
        elif UVB=='FG09':
            data = ib.FaucherGiguere09()
        redshift_UVB = data[0,:]
        if U_MW_z0 is None:
            UVbackground = data[1,:]/2.2e-12 # Divides through by local MW value (that number is in eV/s).  Original reference is unknown.  This probably should be avoided.
        else:
            UVbackground = data[1,:] / data[1,0] * U_MW_z0
    ISRF_floor = 10**np.interp(redshift, redshift_UVB, np.log10(UVbackground)) * np.ones(len(mass))
    
    # Leaving this commented as I have been feeding in values that already have this floor applied.  Note that in future if values fed in don't have the floor applied, these lines should be uncommented.
#    if UV_MW is not None:
#        UV_MW[UV_MW<ISRF_floor] = ISRF_floor

    # Approximate UV field based on average SF density
    sf = (SFR>0) if f_ISM is None else (SFR>0) * f_ISM
    if UV_pos is not None and UV_MW is None and len(sf[sf])>0:
        CoSF = np.sum(mass[sf] * UV_pos[sf].T, axis=1) / np.sum(mass[sf]) # centre of star formation
        Rsqr = np.sum((UV_pos - CoSF)**2, axis=1)
        Rsqr_area = np.max((Rsqr + 0.5*(mass/rho)**(2./3.))[sf])
        Sigma_SFR_cen = np.sum(SFR) / Rsqr_area / np.pi
        ISRF_floor = np.maximum(ISRF_floor, f_esc*Sigma_SFR_cen/Sigma_SFR0/Rsqr* Rsqr_area)


        
    # Dust to gas ratio relative to MW
    D_MW = Z / 0.0127
    
    it_max = 300 # Maximum iterations for calculating f_H2 (arbitrary)
    f_H2_old = np.zeros(len(mass)) # Initialise before iterating
    fneutral_old = np.zeros(len(mass))

	
    #Estimating the local ionizing field
    kernel_fft = createFFTKernel3D(128)
    sf = (SFR>0)
    a = 1.0/(1+redshift)
    if method!=6 and method!=7:
     G0 = cellUV(kernel_fft, radius, a, sf, pos, mass, rho, SFR, Z, X, 0.0, False, 0.0, True, 0.1, 128, 'HM12')#Diemer et al. 2018

    if method==1: # GK11, eq6
        for it in range(it_max):
            if it==it_max-1: print('iterations hit maximum for GK11, eq6 in HI_H2_masses()')
            f_mol = X*fneutral*f_H2_old /  (X+Y)
            if gamma_fixed is None: gamma = (5./3.)*(1-f_mol) + 1.4*f_mol
            if mu_fixed is None: mu = (X + 4*Y) * (1.+ (1.-fneutral)/fneutral) / ((X+Y) * (1.+ 2*(1.-fneutral)/fneutral - f_H2_old/2.))
            if mode=='u':
                temp = u2temp(u, gamma, mu)
                if calc_fneutral and not np.allclose(fneutral, fneutral_old, rtol=5e-3): fneutral = rahmati2013_neutral_frac(redshift, rho/denom, temp, UVB=UVB) # too slow to do every time (hopefully this converges faster)
            Sigma = np.sqrt(gamma * const_ratio * f_th * rho * temp / mu) # Approximate surface density as true density * Jeans length (see eq. 7 of Schaye and Dalla Vecchia 2008)
            area = mass / Sigma # Effective area covered by particle
            Sigma_SFR = SFR / area
            #G0 = np.maximum(ISRF_floor, f_esc * Sigma_SFR / Sigma_SFR0) if UV_MW is None else 1.0*UV_MW # Calculte interstellar radiation field, assuming it's proportional to SFR density, normalised by local Sigma_SFR of solar neighbourhood.
            D_star = 1.5e-3 * np.log(1. + (3.*G0)**1.7)
            alpha = 2.5*G0 / (1.+(0.5*G0)**2.)
            s = 0.04 / (D_star + D_MW)
            g = (1. + alpha*s + s*s) / (1.+s)
            Lambda = np.log(1. + g * D_MW**(3./7.) * (G0/15.)**(4./7.))
            x = Lambda**(3./7.) * np.log(D_MW * fneutral*n_H/(Lambda*25.))
            f_H2 = 1./ (1.+ np.exp(-4.*x - 3.*x**3)) # H2/(HI+H2)
            if np.allclose(f_H2[~fzero], f_H2_old[~fzero], rtol=5e-3): break
            f_H2_old = 1.*f_H2
            fneutral_old = 1.*fneutral

        mass_H2 = f_H2 * fneutral * X * mass
        mass_HI = (1.-f_H2) * fneutral * X * mass
        mass_H2[fzero] = 0.
        mass_HI[fzero] = 0.

    if method==2 or method==0: # GK11, eq10 (entry 0 for method==0)
        for it in range(it_max):
            if it==it_max-1: print('iterations hit maximum for GK11, eq11 in HI_H2_masses()')
            f_mol = X*fneutral*f_H2_old /  (X+Y)
            if gamma_fixed is None: gamma = (5./3.)*(1-f_mol) + 1.4*f_mol
            if mu_fixed is None: mu = (X + 4*Y) * (1.+ (1.-fneutral)/fneutral) / ((X+Y) * (1.+ 2*(1.-fneutral)/fneutral - f_H2_old/2.))
            if mode=='u':
                temp = u2temp(u, gamma, mu)
                if calc_fneutral and not np.allclose(fneutral, fneutral_old, rtol=5e-3): fneutral = rahmati2013_neutral_frac(redshift, rho/denom, temp, UVB=UVB)
            Sigma = np.sqrt(gamma * const_ratio * f_th * rho * temp / mu)
            Sigma_n = fneutral * X * Sigma # neutral hydrogen density
            area = mass / Sigma
            Sigma_SFR = SFR / area
            #G0 = np.maximum(ISRF_floor, f_esc * Sigma_SFR / Sigma_SFR0) if UV_MW is None else 1.0*UV_MW
            D_star = 1.5e-3 * np.log(1. + (3.*G0)**1.7)
            alpha = 2.5*G0 / (1.+(0.5*G0)**2.)
            s = 0.04 / (D_star + D_MW)
            g = (1. + alpha*s + s*s) / (1.+s)
            Lambda = np.log(1. + g * D_MW**(3./7.) * (G0/15.)**(4./7.))
            Sigma_c = 20. * Lambda**(4./7.) / (D_MW * np.sqrt(1.+G0*D_MW**2.))
            f_H2 = (1.+Sigma_c/Sigma_n)**-2. # H2/(HI+H2)
            if np.allclose(f_H2[~fzero], f_H2_old[~fzero], rtol=5e-3): break
            f_H2_old = 1.*f_H2
            fneutral_old = 1.*fneutral

        mass_H2 = f_H2 * fneutral * X * mass
        mass_HI = (1.-f_H2) * fneutral * X * mass
        mass_H2[fzero] = 0.
        mass_HI[fzero] = 0.

    if method==0:
        mHI_list += [mass_HI]
        mH2_list += [mass_H2]

    if method==3: # GD14, eq6
        for it in range(it_max):
            if it==it_max-1: print('iterations hit maximum for GD14, eq6 in HI_H2_masses()')
            f_mol = X*fneutral*f_H2_old /  (X+Y)
            if gamma_fixed is None: gamma = (5./3.)*(1-f_mol) + 1.4*f_mol
            if mu_fixed is None: mu = (X + 4*Y) * (1.+ (1.-fneutral)/fneutral) / ((X+Y) * (1.+ 2*(1.-fneutral)/fneutral - f_H2_old/2.))
            if mode=='u':
                temp = u2temp(u, gamma, mu)
                if calc_fneutral and not np.allclose(fneutral, fneutral_old, rtol=5e-3): fneutral = rahmati2013_neutral_frac(redshift, rho/denom, temp, UVB=UVB)
            Sigma = np.sqrt(gamma * const_ratio * f_th * rho * temp / mu)
            S = Sigma / rho * 0.01 if S_Jeans else (mass/rho)**(1./3) * 0.01 # Spatial scale: either the Jeans length or approx cell length (per 100 pc)
            D_star = 0.17*(2.+S**5.)/(1.+S**5.)
            U_star = 9.*D_star/S
            g = np.sqrt(D_MW*D_MW + D_star*D_star)
            #G0 = np.maximum(ISRF_floor, f_esc * SFR / mass * Sigma / Sigma_SFR0) if UV_MW is None else 1.0*UV_MW # Instellar radiation field in units of MW's local field (assumed to be proportional to local SFR density).  Reduced from several lines in other methods.
            Lambda = np.log(1.+ (0.05/g+G0)**(2./3)*g**(1./3)/U_star)
            n_half = 14. * np.sqrt(D_star) * Lambda / (g*S)
            x = (0.8 + np.sqrt(Lambda)/S**(1./3)) * np.log(fneutral*n_H/n_half)
            f_H2 = 1./ (1 + np.exp(-x*(1-0.02*x+0.001*x*x))) # H2/(HI+H2)
            if np.allclose(f_H2[~fzero], f_H2_old[~fzero], rtol=5e-3): break
            f_H2_old = 1.*f_H2
            fneutral_old = 1.*fneutral

        mass_H2 = f_H2 * fneutral * X * mass
        mass_HI = (1.-f_H2) * fneutral * X * mass
        mass_H2[fzero] = 0.
        mass_HI[fzero] = 0.


    if method==5 or method==0: # GD14, eq8 (entry 1 for method==0)
        # This has now replaced eq6 for the same output position when method=0
        for it in range(it_max):
            if it==it_max-1: print('iterations hit maximum for GD14, eq8 in HI_H2_masses()')
            f_mol = X*fneutral*f_H2_old /  (X+Y)
            if gamma_fixed is None: gamma = (5./3.)*(1-f_mol) + 1.4*f_mol
            if mu_fixed is None: mu = (X + 4*Y) * (1.+ (1.-fneutral)/fneutral) / ((X+Y) * (1.+ 2*(1.-fneutral)/fneutral - f_H2_old/2.))
            if mode=='u':
                temp = u2temp(u, gamma, mu)
                if calc_fneutral and not np.allclose(fneutral, fneutral_old, rtol=5e-3): fneutral = rahmati2013_neutral_frac(redshift, rho/denom, temp, UVB=UVB)
            Sigma = np.sqrt(gamma * const_ratio * f_th * rho * temp / mu)
            S = Sigma / rho * 0.01 if S_Jeans else (mass/rho)**(1./3) * 0.01 # Spatial scale: either the Jeans length or approx cell length (per 100 pc)
            D_star = 0.17*(2.+S**5.)/(1.+S**5.)
            U_star = 9.*D_star/S
            g = np.sqrt(D_MW*D_MW + D_star*D_star)
            #G0 = np.maximum(ISRF_floor, f_esc * SFR / mass * Sigma / Sigma_SFR0) if UV_MW is None else 1.0*UV_MW # Instellar radiation field in units of MW's local field (assumed to be proportional to local SFR density).  Reduced from several lines in other methods.
            alpha = 0.5 + 1./(1. + np.sqrt(G0*D_MW*D_MW/600.))
            Sigma_R1 = 50./g * np.sqrt(0.001+0.1*G0) / (1. + 1.69*np.sqrt(0.001+0.1*G0)) # Note the erratum on the paper for this equation!
            R = (Sigma * fneutral * X / Sigma_R1)**alpha
            f_H2 = R / (R + 1.)
            if np.allclose(f_H2[~fzero], f_H2_old[~fzero], rtol=5e-3): break
            f_H2_old = 1.*f_H2
            fneutral_old = 1.*fneutral
            
        mass_H2 = f_H2 * fneutral * X * mass
        mass_HI = (1.-f_H2) * fneutral * X * mass
        mass_H2[fzero] = 0.
        mass_HI[fzero] = 0.
    
    if method==0:
        mHI_list += [mass_HI]
        mH2_list += [mass_H2]

    
    if method==4 or method==0: # K13, eq10 (entry 2 for method==0)
        f_c = 5.0 # clumping factor
        alpha = 5.0 # relative pressure of turbulence, magnetic fields vs thermal
        zeta_d = 0.33
        f_w = 0.5
        c_w = 8e3 / m_per_pc * s_per_yr # sound speed of warm medium -- could calculate this better
        for it in range(it_max):
            if it==it_max-1: print('iterations hit maximum for K13, eq10 in HI_H2_masses()')
            f_mol = X*fneutral*f_H2_old /  (X+Y)
            if gamma_fixed is None: gamma = (5./3.)*(1-f_mol) + 1.4*f_mol
            if mu_fixed is None: mu = (X + 4*Y) * (1.+ (1.-fneutral)/fneutral) / ((X+Y) * (1.+ 2*(1.-fneutral)/fneutral - f_H2_old/2.))
            if mode=='u':
                temp = u2temp(u, gamma, mu)
                if calc_fneutral and not np.allclose(fneutral, fneutral_old, rtol=5e-3): fneutral = rahmati2013_neutral_frac(redshift, rho/denom, temp, UVB=UVB)
            Sigma = np.sqrt(gamma * const_ratio * f_th * rho * temp / mu)
            Sigma_n = fneutral * X * Sigma # neutral hydrogen density
            #G0 = np.maximum(ISRF_floor, f_esc * SFR / mass * Sigma / Sigma_SFR0) if UV_MW is None else 1.0*UV_MW
            #
            n_CNM2p = 23.*G0 * 4.1 / (1. + 3.1*D_MW**0.365)
            #
            if not Pth_Lagos:
                R_H2 = f_H2_old / (1. - f_H2_old)
                Sigma_HI = (1 - f_H2_old) * Sigma_n
                frac = 32. * zeta_d * alpha * f_w * c_w*c_w * rho_sd / (np.pi*G*Sigma_HI**2.)
                P_th = np.pi*G*Sigma_HI**2./(4.*alpha) * (1. + 2*R_H2 + np.sqrt((1.+ 2*R_H2)**2 + frac))  
            else:
                frac = 32. * zeta_d * alpha * f_w * c_w*c_w * rho_sd / (np.pi*G*Sigma_n**2.)
                P_th = np.pi*G*Sigma_n**2./(4.*alpha) * (1. + np.sqrt(1 + frac))
            n_CNMhydro = P_th / (1.1 * k_B * T_CNMmax) / (m_per_pc*100.)**3.
            #
            n_CNM = np.max(np.array([n_CNM2p,n_CNMhydro]),axis=0)
            if f_ISM is not None: n_CNM[~f_ISM] = n_CNM2p[~f_ISM] # floor doesn't apply to diffuse halo gas
            chi = 7.2*G0 / (0.1*n_CNM)
            tau_c = 0.066 * f_c * D_MW * Sigma_n
            s = np.log(1.+ 0.6*chi + 0.01*chi*chi) / (0.6*tau_c)
            #
            f_H2 = np.zeros(len(mu))
            f_H2[s<2] = 1. - 0.75*s[s<2]/(1. + 0.25*s[s<2])
            if np.allclose(f_H2[~fzero], f_H2_old[~fzero], rtol=5e-3): break
            f_H2_old = 1.*f_H2
            fneutral_old = 1.*fneutral

        mass_H2 = f_H2 * fneutral * X * mass
        mass_HI = (1.-f_H2) * fneutral * X * mass
        mass_H2[fzero] = 0.
        mass_HI[fzero] = 0.

    if method==6:
        pbyk = n_H*temp
        R = (pbyk/1.7e+4)**0.8
        f_H2 = R/(R+1)
        f_H2[SFR==0] = 0
        mass_H2 = f_H2 * fneutral * X * mass
        mass_HI = (1 - f_H2) * fneutral * X * mass
        mass_H2[fzero] = 0
        mass_HI[fzero] = 0

    if method==7:
        pbyk = n_H*temp
        R = (pbyk/4.3e+4)**0.92
        f_H2 = R/(R+1)
        f_H2[SFR==0] = 0
        mass_H2 = f_H2 * fneutral * X * mass
        mass_HI = (1 - f_H2) * fneutral * X * mass
        mass_H2[fzero] = 0
        mass_HI[fzero] = 0

    if method==8:
       f_H2 = 0
       mass_H2 = f_H2 * fneutral * X * mass
       mass_HI = (1 - f_H2) * fneutral * X * mass
       mass_H2[fzero] = 0
       mass_HI[fzero] = 0


    if method==0:
        mHI_list += [mass_HI]
        mH2_list += [mass_H2]

    if method==0:
        if calc_fneutral:
            return mHI_list, mH2_list, fneutral
        else:
            return mHI_list, mH2_list
    else:
        if calc_fneutral:
            return mass_HI, mass_H2, fneutral
        else:
            return mass_HI, mass_H2
