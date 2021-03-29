###################################################################################################
#
# This unit implements functions to estimate the UV field. It does not explicitly need to know 
# anything about the details of the simulation where the particle or map data come from.
#
# The goal is to compute a crude approximation to the UV flux (in MW units) in all gas cells or 
# cells of a map. There are three possible contributions to the UV field:
#
# 1) UV from locally forming stars. We compare the SFR in a gas cell or map element to the local
#    (solar) value to get U_MW. This contribution is deprecated (see below).
# 2) UV propagated away from star-forming regions. We assume or compute an escape fraction and
#    propagate the radiation with an inverse square law. This can also predict the UV strength
#    inside star-forming cells and is preferred over 1).
# 3) UV backround radiation. We use an analytical model for the UVB.
# 
# (c) Benedikt Diemer
#
###################################################################################################

import numpy as np
import os
import scipy.interpolate

#from colossus.utils import constants
constants_M_PROTON = 1.67262178e-24
constants_KPC = 3.08568025E21
constants_M_SUN = 1.98892E33
constants_C = 2.99792458E10

###################################################################################################
# CONSTANTS
###################################################################################################

# The Draine 1978 UV field at 1000A in photons /cm^2 /s /Hz. This number includes the 4pi factor.
flux_1000A_mw_draine = 3.43e-08

# The flux of a 1 Msun/yr forming population at 1 kpc distance, in units of 
# photons / s / cm^s / Hz, at 1000A. From an online SB99 calculation with a Kroupa IMF.
flux_1000A_sb99 = 3.2844e-06

# Absorption cross-section of hydrogen according to Draine 2003, at 1000A (cm^2 / atom). This is
# used when computing escape fractions. 
sigma_H_1000A = 1.406E-21

###################################################################################################

global_uvb_interp = {}

###################################################################################################

# Compute the UV flux in all gas cells.

def cellUV(kernel_fft, r_half_gas_cat,
				snap_a, mask_sf, coor, mass, rho, sfr, metallicity, fH, scale_height,
				uv_from_local_sfr, uv_sfr_local, uv_propagate, uv_escape_frac, uv_ngrid, 
				uv_bg_model):
	
	mask_nsf = np.logical_not(mask_sf)
	uv_mw = np.zeros_like(sfr)
	if uv_bg_model=='FG09':
  	 uv_bg = uvBackground(snap_a, uv_bg_model, uv_bg_calibration)
	if uv_bg_model=='HM12':
	 uv_bg = uvBackground_HM12(snap_a,q='photoheating')/2.2e-12#Haard & Madau 2012
	
	# For star-forming cells, estimate based on the SFR. 
	if uv_from_local_sfr and np.count_nonzero(mask_sf):
		uv_mw[mask_sf] = uvFromSFRVol(mask_sf, mass, rho, sfr, scale_height, uv_sfr_local)

	# For non-star forming cells, get an estimate based on their distance to UV sources. Note that
	# this calculation can be executed only if we have star-forming particles.
	if uv_propagate and np.count_nonzero(mask_sf):
		uv_mw_from_stars = uvPropagationVol(kernel_fft, r_half_gas_cat,
										mask_sf, coor, sfr, metallicity, rho, fH, scale_height, 
										uv_escape_frac, uv_ngrid)
		uv_mw[mask_nsf] = uv_mw_from_stars[mask_nsf]
		uv_mw[mask_sf] = np.maximum(uv_mw[mask_sf], uv_mw_from_stars[mask_sf])

	# Take the UVB as a minimum for all cells
	uv_mw[uv_bg > uv_mw] = uv_bg

	return uv_mw

###################################################################################################

# Compute the UV flux in all pixels of a map.

def mapUV(kernel_fft, map_extent, snap_a, sf_model, gas_rho, Sigma_sfr, metallicity, 
				uv_from_local_sfr, uv_sfr_local, uv_propagate, uv_escape_frac, uv_bg_model, 
				uv_bg_calibration):
	
	mask_sf = (Sigma_sfr > 0.0)
	uv_bg = uvBackground(snap_a, uv_bg_model, uv_bg_calibration)
	uv_mw = np.zeros_like(Sigma_sfr)
	
	if uv_from_local_sfr and np.count_nonzero(mask_sf):
		uv_mw[mask_sf] = uvFromSFRMap(Sigma_sfr[mask_sf], uv_sfr_local)

	# If we are doing star UV, take the max of that and the SFR-based estimate
	if uv_propagate and np.count_nonzero(mask_sf):
		uv_mw_from_stars = uvPropagationMap(kernel_fft, map_extent, mask_sf, sf_model, gas_rho, 
												Sigma_sfr, metallicity, uv_escape_frac)
		uv_mw = np.maximum(uv_mw_from_stars, uv_mw)
	
	# Take the UVB as a minimum for all cells
	uv_mw[uv_bg > uv_mw] = uv_bg
	
	return uv_mw

###################################################################################################

def uvPropagationVol(kernel_fft, r_half_gas_cat, mask_sf, coor, sfr, metallicity, rho, fH, scale_height, uv_escape_frac,
						uv_ngrid, compare_to_direct = False):

	sf_coor = coor[mask_sf, :]
	sf_sfr = sfr[mask_sf]
	sf_metallicity = metallicity[mask_sf]
	
	if uv_escape_frac == 'var':
		Sigma_H = rho[mask_sf] * fH[mask_sf] * scale_height[mask_sf]
		N_H = Sigma_H / constants_M_PROTON
		tau_uv = 0.5 * sigma_H_1000A * N_H
		f_esc = np.exp(-tau_uv * sf_metallicity / 0.0127)
	else:
		f_esc = uv_escape_frac
	
	sf_uv_flux = sf_sfr * flux_1000A_sb99 * f_esc
	
	# Create a 3D SFR grid. From the bins in each dimension, we create a cube of coordinates from
	# which we create square distances that are used to compute the flux due to all cells at a 
	# given cell. For the cell where the flux originates, the coordinate distance would be zero.
	# Thus, we set the distance to 1/2 the size of the cell in each dimension, corresponding to 
	# the average distance from the center of points in a cube which is about 1/2.
	x_min_all = np.min(coor, axis = 0)
	x_max_all = np.max(coor, axis = 0)
	
	x_min_hr = -np.ones((3), np.float) * r_half_gas_cat
	x_max_hr = np.ones((3), np.float) * r_half_gas_cat
	
	# Find the particles that lie inside/outside the high-def region
	mask_in = (coor[:, 0] >= x_min_hr[0]) & (coor[:, 0] <= x_max_hr[0]) \
			& (coor[:, 1] >= x_min_hr[1]) & (coor[:, 1] <= x_max_hr[1]) \
			& (coor[:, 2] >= x_min_hr[2]) & (coor[:, 2] <= x_max_hr[2])
	mask_out = np.logical_not(mask_in)
	mask_in_sf = mask_in[mask_sf]
	
	cell_umw = np.zeros_like(sfr)
	
	# Execute the computation twice, once for the inner region and once for the outer region which
	# can get large and thus degrade the resolution significantly.
	for l in range(2):
		
		if l == 0:
			mask_ptl = mask_in
			mask_src = mask_in_sf
			x_min = x_min_hr
			x_max = x_max_hr
		else:
			mask_ptl = mask_out
			mask_src = np.ones_like(sf_uv_flux, np.bool)
			x_min = x_min_all
			x_max = x_max_all
			
		if np.count_nonzero(mask_ptl) == 0:
			continue
			
		# Use same grid dimensions in all directions for FFT
		ext_dims = x_max - x_min
		max_ext = np.max(ext_dims)
		midpoint = 0.5 * (x_max + x_min)
		x_min = midpoint - 0.5 * max_ext
		x_max = midpoint + 0.5 * max_ext
		
		g_flux_sf, bin_edges = np.histogramdd(sf_coor[mask_src, :], weights = sf_uv_flux[mask_src], 
				bins = uv_ngrid, range = [[x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]]])
		dx = bin_edges[0][1] - bin_edges[0][0]
	
		source_fft = np.fft.fftn(g_flux_sf)
		g_umw = np.real(np.fft.ifftn(source_fft * kernel_fft / dx**2))

		# There is also a numpy way, but it's slower (and uses N^6 memory points)
		#d2 = (gx[..., np.newaxis, np.newaxis, np.newaxis] - gx)**2 \
		#		+ (gy[..., np.newaxis, np.newaxis, np.newaxis] - gy)**2 \
		#		+ (gz[..., np.newaxis, np.newaxis, np.newaxis] - gz)**2
		#d2[d2 == 0.0] = d2_in_cell
		#g_f = np.sum(g_sfr / d2, axis = (3, 4, 5))
		if compare_to_direct:
			gx, gy, gz = np.meshgrid(bin_edges[0][:-1], bin_edges[1][:-1], bin_edges[2][:-1], indexing = 'ij')
			g_umw_loop = np.zeros((uv_ngrid, uv_ngrid, uv_ngrid), np.float)
			d2_in_cell = 3.0 * (0.5 * dx)**2
			for i in range(uv_ngrid):
				for j in range(uv_ngrid):
					for k in range(uv_ngrid):
						d2 = (gx - gx[i, j, k])**2 + (gy - gy[i, j, k])**2 + (gz - gz[i, j, k])**2
						d2[i, j, k] = d2_in_cell
						g_umw_loop[i, j, k] = np.sum(g_flux_sf / d2)
			diff = g_umw / g_umw_loop - 1.0
			print(np.min(diff), np.max(diff), np.median(diff))
		
		# Convert MW units
		g_umw /= flux_1000A_mw_draine
		
		# Find the cell indices in which non-SF gas particles lie
		bin_centers_x = np.array(bin_edges[0][:-1]) + 0.5 * dx
		bin_centers_y = np.array(bin_edges[1][:-1]) + 0.5 * dx
		bin_centers_z = np.array(bin_edges[2][:-1]) + 0.5 * dx
		interp = scipy.interpolate.RegularGridInterpolator((bin_centers_x, bin_centers_y, bin_centers_z), 
						g_umw, bounds_error = False, fill_value = None)
		cell_umw[mask_ptl] = interp(coor[mask_ptl])
		
	return cell_umw

###################################################################################################

def uvPropagationMap(kernel_fft, map_extent, mask_sf, sf_model, Sigma_gas, Sigma_sfr, metallicity, 
						uv_escape_frac, compare_to_direct = False):
	
	if uv_escape_frac == 'var':
		# Note that we are not using the exact hydrogen fraction for this computation. This would
		# mean computing an extra map of a quantity that is very close to sf_model.fH everywhere.
		N_H = Sigma_gas / constants_KPC**2 * constants_MSUN * sf_model.f_H / constants_M_PROTON
		tau_uv = 0.5 * sigma_H_1000A * N_H
		f_esc = np.exp(-tau_uv * metallicity / 0.0127)
	else:
		f_esc = uv_escape_frac
	
	Nx = Sigma_gas.shape[0]
	Ny = Sigma_gas.shape[1]
	dx = (map_extent[1] - map_extent[0]) / Nx
	dy = (map_extent[3] - map_extent[2]) / Ny
	cell_area = dx * dy

	# Compute the map of input flux. The computation works in a square grid, so we need to pad
	# the y-direction for non-square maps.
	sf_uv_flux = Sigma_sfr * flux_1000A_sb99 * f_esc * cell_area
	if Nx == Ny:
		sf_uv_flux_padded = sf_uv_flux
	else:
		sf_uv_flux_padded = np.zeros((Nx, Nx), np.float)
		sf_uv_flux_padded[:Nx, :Ny] = sf_uv_flux
	
	# Compute the convolution of a 1/r^2 kernel with the input flux map via FFT.
	source_fft = np.fft.fftn(sf_uv_flux_padded)
	map_uv_mw = np.real(np.fft.ifftn(source_fft * kernel_fft / cell_area))
	map_uv_mw = map_uv_mw[:Nx, :Ny]

	# Compute directly via for loop; this is useful to verify the FFT calculation
	if compare_to_direct:
		bin_edges_x = np.linspace(map_extent[0], map_extent[1], Nx + 1)
		bin_edges_y = np.linspace(map_extent[2], map_extent[3], Ny + 1)
		gx, gy = np.meshgrid(bin_edges_x[:-1], bin_edges_y[:-1], indexing = 'ij')
		map_uv_mw_direct = np.zeros_like(Sigma_gas)
		for i in range(Nx):
			for j in range(Ny):
				d2 = (gx - gx[i, j])**2 + (gy - gy[i, j])**2
				d2[i, j] = 2.0 * (0.5 * dx)**2
				map_uv_mw_direct[i, j] = np.sum(sf_uv_flux / d2)
		diff = 	map_uv_mw / map_uv_mw_direct - 1.0
		print(np.min(diff), np.max(diff), np.median(diff))

	# Convert MW units
	map_uv_mw /= flux_1000A_mw_draine
	
	return map_uv_mw

###################################################################################################

# Create a 1/r^2 kernel in 3D space and take its FFT.

def createFFTKernel3D(N):

	bins = np.linspace(0.0, N - 1, N)
	gx, gy, gz = np.meshgrid(bins, bins, bins, indexing = 'ij')
	Nhalf = N // 2
	
	# We need to take care of the one element that is 1/0
	mask = np.ones_like(gx, np.bool)
	mask[Nhalf, Nhalf, Nhalf] = False
	kernel = np.zeros_like(gx)
	kernel[mask] = 1.0 / ((gx[mask] - Nhalf)**2 + (gy[mask] - Nhalf)**2 + (gz[mask] - Nhalf)**2)
	kernel[Nhalf, Nhalf, Nhalf] = 1.0 / (3.0 * 0.5**2)
	
	# Shift the kernel into the 0,0 corner with periodic boundaries. This can be done in a single
	# function:
	# 
	# kernel = np.roll(kernel, (-Nhalf, -Nhalf, -Nhalf), axis = (0, 1, 2))
	#
	# but this syntax wasn't introduced until numpy 1.12. which makes it a bit unsafe.
	kernel = np.roll(kernel, -Nhalf, axis = 0)
	kernel = np.roll(kernel, -Nhalf, axis = 1)
	kernel = np.roll(kernel, -Nhalf, axis = 2)
	kernel_fft = np.fft.fftn(kernel)
	
	return kernel_fft

###################################################################################################

# Create a 1/r^2 kernel in 2D space and take its FFT.

def createFFTKernel2D(N):

	bins = np.linspace(0.0, N - 1, N)
	gx, gy = np.meshgrid(bins, bins, indexing = 'ij')
	Nhalf = N // 2
	
	# We need to take care of the one element that is 1/0
	mask = np.ones_like(gx, np.bool)
	mask[Nhalf, Nhalf] = False
	kernel = np.zeros_like(gx)
	kernel[mask] = 1.0 / ((gx[mask] - Nhalf)**2 + (gy[mask] - Nhalf)**2)
	kernel[Nhalf, Nhalf] = 1.0 / (2.0 * 0.5**2)
	
	# Shift the kernel into the 0,0 corner with periodic boundaries. See function above for a new
	# version with only one np.roll call.
	kernel = np.roll(kernel, -Nhalf, axis = 0)
	kernel = np.roll(kernel, -Nhalf, axis = 1)
	kernel_fft = np.fft.fftn(kernel)
	
	return kernel_fft

###################################################################################################

# This function returns the UV background according to the given model in units of the local 
# radiation field. The comparison to local values can be undertaken in different ways. The 
# difference depends on the UVB model because those models predict different spectral shapes as 
# well as different intensities. 

def uvBackground(snap_a, uv_bg_model, uv_bg_calibration):

	if uv_bg_model != 'fauchergiguere09':
		raise Exception('Only fauchergiguere09 background currently implemented.')

	if uv_bg_calibration == '1000A':
		table_idx = 1
	elif uv_bg_calibration == 'lyman-werner':
		table_idx = 2
	elif uv_bg_calibration == 'photoheating':
		table_idx = 3
	else:
		raise Exception('Unknown quantity, %s.' % uv_bg_calibration)

	if not uv_bg_calibration in global_uvb_interp:
		this_dir = os.path.dirname(os.path.realpath(__file__))
		uvb_table = np.loadtxt(this_dir + '/umw.txt', unpack = True)
		uvb_interp = scipy.interpolate.InterpolatedUnivariateSpline(uvb_table[0], uvb_table[table_idx])
		global_uvb_interp[uv_bg_calibration] = uvb_interp
	
	uvb_interp = global_uvb_interp[uv_bg_calibration]
	z = 1.0 / snap_a - 1.0
	U_MW = uvb_interp(z)
	
	return U_MW

###################################################################################################

# The UVB ionization rate (in s^-1) from the Haardt & Madau 2001 model, taken from Table 2 in 
# Rahmati et al. 2013. 

def uvBackground_HM01(snap_a, q = 'photoionization'):

	if q != 'photoionization':
		raise Exception('Found quantity %s, but only photoionization is implemented in HM01 model.' % q)

	Gamma_UVB_z = np.array([8.34E-14, 7.39E-13, 1.5E-12, 1.16E-12, 7.92E-13, 5.43E-13])
	sigma_nu_z = np.array([3.27E-18, 2.76E-18, 2.55E-18, 2.49E-18, 2.45E-18, 2.45E-18])
	zz = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

	snap_z = 1.0 / snap_a - 1.0
	Gamma_UVB = 10**np.interp(snap_z, zz, np.log10(Gamma_UVB_z))
	sigma_nu = 10**np.interp(snap_z, zz, np.log10(sigma_nu_z))
	
	return Gamma_UVB, sigma_nu

###################################################################################################

# The UVB HI ionization rate (in s^-1) or the HI photoheating rate (in eV/s) from the 
# Haardt & Madau 2012 model, taken from Table 3 in their paper.

def uvBackground_HM12(snap_a, q = 'photoionization'):

	data = np.array([[
			0.00, 0.05, 0.10, 0.16, 0.21, 0.27, 0.33, 0.40, 0.47, 0.54, 0.62, 0.69, 0.78, 0.87,
			0.96, 1.05, 1.15, 1.26, 1.37, 1.49, 1.61, 1.74, 1.87, 2.01, 2.16, 2.32, 2.48, 2.65, 
			2.83, 3.02, 3.21, 3.42, 3.64, 3.87, 4.11, 4.36, 4.62, 4.89, 5.18, 5.49, 5.81, 6.14,
			6.49, 6.86, 7.25, 7.65, 8.07, 8.52, 8.99, 9.48, 9.99, 10.50, 11.10, 11.70, 12.30, 
			13.00, 13.70, 14.40, 15.10],
			[0.228E-13, 0.284E-13, 0.354E-13, 0.440E-13, 0.546E-13, 0.674E-13, 0.831E-13, 
			 0.102E-12, 0.125E-12, 0.152E-12, 0.185E-12, 0.223E-12, 0.267E-12, 0.318E-12, 
			 0.376E-12, 0.440E-12, 0.510E-12, 0.585E-12, 0.660E-12, 0.732E-12, 0.799E-12, 
			 0.859E-12, 0.909E-12, 0.944E-12, 0.963E-12, 0.965E-12, 0.950E-12, 0.919E-12, 
			 0.875E-12, 0.822E-12, 0.765E-12, 0.705E-12, 0.647E-12, 0.594E-12, 0.546E-12, 
			 0.504E-12, 0.469E-12, 0.441E-12, 0.412E-12, 0.360E-12, 0.293E-12, 0.230E-12, 
			 0.175E-12, 0.129E-12, 0.928E-13, 0.655E-13, 0.456E-13, 0.312E-13, 0.212E-13, 
			 0.143E-13, 0.959E-14, 0.640E-14, 0.427E-14, 0.292E-14, 0.173E-14, 0.102E-14, 
			 0.592E-15, 0.341E-15, 0.194E-15],
			[0.889E-13, 0.111E-12, 0.139E-12, 0.173E-12, 0.215E-12, 0.266E-12, 0.329E-12,
			 0.405E-12, 0.496E-12, 0.605E-12, 0.734E-12, 0.885E-12, 0.106E-11, 0.126E-11, 
			 0.149E-11, 0.175E-11, 0.203E-11, 0.232E-11, 0.262E-11, 0.290E-11, 0.317E-11, 
			 0.341E-11, 0.360E-11, 0.374E-11, 0.381E-11, 0.382E-11, 0.375E-11, 0.363E-11, 
			 0.346E-11, 0.325E-11, 0.302E-11, 0.279E-11, 0.257E-11, 0.236E-11, 0.218E-11, 
			 0.202E-11, 0.189E-11, 0.178E-11, 0.167E-11, 0.148E-11, 0.123E-11, 0.989E-12, 
			 0.771E-12, 0.583E-12, 0.430E-12, 0.310E-12, 0.219E-12, 0.153E-12, 0.105E-12, 
			 0.713E-13, 0.481E-13, 0.323E-13, 0.217E-13, 0.151E-13, 0.915E-14, 0.546E-14, 
			 0.323E-14, 0.189E-14, 0.110E-14]])

	snap_z = 1.0 / snap_a - 1.0
	if q == 'photoionization':
		ret = 10**np.interp(snap_z, data[0], np.log10(data[1]))
	elif q == 'photoheating':
		ret = 10**np.interp(snap_z, data[0], np.log10(data[2]))
	else:
		raise Exception('Unknown quantity, %s.' % q)
	
	return ret

###################################################################################################

# The UVB HI ionization rate (in s^-1) from the Faucher-Giguere et al. 2009, taken from Table 2 in 
# their paper.

def uvBackground_FG09(snap_a, q = 'photoionization'):
	
	data = np.array([
			[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 
			3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0],
            [0.0384, 0.0728, 0.1295, 0.2082, 0.3048, 0.4074, 0.4975, 0.5630, 0.6013, 0.6142, 
			0.6053, 0.5823, 0.5503, 0.5168, 0.4849, 0.4560, 0.4320, 0.4105, 0.3917, 0.3743, 0.3555, 
			0.3362, 0.3169, 0.3001, 0.2824, 0.2633, 0.2447, 0.2271, 0.2099],
			[0.158, 0.311, 0.569, 0.929, 1.371, 1.841, 2.260, 2.574, 2.768, 2.852, 2.839, 2.762, 
			2.642, 2.511, 2.384, 2.272, 2.171, 2.083, 2.002, 1.921, 1.833, 1.745, 1.661, 1.573, 
			1.487, 1.399, 1.305, 1.216, 1.127]])

	snap_z = 1.0 / snap_a - 1.0

	if q == 'photoionization':
		ret = 10**np.interp(snap_z, data[0], np.log10(data[1] * 1E-12))
	elif q == 'photoheating':
		ret = 10**np.interp(snap_z, data[0], np.log10(data[2] * 1E-12))
	else:
		raise Exception('Unknown quantity, %s.' % q)
	
	return ret

###################################################################################################

# Compute the interstellar radiation field based on the surface density of star formation,
# following Lagos et al. 2015. 

def uvFromSFRVol(mask_sf, mass, rho, sfr, scale_height, uv_sfr_local):

	V_cell = mass[mask_sf] / rho[mask_sf]
	Sigma_sfr = sfr[mask_sf] / V_cell * scale_height[mask_sf] * constants_KPC**2
	uv_mw = Sigma_sfr / uv_sfr_local

	return uv_mw

###################################################################################################

# Compute the interstellar radiation field based on the surface density of star formation. 

def uvFromSFRMap(Sigma_sfr, uv_sfr_local):

	uv_mw = Sigma_sfr / uv_sfr_local

	return uv_mw

###################################################################################################
	
# The Draine 78 spectrum in units of photos /cm^2 /s /Hz (note that 4pi has been multiplied
# in to remove /sr unit). An equivalent calculation can be done in eV units:
#
# E_erg = constants.H * f
# E_eV = E_erg / constants.EV	
# df_per_dEv = constants.H / constants.EV
# F = 1.658E6 * E_eV - 2.152E5 * E_eV**2 + 6.919E3 * E_eV**3
# F *= df_per_dEv * 4.0 * np.pi

def uvSpectrumDraine78(nu):

	lam = constants_C / nu * 1E8
	F = 1.068E-3 / lam - 1.719 / lam**2 + 6.853E2 / lam**3
	
	return F

###################################################################################################
