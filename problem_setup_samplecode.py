import numpy as np
from astropy import coordinates as coord

# Constants
AU  = 1.49598e13         # Astronomical Unit      [cm]
SOLAR_MASS  = 1.98892e33 # Solar mass             [g]
SOLAR_RADIUS  = 6.96e10  # Solar radius           [cm]
G = 6.673e-8             # Gravitational constant [dyne cm^2/g^2]
k = 1.3807e-16           # Boltzmann's constant   [erg/K]
mp  = 1.6726e-24         # Mass of proton         [g]

# Monte Carlo parameters
# 200 million photons for temperature output to be self-consistent
nphot_therm = 200000000
nphot_scat  = 200000000

# Grid parameters (spherical coordinates)
nr = 200         # number of radial grid points
ntheta = 200     # number of polar grid points
nphi = 200       # number of azimuthal grid points
rin = 0.05*AU    # inner radius 
rout = 100*AU    # outer radius

# Disk parameters
# Surface density based on minimum mass solar nebula (Wu & Lithwick 2021) 
surface_density_0 = 20.    # Dust density at 1 AU
inner_inc = 80.            # Inner disk inclination (from Pinilla+ 2018)
innerdisk_edge = 1.0 * AU  # Inner disk outer edge
outer_inc = 0.             # Outer disk inclination 
outer_disk_width = 0.5* AU # Outer disk radial Gaussian width 

# Star parameters for Dipper star J1604 from Sicilia-Aguilar+2020
mstar = 1.24*SOLAR_MASS
rstar = 1.4*SOLAR_RADIUS
tstar = 4730.
pstar = np.array([0.,0.,0.])

def rotate(x,y,inclination):
    """
    Simple rotation counterclockwise
    """
    cos = np.cos(inclination)
    sin = np.sin(inclination)
    x_rotated  = cos*x - sin*y
    y_rotated  = sin*x + cos*y
    return x_rotated,y_rotated

def temp_profile(radius):
    """
    Create temperature profile (motivated by a hot inner rim) 
    as a function of radius.

    Parameters:
        radius (array): radius [cm] for entire grid
    
    Returns:
        temp (array): temperature [K] as function of radius
    """
    temp = np.zeros((nr, ntheta, nphi))

    hot_inner_rim = np.where(radius < 0.07*AU)
    inner_disk = np.where((radius > 0.07*AU) & (radius < 10*AU))
    outer_disk = np.where(radius > 10*AU)

    temp[hot_inner_rim] = 1600 * (radius[hot_inner_rim]/ (0.05*AU) )** (-4)
    temp[inner_disk] = 420 * (radius[inner_disk]/ (0.07*AU) )** (-0.5)
    temp[outer_disk] = 60.

    return temp

def density_profile(rsph, rcyl, height, temp):
    """
    Calculate density for inner and outer disk modelled as:

    Surface density profile = 20 * (1/r) g/cm^2
    Radial density profile = surface density / sqrt(2pi * scale height)
    Vertical density profile = Gaussian dist. with width = scale height 

    At outer disk, add radial Gaussian smoothing

    Parameters:
        rsph (array): spherical radius [cm] for entire grid
        rcyl (array): cylindrical radius [cm] for entire grid
        height (array): height [cm] above disk midplane
        temp (array): temperature [K] as function of radius
    
    Returns:
        density (array): density [g/cm^3] as function of
                         radius, temperature, scale height
    """
    # Calculate isothermal sound speed
    sound_speed = np.sqrt( k*temp / (2.3*mp) )

    # Calculate local scale height
    scale_height = sound_speed/np.sqrt( G*mstar / (rcyl**3) )

    # Calculate density profile
    surface_density  = surface_density_0 * (rsph/AU)**(-1.)
    radial_profile = surface_density / ( np.sqrt(2.*np.pi) * scale_height )
    vertical_profile = np.exp(-0.5 * (height**2/scale_height**2) )
    density = radial_profile * vertical_profile

    # At outer disk, add radial Gaussian smoothing
    outer_disk = np.where(radius > 10*AU)
    density[outer_disk] *= np.exp(-0.5*( (radius[outer_disk]-80*AU) / outer_disk_width )**2)
    
    # No dust between inner and outer disk
    between = np.where((radius > innerdisk_edge) & (radius < 70*AU))
    density[between] = 1e-30

    return np.array(density)

# Make the coordinates
ri       = np.logspace(np.log10(rin),np.log10(rout),nr+1)
thetai   = np.linspace(0.,np.pi,ntheta+1)
phii     = np.linspace(1e-4, np.pi-1e-4, nphi+1)   # Sample hemisphere
radius   = 0.5 * ( ri[:-1] + ri[1:] )              # Take the in-between values
theta   = 0.5 * ( thetai[:-1] + thetai[1:] )
phi     = 0.5 * ( phii[:-1] + phii[1:] )
nr       = len(radius)                             # Recompute nr, because of refinement at inner edge

# Make the grid such that coord [i,j,k] can be found
# as r_mesh[i,:,:], theta_mesh[:,j,:], phi_mesh[:,:,k]
r_mesh, theta_mesh, phi_mesh = np.meshgrid(radius,theta,phi,indexing='ij')

# Coordinate transform
# subtract pi/2 because astropy takes values between -90 and 90 deg
x, y, z = coord.spherical_to_cartesian(r_mesh, theta_mesh-np.pi/2, phi_mesh)
rsph = np.sqrt(x**2 + y**2 + z**2)
rcyl = np.sqrt(x**2 + y**2)

inclination = np.zeros((nr, ntheta, nphi))
inclination[np.where(rsph < 10*AU)] = inner_inc * (np.pi/180.)

# Rotate along y-axis, in order to calculate height above midplane
xd, zd = rotate(x, z, inclination)

# Make the dust density model
temperature = temp_profile(rsph)
density = density_profile(rsph, rcyl, zd, temperature)

# Write the wavelength_micron.inp file
lam1     = 0.1e0
lam2     = 7.0e0
lam3     = 25.e0
lam4     = 1.0e4
n12      = 20
n23      = 100
n34      = 30
lam12    = np.logspace(np.log10(lam1),np.log10(lam2),n12,endpoint=False)
lam23    = np.logspace(np.log10(lam2),np.log10(lam3),n23,endpoint=False)
lam34    = np.logspace(np.log10(lam3),np.log10(lam4),n34,endpoint=True)
lam      = np.concatenate([lam12,lam23,lam34])
nlam     = lam.size

# Write the wavelength file
with open('wavelength_micron.inp','w+') as f:
    f.write('%d\n'%(nlam))
    for value in lam:
        f.write('%13.6e\n'%(value))

# Write the stars.inp file
with open('stars.inp','w+') as f:
    f.write('2\n')
    f.write('1 %d\n\n'%(nlam))
    f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n'%(rstar,mstar,pstar[0],pstar[1],pstar[2]))
    for value in lam:
        f.write('%13.6e\n'%(value))
    f.write('\n%13.6e\n'%(-tstar))

# Write the grid file
with open('amr_grid.inp','w+') as f:
    f.write('1\n')                       # iformat
    f.write('0\n')                       # AMR grid style  (0=regular grid, no AMR)
    f.write('100\n')                     # Coordinate system: spherical
    f.write('0\n')                       # gridinfo
    f.write('1 1 1\n')                   # Include r,theta coordinates
    f.write('%d %d %d\n'%(nr,ntheta,nphi))  # Size of grid
    for value in ri:
        f.write('%13.6e\n'%(value))      # r coordinates (cell walls)
    for value in thetai:
        f.write('%13.6e\n'%(value))      # theta coordinates (cell walls)
    for value in phii:
        f.write('%13.6e\n'%(value))      # phi coordinates (cell walls)

# Write the density file
with open('dust_density.inp','w+') as f:
    f.write('1\n')                       # Format number
    f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
    f.write('1\n')                       # Nr of dust species
    data = density.ravel(order='F')      # Create a 1-D view, fortran-style indexing
    data.tofile(f, sep='\n', format="%13.6e")
    f.write('\n')

# Dust opacity control file
with open('dustopac.inp','w+') as f:
    f.write('2               Format number of this file\n')
    f.write('1               Nr of dust species\n')
    f.write('============================================================================\n')
    f.write('1               Way in which this dust species is read\n')
    f.write('0               0=Thermal grain\n')
    f.write('silicate        Extension of name of dustkappa_***.inp file\n')
    f.write('----------------------------------------------------------------------------\n')

# Write the radmc3d.inp control file
with open('radmc3d.inp','w+') as f:
    f.write('nphot = %d\n'%(nphot_therm))
    f.write('nphot_scat = %d\n'%(nphot_scat))

    # Turn on isotropic scattering
    f.write('scattering_mode_max = 1\n')

    # Turn on modified random walk for faster computation
    # in optically depth regions
    f.write('modified_random_walk = 1\n')

    # Use star of finite radius 
    f.write('istar_sphere = 1\n')
    
    # Use RADMC-3D in parallel with 16 threads
    f.write('setthreads = 16')
