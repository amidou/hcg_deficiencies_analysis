import numpy as np
import pandas as pd
# from scipy.stats import spearmanr
from astroquery.ipac.ned import Ned
from astroquery.irsa_dust import IrsaDust
from astroquery.vizier import Vizier
from astroquery.sdss import SDSS
import astropy.coordinates as coords
import astropy.units as u
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.wcs import WCS
# from astropy.table import Table, QTable
from astropy.nddata import Cutout2D
# from astropy.visualization import make_lupton_rgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import re, os, warnings
# from astropy.utils.exceptions import AstropyWarning
import astropy.visualization as viz # MinMaxInterval, ImageNormalize, LinearStretch, ZScaleInterval, SqrtStretch, LogStretch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties
import subprocess
import yaml
import legacystamps
import splusdata
import pymc as pm
import arviz as az
from spectral_cube import SpectralCube
from regions import Regions

def write_region_file(g):
    '''
    Write a region file in CRTF format for a HCG.
    Uses the YAML file `group_members.yml`.
    Parameters
    ----------
    g: HCG name
    '''
    colors = {'counterparts': '10161a',
              'nocounterparts': 'db3737',
              'noz': '7157d9',
              'core': '2965cc'}
    
    with open('group_members.yml') as f:
        group_members = yaml.safe_load(f)
    members = group_members[g]
    fout = open('%s_regions.crtf' %g.lower().replace(' ',''), 'w')
    c = Ned.query_object(g)
    cra, cdec = c['RA'][0], c['DEC'][0]
    fout.write('#CRTFv0 CASA Region Text Format version 0\n')
    fout.write('circle [[%fdeg, %fdeg], 1800arcsec] coord=ICRS, linewidth=2, linestyle=-, symsize=1,\
    symthick=1, color=10161a, font=Helvetica, fontsize=10, fontstyle=bold, usetex=false\n' %(cra,cdec))
    for key in members.keys():
        if not key in ['confused', 'core_center']:
            sources = members[key]
            color = colors[key]
            if key == 'core':
                sources = [item.strip() for sublist in sources for item in sublist.split(',')]
            for s in sources:
                if len(s) == 1:
                    t = Ned.query_object(g+s.strip())
                else:
                    t = Ned.query_object(s)
                ra, dec = t['RA'][0], t['DEC'][0]
                fout.write('symbol [[%fdeg, %fdeg], s] coord=ICRS, linewidth=2, linestyle=-, symsize=1, symthick=1,\
                color=%s, font=Helvetica, fontsize=10, fontstyle=bold, usetex=false, label="%s", labelcolor=%s, labelpos=top\n'\
                %(ra,dec,color,s,color))
        
    fout.close()

def center_radec(ra, dec, mass):
    '''
    Calculates the RA & Dec coordinates of the centre of mass of a set of objects given their positions and masses
    Params:
    ----------
    ra: array of RAs
    dec: array of Decs
    mass: array of stellar masses
    Returns the ra, dec coordinates of the center of mass
    '''
    # Converting degrees to radians
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    # Converting spherical coordinates to Cartesian coordinates
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    
    # Calculating the weighted mean of Cartesian coordinates
    weighted_x = np.sum(x * mass) / np.sum(mass)
    weighted_y = np.sum(y * mass) / np.sum(mass)
    weighted_z = np.sum(z * mass) / np.sum(mass)

    # Convert weighted mean Cartesian coordinates back to spherical coordinates
    center_ra = np.arctan2(weighted_y, weighted_x)
    center_dec = np.arcsin(weighted_z)

    # Convert radians to degrees
    center_ra_deg = np.degrees(center_ra)
    center_dec_deg = np.degrees(center_dec)
    
    return center_ra_deg, center_dec_deg

def center_of_mass(g):
    '''
    Calculates the center of mass of an HCG
    Params:
    ----------
    g: HCG ID
    '''
    with open('group_members.yml') as f:
        members = yaml.safe_load(f)
    with open('data_files.yml') as f:
        params = yaml.safe_load(f)

    hcg_members = members[g]
    core_members = hcg_members['core']
    name_cores = [item.strip() for sublist in core_members for item in sublist.split(',')]
    name_cores = [g+name_core.strip() if len(name_core)==1 else name_core for name_core in name_cores]

    masses = pd.read_csv(f'stellar_masses_h{g.split()[-1]}.csv', index_col='ID')
    core_masses = masses[masses.Name.isin(name_cores)]
    ra, dec, sm = core_masses.RA, core_masses.Dec, 10**core_masses.logMstar
    cen_ra, cen_dec = center_radec(ra, dec, sm)
    return cen_ra, cen_dec

def make_cutout(fp, ra, dec, majax_am, minax_am, angle):
    '''
    Perform ellipse cutout of 2D or 3D fits.
    Parameters
    ----------
    ra: str
        central R.A of output cutout.
    dec: str
        central Dec. of output cutout.
    majax_am: float
        major axis of output ellipse in arcmin.
    minax_am: float
        minor axis of output ellipse in arcmin.
    angle: float
        PA angle of ellipse in degrees.
    '''
    data, hdr = fits.getdata(fp, header=True)
    wcs = WCS(hdr).celestial
    majax = coords.Angle(majax_am, unit='arcmin')
    minax = coords.Angle(minax_am, unit='arcmin')
    gcoo = coords.SkyCoord(ra, dec, unit='hourangle,degree', frame='icrs')
    if len(data.shape) == 2:
        cutout = Cutout2D(data, position=gcoo, size=1.1*majax, wcs=wcs)
        cutout_data, cutout_wcs = cutout.data, cutout.wcs
    elif len(data.shape) >= 3:
        if len(data.shape) > 3:
            data = data[0]
        cutout_data = []
        for k in range(data.shape[0]):
            cutout = Cutout2D(data[k], position=gcoo, size=1.1*majax, wcs=wcs)
            cutout_data.append(cutout.data)
        cutout_data = np.array(cutout_data)
        cutout_wcs = cutout.wcs
    y, x = np.ogrid[0:cutout.data.shape[0], 0:cutout.data.shape[1]]
    coo = cutout_wcs.pixel_to_world(x, y)
    mask = ((coo.ra.deg - gcoo.ra.deg) * np.cos(np.radians(90-angle)) + 
            (coo.dec.deg - gcoo.dec.deg) * np.sin(np.radians(90-angle)))**2 / (0.5 * majax.deg)**2 + \
            ((coo.ra.deg - gcoo.ra.deg) * np.sin(np.radians(90-angle)) - 
             (coo.dec.deg - gcoo.dec.deg) * np.cos(np.radians(90-angle)))**2 / (0.5 * minax.deg)**2 > 1
    if len(data.shape) == 2:
        cutout_data[mask] = np.nan
    elif len(data.shape) >= 3:
        for k in range(data.shape[0]):
            cutout_data[k][mask] = np.nan
    return cutout_data, cutout_wcs, hdr

def download_grz(g, size=1.5, ddir='/mnt/scratch/HCGs/legacy_images', bands='grz'):
    '''
    Download DECaLS images of an HCG.
    Parameters
    ----------
    g: HCG name
    size: angular size in degrees
    ddir: output directory
    '''
    gtab = Ned.query_object(g)
    gcoo = coords.SkyCoord(gtab['RA'][0], gtab['DEC'][0], unit='deg', frame='icrs')
    fname = legacystamps.download(ra=gcoo.ra.deg, dec=gcoo.dec.deg, mode='fits', bands=bands, size=size, autoscale=True, ddir=ddir, layer='ls-dr10')
    if os.path.isfile(fname):
        if not ddir.endswith('/'):
            ddir += '/'
        newname = ddir + g.replace(' ','') + '_grz.fits'
        os.rename(fname, newname)
        print('File renamed to ', newname)
    else:
        print('!!! Warning: file not downloaded!')
    return newname

def normalize_image(image_data, contrast=0.005, stretch_func='sinh', a=0.1):
    '''
    Normalize an image using the astropy.visualization module.
    Parameters
    ----------
    image_data: HDU data of the image
    '''
    interval = viz.ZScaleInterval(contrast=contrast)

    if stretch_func == 'sinh':
        stretch = viz.SinhStretch(a)
    elif stretch_func == 'asinh':
        stretch = viz.AsinhStretch(a)
    elif stretch_func == 'log':
        stretch = viz.LogStretch(a)
    elif stretch_func == 'sqrt':
        stretch = viz.SqrtStretch()
    
    norm = viz.ImageNormalize(image_data, interval=interval, stretch=stretch)
    return norm(image_data)

def mag_to_Mag(mag, dist_Mpc):
    '''
    Convert apparent magnitude to absolute mag.
    Parameters
    ----------
    mag: apparent magnitude
    dist_Mpc: distance of galaxy in Mpc
    '''
    dist_pc = dist_Mpc * 1e6
    abs_mag = mag - 5 * (np.log10(dist_pc) - 1.)
    return abs_mag
    
def mag_to_smass(gmag, rmag, dist, e_gmag=None, e_rmag=None):
    '''
    Calculate the Bell stellar mass of a galaxy using the Garcia-Benito+ 2019 calibration.
    Parameters
    ----------
    gmag: g magnitude
    rmag: r magnitude
    dist: distance in Mpc
    unc : return associated uncertainty
    '''
    alpha, beta = 1.49, -0.70 # Garcia-Benito+ 2019 fit parameters
    Magr = mag_to_Mag(rmag, dist)
    logms = alpha * (gmag - rmag) + beta - 0.4 * (Magr - 4.67)
    if (e_gmag and e_rmag):
        e_logms = np.sqrt(alpha*np.sqrt(e_gmag**2 + e_rmag**2)**2 + ((0.4*e_gmag)**2 + 0.01)**2)
        return logms, e_logms
    else:
        return logms

def wisea_pattern(name):
    # Define a regular expression pattern to extract RA and Dec parts
    pattern = re.compile(r"WISEA J(\d{6})\.(\d+)([-+])(\d{6})\.(\d+)")

    match = pattern.match(name)
    if match:
        ra_hms = match.group(1)
        dec_dms = match.group(4)
        # Format the extracted parts into the desired string format
        formatted_name = f'WJ{ra_hms}_{dec_dms}'
    else:
        raise Exception('Source name does not match a WISEA nomenclature')
    return(formatted_name)

def id_to_name(g, file_members='group_members.yml'):
    '''
    Return a dictionary of IDs and source names
    ----------
    g: HCG ID name
    '''
    with open(file_members) as f:
        group_members = yaml.safe_load(f)
    gm = group_members[g]
    mlist = []
    for key in ['counterparts', 'nocounterparts', 'noz', 'confused']:
        if key in gm.keys():
            for m in gm[key]:
                mlist.append(m)
    if 'core' in gm.keys():
        m_cores = [item.strip() for sublist in gm['core'] for item in sublist.split(',')]
        for mc in m_cores:
            if len(mc) == 1: mc = g + mc
            mlist.append(mc)
    if 'complexes' in gm.keys():
        for _ in gm['complexes']:
            mlist.append(gm['complexes'][_]['source'])
            if isinstance(gm['complexes'][_]['satellites'], list):
                for ms in gm['complexes'][_]['satellites']:
                    mlist.append(ms)
            else:
                mlist.append(gm['complexes'][_]['satellites'])
    mlist = list(set(mlist))
    idx = [wisea_pattern(s) if s.startswith('WISEA') else s for s in mlist]
    idx = np.array([m.split('.')[0]
                    .replace('-','_')
                    .replace(' ','')
                    .replace('__','')
                    .replace('2MASS','2M')
                    # .replace('WISEA','W')
                    .replace('2dFGRS','2d')
                    .replace('APMUKS(BJ)','AP_')
                    for m in idx])
    id_dict = {}
    for id_, name_ in zip(idx, mlist):
        id_dict[id_] = name_ 
    return id_dict

def query_SDSS_magnitude(ra, dec, mag_type='auto'):
    '''
    Query the SDSS g and r magnitudes of individual galaxies.
    Parameters
    ----------
    ra: RA of galaxy
    dec: Declination of galaxy
    mag_type: type of magnitudes to return (auto, petro, etc.)
    '''
    g, e_g, r, e_r = mag_type+'Mag_g', mag_type+'MagErr_g', mag_type+'Mag_r', mag_type+'MagErr_r'
        
    # Define coordinates using astropy
    coordinates = coords.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')

    # Query SDSS for imaging data
    photo_query = SDSS.query_region(coordinates, spectro=True, radius='1arcmin', photoobj_fields=[g, e_g, r, e_r, 'type', 'ra', 'dec'])

    # Check if the query returned results
    if photo_query is not None and len(photo_query) > 0:
        # Filter galaxies based on the 'type' parameter
        galaxies = photo_query[photo_query['type'] == 3]
        
        if len(galaxies) == 1:
            return galaxies[g][0], galaxies[e_g][0], galaxies[r][0], galaxies[e_r][0]
        elif len(galaxies) > 1:
            q_ra, q_dec = galaxies['ra'], galaxies['dec']
            sep = coordinates.separation(coords.SkyCoord(q_ra, q_dec, unit=u.deg, frame='icrs'))
            gmags, e_gmags, rmags, e_rmags = galaxies[g], galaxies[e_g], galaxies[r], galaxies[e_r]
            idx = np.argsort(sep)
            sep, gmags, e_gmags, rmags, e_rmags = sep[idx], gmags[idx], e_gmags[idx], rmags[idx], e_rmags[idx]
            
            return gmags[0], e_gmags[0], rmags[0], e_rmags[0]
        else:
            return None
    else:
        return None

def get_splus_phot(g, radius_deg=2, user='asorgho', passwd='11sz0795'):
    '''
    Query the S-PLUS photometries for all galaxies around a given group.
    Parameters
    ----------
    g: HCG name
    radius_deg: cone radius in degrees
    '''
    gned = Ned.query_object(g)
    gra, gdec = gned['RA'][0], gned['DEC'][0]
    conn = splusdata.Core(user, passwd)
    result_table = conn.query(f"""
    SELECT
        det.ID,
        det.RA,
        det.DEC,
        g.g_auto, g.e_g_auto,
        r.r_auto, r.e_r_auto,
        det.A AS radius, det.ELONGATION as AB_ratio,
        i.i_auto, i.e_i_auto,
        sqg.PROB_GAL,
        pz.zml,
        det.s2n_DET_auto
    FROM
        dr4_dual.dr4_dual_detection AS det
    JOIN
        dr4_dual.dr4_dual_g AS g ON (g.ID = det.ID)
    JOIN
        dr4_dual.dr4_dual_r AS r ON (r.ID = det.ID)
    JOIN
        dr4_dual.dr4_dual_i AS i ON (i.ID = det.ID)
    JOIN
        dr4_vacs.dr4_star_galaxy_quasar AS sqg ON (sqg.ID = det.ID)
    JOIN
        dr4_vacs.dr4_gal_photoz AS pz ON (pz.ID = det.ID)
    WHERE
        1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {gra}, {gdec}, {radius_deg}))
    AND
        det.s2n_DET_auto > 3
    AND
        sqg.CLASS = 2
    AND
        sqg.PROB_GAL >= 0.9
    AND
        pz.zml < 0.1
    AND
        (pz.zml_84p - pz.zml_16p) < 0.02
    """)
    ascii.write(result_table, 'phot_%s_splus.csv' %g.lower().replace(' ',''), format='csv', overwrite=True)
    return result_table
    
def jy_to_cm2(data,hdr):
    '''
    Convert moment 0 map into column density
    Parameters
    ----------
    data: HDU data array
    hdr: HDU header
    '''
    cell_size = abs(hdr['CDELT1'])
    bunit = hdr['BUNIT']
    if bunit.lower() in ['jy/beam.m/s', 'jy/beam*m/s']:
        fac = 1.249e21
    elif bunit.lower() in ['jy/beam.km/s', 'jy/beam*km/s']:
        fac = 1.249e24
    else:
        fac = 1.249e24
        print('check moment 0 units, considering "Jy/beam.km/s" but could be wrong')
    # cell_area = cell_size * cell_size
    bmaj, bmin = hdr['BMAJ'] * 3600., hdr['BMIN'] * 3600.
    beam_area = bmaj * bmin * np.pi / (4.0 * np.log(2.0))
    nhi = fac * data / beam_area
    return nhi

def velocity_dispersion(g):
    '''
    Calculate the velocity dispersion of a HCG from the systemic velocities of its members.
    Uses NED to get the systemic velocities.
    Parameters
    ----------
    g: HCG name
    '''
    with open('group_members.yml') as f:
        group_members = yaml.safe_load(f)
    with open('data_files.yml') as f:
        params = yaml.safe_load(f)
    df = pd.read_csv('params_tables/HIdef_%s.csv' %g.replace(' ','').lower(), na_values='--')
    members = group_members[g]
    cores = members['core']
    vcen = params[g]['velocity']
    name_cores = [item.strip() for sublist in cores for item in sublist.split(',')]
    name_cores = [g+x if len(x) == 1 else x for x in name_cores]
    vsys = np.zeros(len(name_cores))
    for i,name in enumerate(name_cores):
        try:
            ned_v = Ned.get_table(name, table='redshifts')
        except:
            vsys[i] = np.nan
        else:
            vsys[i] = np.ma.median(ned_v['Published Velocity'])
    delta_v_sq = np.array([(x-vcen)**2 for x in vsys])
    vdisp = np.sqrt(np.nanmean(delta_v_sq))
    return vdisp

# def virial_radius(g):
#     '''
#     Calculate the virial radius rvir (in arcmin) of a HCG group from its velocity dispersion.
#     Uses the approximation in Toribio+ 2011 (eq. 3), and calls the function velocity_dispersion
#     Parameters
#     ----------
#     g: HCG name
#     '''
#     with open('data_files.yml') as f:
#         params = yaml.safe_load(f)
#     d_Mpc = params[g]['distance']
#     vdisp = velocity_dispersion(g)
#     rvir_Mpc = (np.sqrt(3.) / 700.) * vdisp
#     rvir_radian = rvir_Mpc/d_Mpc
#     return rvir_radian * 3437.75

def group_virial_mass(g, G=4.301e-9):
    """
    Calculate the total mass of a galaxy group using the projected mass estimator.
    From Heisler et al. 1985 (https://ui.adsabs.harvard.edu/abs/1985ApJ...298....8H/abstract).
    
    Parameters:
    - center_vel (float): Line-of-sight velocity of the group (in km/s).
    - velocities (array-like): Line-of-sight velocities of galaxies (in km/s).
    - center_position (SkyCoord): SkyCoord object representing the center of the group.
    - positions (SkyCoord array): Array of SkyCoord objects representing the positions of galaxies.
    - group_distance (float): Distance to the galaxy group in megaparsecs (Mpc).
    - G (float): Gravitational constant in (km/s)^2 * Mpc / Msun. Default is 4.302e-9.

    Returns:
    - Mvir (float): Estimated total mass of the group in solar masses (Msun).
    """
    # velocities = np.array(velocities)  # Ensure inputs are numpy arrays

    with open('data_files.yml') as f:
        params = yaml.safe_load(f)
    dist_Mpc = params[g]['distance']

    center_vel = params[g]['velocity']
    center_radec = center_of_mass(g)
    center_coo = coords.SkyCoord(center_radec[0], center_radec[1], unit='deg', frame='icrs')

    t = pd.read_csv('/mnt/scratch/HCGs/params_tables/HIdef_%s.csv' %g.replace(' ','').lower(), na_values='--')
    t.dropna(subset=['Name'], inplace=True)
    ra, dec, = t['RA'], t['Dec']

    ra_cores = t[(t.Name.str.contains('core')) & (~t.Name.str.contains('total'))]['RA']
    dec_cores = t[(t.Name.str.contains('core')) & (~t.Name.str.contains('total'))]['Dec']
    coo_cores = coords.SkyCoord(ra_cores, dec_cores, unit='deg', frame='icrs')
    vsys_cores = t[(t.Name.str.contains('core')) & (~t.Name.str.contains('total'))]['Vsys']

   # Calculate the angular separations between all pairs of galaxies
    angular_sep = center_coo.separation(coo_cores).to(u.radian).value
    projected_distances = dist_Mpc * angular_sep
    
    vel_los = center_vel - vsys_cores

    numerator = np.sum(vel_los**2)  # v^2
    denominator = np.sum(1/projected_distances)  # Sum of 1/R
    
    vdisp = velocity_dispersion(g)
    e_vsys = vdisp / np.sqrt(len(vsys_cores))
    term1 = np.sum(vel_los * (0.02 * vsys_cores))**2
    term2 = np.sum(vel_los)**2 * (e_vsys)**2
    sigma_sv = 2. * np.sqrt(term1 + term2)
    # sigma_sv = 0.2 * np.sqrt(np.sum(vel_los**4))
    
    if denominator == 0:
        raise ValueError("The sum of projected distances cannot be zero.")

    # Compute the mass
    fac = (3. * np.pi * len(vsys_cores)) / (2. * G)
    Mvir = fac * (numerator / denominator)

    e_Mvir = Mvir * sigma_sv / numerator
    
    return Mvir, e_Mvir

def virial_radius_from_mass(Mvir, e_Mvir, G=4.301e-9, H0=70):
    """
    Calculate the virial radius (in kpc) of a galaxy group given its total mass.
    From the definition of Virial mass

    Parameters:
    - Mvir (float): Total mass of the galaxy group in solar masses (Msun).
    - group_distance (float): Distance to the galaxy group in megaparsecs (Mpc).
    - G (float): Gravitational constant in (km/s)^2 * Mpc / Msun. Default is 4.302e-9.
    - H0 (float): Hubble constant in km/s/Mpc. Default is 70.

    Returns:
    - R_vir (float): Virial radius of the galaxy group in kiloparsecs (kpc).
    """
    # Calculate the virial radius
    Rvir = (G * Mvir / (100 * H0**2))**(1/3)
    
    e_Rvir = Rvir * e_Mvir / (3. * Mvir)
    Rvir *= 1e3  # Convert from Mpc to kpc
    e_Rvir *= 1e3 # Convert from Mpc to kpc
    
    return Rvir, e_Rvir

def totflux(data, hdr):
    '''Calculate total flux from an HDU
    Parameters
    ----------
    data: HDU data array
    hdr: HDU header
    '''
    cell_size = abs(hdr['CDELT1'])
    cell_area = cell_size * cell_size
    bmaj, bmin = hdr['BMAJ'], hdr['BMIN']
    beam_area = bmaj * bmin * np.pi / (4.0 * np.log(2.0) * cell_area)
    # box = data.copy()
    sdv = np.nansum(data) / beam_area ### conversion from Jy/beam.km/s to Jy.km/s
    bunit = hdr['BUNIT']
    if bunit.lower().replace(' ','') in ['jy/beam.m/s', 'jy/beam*m/s', 'beam-1jyms-1']: ### convert from Jy/beam.m/s to Jy/beam.km/s
        sdv *= 1e-3
    return sdv

def himass(f, D):
    '''Calculate HI mass from flux f (Jy.km/s) and distance D (Mpc)
    Parameters
    ----------
    f: flux in Jy.km/s
    D: distance in Mpc
    '''
    m = 2.36e5 * D**2 * f
    
    e_f = 0.1 * f
    e_D = 0.05 * D
    e_m = m * np.sqrt((e_f/f)**2 + (2*e_D/D)**2)
    
    return np.log10(m), e_m / (m * np.log(10))

def galactic_extinction(ra, dec, band):
    coo = coords.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    ext = IrsaDust.get_extinction_table(coo)
    for i in range(len(ext)):
        if ext[i]['Filter_name'] == 'SDSS %s' %band:
            Ag = ext[i]['A_SandF']
    try:
        return Ag
    except:
        raise Exception('problem with the table')

def internal_extinction(r, ttype):
    T_to_k = {2: 0.32,
              3: 0.24,
              4: 0.16,
              5: 0.09,
              6: 0.05,
              7: 0.02}
    if np.isnan(ttype):
        ttype = 0
    if ttype < 2:
        k = 0.32
    elif ttype < 8:
        k = T_to_k[ttype]
    else:
        k = 0.02
    Ai = -2.5 * np.log10(1. + (1 - k) * r**0.58)
    if np.isnan(Ai):
        Ai = 0.
    return Ai

def kcorrect(ttype, vsys):
    T_to_ak = {1: 0.125,
               2: 0.1,
               3: 0.075,
               4: 0.065,
               5: 0.055,
               6: 0.045,
               7: 0.035,
               8: 0.025,
               9: 0.015,
               10: 0.005}
    if np.isnan(ttype):
        ttype = 0
    if ttype < 1:
        ak = 0.15
    else:
        ak = T_to_ak[ttype]
    if np.isnan(vsys):
        vsys = 0.
    Ak = ak * vsys / 1e4
    return Ak

def gmag_to_Bmag(gmag, rmag):
    '''g to B magnitude conversion from Jester+ 2005 (https://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php).
    A correction term is applied to correct for the observed offset.'''
    Bmag = gmag + 0.39*(gmag-rmag) + 0.21
    # delta_mag = 0.063 * Bmag - 0.272
    delta_mag = 0.684
    return Bmag - delta_mag

def mag_correction(mag, ra, dec, ba, ttype, vsys, band):
    Ag = galactic_extinction(ra, dec, band)
    Ai = internal_extinction(ba, ttype)
    Ak = kcorrect(ttype, vsys)
    return mag - Ag - Ai - Ak

# def Bmag_to_Btot(Bmag, ra, dec, ba, ttype, vsys):
#     Ag = galactic_extinction(ra, dec)
#     Ai = internal_extinction(ba, ttype)
#     Ak = kcorrect(ttype, vsys)
#     Btot = Bmag - Ag - Ai - Ak
#     return Btot

def mj18_d25_to_mhi(d25, D, ttype):
    '''Expected HI mass: MHI vs. D25 (kpc) relation from the Jones+ 2018 regression.
    Returns log(MHI).
    Params:
    ----------
    d25: radius d25 in arcmin
    D: distance D (Mpc)
    ttype: T type
    '''
    if ttype < 3:
        a, b, sigma = 1.04, 6.44, 0.27
    elif ttype <= 5:
        a, b, sigma = 0.93, 7.14, 0.16
    elif ttype > 5 or np.isnan(ttype):
        a, b, sigma = 0.81, 7.53, 0.17
    dkpc = d25 * D * 0.291 ### arcmin to kpc
    logm = a * 2*np.log10(dkpc) + b
    return logm, sigma

def mj18_LB_to_mhi(logLB, ttype):
    '''Expected HI mass: MHI vs. LB (Lsol) relation from the Jones+ 2018 regression.
    Returns log(MHI).
    Params:
    ----------
    LB: B total luminosity in Lsol
    ttype: T type
    '''
    if np.isnan(ttype):
        ttype = 0
    if ttype < 3:
        a, b, sigma = 1.03, -1.33, 0.
    elif ttype <= 5:
        a, b, sigma = 0.85, 0.69, 0.
    else:
        a, b, sigma = 0.84, 0.90, 0.
    logm = a * logLB + b
    return logm, sigma

def mag_to_lum(mag, dist, band='r'):
    '''Mbol values from Willmer 2018'''
    Mbol =  {'g': 5.11, 'r': 4.65}
    logL = 10. + 2. * np.log10(dist) + 0.4 * (Mbol[band] - mag)
    return logL

def rmag_to_mhi(rmag, dist, ttype, morph=True):
    '''Expected HI mass: MHI vs. Lg (Lsol) relation constructed the AMIGA sample.
    Returns log(MHI).
    Params:
    ----------
    LB: B total luminosity in Lsol
    ttype: T type
    morph: use separate calibration for each morphology bin (True) or not (False)
    '''
    if morph:
        if ttype < 3:
            morph = 'early'
        elif ttype <= 5:
            morph = 'late'
        else:
            morph = 'irregular'
    else:
        morph = 'total'
    with open('/mnt/scratch/HCGs/lum_mass_fitres.yml') as lm_fitres:
        fitres = yaml.safe_load(lm_fitres)
    a, b, sigma = fitres[morph]['a'], fitres[morph]['b'], fitres[morph]['sigma']

    logLr = mag_to_lum(rmag, dist)
    logm = a * logLr + b
    return logm, sigma

def ms_to_mhi_bok20(logms):
    return 0.44 * logms + 5.19, 0.33

def rmag_to_mhi_toribio(rmag, dist):
    Magr = mag_to_Mag(rmag, dist)
    a0, a1 = 6.44, -0.18
    sigma = 0.25
    return a0 + a1 * Magr, sigma

def rmag_to_mhi_bradford(gmag, rmag, dist):
    try:
        logms, _ = mag_to_smass(gmag, rmag, dist)
    except TypeError:
        logms = mag_to_smass(gmag, rmag, dist)
    if logms < 8.6:
        a, b, sigma = 1.052, 0.236, 0.285
    else:
        a, b, sigma = 0.461, 5.329, 0.221
    return a * logms + b - 0.15, sigma
    
def predicted_mhi(rmag, gmag, dist, ttype, logms):
    if rmag < 8.:
        rmag = np.nan
    if logms > 9.2:
        logmhi, sigma = rmag_to_mhi(rmag, dist, ttype)
    else:
        logmhi, sigma = rmag_to_mhi_bradford(gmag, rmag, dist)
    return logmhi, sigma

def Btot_to_lumB(Btot, dist, h=0.7):
    '''
    Equation 3 from Jones+ 2018
    Params:
    ----------
    gmag: g magnitude
    rmag: r magnitude
    dist: galaxy distance in Mpc
    '''
    Mbol = 4.88
    logLBh2 = 10. + 2. * np.log10(dist*h) + 0.4 * (Mbol - Btot)
    logLB = logLBh2 - 2. * np.log10(h)
    return logLB
        
def hidef(mobs, mpred, mobs_err=None, mpred_err=None):
    '''
    Calculate the HI deficiency
    Parameters:
    ----------
    mobs: log of measured HI mass
    mpred: log of predicted HI mass
    '''
    defhi = mpred - mobs
    if mobs_err and mpred_err:
        defhi_err = np.sqrt(mpred_err**2 + mobs_err**2)
    else:
        defhi_err = np.nan
    return defhi, defhi_err

def get_params(g, s):
    '''
    Get the LEDA morphological type, Vsys AND diameter of a galaxy.
    Parameters
    ----------
    s: source NED name
    '''
    df_smass = pd.read_csv('stellar_masses_h%s.csv' %g.split()[-1])
    table = df_smass[df_smass.Name.str.lower() == s.lower()]
    if len(table) > 0:
        ra, dec = table['RA'].iloc[0], table['Dec'].iloc[0]
        d25 = 2. * table['radius_r'].iloc[0]
        ba = table['ba_ratio'].iloc[0]
        if np.isnan(d25):
            d25 = 2. * table['radius_g'].iloc[0]

        gmag, gmag_err = table.gmag.iloc[0], table.e_gmag.iloc[0]
        rmag, rmag_err = table.rmag.iloc[0], table.e_rmag.iloc[0]
        logms, e_logms = table.logMstar.iloc[0], table.e_logMstar.iloc[0]
    else:
        nedtab = Ned.query_object(s)
        ra, dec = nedtab['RA'][0], nedtab['DEC'][0]
        rmag, rmag_err, gmag, gmag_err = np.nan, np.nan, np.nan, np.nan
        d25, ba = np.nan, np.nan
        logms, e_logms = np.nan, np.nan
        
    with open('mtype.yml') as f:
        mtype_dict = yaml.safe_load(f)
    with open('morphologies.yml') as f:
        morph_dict = yaml.safe_load(f)
    g_morphs = morph_dict[g]
    
    try:
        vel_tab = Ned.get_table(s, table='redshifts')
        vsys = np.ma.median(vel_tab['Published Velocity'])
    except:
        vsys = np.nan
    try:
        mtype = g_morphs[s]
        if mtype.endswith('*'): mtype = mtype[:-1]
    except KeyError:
        mtype = np.nan
        print(f'Warning: {s} has no morphological type')
    try:
        ttype = [k for k in mtype_dict.keys() if mtype in mtype_dict[k]][0]
    except IndexError:
        ttype = np.nan
    return ra, dec, vsys, mtype, ttype, d25, ba, gmag, gmag_err, rmag, rmag_err, logms, e_logms

def skycoo(ra, dec):
    '''
    Transfrom ra, dec in SKyCoord element
    Parameters
    ----------
    ra: RA
    dec: Dec
    '''
    try:
        ra, dec = float(ra), float(dec)
        coo = coords.SkyCoord(ra, dec, unit='degree', frame='icrs')
    except ValueError:
        coo = coords.SkyCoord(ra, dec, unit='hourangle,degree', frame='icrs')
    return coo

def flux_limit(data, wcs, ra, dec, size, vdelt, beam_area, nsigma=3., dv=20.):
    '''
    Estimate the limit flux of a non-HI detected galaxy.
    Returns the total flux in Jy.km/s
    Params:
    ----------
    data: HDU data
    wcs: WCS element
    ra: RA in degrees
    dec: Dec in degrees
    size: quantity element of the angular size
    vdelt: velocity width of the channels
    beam_area: beam area in degrees (units to be double-checked)
    nsigma: level above the noise in units of sigma
    dv: total velocity width to be considered for a dectection
    '''
    box = Cutout2D(data, position=coords.SkyCoord(ra,dec,unit='deg',frame='icrs'), size=coords.Angle(size,unit='deg'), wcs=wcs)
    sflux = nsigma * np.nansum(box.data) * np.sqrt(dv/vdelt)
    sdv = sflux * dv / beam_area ### conversion from Jy/beam to Jy.km/s
    return sdv

def get_cubelet(g, ra, dec, params, radius=60, peak=True, multi=False):
    '''
    Find the HI cubelet corresponding to a detected galaxy.
    Returns the HDU data and header.
    Parameters
    ----------
    g: HCG name
    ra: RA in degree or HMS
    dec: Dec in degree or DMS
    radius: radius of search cone in arcsec
    '''
    catfile = params[g]['rootdir'] + '/' + params[g]['catalog']
    cat = pd.read_csv(catfile, sep='\s+', engine='python', header=18, skiprows=[19])
    cat.drop(columns='#', inplace=True)
    try:
        ra, dec = float(ra), float(dec)
        scoo = coords.SkyCoord(ra, dec, unit='degree,degree', frame='icrs')
    except ValueError:
        scoo = coords.SkyCoord(ra, dec, unit='hourangle,degree', frame='icrs')
    if peak:
        cat_ra, cat_dec = cat.ra_peak, cat.dec_peak
    else:
        cat_ra, cat_dec = cat.ra, cat.dec
    sofia_coo = coords.SkyCoord(cat_ra, cat_dec, unit='degree,degree', frame='icrs')
    root = params[g]['rootdir']
    dirname = params[g]['dir_cubelets']
    df_matches = pd.DataFrame(columns=['separation','file_name'])
    for sid,coo in zip(cat.id,sofia_coo):
        sep = coo.separation(scoo)
        if sep.arcsec < radius:
            fits_name = root +'/' + dirname +'/' + dirname[:-8]+str(sid)+'_mom0.fits'
            df_matches.loc[sid] = [sep.arcsec, fits_name]
    if multi:
        fits_files = df_matches.file_name.values
        subdata, subhdr = [], []
        for f in fits_files:
            with fits.open(f) as hdu:
                subdata.append(hdu[0].data)
                subhdr.append(hdu[0].header)
    else:
        actual_fits = df_matches[df_matches.separation == df_matches.separation.min()].file_name.values
        if len(actual_fits) > 0:
            subdata, subhdr = fits.getdata(actual_fits[0], header=True)
        else:
            raise(ValueError)
    return subdata, subhdr

def get_global_profile(g, ra, dec, params, radius=60, peak=True, multi=False):
    '''
    Find the HI profile corresponding to a detected galaxy.
    Returns the HDU data and header.
    Parameters
    ----------
    g: HCG name
    ra: RA in degree or HMS
    dec: Dec in degree or DMS
    radius: radius of search cone in arcsec
    '''
    catfile = params[g]['rootdir'] + '/' + params[g]['catalog']
    cat = pd.read_csv(catfile, sep='\s+', engine='python', header=18, skiprows=[19])
    cat.drop(columns='#', inplace=True)
    try:
        ra, dec = float(ra), float(dec)
        scoo = coords.SkyCoord(ra, dec, unit='degree,degree', frame='icrs')
    except ValueError:
        scoo = coords.SkyCoord(ra, dec, unit='hourangle,degree', frame='icrs')
    if peak:
        cat_ra, cat_dec = cat.ra_peak, cat.dec_peak
    else:
        cat_ra, cat_dec = cat.ra, cat.dec
    sofia_coo = coords.SkyCoord(cat_ra, cat_dec, unit='degree,degree', frame='icrs')
    root = params[g]['rootdir']
    dirname = params[g]['dir_cubelets']
    df_matches = pd.DataFrame(columns=['separation','file_name'])
    for sid,coo in zip(cat.id,sofia_coo):
        sep = coo.separation(scoo)
        if sep.arcsec < radius:
            spec_name = root +'/' + dirname +'/' + dirname[:-8]+str(sid)+'_spec.txt'
            df_matches.loc[sid] = [sep.arcsec, spec_name]
    if multi:
        true_spec = df_matches.file_name.values
    else:
        specs = df_matches[df_matches.separation == df_matches.separation.min()].file_name.values
        if len(specs) > 0:
            true_spec = specs[0]
        else:
            raise(ValueError)
    return true_spec

def crop_hcg_fits(hcg, input_type='image'):
    '''
    Crop the fits file of a Phase 2 HCG (image or mask) to be used for HI mass calculation of individual members.
    '''
    chan_range = {16: [3486.1, 4278.1],
              31: [3849.6, 4444.7],
              91: [7508.3, 6613.0]}
    image_files = {16: '/mnt/scratch/ianja/HCG16/HCG16_OUT_LOW60/cubes/cube_1/hcg16_HCG16_hcg16_line60_masked.pb_corr.vopt.fits',
                   31: '/mnt/scratch/ianja/HCG31/HCG31_OUT_LOW15/cubes/cube_1/hcg31_largebw_HCG31_hcg31_line15_masked_vopt.image.fits',
                   91: '/mnt/scratch/ianja/HCG91/HCG91_OUT_LOW60/cubes/cube_1/hcg91_large_HCG91_hcg91_line60_masked.pb_corr.fits'
                   }
    mask_files = {16: '/mnt/scratch/ianja/HCG16/HCG16_OUT_LOW60/cubes/cube_1/sofia_outputs_mask/hcg16_HCG16_hcg16_line60_masked.image_mask.fits',
                  31: '/mnt/scratch/ianja/HCG31/HCG31_OUT_LOW15/cubes/cube_1/sofia_outputs_mask/hcg31_largebw_HCG31_hcg31_line15_masked_vopt.image.sofia_mask.fits',
                  91: '/mnt/scratch/ianja/HCG91/HCG91_OUT_LOW60/cubes/cube_1/sofia_outputs_mask/hcg91_large_HCG91_hcg91_line60_masked.image.sofia_mask.fits'
                  }
    if hcg == 31:
        suffix = '_15as'
    else:
        suffix = ''
    if input_type == 'image':
        f = image_files[hcg]
    elif input_type == 'mask':
        f = mask_files[hcg]
    freg = f'/mnt/scratch/HCGs/separate_hi_discs/box_hcg{hcg}.reg'
    chans = chan_range[hcg]
    reg = Regions.read(freg, format='crtf')
    cube = SpectralCube.read(f)
    cube_kms = cube.with_spectral_unit(u.km/u.s, velocity_convention='optical', rest_value=1420405752*u.Hz)
    if len(reg) == 1:
        ccoo = reg[0].center
        width, height = reg[0].width, reg[0].height
    else:
        raise Exception('region must be a single box')
    sub_cube_radec = cube_kms.subcube(xlo=ccoo.ra+0.5*width, xhi=ccoo.ra-0.5*width,
                                  ylo=ccoo.dec-0.5*height, yhi=ccoo.dec+0.5*height)
    sub_cube = sub_cube_radec.spectral_slab(int(min(chans))*u.km/u.s, int(max(chans))*u.km/u.s)

    sub_cube.hdu.writeto(f'/mnt/scratch/HCGs/separate_hi_discs/hcg{hcg}_{append}cropped{suffix}_untested.fits', overwrite=True)
    return sub_cube.hdu.data

def calculate_individual_masses(hcg, show_plot=False):
    '''
    Calculate MHI of Phase 2 HCGs individual member galaxies discs and plot their contours on an optical image.
    Returns a Pandas dataframe containing the HI masses of the member galaxies.
    '''

    with open('data_files.yml') as f:
        params = yaml.safe_load(f)
    dist = params[f'HCG {hcg}']['distance']
    warnings.filterwarnings(action='ignore', category=UserWarning)
    mydir = 'slicerastro_output/'
    colors = ['w', 'b', 'cyan', 'magenta', 'yellow']
    with open('/mnt/scratch/HCGs/separate_hi_discs/ids_to_galnames.yml') as fp:
        ids_to_galnames = yaml.safe_load(fp)
    id_to_galname = ids_to_galnames[f'HCG {hcg}']
    opt_d, opt_h = fits.getdata(f'legacy_images/HCG{hcg}_grz.fits', header=True)
    if hcg == 31:
        suffix = '_15as'
    else:
        suffix = ''    
    cube = SpectralCube.read(f'/mnt/scratch/HCGs/separate_hi_discs/hcg{hcg}_cropped{suffix}.fits')
    cube_kms = cube.with_spectral_unit(u.km/u.s, velocity_convention='optical', rest_value=1420405752*u.Hz)
    mask_array = fits.getdata(f'/mnt/scratch/HCGs/separate_hi_discs/slicerastro_output/hcg{hcg}_all{suffix}_mask.fits')
    raw_mask_array = fits.getdata(f'/mnt/scratch/HCGs/separate_hi_discs/hcg{hcg}_mask_cropped{suffix}.fits')
    if len(raw_mask_array.shape) > 3:
        raw_mask_array = raw_mask_array[0]
    reg = Regions.read(f'/mnt/scratch/HCGs/separate_hi_discs/box_hcg{hcg}.reg', format='crtf')
    if len(reg) == 1:
        ccoo = reg[0].center
        width, height = reg[0].width, reg[0].height

    cutout = Cutout2D(opt_d[0], position=ccoo, size=(height,width), wcs=WCS(opt_h).celestial)
    r, wcs = cutout.data, cutout.wcs
    vmin=np.nanmedian(r)-0.05*np.nanstd(r)
    vmax=np.nanmedian(r)+0.1*np.nanstd(r)
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(projection=wcs.celestial)
    ax.imshow(r, vmin=vmin, vmax=vmax, cmap='gray_r')

    all_hi_mask = raw_mask_array > 0.
    all_hi = cube_kms.with_mask(all_hi_mask)
    # igm_mask = mask_array == 0.
    # igm_cube = all_hi.with_mask(igm_mask)
    # igm_m0 = igm_cube.moment(order=0)
    # levels = [6.6e18 * 2**x for x in range(15)]
    # igm_nhi = jy_to_cm2(igm_m0.hdu.data, igm_m0.hdu.header)
    # igm_nhi[np.isnan(igm_nhi)] == 0.
    all_hi_m0 = all_hi.moment(order=0)
    levels = [6.6e18 * 2**x for x in range(15)]
    all_hi_nhi = jy_to_cm2(all_hi_m0.hdu.data, all_hi_m0.hdu.header)
    all_hi_nhi[np.isnan(all_hi_nhi)] == 0.
    ax.contour(all_hi_nhi, levels=levels, colors='k', linewidths=0.7, transform=ax.get_transform(all_hi_m0.wcs))
    mhi_tot = 0.
    df = pd.DataFrame(columns=['HCG', 'member', 'logMHI'])
    
    ### Solve dimension mismatch issue between mask and data (106 vs 107) for HCG 31
    if hcg == 31:
        nan_plane = np.full((1, mask_array.shape[1], mask_array.shape[2]), np.nan)
        mask_array = np.concatenate((mask_array, nan_plane), axis=0)

    for i, key in enumerate(id_to_galname.keys()):
        mask = mask_array == float(key)
        masked_cube = cube_kms.with_mask(mask)
        m0 = masked_cube.moment(order=0)
        galname = id_to_galname[key].lower().replace(' ','')
        m0.hdu.writeto(f'/mnt/scratch/HCGs/separate_hi_discs/hcg{hcg}_mom0_{galname}.fits', overwrite=True)
        m0_data = m0.hdu.data
        flux = totflux(m0_data, m0.hdu.header)
        mhi, e_mhi = himass(flux, dist)
        mhi_tot += 10**mhi
        nhi_data = jy_to_cm2(m0_data, m0.hdu.header)
        nhi_data[np.isnan(nhi_data)] = 0
        ax.contour(nhi_data, levels=levels, colors=colors[int(key)-1], linewidths=1, transform=ax.get_transform(m0.wcs))
        df.loc[i] = [hcg, galname.upper().replace('HCG',''), mhi]
    #     print(f'{galname.upper()}: {mhi:.1f}')
    df.loc[i+1] = [hcg, 'total', np.log10(mhi_tot)]
    # print(f'Total: {np.log10(mhi_tot): .2f}')
    ax.coords[0].set_axislabel('RA (J2000)'); ax.coords[1].set_axislabel('Dec (J2000)')
    if hcg == 31:
        ax.set_xlim((0.5*r.shape[1]-0.25*r.shape[1]), (0.5*r.shape[1]+0.25*r.shape[1]))
        ax.set_ylim((0.5*r.shape[0]-0.25*r.shape[0]), (0.5*r.shape[0]+0.25*r.shape[0]))
    plt.savefig(f'/mnt/scratch/HCGs/maps_figures/hcg{hcg}_core_members{suffix}.pdf')
    df.to_csv(f'/mnt/scratch/HCGs/individual_masses_hcg{hcg}.csv')
    if show_plot:
        plt.show()
    else:
        plt.clf()
    return df

def vrad2vopt(vrad):
    '''Convert radio velocity to optical velocity.'''
    c = 299792.458 # km/s
    freq0 = 1420405751.768 # Hz
    freq = freq0 * (1. - vrad/c)
    return c/freq * (freq0 - freq)

def HIvsys(spec):
    fitres = subprocess.run(['busyfit', '-c', '2', '3', '-noplot', f'{spec}'], capture_output=True, text=True)
    for line in fitres.stdout.split('\n'):
        if line.startswith('Pos') and line.endswith('[spec]'):
            vsys = float(line.split()[2]) * 1e-3
    # vsys = vrad2vopt(vsys)
    return vsys

def bayes_fit(x_det, x_ul, y_det, y_ul):
    '''
    Linear regression of censored data based on Bayesian method.
    Parameters
    ----------
    x_det: x-array of determined (observed) data
    x_ul : x-array of censored (upper limit) data
    y_det: y-array of determined (observed) data
    y_ul : y-array of censored (upper limit) data
    '''
    # Bayesian model using PyMC
    with pm.Model() as model:
        
        np.random.seed(42)
        
        # Priors for intercept and slope
        slope = pm.Normal('slope', mu=0, sigma=10)
        intercept = pm.Normal('intercept', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Linear model for detected data
        mu_detected = intercept + slope * (x_det - 10.)

        # Likelihood for detected data
        likelihood_detected = pm.Normal('y_det', mu=mu_detected, sigma=sigma, observed=y_det)

        # Linear model for censored data
        mu_censored = intercept + slope * (x_ul - 10.)

        # Censored data likelihood adjustment using Potential
        censored_likelihood = pm.Censored('y_cen',
                               pm.Normal.dist(mu=mu_censored, sigma=sigma), 
                               lower=None,
                               upper=None,
                               observed=y_ul)

        # Sampling from the posterior
        trace = pm.sample(2000, return_inferencedata=True, tune=1000, target_accept=0.95)

    return trace

def fit_lum_mass(split_morph=False):
    '''
    Perform fit of luminosity = f(mass) using the Bayesian fit.
    If split_morph = True, split the data by morphology before performing the fit.
    '''
    h=0.7
    df_sga = pd.read_csv('/mnt/scratch/HCGs/SGA_AMIGA_GRZMAG.csv', index_col='cig')
    dfj = pd.read_csv('/mnt/scratch/HCGs/J18_TableB2.csv', index_col='CIG')
    dfm = pd.read_csv('/mnt/scratch/HCGs/CIG_MORPHOLOGY.csv', index_col='cig')
    dfm_sub = dfm[['MORPH_TYPE', 'E_MORPH_TYPE', 'MORPH_LETTER_RC3']]
    df = pd.merge(df_sga, dfj, left_index=True, right_index=True, how='inner')
    df = pd.merge(df, dfm_sub, left_index=True, right_index=True, how='inner')
    m = df.R_MAG_SB26 > 0
    df = df[m]
    df_det = df[df.Limit == 0]
    df_ul = df[df.Limit > 0]
    logL_det = mag_to_lum(df_det.R_MAG_SB26, df_det.Dmod)
    logMHI_det = df_det.logMHI
    logLg_ul = mag_to_lum(df_ul.R_MAG_SB26, df_ul.Dmod)
    logMHI_ul = df_ul.logMHI_lim
    logMHI_det, logMHI_ul = logMHI_det + 2.*np.log10(h), logMHI_ul + 2.*np.log10(h)

    if split_morph:
        early_mask_det = df_det.MORPH_TYPE < 3
        int_mask_det   = (df_det.MORPH_TYPE >= 3) & (df_det.MORPH_TYPE <= 5)
        late_mask_det  = df_det.MORPH_TYPE > 5

        early_mask_ul = df_ul.MORPH_TYPE < 3
        int_mask_ul   = (df_ul.MORPH_TYPE >= 3) & (df_ul.MORPH_TYPE <= 5)
        late_mask_ul  = df_ul.MORPH_TYPE > 5

        det_masks = [early_mask_det, int_mask_det, late_mask_det]
        ul_masks  = [early_mask_ul, int_mask_ul, late_mask_ul]
        
        trace = []
        input_arrays = [[] for _ in range(3)]
        for i in range(3):

            logL = logL_det[det_masks[i]].values
            logL_ul = logLg_ul[ul_masks[i]].values
            logM = logMHI_det[det_masks[i]].values
            logM_ul = logMHI_ul[ul_masks[i]].values
            
            input_arrays[i] = [logL, logL_ul, logM, logM_ul]
            trace.append(bayes_fit(logL, logL_ul, logM, logM_ul))
            
    else:
        logL = logL_det.values
        logL_ul = logLg_ul.values
        logM = logMHI_det.values
        logM_ul = logMHI_ul.values
        
        input_arrays = [logL, logL_ul, logM, logM_ul]
        trace = bayes_fit(logL, logL_ul, logM, logM_ul)

    return input_arrays, trace

def plot_fit(arrays, trace, trace_total=None, figdir='/mnt/scratch/HCGs/hidef_figures/'):
    '''
    Plot the Bayesian fit.
    arrays and trace are outputs of the function fit_lum_mass.
    trace_total is the trace of the non-split fit.
    '''
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    handles, labels = plt.gca().get_legend_handles_labels()
    xleft, ytop = 0.05, 0.95
    
    xarr = np.linspace(6, 12, 50)
    
    if isinstance(trace, list):
        colors = ['orange', 'green', 'blue']
        T_text = ['T<3', 'T=3-5', 'T>5']
        for i, (x, t) in enumerate(zip(arrays, trace)):
            # if i != 1:
            #     continue
            ax.scatter(x[0], x[2], s=15, marker='o', ec=colors[i], fc='None')
            ax.scatter(x[1], x[3], s=15, marker=r'$\bf\downarrow$', c=colors[i])

            slope, intercept, sigma = az.summary(t, var_names=['slope', 'intercept', 'sigma'], round_to=2)['mean'].values
            ax.plot(xarr, slope*(xarr-10.)+intercept, ls='--', dashes=[10,5], lw=2, color=colors[i])
            if intercept > 0: sign = '+'
            else: sign = '-'
            xpos, ypos = xleft, ytop-0.04*i
            ax.text(xpos, ypos, r'$%s: y = %.2f(x-10) %s %.2f$' %(T_text[i],slope,sign,abs(intercept)), ha='left', va='top', color=colors[i], size=12, transform=ax.transAxes)
        arrow = mpl.lines.Line2D([0], [0], label='non-detection', marker=r'$\bf\downarrow$', markersize=10, markeredgecolor='k', markerfacecolor='k', linestyle='')
        dot = mpl.lines.Line2D([0], [0], label='detection', marker='o', markersize=7, markeredgecolor='k', markerfacecolor='None', linestyle='')
        handles.extend([dot, arrow])
             
    else:
        ax.scatter(arrays[0], arrays[2], s=15, marker='o', ec='gray', fc='None')
        ax.scatter(arrays[1], arrays[3], s=15, marker='x', c='gray')
        slope, intercept, sigma = az.summary(trace, var_names=['slope', 'intercept', 'sigma'], round_to=2)['mean'].values
        ax.plot(xarr, slope*(xarr-10.)+intercept, ls='-', lw=1, color='k')
        if intercept > 0: sign = '+'
        else: sign = '-'
        ax.text(xleft, ytop, r'$all: y = %.2f(x-10) %s %.2f$' %(slope,sign,abs(intercept)), ha='left', va='top', color='k', size=12, transform=ax.transAxes)

    if trace_total:
        tot_slope, tot_intercept, tot_sigma = az.summary(trace_total, var_names=['slope', 'intercept', 'sigma'], round_to=2)['mean'].values
        ax.plot(xarr, tot_slope*(xarr-10.)+tot_intercept, ls='-', lw=2, color='k')
        if tot_intercept > 0: sign = '+'
        else: sign = '-'
        ax.text(xpos, ypos-0.04, r'$all: y = %.2f(x-10) %s %.2f$' %(tot_slope,sign,abs(tot_intercept)), ha='left', va='top', color='k', size=12, transform=ax.transAxes)
    
    ax.tick_params(axis='both', direction='in', right=True, top=True, length=5, width=1)
    ax.legend(handles=handles, loc='lower right', fontsize=10)
    plt.xlim(8.1,11.9); plt.ylim(6.6,11.4)
    plt.xlabel(r'$\rm \log{(L_{\it r}/L_\odot)}$')
    plt.ylabel(r'$\rm log(M_{HI}^{obs}/M_\odot)$')
    plt.savefig(figdir+'luminosity_mass_fit.pdf')
    plt.show()

def plot_map(fig, g, gcoo, f_mom0, df, nsigma=0.5, cmap='gray_r', color=False):
    '''
    Produces a plot of the detections overlaid on an optical r-band or composite color image.
    This is a dependency of the `hcg_hi_content` function.
    Parameters
    ----------
    fig: a figure element
    g: the HCG name
    gcoo: the central position coordinates of the HCG; an astropy.coordinate element
    f_mom0: the moment 0 file name
    df: a dataframe output of `hcg_hi_content`
    nsigma: sets the display range of the optical r-band image
    cmap: matplotlib colormap for the optical image
    color: plot a color composite image (True) or r-band image (False)
    '''
    colors = {'counterparts': 'blue',
              'nocounterparts': 'red',
              'noz': 'green',
              'core': 'orange'}
    grz, opthdr = fits.getdata('/mnt/scratch/HCGs/legacy_images/%s_grz.fits' %g.replace(' ',''), header=True)
    gband, rband, zband = grz
    
    optwcs = WCS(opthdr).celestial
    hidata, hihdr   = fits.getdata(f_mom0, header=True)
    nhi = jy_to_cm2(hidata,hihdr)
    
    try:
        pixsize = abs(opthdr['CDELT1'])
    except KeyError:
        pixsize = abs(opthdr['CD1_1'])
    
    nhi_3s_levels = {'HCG 16': 3.5,
                 'HCG 31': 3.3,
                 'HCG 91': 3.7,
                 'HCG 30': 3.3,
                 'HCG 90': 3.4,
                 'HCG 97': 3.2}
    
    levs = [nhi_3s_levels[g] * 1e18 * 2**(2*x) for x in range(10)]

    hiwcs = WCS(hihdr)
    
    ax = fig.add_subplot(111, projection=optwcs)
    if color:
        # optdata = make_lupton_rgb(gband, rband, zband, Q=10, stretch=0.01, filename='%s.jpeg' %g.replace(' ','').lower())
        norm_images = [normalize_image(band, contrast=0.1, stretch_func='asinh', a=0.1) for band in grz]
        norm_images.reverse()
        optdata = np.stack(norm_images, axis=-1)
        ax.imshow(optdata, cmap=cmap, aspect='equal')
    else:
        optdata = rband #fits.getdata(opt_dir+'%s_g_r_binned.fits' %g.replace(' ',''), header=True)
        norm = viz.ImageNormalize(optdata, interval=viz.PercentileInterval(99.5), stretch=viz.LogStretch())
        ax.imshow(optdata, cmap=cmap, norm=norm, aspect='equal')
    
    ax.contour(nhi, levels=levs, linewidths=1, colors='w', transform=ax.get_transform(hiwcs))
    pb = mpl.patches.Ellipse((gcoo.ra.deg, gcoo.dec.deg), width=1, height=1, angle=0, ec='k', fc='None', ls='--', transform=ax.get_transform('world'))
    ax.add_artist(pb)

    for det_type in colors.keys():
        df_class = df[df['class'] == det_type]
        if not df_class.empty:
            for i in df_class.index:
                row = df_class.loc[i]
                try:
                    nhi = jy_to_cm2(row.data,row.header)
                    wcs = WCS(row.header)
                    ax.contour(nhi, levels=levs, linewidths=1, colors=colors[det_type], transform=ax.get_transform(wcs))
                except:
                    ax.scatter(row.ra, row.dec, marker='x', s=20, color=colors[det_type], transform=ax.get_transform('world'))
    ax.tick_params(which='both', direction='in')
    ax.coords[0].set_axislabel('RA (J2000)'); ax.coords[1].set_axislabel('Dec (J2000)')
    ax.coords[0].set_major_formatter('hh:mm:ss'); ax.coords[1].set_major_formatter('dd:mm')
    ax.coords[0].set_ticklabel(exclude_overlapping=True); ax.coords[1].set_ticklabel(exclude_overlapping=True)    

def inset_plot(g, nsigma=0.2, contrast=0.05, stretch_func='log', norm_a=1, mapdir='/mnt/scratch/HCGs/maps_figures/'):
    '''
    Produces a plot of HI contours on an optical r-band image and shows detections in inset boxes overlaid on composite color images.
    Dependency files: group_members.yml; data_files.yml; detections_positions.yml
    Parameters:
    ----------
    g: the HCG name
    nsigma: sets the display range of the optical r-band image    
    '''
    mpl.rcParams['hatch.linewidth'] = 0.5
    
    with open('group_members.yml') as f:
        group_members = yaml.safe_load(f)
    with open('data_files.yml') as f:
        params = yaml.safe_load(f)
    with open('detections_positions.yml') as f:
        positions = yaml.safe_load(f)
    
    grz, opthdr = fits.getdata('/mnt/scratch/HCGs/legacy_images/%s_grz.fits' %g.replace(' ',''), header=True)
    
    nhi_3s_levels = {'HCG 16': 3.5,
                 'HCG 31': 3.3,
                 'HCG 91': 3.7,
                 'HCG 30': 3.3,
                 'HCG 90': 3.4,
                 'HCG 97': 3.2}
    
    levs = [nhi_3s_levels[g] * 1e18 * 2**(2*x) for x in range(10)]
    
    optwcs = WCS(opthdr).celestial
    
    fits_m0 = params[g]['rootdir'] + '/' + params[g]['moment_0']
    hidata, hihdr = fits.getdata(fits_m0, header=True)
    hiwcs = WCS(hihdr)
    nhi_ent = jy_to_cm2(hidata,hihdr)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection=optwcs)
    ax.imshow(grz[1], origin='lower', cmap='gray_r', norm=viz.ImageNormalize(grz[1], interval=viz.PercentileInterval(99.5), stretch=viz.LogStretch()))#vmin=vmin, vmax=vmax)
    ax.contour(nhi_ent, levels=levs, linewidths=0.5, colors='b', transform=ax.get_transform(hiwcs))
    gned = Ned.query_object(g)
    gra, gdec = gned['RA'][0], gned['DEC'][0]
    pb = mpl.patches.Ellipse((gra, gdec), width=1, height=1, angle=0, ec='k', fc='None', ls='--', transform=ax.get_transform('world'))
    ax.add_artist(pb)

    subdir = 'color_plots/%s/' %(g.replace(' ',''))

    detections = group_members[g]['counterparts'] + group_members[g]['noz']
    inax_size = 0.2
    
    try:
        pixsize = abs(opthdr['CDELT1'])
    except KeyError:
        pixsize = abs(opthdr['CD1_1'])
    scale = u.Quantity(10.,unit='kpc')
    dist = u.Quantity(params[g]['distance'], unit='Mpc')
    scale_angle = coords.Angle(float(scale.to('kpc') / dist.to('kpc')), unit='radian')
    scale_pix = scale_angle.degree / pixsize

    bmaj, bmin, bpa = hihdr['BMAJ'], hihdr['BMIN'], hihdr['BPA']
    bmaj, bmin = coords.Angle(bmaj, unit='deg'), coords.Angle(bmin, unit='deg')
    if np.isnan(bpa): bpa = 0.0
    
    fontprops = FontProperties(size=8, family='monospace')

    boxpos = positions[g]
    members = group_members[g]
    for i,source in enumerate(detections):
        ned = Ned.query_object(source)
        ra, dec = ned['RA'][0], ned['DEC'][0]
        try:
            subdata, subhdr = get_cubelet(g, ra, dec, params, radius=45)
        except ValueError:
            print('No HI match: ',source)
        else:
            sub_hiwcs = WCS(subhdr)
            sub_coo = sub_hiwcs.pixel_to_world(0.5*subhdr['NAXIS1'], 0.5*subhdr['NAXIS2'])
            xs, ys = subhdr['NAXIS1'] * abs(subhdr['CDELT1']), subhdr['NAXIS2'] * abs(subhdr['CDELT2'])
            size = [coords.Angle(ys, unit='deg'), coords.Angle(xs, unit='deg')]
            cutout_grz = [Cutout2D(band, position=sub_coo, size=size, wcs=optwcs) for band in grz]
            cutout_grz_data = [normalize_image(cutout.data, contrast=contrast, stretch_func=stretch_func, a=norm_a) for cutout in cutout_grz]
            cutout_grz_data.reverse()
            cutout_rgb = np.stack(cutout_grz_data, axis=-1)
            try:
                flat_pos = [x for subpos in boxpos[source] for x in subpos]
            except TypeError:
                flat_pos = [x for x in boxpos[source]]
            if len(flat_pos) == 4:
                inax_pos = boxpos[source][0]
                xy_pos = boxpos[source][1]
            else:
                inax_pos = boxpos[source]
            inax = ax.inset_axes([inax_pos[0],inax_pos[1],inax_size,inax_size], projection=cutout_grz[0].wcs)
            inax.imshow(cutout_rgb, cmap='plasma')
            if 'multi_id' in members.keys() and source in members['multi_id']:
                subnhi = Cutout2D(nhi_ent, position=sub_coo, size=size, wcs=hiwcs)
                subnhi_data, subnhi_wcs = subnhi.data, subnhi.wcs
                inax.contour(subnhi_data, levels=[5e18]+levs, linewidths=0.5, colors='w', transform=inax.get_transform(subnhi_wcs))
            else:
                nhi = jy_to_cm2(subdata, subhdr)
                inax.contour(nhi, levels=levs, linewidths=0.5, colors='w', transform=inax.get_transform(WCS(subhdr)))
            scalebar = AnchoredSizeBar(transform=inax.transData, size=scale_pix, label='10 kpc', loc='lower left', pad=0.1, color='white', frameon=False, fontproperties=fontprops)
            inax.add_artist(scalebar)
            
            el_x = inax.get_xlim()[1] - 0.5*bmaj.deg/pixsize - 0.05 * (inax.get_xlim()[1] - inax.get_xlim()[0])
            el_y = inax.get_ylim()[0] + 0.5*bmaj.deg/pixsize + 0.05 * (inax.get_ylim()[1] - inax.get_ylim()[0])
            beam = mpl.patches.Ellipse((el_x, el_y), width=bmin.deg/pixsize, height=bmaj.deg/pixsize, angle=bpa, lw=0.5, ec='w', fc='None', hatch='/////', transform=inax.transData)  # in data coordinates!
            inax.add_artist(beam)
            
            inax.text(0.5,0.9,source, ha='center', va='center', fontsize=6, color='w', transform=inax.transAxes)
            inax.coords[0].set_ticks_visible(False); inax.coords[1].set_ticks_visible(False)
            inax.coords[0].set_ticklabel_visible(False); inax.coords[1].set_ticklabel_visible(False)            
            inax.coords[0].set_axislabel(''); inax.coords[1].set_axislabel('')
            if len(flat_pos) == 4:
                arrow = mpl.patches.ConnectionPatch(xyA=(xy_pos[0],xy_pos[1]), coordsA=inax.transAxes, xyB=(sub_coo.ra.deg,sub_coo.dec.deg), shrinkB=7, coordsB=ax.get_transform('world'), arrowstyle='->')
                ax.add_artist(arrow)
    ax.tick_params(which='both', direction='in')
    ax.coords[0].set_axislabel('RA (J2000)'); ax.coords[1].set_axislabel('Dec (J2000)')
    ax.coords[0].set_major_formatter('hh:mm:ss'); ax.coords[1].set_major_formatter('dd:mm')
    ax.coords[0].set_ticklabel(exclude_overlapping=True); ax.coords[1].set_ticklabel(exclude_overlapping=True)
    plt.tight_layout()
    plt.savefig(mapdir+'%s_annotated.pdf' %g.replace(' ',''))

def plot_defhi(glist, tabdir='./params_tables/', par='hidef', figdir='./hidef_figures/',savefig=True):
    '''
    Plots the deficiency vs. separation or mhi_obs vs mhi_pred.
    Parameters:
    ----------
    glist: list of HCGs to plot
    tabdir: directory where parameter files are stored in csv format; output of `hcg_hi_content`
    par: parameters to plot; hidef = deficiency vs. spearation, mass = mhi_obs vs mhi_pred.
    '''
    with open('data_files.yml') as f:
        params = yaml.safe_load(f)
    pb_radian = coords.Angle(30., unit='arcmin').radian
    
    mcolor = {'HCG 16': '#109310',
               'HCG 30': '#DC143C',
               'HCG 31': '#0000FF',
               'HCG 90': '#FF66FF',
               'HCG 91': '#3399FF',
               'HCG 97': '#FF8C00'
              }
    msize = 50
    if par == 'hidef':
        xaxis, yaxis = 'sep_norm', 'defHI'
        e_xaxis, e_yaxis = 'e_sepnorm', 'e_defHI'
        xlabel, ylabel, arrow = r'$\Delta r_{\rm proj} / r_{\rm vir}$', r'$\rm def_{\textsc{Hi}}\,\, (dex)$', r'$\bf\downarrow$'
        xpos, ypos = 0.85, 0.3
        fname = 'hidef_vs_sep.pdf'
    elif par == 'mass':
        xaxis, yaxis = 'logMHIexp', 'logMHIobs'
        e_xaxis, e_yaxis = 'e_logMHIexp', 'e_logMHIobs'
        xlabel, ylabel, arrow = r'$\rm log(M_{HI}^{pred}/M_\odot)$', r'$\rm log(M_{HI}^{obs}/M_\odot)$', r'$\bf\downarrow$'
        xpos, ypos = 0.12, 0.5
        fname = 'himass_pred_vs_obs.pdf'
    elif par == 'vsys':
        xaxis, yaxis = 'sep_norm', 'Vsys'
        xlabel, ylabel, arrow = r'$\Delta r_{\rm proj} / r_{\rm vir}$', r'$(V-V_{\rm HCG})/\sigma$', 'D'
        xpos, ypos = 0.85, 0.05
        fname = 'vsys_vs_sep.pdf'
    pb_norm = np.zeros(len(glist))
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    mean_ex, mean_ey = np.zeros(len(glist)), np.zeros(len(glist))
    for i,g in enumerate(glist):
        vsys = params[g]['velocity']
        dist_Mpc = params[g]['distance']
        df_file = tabdir+'HIdef_%s.csv' %g.replace(' ','').lower()
        pb_kpc = pb_radian * dist_Mpc * 1e3
        Mvir, e_Mvir = group_virial_mass(g)
        Rvir, e_Rvir = virial_radius_from_mass(Mvir, e_Mvir)
        pb_norm[i] = pb_kpc / Rvir
        vdisp = velocity_dispersion(g)
        # cenra, cendec = center_of_mass(g)
        # cen_coo = coords.SkyCoord(cenra, cendec, unit=('deg','deg'), frame='icrs')
        if os.path.isfile(df_file):
            df = pd.read_csv(df_file, na_values='--')
        else:
            df = hcg_hi_content(g)
        df.dropna(subset=['Name'], inplace=True)
        mflag = df.MHIlim_flag
        df_mtrue, df_mlim = df[mflag == 0], df[mflag == 1]
        mask_core_mtrue = df_mtrue.Name.str.contains('core:')
        mask_core_mlim = df_mlim.Name.str.contains('core:')
        df_core_mtrue, df_nocore_mtrue = df_mtrue[mask_core_mtrue], df_mtrue[~mask_core_mtrue]
        df_core_mlim, df_nocore_mlim = df_mlim[mask_core_mlim], df_mlim[~mask_core_mlim]
        # df_cores = df[df.Name.str.contains('core:')]
        # core_coos = coords.SkyCoord(df_cores.RA, df_cores.Dec, unit=('deg','deg'), frame='icrs')
        # max_sep = np.max(cen_coo.separation(core_coos))
        # max_sep = max_sep.radian * params[g]['distance'] * 1e3
        df_tot_core = df[df.Name.str.replace(' ','') == 'core:total']
        if par == 'vsys':
            ax.scatter(df_nocore_mtrue[xaxis], (df_nocore_mtrue[yaxis]-vsys)/vdisp, marker='x', s=msize, c=mcolor[g])
            ax.scatter(df_core_mtrue[xaxis], (df_core_mtrue[yaxis]-vsys)/vdisp, marker='o', s=msize, c=mcolor[g])
            ax.scatter(df_mlim[xaxis], (df_mlim[yaxis]-vsys)/vdisp, marker=arrow, s=msize, fc='None', ec=mcolor[g])
            ax.scatter(df_core_mlim[xaxis], (df_core_mlim[yaxis]-vsys)/vdisp, marker='o', s=3.5*msize, fc='None', ec=mcolor[g])
            ax.scatter(df_tot_core[xaxis], (df_tot_core[yaxis]-vsys)/vdisp, marker='s', s=msize, c=mcolor[g])
            ax.scatter(df_tot_core[xaxis], (df_tot_core[yaxis]-vsys)/vdisp, marker='s', s=4*msize, fc='None', ec=mcolor[g])
            ax.axhline(y=0, ls='--', color='gray', lw=1, dashes=[5,10])
        else:
            ax.scatter(df_nocore_mtrue[xaxis], df_nocore_mtrue[yaxis], marker='x', s=msize, c=mcolor[g])
            ax.scatter(df_core_mtrue[xaxis], df_core_mtrue[yaxis], marker='o', s=msize, c=mcolor[g])
            ax.scatter(df_mlim[xaxis], df_mlim[yaxis], marker=arrow, s=msize, c=mcolor[g])
            ax.scatter(df_core_mlim[xaxis], df_core_mlim[yaxis], marker='o', s=3.5*msize, fc='None', ec=mcolor[g])
            ax.scatter(df_tot_core[xaxis], df_tot_core[yaxis], marker='s', s=msize, c=mcolor[g])
            ax.scatter(df_tot_core[xaxis], df_tot_core[yaxis], marker='s', s=4*msize, fc='None', ec=mcolor[g])
        
        try:
            mean_ex[i] = np.nanmean(df_mtrue[~df_mtrue.Name.str.contains('total')][e_xaxis])
        except:
            pass
        try:
            mean_ey[i] = np.nanmean(df_mtrue[~df_mtrue.Name.str.contains('total')][e_yaxis])
        except:
            pass
        if par == 'hidef':
            sigma = 0.42
            ax.axhspan(ymin=-sigma, ymax=sigma, fc='gray', alpha=0.07, zorder=0)
            ax.axhline(y=0.0, ls='--', color='k', lw=1, dashes=[5,10], alpha=0.5)
            
        ax.set_xlabel(xlabel, size=15)
        ax.set_ylabel(ylabel, size=15);
        ax.tick_params(axis='both', direction='in', right=True, top=True, length=5, width=1)
        if i > 2: ix, iy = 1, i - 3
        else: ix, iy = 0, i
        ax.text(xpos+0.12*ix, ypos+(2-iy)*0.05, r'$\bf %s\,%s$' %(g.split()[0],g.split()[1]), color=mcolor[g], size=12, ha='right', va='bottom', transform=ax.transAxes)
        # ax.legend(loc='best', fontsize=12, facecolor='None')
    if par in ['hidef', 'vsys']:
        ax.set_ylim(ymax=max(ax.get_ylim()[1],abs(ax.get_ylim()[0])))
        # ax.invert_yaxis()
        for i,g in enumerate(glist):
            length = 0.07 * abs(ax.get_ylim()[1]-ax.get_ylim()[0])
            width = 1e-5 * (ax.get_xlim()[1]-ax.get_xlim()[0])
            head_width = 0.01 * (ax.get_xlim()[1]-ax.get_xlim()[0])
            head_length = 0.4 * length
            # overhang = 1
            ax.arrow(x=pb_norm[i], y=ax.get_ylim()[1]-length, dx=0, dy=length, color=mcolor[g], width=width, head_width=head_width, head_length=head_length, overhang=1, length_includes_head=True)
            # ax.set_ylim()
    elif par == 'mass':
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.plot([0,20],[0,20], ls='--', dashes=[5,2], color='gray', lw=1)
        x = np.arange(0,20)
        sigma = 0.42 #np.mean([0.27,0.16,0.17])
        ax.fill_between(x, y1=x-sigma, y2=x+sigma, color='gray', alpha=0.3, zorder=0)
        ax.set_xlim(6.8,10.6); ax.set_ylim(6.8,10.6)
    if par in 'mass':
        mean_ex = 0.5 * mean_ex.mean()
        mean_ey = mean_ey.mean()
        ax.errorbar(8.0, 9.5, xerr=mean_ex, yerr=mean_ey, fmt='none', ecolor='k', elinewidth=1, capsize=4)
    elif par == 'hidef':
        mean_ex = mean_ex.mean()
        mean_ey = 0.5 * mean_ey.mean()
        ax.errorbar(4.5, 1.5, xerr=mean_ex, yerr=mean_ey, fmt='none', ecolor='k', elinewidth=1, capsize=4)
    sq = mpl.lines.Line2D([0], [0], label='total core', marker='s', markersize=10, markeredgecolor='k', markerfacecolor='gray', linestyle='')
    dot = mpl.lines.Line2D([0], [0], label='core member', marker='o', markersize=10, markeredgecolor='k', markerfacecolor='gray', linestyle='')
    cross = mpl.lines.Line2D([0], [0], label='outskirts', marker='x', markersize=10, markeredgecolor='k', markerfacecolor='k', linestyle='')
    arrw = mpl.lines.Line2D([0], [0], label='non-detection', marker=arrow, markersize=10, markeredgecolor='k', markerfacecolor='k', linestyle='')
    handles.extend([sq, dot, cross, arrw])
    ax.legend(handles=handles, loc='best', fontsize=10)
    if par == 'vsys':
        ylim = (-19, 19)
    if par == 'hidef':
        ax.invert_yaxis()
    if savefig:
        plt.savefig(figdir+fname)
    plt.tight_layout()
    plt.show()

def plot_hi_star(glist, tabdir='./params_tables/', figdir='./hidef_figures/',savefig=True):
    '''
    Plots the HI vs. stellar mass
    Parameters:
    ----------
    glist: list of HCGs to plot
    tabdir: directory where parameter files are stored in csv format; output of `hcg_hi_content`
    '''
    # with open('data_files.yml') as f:
    #     params = yaml.safe_load(f)
    # pb_amin = 30.
    mcolor = {'HCG 16': '#109310',
               'HCG 30': '#DC143C',
               'HCG 31': '#0000FF',
               'HCG 90': '#FF66FF',
               'HCG 91': '#3399FF',
               'HCG 97': '#FF8C00'
              }
    msize = 50
    arrow = r'$\bf\downarrow$'
    
    maddox = Table.read('maddox2015.txt', format='ascii')
    logms_maddox = maddox['logMs']
    logmhi_maddox = maddox['logMHI']
    hicat = pd.read_csv('HICAT_table8.csv')
    df_j18 = pd.read_csv('/mnt/scratch/HCGs/Jones18_Table2_HIderived.csv', index_col='CIG')
    df_b20 = pd.read_csv('/mnt/scratch/HCGs/Bok2020_table.csv', index_col='cig')
    df_cig = pd.merge(df_j18, df_b20, left_index=True, right_index=True, how='inner')
    
    mx = np.linspace(6,13,10)
    fig, ax = plt.subplots(figsize=(8,6))
    # ax.plot(logms_maddox, logmhi_maddox, 'k-', lw=1)
    # ax.errorbar(logms_maddox, logmhi_maddox, yerr=maddox['1s'], capsize=4, c='k')
    # ax.scatter(hicat['logM*'], hicat['logMHI'], s=10, c='gray', marker='x', alpha=0.6)
    ax.scatter(df_cig['Log(smass)'], df_cig['logMHI'], s=30, c='gray', marker='+', alpha=0.6, zorder=0)
    
    handles, labels = plt.gca().get_legend_handles_labels()

    mean_ex, mean_ey = np.zeros(len(glist)), np.zeros(len(glist))
    
    for i,g in enumerate(glist):
        df_file = tabdir+'HIdef_%s.csv' %g.replace(' ','').lower()
        if os.path.isfile(df_file):
            df = pd.read_csv(df_file, na_values='--')
        else:
            print('Running `hcg_hi_content...`')
            df = hcg_hi_content(g)
        df.dropna(subset=['Name'], inplace=True)
        mflag = df.MHIlim_flag
        df_mtrue, df_mlim = df[mflag == 0], df[mflag == 1]
        mask_core_mtrue = df_mtrue.Name.str.contains('core:')
        mask_core_mlim = df_mlim.Name.str.contains('core:')
        df_core_mtrue, df_nocore_mtrue = df_mtrue[mask_core_mtrue], df_mtrue[~mask_core_mtrue]
        df_core_mlim, df_nocore_mlim = df_mlim[mask_core_mlim], df_mlim[~mask_core_mlim]
        df_tot_core = df[df.Name.str.replace(' ','') == 'core:total']
        
        ax.scatter(df_nocore_mtrue['logMs'], df_nocore_mtrue['logMHIobs'], marker='x', s=msize, c=mcolor[g])
        ax.scatter(df_core_mtrue['logMs'], df_core_mtrue['logMHIobs'], marker='o', s=msize, c=mcolor[g])
        ax.scatter(df_mlim['logMs'], df_mlim['logMHIobs'], marker=arrow, s=msize, c=mcolor[g])
        ax.scatter(df_core_mlim['logMs'], df_core_mlim['logMHIobs'], marker='o', s=3.5*msize, fc='None', ec=mcolor[g])
        ax.scatter(df_tot_core['logMs'], df_tot_core['logMHIobs'], marker='s', s=msize, c=mcolor[g])
        ax.scatter(df_tot_core['logMs'], df_tot_core['logMHIobs'], marker='s', s=4*msize, fc='None', ec=mcolor[g])
        if i > 2: ix, iy = 1, i - 3
        else: ix, iy = 0, i
        ax.text(0.12+0.12*ix, 0.60+(2-iy)*0.05, r'$\bf %s\,%s$' %(g.split()[0],g.split()[1]), color=mcolor[g], size=12, ha='right', va='bottom', transform=ax.transAxes)
        try:
            mean_ex[i] = np.nanmean(df_mtrue[~df_mtrue.Name.str.contains('total')]['e_logMs'])
        except:
            pass
        try:
            mean_ey[i] = np.nanmean(df_mtrue[~df_mtrue.Name.str.contains('total')]['e_logMHIobs'])
        except:
            pass
    sq = mpl.lines.Line2D([0], [0], label='total core', marker='s', markersize=10, markeredgecolor='k', markerfacecolor='gray', linestyle='')
    dot = mpl.lines.Line2D([0], [0], label='core member', marker='o', markersize=10, markeredgecolor='k', markerfacecolor='gray', linestyle='')
    cross = mpl.lines.Line2D([0], [0], label='outskirts', marker='x', markersize=10, markeredgecolor='k', markerfacecolor='k', linestyle='')
    arrw = mpl.lines.Line2D([0], [0], label='non-detection', marker=arrow, markersize=10, markeredgecolor='k', markerfacecolor='k', linestyle='')
    plus = mpl.lines.Line2D([0], [0], label='AMIGA', marker='+', markersize=7, markeredgecolor='gray', markerfacecolor='gray', linestyle='')
    handles.extend([sq, dot, cross, arrw, plus])
    ax.axvline(x=9.2, lw=0.75, ls='--', dashes=[10,5], color='gray')
    ax.tick_params(axis='both', direction='in', right=True, top=True, length=5, width=1)
    ax.legend(handles=handles, loc='best', fontsize=10)
    ax.set_xlabel(r'$\rm \log(M_{star}/M_\odot)$')
    ax.set_ylabel(r'$\rm \log(M_\textsc{Hi}/M_\odot)$')
    ax.errorbar(8.7, 10.2, xerr=1.5*mean_ex.mean(), yerr=mean_ey.mean(), fmt='none', ecolor='k', elinewidth=1.5, capsize=1)
    if savefig:
        plt.savefig(figdir+'mhi_mstar_amiga.pdf')
    plt.tight_layout()
    plt.show()
    
def hcg_hi_content(g, dv=20., plot=False, color=False, mapdir='./maps_figures/', mem_file='group_members.yml', tabdir='./params_tables/'):
    '''
    Compute HI and optical parameters of detections in an HCG group
    Parameters:
    ----------
    g: the HCG name
    dv: total velocity width to be considered to estimate the maximum flux of a non-dectection
    plot: produce plot (using `plot_maps`)
    color: whether plot should be composite color or r-band
    mapdir: output directory of plots
    tabdir: output directory of parameter files in csv format
    '''
    with open('data_files.yml') as f:
        params = yaml.safe_load(f)
    with open(mem_file) as f:
        group_members = yaml.safe_load(f)

    noise_cube = params[g]['rootdir'] + params[g]['noise_cube']
    noise_cube = noise_cube.replace('results','outputs')
    group_dist = params[g]['distance']
    group_vsys = params[g]['velocity']
            
    Mvir, e_Mvir = group_virial_mass(g)
    group_diam, e_rvir = virial_radius_from_mass(Mvir, e_Mvir)
    
    cra, cdec = center_of_mass(g)
    ccoo = coords.SkyCoord(cra, cdec, unit='deg', frame='icrs')

    vlow, vhigh = group_vsys-1000., group_vsys+1000.
    
    noise_data, noise_hdr = fits.getdata(noise_cube, header=True)
    pb_path = '/'.join(params[g]['rootdir'].split('/')[:-4]) + '/data/%s/' %g.lower().replace(' ','')
    pb_fits = pb_path + '%s_line60_masked.pb.fits' %g.lower().replace(' ','')
    pb_data = fits.getdata(pb_fits)
    noise_median = np.nanmedian(noise_data, axis=0)
    pb_median = np.nanmedian(np.squeeze(pb_data), axis=0)
    noise_median /= pb_median
    cell_size = abs(noise_hdr['CDELT1'])
    cell_area = cell_size * cell_size
    bmaj, bmin = noise_hdr['BMAJ'], noise_hdr['BMIN']
    beam_area = bmaj * bmin * np.pi / (4.0 * np.log(2.0) * cell_area)
    wcs = WCS(noise_hdr).celestial
    vdelt = np.abs(noise_hdr['CDELT3'])
    if vdelt > 1e3: vdelt *= 1e-3
            
    cols = ['HCG', 'Name', 'RA', 'Dec', 'Vsys', 'M_type', 'T_type', 'sep_kpc', 'sep_norm', 'e_sepnorm',
            'D25_am', 'D25_kpc', 'flux', 'logMHIobs', 'e_logMHIobs', 'logMHIexp', 'e_logMHIexp', 'defHI', 'e_defHI', 'OPT_HI', 'zflag',
            'HIflag', 'MHIlim_flag', 'gmag', 'e_gmag', 'rmag', 'e_rmag', 'Btot', 'logLB', 'logMs', 'e_logMs']
    plotdf_cols = ['HCG', 'Name', 'ra', 'dec', 'data', 'header', 'class']
    ### OPT_HI: detected in optical or HI; 0 = optical only, 1 = HI only, 2 = both
    ### zflag: velocity flag; 0 = no optical velocity, 1 = optical velocity found
    ### HIflag: detected in HI? 0 = no, 1 = yes
    ### MHIlim_flag: HI mass limit? 0 = no, 1 = yes
    
    members = group_members[g]
    cores = members['core']
    vcen = params[g]['velocity']
    core_members = [item.strip() for sublist in cores for item in sublist.split(',')]
    core_members = [g+x if len(x) == 1 else x for x in core_members]
    df_params = pd.DataFrame()
    df_cubelets = pd.DataFrame()
    if 'counterparts' in members.keys():
        sources = members['counterparts']
        sources = [x for x in sources if not x in core_members]
        df_ctp = pd.DataFrame(columns=cols, index=sources)
        df_ctp_cbl = pd.DataFrame(columns=plotdf_cols, index=sources, dtype='object')
        for i, source in enumerate(sources):
            ra, dec, vsys, mtype, ttype, d25, ba, gmag, gmag_err, rmag, rmag_err, logms, e_logms = get_params(g, source)
            try:
                spec = get_global_profile(g, ra, dec, params)
                vsys_hi = HIvsys(spec)
            except ValueError:
                vsys_hi = np.nan
            if not (vsys_hi-1000. <= vcen <= vsys_hi+1000.):
                continue
            if 'multi_id' in members.keys() and source in members['multi_id']:
                listdata, listhdr = get_cubelet(g, ra, dec, params, multi=True)
                flux = 0.0
                for subdata, subhdr in zip(listdata, listhdr):
                    flux += totflux(subdata, subhdr)
                mhi_obs, e_mhi_obs = himass(flux, group_dist)
                gmag, rmag = mag_correction(gmag, ra, dec, ba, ttype, vsys, 'g'), mag_correction(rmag, ra, dec, ba, ttype, vsys, 'r')
                Btot = gmag_to_Bmag(gmag, rmag)
                logLB = Btot_to_lumB(Btot, group_dist)
                # mhi_pred, _ = mj18_LB_to_mhi(logLB, ttype)
                mhi_pred, e_mhi_pred = predicted_mhi(rmag, gmag, group_dist, ttype, logms)
                def_hi, e_def_hi = hidef(mhi_obs, mhi_pred, e_mhi_obs, e_mhi_pred)
                hi_flag = 1
            else:
                try:
                    subdata, subhdr = get_cubelet(g, ra, dec, params)
                except ValueError:
                    subdata, subhdr = np.nan, np.nan
                    flux, mhi_obs = np.nan, np.nan
                    gmag, rmag = mag_correction(gmag, ra, dec, ba, ttype, vsys, 'g'), mag_correction(rmag, ra, dec, ba, ttype, vsys, 'r')
                    Btot = gmag_to_Bmag(gmag, rmag)
                    logLB = Btot_to_lumB(Btot, group_dist)
                    # mhi_pred, _ = mj18_LB_to_mhi(logLB, ttype)
                    mhi_pred, e_mhi_pred = predicted_mhi(rmag, gmag, group_dist, ttype, logms)
                    def_hi = np.nan
                    hi_flag = 0
                else:
                    hi_flag = 1
                    flux = totflux(subdata, subhdr)
                    mhi_obs, e_mhi_obs = himass(flux, group_dist)
                    gmag, rmag = mag_correction(gmag, ra, dec, ba, ttype, vsys, 'g'), mag_correction(rmag, ra, dec, ba, ttype, vsys, 'r')
                    Btot = gmag_to_Bmag(gmag, rmag)
                    logLB = Btot_to_lumB(Btot, group_dist)
                    # mhi_pred, _ = mj18_LB_to_mhi(logLB, ttype)
                    mhi_pred, e_mhi_pred = predicted_mhi(rmag, gmag, group_dist, ttype, logms)
                    def_hi, e_def_hi = hidef(mhi_obs, mhi_pred, e_mhi_obs, e_mhi_pred)
            scoo = skycoo(ra, dec)
            sep = scoo.separation(ccoo) ### separation from the group centre
            sep_kpc = sep.radian * group_dist * 1e3 ### separation in kpc
            sep_norm = sep_kpc / group_diam
            e_sep_norm = sep_norm * e_rvir / group_diam
            d25_kpc = coords.Angle(d25, unit='arcmin').radian * group_dist * 1e3
            if np.isnan(vsys):
                vsys = vsys_hi
            if vsys >= vlow and vsys <= vhigh:
                df_ctp.iloc[i] = [int(g.split()[-1]), source, scoo.ra.deg, scoo.dec.deg, vsys, mtype, ttype, sep_kpc, sep_norm, e_sep_norm, d25, d25_kpc, flux, mhi_obs, e_mhi_obs, mhi_pred, e_mhi_pred, def_hi, e_def_hi, 2, 1, hi_flag, 0, gmag, gmag_err, rmag, rmag_err, Btot, logLB, logms, e_logms]
                df_ctp_cbl.iloc[i] = [int(g.split()[-1]), source, scoo.ra.deg, scoo.dec.deg, subdata, subhdr, 'counterparts']
        df_params = pd.concat([df_params, df_ctp], ignore_index=True)
        df_cubelets = pd.concat([df_cubelets, df_ctp_cbl], ignore_index=True)
            
    if 'noz' in members.keys():
        noz_sources = members['noz']
        noz_sources = [x for x in noz_sources if not x in core_members]
        df_noz = pd.DataFrame(columns=cols, index=noz_sources)
        df_noz_cbl = pd.DataFrame(columns=plotdf_cols, index=noz_sources, dtype='object')
        for i, source in enumerate(noz_sources):
            ra, dec, vsys, mtype, ttype, d25, ba, gmag, gmag_err, rmag, rmag_err, logms, e_logms = get_params(g, source)
            try:
                subdata, subhdr = get_cubelet(g, ra, dec, params, radius=60)
            except ValueError:
                subdata, subhdr = np.nan, np.nan
                flux, mhi_obs = np.nan, np.nan
                gmag, rmag = mag_correction(gmag, ra, dec, ba, ttype, vsys, 'g'), mag_correction(rmag, ra, dec, ba, ttype, vsys, 'r')
                Btot = gmag_to_Bmag(gmag, rmag)
                logLB = Btot_to_lumB(Btot, group_dist)
                # mhi_pred, _ = mj18_LB_to_mhi(logLB, ttype)
                mhi_pred, e_mhi_pred = predicted_mhi(rmag, gmag, group_dist, ttype, logms)
                def_hi = np.nan
                hi_flag = 0
            else:
                flux = totflux(subdata, subhdr)
                mhi_obs, e_mhi_obs = himass(flux, group_dist)
                gmag, rmag = mag_correction(gmag, ra, dec, ba, ttype, vsys, 'g'), mag_correction(rmag, ra, dec, ba, ttype, vsys, 'r')
                Btot = gmag_to_Bmag(gmag, rmag)
                logLB = Btot_to_lumB(Btot, group_dist)
                # mhi_pred, _ = mj18_LB_to_mhi(logLB, ttype)
                mhi_pred, e_mhi_pred = predicted_mhi(rmag, gmag, group_dist, ttype, logms)
                def_hi, e_def_hi = hidef(mhi_obs, mhi_pred, e_mhi_obs, e_mhi_pred)
            scoo = skycoo(ra, dec)
            sep = scoo.separation(ccoo) ### separation from the group centre
            sep_kpc = sep.radian * group_dist * 1e3 ### separation in kpc
            sep_norm = sep_kpc / group_diam
            e_sep_norm = sep_norm * e_rvir / group_diam
            d25_kpc = coords.Angle(d25, unit='arcmin').radian * group_dist * 1e3
            if np.isnan(vsys):
                try:
                    spec = get_global_profile(g, ra, dec, params)
                    vsys = HIvsys(spec)
                except ValueError:
                    pass
            if vsys >= vlow and vsys <= vhigh:
                df_noz.iloc[i] = [int(g.split()[-1]), source, scoo.ra.deg, scoo.dec.deg, vsys, mtype, ttype, sep_kpc, sep_norm, e_sep_norm, d25, d25_kpc, flux, mhi_obs, e_mhi_obs, mhi_pred, e_mhi_pred, def_hi, e_def_hi, 1, 0, hi_flag, 0, gmag, gmag_err, rmag, rmag_err, Btot, logLB, logms, e_logms]
                df_noz_cbl.iloc[i] = [int(g.split()[-1]), source, scoo.ra.deg, scoo.dec.deg, subdata, subhdr, 'noz']
        df_params = pd.concat([df_params, df_noz], ignore_index=True)
        df_cubelets = pd.concat([df_cubelets, df_noz_cbl], ignore_index=True)

    if 'complexes' in members.keys():
        complexes = members['complexes']
        centrals = [complexes[key]['source'] for key in complexes.keys()]
        satellites = [complexes[key]['satellites'] for key in complexes.keys()]
        extra_rows = ['all : ' + name for name in centrals]
        all_sources = [item for central, satellite, extra_row in zip(centrals, satellites, extra_rows) for item in ([central] + satellite if isinstance(satellite, list) else [central, satellite]) + [extra_row]]
        df = pd.DataFrame(columns=cols, index=all_sources)
        df_cbl = pd.DataFrame(columns=plotdf_cols, index=all_sources, dtype='object')
        i = 0
        for ic, central in enumerate(centrals):
            ra, dec, vsys, mtype, ttype, d25, ba, gmag, gmag_err, rmag, rmag_err, logms, e_logms = get_params(g, central)
            subdata, subhdr = get_cubelet(g, ra, dec, params)
            flux = totflux(subdata, subhdr)
            mhi_obs_all, e_mhi_obs_all = himass(flux, group_dist)
            gmag, rmag = mag_correction(gmag, ra, dec, ba, ttype, vsys, 'g'), mag_correction(rmag, ra, dec, ba, ttype, vsys, 'r')
            Btot = gmag_to_Bmag(gmag, rmag)
            logLB = Btot_to_lumB(Btot, group_dist)
            # mhi_pred_cen, _ = mj18_LB_to_mhi(logLB, ttype)
            mhi_pred_cen, e_mhi_pred_cen = predicted_mhi(rmag, gmag, group_dist, ttype, logms)
            src_coo = skycoo(ra, dec)
            sep = src_coo.separation(ccoo) ### separation from the group centre
            sep_kpc = sep.radian * group_dist * 1e3 ### separation in kpc
            sep_norm = sep_kpc / group_diam
            e_sep_norm = sep_norm * e_rvir / group_diam
            d25_kpc = coords.Angle(d25, unit='arcmin').radian * group_dist * 1e3
            if vsys >= vlow and vsys <= vhigh:
                df.iloc[i] = [int(g.split()[-1]), 'complex: %s' %central, ra, dec, vsys, mtype, ttype, sep_kpc, sep_norm, e_sep_norm, d25, d25_kpc, np.nan, np.nan, np.nan, mhi_pred_cen, e_mhi_pred_cen, np.nan, np.nan, 2, 1, 1, 0, gmag, gmag_err, rmag, rmag_err, Btot, logLB, logms, e_logms]
                df_cbl.iloc[i] = [int(g.split()[-1]), 'complex: %s' %central, ra, dec, np.nan, np.nan, 'complex']
            sat_source = satellites[ic]
            i += 1
            if isinstance(sat_source, list):
                mhi_pred_sat, e_mhi_pred_sat = np.zeros(len(sat_source)), np.zeros(len(sat_source))
                logms_sat = np.zeros(len(sat_source))
                for i_, sat in enumerate(sat_source):
                    ra_, dec_, vsys_, mtype_, ttype_, d25_, ba_, gmag_, rmag_, gmag_err_, rmag_err_, logms_, e_logms_ = get_params(g, sat)
                    logms_sat[i_] = logms_
                    gmag_, rmag_ = mag_correction(gmag_, ra_, dec_, ba_, ttype_, vsys_, 'g'), mag_correction(rmag_, ra_, dec_, ba_, ttype_, vsys_, 'r')
                    Btot_ = gmag_to_Bmag(gmag_, rmag_)
                    logLB_ = Btot_to_lumB(Btot_, group_dist)
                    # mhi_pred_sat[i_], _ = mj18_LB_to_mhi(logLB_, ttype_)
                    mhi_pred_sat[i_], e_mhi_pred_sat[i_] = predicted_mhi(rmag_, gmag_, group_dist, ttype_, logms_)
                    src_coo_ = skycoo(ra_, dec_)
                    sep_ = src_coo_.separation(ccoo) ### separation from the group centre
                    sep_kpc_ = sep.radian * group_dist * 1e3 ### separation in kpc
                    sep_norm_ = sep_kpc / group_diam
                    e_sep_norm_ = sep_norm_ * e_rvir / group_diam
                    d25_kpc_ = coords.Angle(d25_, unit='arcmin').radian * group_dist * 1e3
                    if vsys_ >= vlow and vsys_ <= vhigh:
                        df.iloc[i] = [int(g.split()[-1]), 'complex: %s' %sat, ra_, dec_, vsys_, mtype_, ttype_, sep_kpc_, sep_norm_, e_sep_norm_, d25_, d25_kpc_, np.nan, np.nan, np.nan, mhi_pred_sat[i_], e_mhi_pred_sat[i_], np.nan, np.nan, 2, 1, 1, 0, gmag_, gmag_err_, rmag_, rmag_err_, Btot_, logLB_, logms_, e_logms_]
                        df_cbl.iloc[i] = [int(g.split()[-1]), 'complex: %s' %sat, ra_, dec_, np.nan, np.nan, 'complex']
                        i += 1
                mhi_pred_all = np.log10(10**mhi_pred_cen + np.nansum(10**mhi_pred_sat))
                e_mhi_pred_all = np.sqrt(np.nanmean(e_mhi_pred_sat**2) + e_mhi_pred_cen**2)
                logms_all = np.log10(10**logms + np.nansum(10**logms_sat))
                def_hi, e_def_hi = hidef(mhi_obs_all, mhi_pred_all, e_mhi_obs_all, e_mhi_pred_all)
                df.iloc[i] = [int(g.split()[-1]), 'all: %s' %central, ra, dec, vsys_, np.nan, np.nan, sep_kpc, sep_norm, e_sep_norm, np.nan, np.nan, flux, mhi_obs_all, e_mhi_obs_all, mhi_pred_all, e_mhi_pred_all, def_hi, e_def_hi, 2, 1, 1, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, logms_all, np.nan]
                df_cbl.iloc[i] = [int(g.split()[-1]), 'all: %s' %central, ra, dec, subdata, subhdr, 'complex']
            else:
                ra_, dec_, vsys_, mtype_, ttype_, d25_, ba_, gmag_, rmag_, gmag_err_, rmag_err_, logms_, e_logms_ = get_params(g, sat_source)
                d25_kpc_ = coords.Angle(d25_, unit='arcmin').radian * group_dist * 1e3
                gmag_, rmag_ = mag_correction(gmag_, ra_, dec_, ba_, ttype_, vsys_, 'g'), mag_correction(rmag_, ra_, dec_, ba_, ttype_, vsys_, 'r')
                Btot_ = gmag_to_Bmag(gmag_, rmag_)
                logLB_ = Btot_to_lumB(Btot_, group_dist)
                # mhi_pred_sat, _ = mj18_LB_to_mhi(logLB_, ttype_)
                mhi_pred_sat, e_mhi_pred_sat = predicted_mhi(rmag_, gmag_, group_dist, ttype_, logms_)
                src_coo_ = skycoo(ra_, dec_)
                sep_ = src_coo_.separation(ccoo) ### separation from the group centre
                sep_kpc_ = sep_.radian * group_dist * 1e3 ### separation in kpc
                sep_norm_ = sep_kpc_ / group_diam
                e_sep_norm_ = sep_norm_ * e_rvir / group_diam
                mhi_pred_all = np.log10(10**mhi_pred_cen + 10**mhi_pred_sat)
                e_mhi_pred_all = np.sqrt(e_mhi_pred_sat**2 + e_mhi_pred_cen**2)
                logms_all = np.log10(10**logms + 10**logms_)
                def_hi, e_def_hi = hidef(mhi_obs_all, mhi_pred_all, e_mhi_obs_all, e_mhi_pred_all)
                if vsys_ >= vlow and vsys_ <= vhigh:
                    df.iloc[i] = [int(g.split()[-1]), 'complex: %s' %sat_source, ra_, dec_, vsys_, mtype_, ttype_, sep_kpc_, sep_norm_, e_sep_norm_, d25_, d25_kpc_, np.nan, np.nan, np.nan, mhi_pred_sat, e_mhi_pred_sat, np.nan, np.nan, 2, 1, 1, 0, gmag_, gmag_err_, rmag_, rmag_err_, Btot_, logLB_, logms_, e_logms_]
                    df_cbl.iloc[i] = [int(g.split()[-1]), 'complex: %s' %sat_source, ra_, dec_, np.nan, np.nan, 'complex']
                df.iloc[i+1] = [int(g.split()[-1]), 'all: %s' %central, ra, dec, vsys, np.nan, np.nan, sep_kpc, sep_norm, e_sep_norm, np.nan, np.nan, flux, mhi_obs_all, e_mhi_obs_all, mhi_pred_all, e_mhi_pred_all, def_hi, e_def_hi, 2, 1, 1, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, logms_all, np.nan]
                df_cbl.iloc[i+1] = [int(g.split()[-1]), 'all: %s' %central, ra, dec, subdata, subhdr, 'complex']
                i += 1
            i += 1
        df_params = pd.concat([df_params, df], ignore_index=True)
        df_cubelets = pd.concat([df_cubelets, df_cbl], ignore_index=True)
        
    if 'core' in members.keys():
        cores = members['core']
        name_cores = [item.strip() for sublist in cores for item in sublist.split(',')]
        mhi_pred, e_mhi_pred = np.zeros(len(name_cores)), np.zeros(len(name_cores))
        logms_mem = np.zeros(len(name_cores))
        name_tot_core = 'core: total'
        df_core = pd.DataFrame(columns=cols, index=name_cores+[name_tot_core])
        df_core_cbl = pd.DataFrame(columns=plotdf_cols, index=name_cores+[name_tot_core], dtype='object')
        
        if 'core_center' in members.keys():
            core_cen = members['core_center']
            if len(core_cen) == 1:
                ra, dec = core_cen[0].split(',')
            else:
                print('Warning: there are more than 1 confusion centre coordinates; taking the first')
            cendata, cenhdr = get_cubelet(g, float(ra), float(dec), params, radius=30, peak=False)
            flux = totflux(cendata, cenhdr)
            mhi_obs, e_mhi_obs = himass(flux, group_dist)
            for i,core in enumerate(name_cores):
                if len(core) == 1:
                    core = g+core.strip()
                ra, dec, vsys, mtype, ttype, d25, ba, gmag, gmag_err, rmag, rmag_err, logms, e_logms = get_params(g, core)
                logms_mem[i] = logms
                gmag, rmag = mag_correction(gmag, ra, dec, ba, ttype, vsys, 'g'), mag_correction(rmag, ra, dec, ba, ttype, vsys, 'r')
                Btot = gmag_to_Bmag(gmag, rmag)
                logLB = Btot_to_lumB(Btot, group_dist)
                # mhi_pred[i], _ = mj18_LB_to_mhi(logLB, ttype)
                mhi_pred[i], e_mhi_pred[i] = predicted_mhi(rmag, gmag, group_dist, ttype, logms)
                f_mom0 = '/mnt/scratch/HCGs/separate_hi_discs/%s_mom0_%s.fits' %(g.replace(' ','').lower(), core.replace(' ','').lower())
                try:
                    core_data, core_hdr = fits.getdata(f_mom0, header=True)
                    core_flux = totflux(core_data, core_hdr)
                    hi_flag, hilim = 1, 0
                except FileNotFoundError:
                    core_flux = flux_limit(data=noise_median, wcs=wcs, ra=ra, dec=dec, size=bmaj, vdelt=vdelt, beam_area=beam_area)
                    hi_flag, hilim = 0, 1
                    print(f'{g} -- {core}: no moment map found, assuming a non-detection')
                    
                core_mhi, e_core_mhi = himass(core_flux, group_dist)
                def_hi, e_def_hi = hidef(core_mhi, mhi_pred[i], e_core_mhi, e_mhi_pred[i])
                src_coo = skycoo(ra, dec)
                sep = src_coo.separation(ccoo) ### separation from the group centre
                sep_kpc = sep.radian * group_dist * 1e3 ### separation in kpc
                sep_norm = sep_kpc / group_diam
                e_sep_norm = sep_norm * e_rvir / group_diam
                d25_kpc = coords.Angle(d25, unit='arcmin').radian * group_dist * 1e3
                if np.isnan(vsys):
                    try:
                        spec = get_global_profile(g, ra, dec, params)
                        vsys = HIvsys(spec)
                    except ValueError:
                        pass
                df_core.iloc[i] = [int(g.split()[-1]), 'core: %s' %core, src_coo.ra.deg, src_coo.dec.deg, vsys, mtype, ttype, sep_kpc, sep_norm, e_sep_norm, d25, d25_kpc, core_flux, core_mhi, e_core_mhi, mhi_pred[i], e_mhi_pred[i], def_hi, e_def_hi, 2, 1, hi_flag, hilim, gmag, gmag_err, rmag, rmag_err, Btot, logLB, logms, e_logms]
                df_core_cbl.iloc[i] = [int(g.split()[-1]), 'core: %s' %core, src_coo.ra.deg, src_coo.dec.deg, np.nan, np.nan, 'core']
        else:
            flux, mhi_obs, e_mhi_obs, def_hi, e_def_hi = np.zeros(len(name_cores)), np.zeros(len(name_cores)), np.zeros(len(name_cores)), np.zeros(len(name_cores)), np.zeros(len(name_cores))
            cendata, cenhdr = np.nan, np.nan
            for i, core in enumerate(name_cores):
                if len(core) == 1:
                    core = g+core.strip()
                ra, dec, vsys, mtype, ttype, d25, ba, gmag, gmag_err, rmag, rmag_err, logms, e_logms = get_params(g, core)
                if np.isnan(vsys): zflag = 0
                else: zflag = 1
                logms_mem[i] = logms
                gmag, rmag = mag_correction(gmag, ra, dec, ba, ttype, vsys, 'g'), mag_correction(rmag, ra, dec, ba, ttype, vsys, 'r')
                Btot = gmag_to_Bmag(gmag, rmag)
                logLB = Btot_to_lumB(Btot, group_dist)
                mhi_pred[i], e_mhi_pred[i] = predicted_mhi(rmag, gmag, group_dist, ttype, logms)
                if core in members['nocounterparts']:
                    flux[i] = flux_limit(data=noise_median, wcs=wcs, ra=ra, dec=dec, size=bmaj, vdelt=vdelt, beam_area=beam_area)
                    hiflag, hilim, opt_hi = 0, 1, 0
                else:
                    try:
                        subdata, subhdr = get_cubelet(g, float(ra), float(dec), params, radius=30, peak=False)
                        flux[i] = totflux(subdata, subhdr)
                        hiflag, hilim, opt_hi = 1, 0, 2
                    except ValueError:
                        flux[i] = np.nan
                        hiflag, hilim, opt_hi = 0, 0, 0
                src_coo = skycoo(ra, dec)
                sep = src_coo.separation(ccoo) ### separation from the group centre
                sep_kpc = sep.radian * group_dist * 1e3 ### separation in kpc
                sep_norm = sep_kpc / group_diam
                e_sep_norm = sep_norm * e_rvir / group_diam
                d25_kpc = coords.Angle(d25, unit='arcmin').radian * group_dist * 1e3
                if np.isnan(vsys):
                    spec = get_global_profile(g, ra, dec, params)
                    vsys = HIvsys(spec)
                mhi_obs[i], e_mhi_obs[i] = himass(flux[i], group_dist)
                def_hi[i], e_def_hi[i] = hidef(mhi_obs[i], mhi_pred[i], e_mhi_obs[i], e_mhi_pred[i])
                df_core.iloc[i] = [int(g.split()[-1]), 'core: %s' %core, src_coo.ra.deg, src_coo.dec.deg, vsys, mtype, ttype, sep_kpc, sep_norm, e_sep_norm, d25, d25_kpc, flux[i], mhi_obs[i], e_mhi_obs[i], mhi_pred[i], e_mhi_pred[i], def_hi[i], e_def_hi[i], opt_hi, zflag, hiflag, hilim, gmag, gmag_err, rmag, rmag_err, Btot, logLB, logms, e_logms]
                df_core_cbl.iloc[i] = [int(g.split()[-1]), 'core: %s' %core, src_coo.ra.deg, src_coo.dec.deg, np.nan, np.nan, 'core']

        tot_mhi_pred = np.log10(np.nansum(10**mhi_pred))
        e_tot_mhi_pred = np.sqrt(np.nansum(e_mhi_pred**2))
        logms_tot = np.log10(np.nansum(10**logms_mem))
        if isinstance(mhi_obs, np.ndarray):
            tot_flux = np.nansum(flux)
            tot_mhi_obs = np.log10(np.nansum(10**mhi_obs))
            e_tot_mhi_obs = np.sqrt(np.nansum(e_mhi_obs**2))
        else:
            tot_flux, tot_mhi_obs, e_tot_mhi_obs = flux, mhi_obs, e_mhi_obs
        tot_def_hi, e_tot_def_hi = hidef(tot_mhi_obs, tot_mhi_pred, e_tot_mhi_obs, e_tot_mhi_pred)
        df_core.iloc[i+1] = [int(g.split()[-1]), name_tot_core, ccoo.ra.deg, ccoo.dec.deg, group_vsys, np.nan, np.nan, 0.0, 0.0, 0.0, np.nan, np.nan, tot_flux, tot_mhi_obs, e_tot_mhi_obs, tot_mhi_pred, e_tot_mhi_pred, tot_def_hi, e_tot_def_hi, 2, 1, 1, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, logms_tot, np.nan]
        df_core_cbl.iloc[i+1] = [int(g.split()[-1]), name_tot_core, ccoo.ra.deg, ccoo.dec.deg, cendata, cenhdr, 'core']
        df_params = pd.concat([df_params, df_core], ignore_index=True)
        df_cubelets = pd.concat([df_cubelets, df_core_cbl], ignore_index=True)

    if 'nocounterparts' in members.keys():
        sources = members['nocounterparts']
        sources = [x for x in sources if not x in core_members]
        df_ncp = pd.DataFrame(columns=cols, index=sources)
        df_ncp_cbl = pd.DataFrame(columns=plotdf_cols, index=sources, dtype='object')
        for i,source in enumerate(sources):
            ra, dec, vsys, mtype, ttype, d25, ba, gmag, gmag_err, rmag, rmag_err, logms, e_logms = get_params(g, source)
            flux = flux_limit(data=noise_median, wcs=wcs, ra=ra, dec=dec, size=bmaj, vdelt=vdelt, beam_area=beam_area)
            mhi_obs, e_mhi_obs = himass(flux, group_dist)
            if np.isnan(vsys):
                zflag = 0
            else:
                zflag = 1
            # if np.isnan(d25):
            #     mhi_pred, def_hi = np.nan, np.nan
            # else:
            gmag, rmag = mag_correction(gmag, ra, dec, ba, ttype, vsys, 'g'), mag_correction(rmag, ra, dec, ba, ttype, vsys, 'r')
            Btot = gmag_to_Bmag(gmag, rmag)
            logLB = Btot_to_lumB(Btot, group_dist)
            # mhi_pred, _ = mj18_LB_to_mhi(logLB, ttype)
            mhi_pred, e_mhi_pred = predicted_mhi(rmag, gmag, group_dist, ttype, logms)
            def_hi, e_def_hi = hidef(mhi_obs, mhi_pred, e_mhi_obs, e_mhi_pred)
            scoo = skycoo(ra, dec)
            sep = scoo.separation(ccoo) ### separation from the group centre
            sep_kpc = sep.radian * group_dist * 1e3 ### separation in kpc
            sep_norm = sep_kpc / group_diam
            e_sep_norm = sep_norm * e_rvir / group_diam
            d25_kpc = coords.Angle(d25, unit='arcmin').radian * group_dist * 1e3
            if np.isnan(vsys):
                try:
                    spec = get_global_profile(g, ra, dec, params)
                    vsys = HIvsys(spec)
                except ValueError:
                    pass
            if vsys >= vlow and vsys <= vhigh:
                df_ncp.iloc[i] = [int(g.split()[-1]), source, scoo.ra.deg, scoo.dec.deg, vsys, mtype, ttype, sep_kpc, sep_norm, e_sep_norm, d25, d25_kpc, flux, mhi_obs, e_mhi_obs, mhi_pred, e_mhi_pred, def_hi, e_def_hi, 0, zflag, 0, 1, gmag, gmag_err, rmag, rmag_err, Btot, logLB, logms, e_logms]
                df_ncp_cbl.iloc[i] = [int(g.split()[-1]), source, scoo.ra.deg, scoo.dec.deg, np.nan, np.nan, 'nocounterparts']
        df_params = pd.concat([df_params, df_ncp], ignore_index=True)
        df_cubelets = pd.concat([df_cubelets, df_ncp_cbl], ignore_index=True)

    for col in ['RA', 'Dec']:
        df_params[col] = df_params[col].map(lambda x: '{0:.6f}'.format(x))
    for col in ['sep_kpc', 'sep_norm', 'D25_am', 'D25_kpc', 'flux', 'logMHIobs', 'e_logMHIobs', 'logMHIexp', 'defHI', 'gmag', 'e_gmag', 'rmag', 'e_rmag', 'Btot', 'logLB', 'logMs', 'e_logMs']:
        df_params[col] = df_params[col].map(lambda x: '{0:.3f}'.format(x))
        
    if not os.path.isdir(tabdir):
        os.mkdir(tabdir)
    if not tabdir.endswith('/'):
        tabdir += '/'    
    outfile = tabdir+'HIdef_%s.csv' %g.replace(' ','').lower()
    df_params.to_csv(outfile, na_rep='NaN')
        
    if plot:
        if not os.path.isdir(mapdir):
            os.mkdir(mapdir)
        if not mapdir.endswith('/'):
            mapdir += '/'
        fits_m0 = params[g]['rootdir'] + '/' + params[g]['moment_0']
        fig = plt.figure(figsize=(8,8))
        if color:
            figname = mapdir+g.replace(' ','').upper()+'_grz.pdf'
            plot_map(fig, g, gcoo=ccoo, f_mom0=fits_m0, df=df_cubelets, color=True)
        else:
            figname = mapdir+g.replace(' ','').upper()+'.pdf'
            plot_map(fig, g, gcoo=ccoo, f_mom0=fits_m0, df=df_cubelets, color=False)
        plt.savefig(figname)
            
    return df_params, df_cubelets