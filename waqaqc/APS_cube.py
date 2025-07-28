import numpy as np
from astropy.io import fits
import os
import spectres
import multiprocessing as mp
import gc
import configparser
from astropy.wcs import WCS
import tqdm

_wave = None
_n_wave = None


def init_globals(wave, n_wave):
    global _wave, _n_wave
    _wave = wave
    _n_wave = n_wave


def forloop(args):
    i, c_spec, c_espec = args

    n_flux, n_err = spectres.spectres(_n_wave, _wave, c_spec, c_espec)

    n_flux = np.array(n_flux, dtype=np.float32)
    n_err = np.array(n_err, dtype=np.float32)

    return n_flux, n_err


def cube_creator(self):
    config = configparser.ConfigParser()
    config.read(self)

    file_dir = config.get('APS_cube', 'file_dir')

    wcs_c = fits.open(file_dir + config.get('QC_plots', 'blue_cube'))
    c = fits.open(file_dir + config.get('APS_cube', 'aps_file'))

    gal_id = c[1].data['CNAME'][0]
    gal_dir = gal_id + '_' + wcs_c[0].header['MODE'] + '_' + str(wcs_c[0].header['OBID'])

    os.makedirs(gal_dir, exist_ok=True)

    aps_id = c[2].data['APS_ID']
    bin_id = c[2].data['BIN_ID']
    r_bin_id = c[3].data['BIN_ID']

    wave = np.exp(c[1].data['LOGLAM'][0])
    if wcs_c[0].header['MODE'] == 'HIGHRES':
        n_wave = np.arange(min(wave) + 0.1, max(wave), 0.1)
    elif wcs_c[0].header['MODE'] == 'LOWRES':
        n_wave = np.arange(min(wave) + 0.5, max(wave), 0.5)

    axis_header = fits.Header()
    axis_header['NAXIS1'] = wcs_c[1].header['NAXIS1']
    axis_header['NAXIS2'] = wcs_c[1].header['NAXIS2']
    axis_header['CD1_1'] = wcs_c[1].header['CD1_1']
    axis_header['CD2_2'] = wcs_c[1].header['CD2_2']
    axis_header['CRPIX1'] = wcs_c[1].header['CRPIX1']
    axis_header['CRPIX2'] = wcs_c[1].header['CRPIX2']
    axis_header['CRVAL1'] = wcs_c[1].header['CRVAL1']
    axis_header['CRVAL2'] = wcs_c[1].header['CRVAL2']
    axis_header['CTYPE1'] = wcs_c[1].header['CTYPE1']
    axis_header['CTYPE2'] = wcs_c[1].header['CTYPE2']
    axis_header['CUNIT1'] = wcs_c[1].header['CUNIT1']
    axis_header['CUNIT2'] = wcs_c[1].header['CUNIT2']

    wcs = WCS(axis_header)

    aps_ra = c[2].data['X_0'] + (c[2].data['X'] / 3600)
    aps_dec = c[2].data['Y_0'] + (c[2].data['Y'] / 3600)

    aps_ra_dec = np.vstack((aps_ra, aps_dec)).T

    pix_map = np.round(wcs.wcs_world2pix(aps_ra_dec, 0), 0)

    pix_mapt = pix_map.T.astype(int)
    pix_mapt[0] = pix_mapt[0] - np.min(pix_mapt[0])
    pix_mapt[1] = pix_mapt[1] - np.min(pix_mapt[1])

    pix_mapt = pix_mapt.T

    x_pix, y_pix = pix_map.T.astype(int)

    print('')
    print('Recreating original datacube from APS file. This may take a few minutes...')
    ext = 1

    with mp.Pool(int(config.get('APS_cube', 'n_proc')), initializer=init_globals, initargs=(wave, n_wave)) as pool:
        rss = pool.starmap(forloop, tqdm.tqdm(zip((i, c[ext].data['SPEC'][i], c[ext].data['ESPEC'][i])
                                                  for i in np.arange(c[ext].data['SPEC'].shape[0])),
                                              total=c[ext].data['SPEC'].shape[0]))

    rss_data = np.empty((c[1].data['SPEC'].shape[0], len(n_wave)), dtype=np.float32)
    rss_err = np.empty((c[1].data['SPEC'].shape[0], len(n_wave)), dtype=np.float32)

    for i in np.arange(c[ext].data['SPEC'].shape[0]):
        rss_data[i] = rss[i][0]
        rss_err[i] = rss[i][1]

    del rss
    gc.collect()

    vorbin_cube_data = np.zeros((c[3].data['SPEC'].shape[0], len(n_wave)), dtype=np.float32)
    vorbin_cube_err = np.zeros((c[3].data['SPEC'].shape[0], len(n_wave)), dtype=np.float32)

    apsid_map = np.zeros((np.max(y_pix) - np.min(y_pix) + 1, np.max(x_pix) - np.min(x_pix) + 1)) * np.nan
    vorbin_map = np.zeros((np.max(y_pix) - np.min(y_pix) + 1, np.max(x_pix) - np.min(x_pix) + 1)) * np.nan
    stel_vel_map = np.zeros((np.max(y_pix) - np.min(y_pix) + 1, np.max(x_pix) - np.min(x_pix) + 1)) * np.nan
    aps_maps = np.zeros(
        (len(c[4].data.names) - 1, np.max(y_pix) - np.min(y_pix) + 1, np.max(x_pix) - np.min(x_pix) + 1)) * np.nan

    print('')
    print('Recreating Voronoi binning datacube from APS file. This may take a few minutes...')
    ext = 3
    with mp.Pool(int(config.get('APS_cube', 'n_proc')), initializer=init_globals, initargs=(wave, n_wave)) as pool:
        # vorbin = pool.starmap(forloop, tqdm.tqdm(zip((i, c[ext].data['SPEC'][i], c[ext].data['ESPEC'][i])
        #                                              for i in np.arange(c[ext].data['SPEC'].shape[0])),
        #                                          total=c[ext].data['SPEC'].shape[0]))
        vorbin = list(tqdm.tqdm(pool.imap_unordered(forloop,
                                                    ((i, c[ext].data['SPEC'][i], c[ext].data['ESPEC'][i])
                                                     for i in range(100))),
                                total=100))

    print('oi0')

    for i in np.arange(c[ext].data['SPEC'].shape[0]):
        vorbin_cube_data[i] = vorbin[i][0]
        vorbin_cube_err[i] = vorbin[i][1]

    print('oi1')

    del vorbin
    gc.collect()

    cube_data = np.zeros((len(n_wave), np.max(y_pix) - np.min(y_pix) + 1, np.max(x_pix) - np.min(x_pix) + 1),
                         dtype=np.float32)
    cube_err = np.zeros((len(n_wave), np.max(y_pix) - np.min(y_pix) + 1, np.max(x_pix) - np.min(x_pix) + 1),
                        dtype=np.float32)
    vorbin_data = np.zeros((len(n_wave), np.max(y_pix) - np.min(y_pix) + 1, np.max(x_pix) - np.min(x_pix) + 1),
                           dtype=np.float32)
    vorbin_err = np.zeros((len(n_wave), np.max(y_pix) - np.min(y_pix) + 1, np.max(x_pix) - np.min(x_pix) + 1),
                          dtype=np.float32)

    aps_maps_names = []

    cnt = 0
    #
    for j in np.arange(len(c[4].data.names) - 1):
        aps_maps_names.append(c[4].data.names[j + 1])
    for i in pix_mapt:
        apsid_map[i[1], i[0]] = aps_id[cnt]
        vorbin_map[i[1], i[0]] = bin_id[cnt]
        cnt += 1

    print('oi2')

    cnt = 0
    for i in pix_mapt:
        if apsid_map[i[1], i[0]] >= 0:
            cube_data[:, i[1], i[0]] = rss_data[aps_id == apsid_map[i[1], i[0]]][0]
            cube_err[:, i[1], i[0]] = rss_err[aps_id == apsid_map[i[1], i[0]]][0]
        if vorbin_map[i[1], i[0]] >= 0:
            vorbin_data[:, i[1], i[0]] = vorbin_cube_data[r_bin_id == vorbin_map[i[1], i[0]]][0] / \
                                         len(np.where(vorbin_map == vorbin_map[i[1], i[0]])[0])
            vorbin_err[:, i[1], i[0]] = vorbin_cube_err[r_bin_id == vorbin_map[i[1], i[0]]][0] / \
                                        len(np.where(vorbin_map == vorbin_map[i[1], i[0]])[0])
            stel_vel_map[i[1], i[0]] = c[4].data['V'][r_bin_id == bin_id[cnt]]
            for j in np.arange(len(c[4].data.names) - 1):
                if len(c[4].data[c[4].data.names[j + 1]][r_bin_id == bin_id[cnt]].shape) == 1:
                    aps_maps[j, i[1], i[0]] = c[4].data[c[4].data.names[j + 1]][r_bin_id == bin_id[cnt]]
                else:
                    np.delete(aps_maps, j, axis=0)
        print('Rearranging into datacube formats: ' + str(
            round(100. * cnt / pix_mapt.shape[0], 2)) + '%', end='\r')
        cnt += 1

    print('oi3')

    del rss_data, rss_err, vorbin_cube_data, vorbin_cube_err
    gc.collect()

    print('')

    cpix = [c[2].data['X_0'][0] + (min(c[2].data['X'], key=abs) / 3600), c[2].data['Y_0'][0] +
            (min(c[2].data['Y'], key=abs) / 3600)]

    cpix_x, cpix_y = np.round(wcs.wcs_world2pix(np.array([cpix]), 0))[0]

    # create RSS file

    cube_head = c[0].header.copy()
    cube_head['SIMPLE'] = True
    cube_head['BITPIX'] = -32
    cube_head['NAXIS'] = 3
    cube_head['NAXIS1'] = cube_data.shape[2]
    cube_head['NAXIS2'] = cube_data.shape[1]
    cube_head['NAXIS3'] = cube_data.shape[0]
    cube_head['CTYPE3'] = 'WAVELENGTH'
    cube_head['CUNIT3'] = 'Angstrom'
    if wcs_c[0].header['MODE'] == 'HIGHRES':
        cube_head['CDELT3'] = 0.1
    if wcs_c[0].header['MODE'] == 'LOWRES':
        cube_head['CDELT3'] = 0.5
    cube_head['DISPAXIS'] = 1
    cube_head['CRVAL3'] = min(n_wave)
    cube_head['CRPIX3'] = 1
    cube_head['CRPIX1'] = cpix_x
    cube_head['CRPIX2'] = cpix_y
    cube_head['CRVAL1'] = cpix[0]
    cube_head['CRVAL2'] = cpix[1]
    cube_head['CDELT1'] = axis_header['CD1_1']
    cube_head['CDELT2'] = axis_header['CD2_2']
    cube_head['CTYPE1'] = 'RA---TAN'
    cube_head['CTYPE2'] = 'DEC--TAN'
    cube_head['CUNIT1'] = 'deg'
    cube_head['CUNIT2'] = 'deg'
    cube_head['CCNAME'] = gal_id
    cube_head['OBSMODE'] = c[0].header['OBSMODE']
    cube_head['MODE'] = wcs_c[0].header['MODE']

    map_head = fits.Header()
    map_head['SIMPLE'] = True
    map_head['BITPIX'] = -32
    map_head['NAXIS'] = 2
    map_head['NAXIS1'] = vorbin_map.shape[1]
    map_head['NAXIS2'] = vorbin_map.shape[0]
    map_head['DISPAXIS'] = 1
    map_head['CRPIX1'] = cpix_x
    map_head['CRPIX2'] = cpix_y
    map_head['CRVAL1'] = cpix[0]
    map_head['CRVAL2'] = cpix[1]
    map_head['CDELT1'] = axis_header['CD1_1']
    map_head['CDELT2'] = axis_header['CD2_2']
    map_head['CTYPE1'] = 'RA---TAN'
    map_head['CTYPE2'] = 'DEC--TAN'
    map_head['CUNIT1'] = 'deg'
    map_head['CUNIT2'] = 'deg'

    n_cube = fits.HDUList([fits.PrimaryHDU(),
                           fits.ImageHDU(data=cube_data, header=cube_head, name='DATA'),
                           fits.ImageHDU(data=cube_err, header=cube_head, name='ERROR')])
    n_vorbin = fits.HDUList([fits.PrimaryHDU(),
                             fits.ImageHDU(data=vorbin_data, header=cube_head, name='DATA'),
                             fits.ImageHDU(data=vorbin_err, header=cube_head, name='ERROR')])
    maps_HDU = fits.HDUList([fits.PrimaryHDU()])
    for i in np.arange(len(aps_maps)):
        maps_HDU.append(fits.ImageHDU(data=aps_maps[i], header=map_head, name=aps_maps_names[i]))

    n_cube.writeto(gal_dir + '/' + gal_id + '_cube.fits', overwrite=True)
    n_vorbin.writeto(gal_dir + '/' + gal_id + '_vorbin_cube.fits', overwrite=True)
    maps_HDU.writeto(gal_dir + '/' + gal_id + '_APS_maps.fits', overwrite=True)
    fits.writeto(gal_dir + '/' + 'vorbin_map.fits', vorbin_map, header=map_head, overwrite=True)
