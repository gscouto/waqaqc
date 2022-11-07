import numpy as np
from astropy.io import fits
import os
import time
import spectres
import multiprocessing as mp
import gc
import configparser
from astropy.wcs import WCS


def forloop(args):
    i, n_wave, wave, c_spec, c_espec = args

    n_flux, n_err = spectres.spectres(n_wave, wave, c_spec, c_espec)

    if (i / 500.).is_integer():
        print(i)

    return n_flux, n_err


def cube_creator(self):
    s_time = time.time()

    config = configparser.ConfigParser()
    config.read(self)

    file_dir = config.get('APS_cube', 'file_dir')
    list_file = os.listdir(file_dir)

    c = fits.open(file_dir + [s for s in list_file if 'APS.fits' in s][0])

    gal_id = c[0].header['CCNAME1']
    gal_dir = gal_id

    os.makedirs(gal_dir, exist_ok=True)

    xx = np.around(c[2].data['X'], 1)
    yy = np.around(c[2].data['Y'], 1)

    aps_id = c[2].data['APS_ID']
    bin_id = c[2].data['BIN_ID']
    r_bin_id = c[3].data['BIN_ID']

    wave = np.exp(c[1].data['LOGLAM'][0])
    if c[0].header['RES-OBS'] == 'HR':
        n_wave = np.arange(min(wave) + 0.01, max(wave), 0.1)
    else:
        n_wave = np.arange(min(wave) + 0.01, max(wave), 0.3)

    rss_data = np.zeros((c[1].data['SPEC'].shape[0], len(n_wave)), dtype=np.float32)
    rss_err = np.zeros((c[1].data['SPEC'].shape[0], len(n_wave)), dtype=np.float32)
    vorbin_cube_data = np.zeros((c[3].data['SPEC'].shape[0], len(n_wave)), dtype=np.float32)
    vorbin_cube_err = np.zeros((c[3].data['SPEC'].shape[0], len(n_wave)), dtype=np.float32)

    xx_map = np.zeros((np.unique(yy).shape[0], np.unique(xx).shape[0]))
    yy_map = np.zeros((np.unique(yy).shape[0], np.unique(xx).shape[0]))
    apsid_map = np.zeros((np.unique(yy).shape[0], np.unique(xx).shape[0])) * np.nan
    vorbin_map = np.zeros((np.unique(yy).shape[0], np.unique(xx).shape[0])) * np.nan

    print('')
    print('Recreating original datacube from APS file. This may take a few minutes...')
    ext = 1

    with mp.Pool(int(config.get('APS_cube', 'n_proc'))) as pool:
        rss = pool.starmap(forloop, zip((i, n_wave, wave, c[ext].data['SPEC'][i], c[ext].data['ESPEC'][i])
                                        for i in np.arange(c[ext].data['SPEC'].shape[0])))

    for i in np.arange(c[ext].data['SPEC'].shape[0]):
        rss_data[i] = rss[i][0]
        rss_err[i] = rss[i][1]

    del rss
    gc.collect()

    print('')
    print('Recreating Voronoi binning datacube from APS file. This may take a few minutes...')
    ext = 3
    with mp.Pool(int(config.get('APS_cube', 'n_proc'))) as pool:
        vorbin = pool.starmap(forloop, zip((i, n_wave, wave, c[ext].data['SPEC'][i], c[ext].data['ESPEC'][i])
                                           for i in np.arange(c[ext].data['SPEC'].shape[0])))

    for i in np.arange(c[ext].data['SPEC'].shape[0]):
        vorbin_cube_data[i] = vorbin[i][0]
        vorbin_cube_err[i] = vorbin[i][1]

    del vorbin
    gc.collect()

    cube_data = np.zeros((len(n_wave), np.unique(yy).shape[0], np.unique(xx).shape[0]))
    cube_err = np.zeros((len(n_wave), np.unique(yy).shape[0], np.unique(xx).shape[0]))
    vorbin_data = np.zeros((len(n_wave), np.unique(yy).shape[0], np.unique(xx).shape[0]))
    vorbin_err = np.zeros((len(n_wave), np.unique(yy).shape[0], np.unique(xx).shape[0]))

    print('')

    for i in np.arange(np.unique(xx).shape[0]):
        for j in np.arange(np.unique(yy).shape[0]):
            cnt = + 1
            if np.size(np.where((xx == np.unique(xx)[i]) & (yy == np.unique(yy)[j]))) > 0:
                xx_map[j, i] = xx[np.where((xx == np.unique(xx)[i]) & (yy == np.unique(yy)[j]))]
                yy_map[j, i] = yy[np.where((xx == np.unique(xx)[i]) & (yy == np.unique(yy)[j]))]
                apsid_map[j, i] = aps_id[np.where((xx == np.unique(xx)[i]) & (yy == np.unique(yy)[j]))]
                vorbin_map[j, i] = bin_id[np.where((xx == np.unique(xx)[i]) & (yy == np.unique(yy)[j]))]
            if apsid_map[j, i] >= 0:
                cube_data[:, j, i] = rss_data[aps_id == apsid_map[j, i]][0]
                cube_err[:, j, i] = rss_err[aps_id == apsid_map[j, i]][0]
            if vorbin_map[j, i] >= 0:
                vorbin_data[:, j, i] = vorbin_cube_data[r_bin_id == vorbin_map[j, i]][0]
                vorbin_err[:, j, i] = vorbin_cube_err[r_bin_id == vorbin_map[j, i]][0]
            print('Rearranging into datacube formats: ' + str(
                round(100. * cnt / np.unique(xx).shape[0] * np.unique(yy).shape[0], 2)) + '%', end='\r')

    del rss_data, rss_err, vorbin_cube_data, vorbin_cube_err
    gc.collect()

    print('')

    ncube_data = np.flip(cube_data, axis=2)
    ncube_err = np.flip(cube_err, axis=2)
    nvorbin_data = np.flip(vorbin_data, axis=2)
    nvorbin_err = np.flip(vorbin_err, axis=2)

    nvorbin_map = np.flip(vorbin_map, axis=1)

    # create RSS file

    cube_head = c[0].header.copy()
    cube_head['SIMPLE'] = True
    cube_head['BITPIX'] = -32
    cube_head['NAXIS'] = 3
    cube_head['NAXIS1'] = ncube_data.shape[2]
    cube_head['NAXIS2'] = ncube_data.shape[1]
    cube_head['NAXIS3'] = ncube_data.shape[0]
    cube_head['CTYPE3'] = 'WAVELENGTH'
    cube_head['CUNIT3'] = 'Angstrom'
    cube_head['CDELT3'] = 0.1
    cube_head['DISPAXIS'] = 1
    cube_head['CRVAL3'] = min(wave)
    cube_head['CRPIX3'] = 1
    cube_head['CRPIX1'] = \
        np.where(np.unique(c[2].data['X']) == np.min(np.unique(c[2].data['X'])[np.unique(c[2].data['X']) > 0]))[0][
            0] + 1  # X from APS with lower absolute value
    cube_head['CRPIX2'] = \
        np.where(np.unique(c[2].data['Y']) == np.min(np.unique(c[2].data['Y'])[np.unique(c[2].data['Y']) > 0]))[0][
            0] + 1  # Y from APS with lower absolute value
    cube_head['CRVAL1'] = np.unique(c[2].data['X_0'])[0] + (np.min(
        np.unique(c[2].data['X'])[np.unique(c[2].data['X']) > 0]) / 3600.)  # X from central pixel plus position X_0
    cube_head['CRVAL2'] = np.unique(c[2].data['Y_0'])[0] + (np.min(
        np.unique(c[2].data['Y'])[np.unique(c[2].data['Y']) > 0]) / 3600.)  # Y from central pixel plus position Y_0
    cube_head['CDELT1'] = -(np.unique(c[2].data['X'])[1] - np.unique(c[2].data['X'])[0]) / 3600.
    cube_head['CDELT2'] = (np.unique(c[2].data['Y'])[1] - np.unique(c[2].data['Y'])[0]) / 3600.
    cube_head['CTYPE1'] = 'RA---TAN'
    cube_head['CTYPE2'] = 'DEC--TAN'
    cube_head['CUNIT1'] = 'deg'
    cube_head['CUNIT2'] = 'deg'
    cube_head['CCNAME1'] = c[0].header['CCNAME1']
    cube_head['OBSMODE'] = c[0].header['OBSMODE']
    cube_head['RES-OBS'] = c[0].header['RES-OBS']

    map_head = fits.Header()
    map_head['SIMPLE'] = True
    map_head['BITPIX'] = -32
    map_head['NAXIS'] = 2
    map_head['NAXIS1'] = nvorbin_map.shape[1]
    map_head['NAXIS2'] = nvorbin_map.shape[0]
    map_head['DISPAXIS'] = 1
    map_head['CRPIX1'] = \
        np.where(np.unique(c[2].data['X']) == np.min(np.unique(c[2].data['X'])[np.unique(c[2].data['X']) > 0]))[0][
            0] + 1  # X from APS with lower absolute value
    map_head['CRPIX2'] = \
        np.where(np.unique(c[2].data['Y']) == np.min(np.unique(c[2].data['Y'])[np.unique(c[2].data['Y']) > 0]))[0][
            0] + 1  # Y from APS with lower absolute value
    map_head['CRVAL1'] = np.unique(c[2].data['X_0'])[0] + (np.min(
        np.unique(c[2].data['X'])[np.unique(c[2].data['X']) > 0]) / 3600.)  # X from central pixel plus position X_0
    map_head['CRVAL2'] = np.unique(c[2].data['Y_0'])[0] + (np.min(
        np.unique(c[2].data['Y'])[np.unique(c[2].data['Y']) > 0]) / 3600.)  # Y from central pixel plus position Y_0
    map_head['CDELT1'] = -(np.unique(c[2].data['X'])[1] - np.unique(c[2].data['X'])[0]) / 3600.
    map_head['CDELT2'] = (np.unique(c[2].data['Y'])[1] - np.unique(c[2].data['Y'])[0]) / 3600.
    map_head['CTYPE1'] = 'RA---TAN'
    map_head['CTYPE2'] = 'DEC--TAN'
    map_head['CUNIT1'] = 'deg'
    map_head['CUNIT2'] = 'deg'

    n_cube = fits.HDUList([fits.PrimaryHDU(data=ncube_data, header=cube_head),
                           fits.ImageHDU(data=ncube_err, header=cube_head, name='ERROR')])
    n_vorbin = fits.HDUList([fits.PrimaryHDU(data=nvorbin_data, header=cube_head),
                             fits.ImageHDU(data=nvorbin_err, header=cube_head, name='ERROR')])

    n_cube.writeto(gal_dir + '/' + gal_id + '_cube.fits', overwrite=True)
    n_vorbin.writeto(gal_dir + '/' + gal_id + '_vorbin_cube.fits', overwrite=True)
    fits.writeto(gal_dir + '/' + 'vorbin_map.fits', nvorbin_map, header=map_head, overwrite=True)

    print('This run took ' + str(round(time.time() - s_time, 2)) + ' secs')


def cube_creator_new(self):

    config = configparser.ConfigParser()
    config.read(self)

    wcs_c = fits.open(config.get('QC_plots', 'blue_cube'))
    file_dir = config.get('APS_cube', 'file_dir')
    list_file = os.listdir(file_dir)

    c = fits.open(file_dir + [s for s in list_file if 'APS.fits' in s][0])

    gal_id = c[0].header['CCNAME1']
    gal_dir = gal_id

    os.makedirs(gal_dir, exist_ok=True)

    aps_id = c[2].data['APS_ID']
    bin_id = c[2].data['BIN_ID']
    r_bin_id = c[3].data['BIN_ID']

    wave = np.exp(c[1].data['LOGLAM'][0])
    if c[0].header['RES-OBS'] == 'HR':
        n_wave = np.arange(min(wave) + 0.01, max(wave), 0.1)
    else:
        n_wave = np.arange(min(wave) + 0.01, max(wave), 0.3)

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

    rss_data = np.zeros((c[1].data['SPEC'].shape[0], len(n_wave)), dtype=np.float32)
    rss_err = np.zeros((c[1].data['SPEC'].shape[0], len(n_wave)), dtype=np.float32)
    vorbin_cube_data = np.zeros((c[3].data['SPEC'].shape[0], len(n_wave)), dtype=np.float32)
    vorbin_cube_err = np.zeros((c[3].data['SPEC'].shape[0], len(n_wave)), dtype=np.float32)

    apsid_map = np.zeros((np.max(y_pix) - np.min(y_pix) + 1, np.max(x_pix) - np.min(x_pix) + 1)) * np.nan
    vorbin_map = np.zeros((np.max(y_pix) - np.min(y_pix) + 1, np.max(x_pix) - np.min(x_pix) + 1)) * np.nan

    print('')
    print('Recreating original datacube from APS file. This may take a few minutes...')
    ext = 1

    with mp.Pool(int(config.get('APS_cube', 'n_proc'))) as pool:
        rss = pool.starmap(forloop, zip((i, n_wave, wave, c[ext].data['SPEC'][i], c[ext].data['ESPEC'][i])
                                        for i in np.arange(c[ext].data['SPEC'].shape[0])))

    for i in np.arange(c[ext].data['SPEC'].shape[0]):
        rss_data[i] = rss[i][0]
        rss_err[i] = rss[i][1]

    del rss
    gc.collect()

    print('')
    print('Recreating Voronoi binning datacube from APS file. This may take a few minutes...')
    ext = 3
    with mp.Pool(int(config.get('APS_cube', 'n_proc'))) as pool:
        vorbin = pool.starmap(forloop, zip((i, n_wave, wave, c[ext].data['SPEC'][i], c[ext].data['ESPEC'][i])
                                           for i in np.arange(c[ext].data['SPEC'].shape[0])))

    for i in np.arange(c[ext].data['SPEC'].shape[0]):
        vorbin_cube_data[i] = vorbin[i][0]
        vorbin_cube_err[i] = vorbin[i][1]

    del vorbin
    gc.collect()

    cube_data = np.zeros((len(n_wave), np.max(y_pix) - np.min(y_pix) + 1, np.max(x_pix) - np.min(x_pix) + 1))
    cube_err = np.zeros((len(n_wave), np.max(y_pix) - np.min(y_pix) + 1, np.max(x_pix) - np.min(x_pix) + 1))
    vorbin_data = np.zeros((len(n_wave), np.max(y_pix) - np.min(y_pix) + 1, np.max(x_pix) - np.min(x_pix) + 1))
    vorbin_err = np.zeros((len(n_wave), np.max(y_pix) - np.min(y_pix) + 1, np.max(x_pix) - np.min(x_pix) + 1))

    cnt = 0
    for i in pix_mapt:
        apsid_map[i[1], i[0]] = aps_id[cnt]
        vorbin_map[i[1], i[0]] = bin_id[cnt]
        if apsid_map[i[1], i[0]] >= 0:
            cube_data[:, i[1], i[0]] = rss_data[aps_id == apsid_map[i[1], i[0]]][0]
            cube_err[:, i[1], i[0]] = rss_err[aps_id == apsid_map[i[1], i[0]]][0]
        if vorbin_map[i[1], i[0]] >= 0:
            vorbin_data[:, i[1], i[0]] = vorbin_cube_data[r_bin_id == vorbin_map[i[1], i[0]]][0]
            vorbin_err[:, i[1], i[0]] = vorbin_cube_err[r_bin_id == vorbin_map[i[1], i[0]]][0]
        print('Rearranging into datacube formats: ' + str(
            round(100. * cnt / pix_mapt.shape[0], 2)) + '%', end='\r')
        cnt += 1

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
    cube_head['CDELT3'] = 0.1
    cube_head['DISPAXIS'] = 1
    cube_head['CRVAL3'] = min(wave)
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
    cube_head['CCNAME1'] = c[0].header['CCNAME1']
    cube_head['OBSMODE'] = c[0].header['OBSMODE']
    cube_head['RES-OBS'] = c[0].header['RES-OBS']

    map_head = fits.Header()
    map_head['SIMPLE'] = True
    map_head['BITPIX'] = -32
    map_head['NAXIS'] = 2
    map_head['NAXIS1'] = vorbin_map.shape[1]
    map_head['NAXIS2'] = vorbin_map.shape[0]
    map_head['DISPAXIS'] = 1
    cube_head['CRPIX1'] = cpix_x
    cube_head['CRPIX2'] = cpix_y
    cube_head['CRVAL1'] = cpix[0]
    cube_head['CRVAL2'] = cpix[1]
    cube_head['CDELT1'] = axis_header['CD1_1']
    cube_head['CDELT2'] = axis_header['CD2_2']
    map_head['CTYPE1'] = 'RA---TAN'
    map_head['CTYPE2'] = 'DEC--TAN'
    map_head['CUNIT1'] = 'deg'
    map_head['CUNIT2'] = 'deg'

    n_cube = fits.HDUList([fits.PrimaryHDU(data=cube_data, header=cube_head),
                           fits.ImageHDU(data=cube_err, header=cube_head, name='ERROR')])
    n_vorbin = fits.HDUList([fits.PrimaryHDU(data=vorbin_data, header=cube_head),
                             fits.ImageHDU(data=vorbin_err, header=cube_head, name='ERROR')])

    n_cube.writeto(gal_dir + '/' + gal_id + '_cube.fits', overwrite=True)
    n_vorbin.writeto(gal_dir + '/' + gal_id + '_vorbin_cube.fits', overwrite=True)
    fits.writeto(gal_dir + '/' + 'vorbin_map.fits', vorbin_map, header=map_head, overwrite=True)
