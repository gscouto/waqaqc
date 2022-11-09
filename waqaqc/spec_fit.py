import numpy as np
from astropy.io import fits
import os
from datetime import datetime
import configparser


def specs(self):
    config = configparser.ConfigParser()
    config.read(self)

    file_dir = config.get('APS_cube', 'file_dir')

    blue_cube = fits.open(file_dir + config.get('QC_plots', 'blue_cube'))

    gal = blue_cube[0].header['CCNAME1']

    gal_dir = gal + '/' + 'pyp_results/' + datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    os.makedirs(gal_dir, exist_ok=True)

    if int(config.get('spec_fit', 'aps_fit')) == 1:

        c = fits.open(gal + '/' + gal + '_vorbin_cube.fits')
        vorbin_map = fits.getdata(gal + '/vorbin_map.fits')

        # create RSS file

        wave = c[0].header['CRVAL3'] + (c[0].header['CDELT3'] * np.arange(c[0].header['NAXIS3']))

        rss_data = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), len(wave)), dtype=np.float32)
        rss_err = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), len(wave)), dtype=np.float32)

        for i in np.unique(vorbin_map[vorbin_map >= 0]):
            rss_data[int(i)] = c[0].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
            rss_err[int(i)] = c[1].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]

        rss_head = fits.Header()
        rss_head['SIMPLE'] = True
        rss_head['BITPIX'] = -32
        rss_head['NAXIS'] = 2
        rss_head['NAXIS1'] = rss_data.shape[1]
        rss_head['NAXIS2'] = rss_data.shape[0]
        rss_head['CTYPE1'] = 'WAVELENGTH'
        rss_head['CUNIT1'] = 'Angstrom'
        rss_head['CDELT1'] = c[0].header['CDELT3']
        rss_head['DISPAXIS'] = 1
        rss_head['CRVAL1'] = c[0].header['CRVAL3']
        rss_head['CRPIX1'] = c[0].header['CRPIX3']

        rss_ima = fits.HDUList([fits.PrimaryHDU(data=rss_data, header=rss_head),
                                fits.ImageHDU(data=rss_err, header=rss_head, name='ERROR')])

        rss_ima.writeto(gal + '_vorbin_RSS.fits', overwrite=True)

        print('')
        print('Running PyParadise best fit')

        os.system(
            'ParadiseApp.py ' + gal + '_vorbin_RSS.fits ' + gal + '_vorbin ' + config.get('spec_fit', 'fwhm') +
            ' --SSP_par parameters_stellar --line_par parameters_eline --parallel ' + config.get('APS_cube', 'n_proc')
            + ' --verbose')

        print('')
        print('Running bootstrap models')

        os.system(
            'ParadiseApp.py ' + gal + '_vorbin_RSS.fits ' + gal + '_vorbin ' + config.get('spec_fit', 'fwhm') +
            ' --SSP_par parameters_stellar --line_par parameters_eline --bootstraps 100 --modkeep 80 --parallel ' +
            config.get('APS_cube', 'n_proc') + ' --verbose')

        os.system('mv ' + gal + '*.fits ' + gal_dir + '/.')
        os.system('cp excl* lines.fit par.lines parameters_eline parameters_stellar ' + gal_dir + '/.')

    if int(config.get('spec_fit', 'blue_fit')) == 1:

        wave = blue_cube[1].header['CRVAL3'] + (blue_cube[1].header['CD3_3'] *
                                                np.arange(blue_cube[1].header['NAXIS3']))
        vorbin_map = fits.getdata(gal + '/vorbin_map_blue.fits')

        rss_data = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), len(wave)), dtype=np.float32)
        rss_err = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), len(wave)), dtype=np.float32)

        for i in np.unique(vorbin_map[vorbin_map >= 0]):
            rss_data[int(i)] = blue_cube[1].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
            rss_err[int(i)] = blue_cube[2].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]

        rss_head = fits.Header()
        rss_head['SIMPLE'] = True
        rss_head['BITPIX'] = -32
        rss_head['NAXIS'] = 2
        rss_head['NAXIS1'] = rss_data.shape[1]
        rss_head['NAXIS2'] = rss_data.shape[0]
        rss_head['CTYPE1'] = 'WAVELENGTH'
        rss_head['CUNIT1'] = 'Angstrom'
        rss_head['CDELT1'] = blue_cube[1].header['CD3_3']
        rss_head['DISPAXIS'] = 1
        rss_head['CRVAL1'] = blue_cube[1].header['CRVAL3']
        rss_head['CRPIX1'] = blue_cube[1].header['CRPIX3']