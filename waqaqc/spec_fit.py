import numpy as np
from astropy.io import fits
import os
import time
from datetime import datetime
import configparser


def specs(self):
    s_time = time.time()

    config = configparser.ConfigParser()
    config.read(self)

    gal = config.get('spec_fit', 'gal_id')

    c = fits.open(config.get('spec_fit', 'file_dir') + gal + '/' + gal + '_vorbin_cube.fits')
    vorbin_map = fits.getdata(config.get('spec_fit', 'file_dir') + gal + '/vorbin_map.fits')

    gal_dir = 'pyp_results/' + gal + '/' + datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    os.makedirs(gal_dir, exist_ok=True)

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
    rss_head['CDELT1'] = 0.1
    rss_head['DISPAXIS'] = 1
    rss_head['CRVAL1'] = min(wave)
    rss_head['CRPIX1'] = 1

    rss_ima = fits.HDUList([fits.PrimaryHDU(data=rss_data, header=rss_head),
                            fits.ImageHDU(data=rss_err, header=rss_head, name='ERROR')])

    rss_ima.writeto(gal + '_vorbin_RSS.fits', overwrite=True)

    print('')
    print('Running PyParadise best fit')

    os.system(
        'ParadiseApp.py ' + gal + '_vorbin_RSS.fits ' + gal + '_vorbin 6.0 --SSP_par parameters_stellar '
                                                              '--line_par parameters_eline --parallel ' +
        config.get('APS_cube', 'n_proc') + ' --verbose')

    print('')
    print('Running bootstrap models')

    os.system(
        'ParadiseApp.py ' + gal + '_vorbin_RSS.fits ' + gal + '_vorbin 6.0 --SSP_par parameters_stellar '
                                                              '--line_par parameters_eline --bootstraps 100 '
                                                              '--modkeep 80 --parallel ' +
        config.get('APS_cube', 'n_proc') + ' --verbose')

    os.system('mv ' + gal + '*.fits ' + gal_dir + '/.')
    os.system('cp excl* lines.fit par.lines parameters_eline parameters_stellar ' + gal_dir + '/.')

    print('This run took ' + str(round(time.time() - s_time, 2)) + ' secs')
