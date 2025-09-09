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
    
    gal_dir = blue_cube[0].header['CCNAME1'] + '_' + blue_cube[0].header['MODE'] + '_' + str(blue_cube[0].header['OBID'])

    # =================== running for red cube ===========================

    if int(config.get('pyp_params', 'red_fit')) == 1:

        res_dir = gal_dir + '/' + 'pyp_results/RED_' + datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        os.makedirs(res_dir, exist_ok=True)

        c = fits.open(gal_dir + '/red_cube_vorbin.fits')
        vorbin_map = fits.getdata(gal_dir + '/vorbin_map_red.fits')

        if int(config.get('spec_fit', 'resol_flag')) == 1:
            fwhm_str = gal_dir + '/resol_table_red.txt'
        if int(config.get('spec_fit', 'resol_flag')) == 0:
            fwhm_str = config.get('spec_fit', 'fwhm_red')

        # create RSS file

        wave = c[1].header['CRVAL3'] + (c[1].header['CDELT3'] * np.arange(c[1].header['NAXIS3']))

        rss_data = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), len(wave)), dtype=np.float32)
        rss_err = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), len(wave)), dtype=np.float32)

        for i in np.unique(vorbin_map[vorbin_map >= 0]):
            rss_data[int(i)] = c[1].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
            rss_err[int(i)] = c[2].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
            if int(config.get('spec_fit', 'cosm_flag')) == 1:
                diff = np.diff(rss_data[int(i)])
                limit = int(config.get('spec_fit', 'cosm_limit')) * np.nanstd(diff)
                for j in np.where(diff > limit)[0]:
                    rss_data[int(i)][j - 4:j + 5] = (np.nanmedian(rss_data[int(i)][j - 10:j - 5]) +
                                                     np.nanmedian(rss_data[int(i)][j + 5:j + 10])) / 2.

        rss_head = fits.Header()
        rss_head['SIMPLE'] = True
        rss_head['BITPIX'] = -32
        rss_head['NAXIS'] = 2
        rss_head['NAXIS1'] = rss_data.shape[1]
        rss_head['NAXIS2'] = rss_data.shape[0]
        rss_head['CTYPE1'] = 'WAVELENGTH'
        rss_head['CUNIT1'] = 'Angstrom'
        rss_head['CDELT1'] = c[1].header['CDELT3']
        rss_head['DISPAXIS'] = 1
        rss_head['CRVAL1'] = c[1].header['CRVAL3']
        rss_head['CRPIX1'] = c[1].header['CRPIX3']

        rss_ima = fits.HDUList([fits.PrimaryHDU(data=rss_data, header=rss_head),
                                fits.ImageHDU(data=rss_err, header=rss_head, name='ERROR')])

        rss_ima.writeto(gal + '_red_vorbin_RSS.fits', overwrite=True)

        print('')
        print('Running PyParadise best fit')

        if int(config.get('spec_fit', 'EL_flag')):
            os.system('ParadiseApp.py ' + gal + '_red_vorbin_RSS.fits ' + gal + '_red_vorbin ' +
                      fwhm_str + ' --SSP_par parameters_stellar_red --line_par parameters_eline_red --parallel ' +
                      config.get('APS_cube', 'n_proc') + ' --verbose')
        else:
            os.system('ParadiseApp.py ' + gal + '_red_vorbin_RSS.fits ' + gal + '_red_vorbin ' + fwhm_str +
                      ' --SSP_par parameters_stellar_red --parallel ' + config.get('APS_cube', 'n_proc') + ' --verbose')

        if int(config.get('spec_fit', 'boots_flag')):
            print('')
            print('Running bootstrap models')

            os.system(
                'ParadiseApp.py ' + gal + '_red_vorbin_RSS.fits ' + gal + '_red_vorbin ' + fwhm_str +
                ' --SSP_par parameters_stellar_red --line_par parameters_eline_red --bootstraps 100 --modkeep 80 '
                '--parallel ' + config.get('APS_cube', 'n_proc') + ' --verbose')

        os.system('mv ' + gal + '*.fits ' + res_dir + '/.')
        os.system('cp config_file.env excl_red* lines.fit par_red.lines parameters_eline_red parameters_stellar_red '
                  + res_dir + '/.')

    # =================== running for blue cube ===========================

    if int(config.get('pyp_params', 'blue_fit')) == 1:

        res_dir = gal_dir + '/' + 'pyp_results/BLUE_' + datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        os.makedirs(res_dir, exist_ok=True)

        c = fits.open(gal_dir + '/blue_cube_vorbin.fits')
        vorbin_map = fits.getdata(gal_dir + '/vorbin_map_blue.fits')

        if int(config.get('spec_fit', 'resol_flag')) == 1:
            fwhm_str = gal_dir + '/resol_table_blue.txt'
        if int(config.get('spec_fit', 'resol_flag')) == 0:
            fwhm_str = config.get('spec_fit', 'fwhm_blue')

        # create RSS file

        wave = c[1].header['CRVAL3'] + (c[1].header['CDELT3'] * np.arange(c[1].header['NAXIS3']))

        rss_data = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), len(wave)), dtype=np.float32)
        rss_err = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), len(wave)), dtype=np.float32)

        for i in np.unique(vorbin_map[vorbin_map >= 0]):
            rss_data[int(i)] = c[1].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
            rss_err[int(i)] = c[2].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
            if int(config.get('spec_fit', 'cosm_flag')) == 1:
                diff = np.diff(rss_data[int(i)])
                limit = int(config.get('spec_fit', 'cosm_limit')) * np.nanstd(diff)
                for j in np.where(diff > limit)[0]:
                    rss_data[int(i)][j - 4:j + 5] = (np.nanmedian(rss_data[int(i)][j - 10:j - 5]) +
                                                     np.nanmedian(rss_data[int(i)][j + 5:j + 10])) / 2.

        rss_head = fits.Header()
        rss_head['SIMPLE'] = True
        rss_head['BITPIX'] = -32
        rss_head['NAXIS'] = 2
        rss_head['NAXIS1'] = rss_data.shape[1]
        rss_head['NAXIS2'] = rss_data.shape[0]
        rss_head['CTYPE1'] = 'WAVELENGTH'
        rss_head['CUNIT1'] = 'Angstrom'
        rss_head['CDELT1'] = c[1].header['CDELT3']
        rss_head['DISPAXIS'] = 1
        rss_head['CRVAL1'] = c[1].header['CRVAL3']
        rss_head['CRPIX1'] = c[1].header['CRPIX3']

        rss_ima = fits.HDUList([fits.PrimaryHDU(data=rss_data, header=rss_head),
                                fits.ImageHDU(data=rss_err, header=rss_head, name='ERROR')])

        rss_ima.writeto(gal + '_blue_vorbin_RSS.fits', overwrite=True)

        print('')
        print('Running PyParadise best fit')

        if int(config.get('spec_fit', 'EL_flag')):
            os.system('ParadiseApp.py ' + gal + '_blue_vorbin_RSS.fits ' + gal + '_blue_vorbin ' +
                      fwhm_str + ' --SSP_par parameters_stellar_blue --line_par parameters_eline_blue --parallel ' +
                      config.get('APS_cube', 'n_proc') + ' --verbose')
        else:
            os.system('ParadiseApp.py ' + gal + '_blue_vorbin_RSS.fits ' + gal + '_blue_vorbin ' + fwhm_str +
                      ' --SSP_par parameters_stellar_blue --parallel ' + config.get('APS_cube', 'n_proc') + ' --verbose')

        if int(config.get('spec_fit', 'boots_flag')):
            print('')
            print('Running bootstrap models')

            os.system(
                'ParadiseApp.py ' + gal + '_blue_vorbin_RSS.fits ' + gal + '_blue_vorbin ' + fwhm_str +
                ' --SSP_par parameters_stellar_blue --line_par parameters_eline_blue --bootstraps 100 --modkeep 80 '
                '--parallel ' + config.get('APS_cube', 'n_proc') + ' --verbose')

        os.system('mv ' + gal + '*.fits ' + res_dir + '/.')
        os.system('cp config_file.env excl_blue* lines.fit par_blue.lines parameters_eline_blue parameters_stellar_blue '
                  + res_dir + '/.')

    # =================== running for APS ===========================

    if int(config.get('pyp_params', 'aps_fit')) == 1:

        res_dir = gal_dir + '/' + 'pyp_results/APS_' + datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        os.makedirs(res_dir, exist_ok=True)

        # c = fits.open(gal + '/APS_cube_vorbin.fits')
        # vorbin_map = fits.getdata(gal + '/vorbin_map_aps.fits')

        c = fits.open(gal_dir + '/' + gal + '_vorbin_cube.fits')
        vorbin_map = fits.getdata(gal_dir + '/vorbin_map.fits')

        if int(config.get('spec_fit', 'resol_flag')) == 1:
            fwhm_str = gal_dir + '/resol_table_aps.txt'
        if int(config.get('spec_fit', 'resol_flag')) == 0:
            fwhm_str = config.get('spec_fit', 'fwhm_APS')

        # create RSS file

        wave = c[0].header['CRVAL3'] + (c[0].header['CDELT3'] * np.arange(c[0].header['NAXIS3']))

        rss_data = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), len(wave)), dtype=np.float32)
        rss_err = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), len(wave)), dtype=np.float32)

        for i in np.unique(vorbin_map[vorbin_map >= 0]):
            rss_data[int(i)] = c[0].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
            rss_err[int(i)] = c[1].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
            if int(config.get('spec_fit', 'cosm_flag')) == 1:
                diff = np.diff(rss_data[int(i)])
                limit = int(config.get('spec_fit', 'cosm_limit')) * np.nanstd(diff)
                for j in np.where(diff > limit)[0]:
                    rss_data[int(i)][j - 4:j + 5] = (np.nanmedian(rss_data[int(i)][j - 10:j - 5]) +
                                                     np.nanmedian(rss_data[int(i)][j + 5:j + 10])) / 2.

        rss_head = fits.Header()
        rss_head['SIMPLE'] = True
        rss_head['BITPIX'] = -32
        rss_head['NAXIS'] = 2
        rss_head['NAXIS1'] = rss_data.shape[1]
        rss_head['NAXIS2'] = rss_data.shape[0]
        rss_head['CTYPE1'] = 'WAVELENGTH'
        rss_head['CUNIT1'] = 'Angstrom'
        rss_head['CDELT1'] = c[1].header['CDELT3']
        rss_head['DISPAXIS'] = 1
        rss_head['CRVAL1'] = c[1].header['CRVAL3']
        rss_head['CRPIX1'] = c[1].header['CRPIX3']

        rss_ima = fits.HDUList([fits.PrimaryHDU(data=rss_data, header=rss_head),
                                fits.ImageHDU(data=rss_err, header=rss_head, name='ERROR')])

        rss_ima.writeto(gal + '_APS_vorbin_RSS.fits', overwrite=True)

        print('')
        print('Running PyParadise best fit')

        if int(config.get('spec_fit', 'EL_flag')):
            os.system('ParadiseApp.py ' + gal + '_APS_vorbin_RSS.fits ' + gal + '_APS_vorbin ' + fwhm_str +
                      ' --SSP_par parameters_stellar_aps --line_par parameters_eline_aps --parallel ' +
                      config.get('APS_cube', 'n_proc') + ' --verbose')
        else:
            os.system('ParadiseApp.py ' + gal + '_APS_vorbin_RSS.fits ' + gal + '_APS_vorbin ' + fwhm_str +
                      ' --SSP_par parameters_stellar_aps --parallel ' + config.get('APS_cube', 'n_proc') + ' --verbose')

        if int(config.get('spec_fit', 'boots_flag')):
            print('')
            print('Running bootstrap models')

            os.system(
                'ParadiseApp.py ' + gal + '_APS_vorbin_RSS.fits ' + gal + '_APS_vorbin ' + fwhm_str +
                ' --SSP_par parameters_stellar_aps --line_par parameters_eline_aps --bootstraps 100 --modkeep 80 '
                '--parallel ' + config.get('APS_cube', 'n_proc') + ' --verbose')

        os.system('mv ' + gal + '*.fits ' + res_dir + '/.')
        os.system('cp ' + self + ' excl_aps* lines.fit par_aps.lines parameters_eline_aps parameters_stellar_aps '
                  + res_dir + '/.')
