import numpy as np
from astropy.io import fits
import os
from datetime import datetime
import configparser


def specs(ob, args):
    # config = configparser.ConfigParser()
    # config.read(self)

    file_dir = args.data_path + ob + '/'

    blue_cube = fits.open(file_dir + np.sort([x for x in os.listdir(file_dir) if ('stackcube' in x)])[1])

    gal = blue_cube[0].header['CCNAME1']

    # gal_dir = blue_cube[0].header['CCNAME1'] + '_' + blue_cube[0].header['MODE'] + '_' + str(
    #     blue_cube[0].header['OBID'])
    gal_dir = str(blue_cube[0].header['OBID']) + '_' + gal + '_' + blue_cube[0].header['MODE'] + '/'

    # =================== running for red cube ===========================

    if args.red_fit_flag == 1:

        res_dir = gal_dir + '/' + 'pyp_results/RED_' + datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        os.makedirs(res_dir, exist_ok=True)

        # create RSS file

        if args.vorbin_flag == 1:
            c = fits.open(gal_dir + '/red_cube_vorbin.fits')
            vorbin_map = fits.getdata(gal_dir + '/vorbin_map_red.fits')
            rss_data = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), c[1].data.shape[0]), dtype=np.float32)
            rss_err = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), c[1].data.shape[0]), dtype=np.float32)

            for i in np.unique(vorbin_map[vorbin_map >= 0]).astype(int):
                rss_data[i] = c[1].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
                rss_err[i] = c[2].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
                if args.sigmaclip_flag == 1:
                    diff = np.diff(rss_data[i])
                    limit = args.sigmaclip_limit * np.nanstd(diff)
                    for j in np.where(diff > limit)[0]:
                        rss_data[i][j - 4:j + 5] = (np.nanmedian(rss_data[i][j - 10:j - 5]) +
                                                    np.nanmedian(rss_data[i][j + 5:j + 10])) / 2.

        elif args.vorbin_flag == 0:
            c = fits.open(file_dir + np.sort([x for x in os.listdir(file_dir) if ('stackcube' in x)])[0])
            rss_data = c[1].data.reshape(c[1].data.shape[2] * c[1].data.shape[1],
                                         c[1].data.shape[0]) * np.mean(c[5].data[:], axis=0)
            rss_err = c[2].data.reshape(c[1].data.shape[2] * c[1].data.shape[1],
                                        c[1].data.shape[0]) * np.mean(c[5].data[:], axis=0)

            for i in np.arange(len(rss_data)):
                if args.sigmaclip_flag == 1:
                    diff = np.diff(rss_data[i])
                    limit = args.sigmaclip_limit * np.nanstd(diff)
                    for j in np.where(diff > limit)[0]:
                        rss_data[i][j - 4:j + 5] = (np.nanmedian(rss_data[i][j - 10:j - 5]) +
                                                    np.nanmedian(rss_data[i][j + 5:j + 10])) / 2.
        else:
            raise ValueError(f"Invalid input: {value}. Expected 0 or 1.")

        if args.resol_flag == 1:
            fwhm_str = gal_dir + '/resol_table_red.txt'
        if args.resol_flag == 0:
            fwhm_str = str(args.fwhm_red)

        rss_head = fits.Header()
        rss_head['SIMPLE'] = True
        rss_head['BITPIX'] = -32
        rss_head['NAXIS'] = 2
        rss_head['NAXIS1'] = rss_data.shape[1]
        rss_head['NAXIS2'] = rss_data.shape[0]
        rss_head['CTYPE1'] = 'WAVELENGTH'
        rss_head['CUNIT1'] = 'Angstrom'
        if args.vorbin_flag == 1:
            rss_head['CDELT1'] = c[1].header['CDELT3']
        if args.vorbin_flag == 0:
            rss_head['CDELT1'] = c[1].header['CD3_3']
        rss_head['DISPAXIS'] = 1
        rss_head['CRVAL1'] = c[1].header['CRVAL3']
        rss_head['CRPIX1'] = c[1].header['CRPIX3']

        rss_ima = fits.HDUList([fits.PrimaryHDU(data=rss_data, header=rss_head),
                                fits.ImageHDU(data=rss_err, header=rss_head, name='ERROR')])

        if args.vorbin_flag == 1:
            f_name = 'red_vorbin'
        if args.vorbin_flag == 0:
            f_name = 'red'

        rss_ima.writeto(gal + '_' + f_name + '_RSS.fits', overwrite=True)

        print('')
        print('Running PyParadise best fit')

        if args.el_flag == 1:
            os.system('ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' +
                      fwhm_str + ' --SSP_par parameters_stellar_red --line_par parameters_eline_red --parallel ' +
                      str(args.nproc) + ' --verbose')
        else:
            os.system('ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' + fwhm_str +
                      ' --SSP_par parameters_stellar_red --parallel ' + str(args.nproc) + ' --verbose')

        if args.boot_flag == 1:
            print('')
            print('Running bootstrap models')

            os.system(
                'ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' + fwhm_str +
                ' --SSP_par parameters_stellar_red --line_par parameters_eline_red --bootstraps 100 --modkeep 80 '
                '--parallel ' + str(args.nproc) + ' --verbose')

        os.system('mv ' + gal + '*.fits ' + res_dir + '/.')
        os.system('cp excl_red* lines_red.fit par_red.lines parameters_eline_red '
                                 'parameters_stellar_red ' + res_dir + '/.')


    # =================== running for blue cube ===========================

    if args.blue_fit_flag == 1:

        res_dir = gal_dir + '/' + 'pyp_results/BLUE_' + datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        os.makedirs(res_dir, exist_ok=True)

        if args.vorbin_flag == 1:
            c = fits.open(gal_dir + '/blue_cube_vorbin.fits')
            vorbin_map = fits.getdata(gal_dir + '/vorbin_map_blue.fits')
            rss_data = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), c[1].data.shape[0]), dtype=np.float32)
            rss_err = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), c[1].data.shape[0]), dtype=np.float32)
        if args.vorbin_flag == 0:
            c = fits.open(file_dir + np.sort([x for x in os.listdir(file_dir) if ('stackcube' in x)])[1])
            rss_data = blue_cube[1].data.reshape(blue_cube[1].data.shape[2] * blue_cube[1].data.shape[1],
                                                 blue_cube[1].data.shape[0]) * np.mean(blue_cube[5].data[:], axis=0)
            rss_err = blue_cube[2].data.reshape(blue_cube[1].data.shape[2] * blue_cube[1].data.shape[1],
                                                blue_cube[1].data.shape[0]) * np.mean(blue_cube[5].data[:], axis=0)

        if args.resol_flag == 1:
            fwhm_str = gal_dir + '/resol_table_blue.txt'
        if args.resol_flag == 0:
            fwhm_str = str(args.fwhm_blue)

        # create RSS file

        if args.vorbin_flag == 1:
            for i in np.unique(vorbin_map[vorbin_map >= 0]).astype(int):
                rss_data[i] = c[1].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
                rss_err[i] = c[2].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
                if args.sigmaclip_flag == 1:
                    diff = np.diff(rss_data[i])
                    limit = args.sigmaclip_limit * np.nanstd(diff)
                    for j in np.where(diff > limit)[0]:
                        rss_data[i][j - 4:j + 5] = (np.nanmedian(rss_data[i][j - 10:j - 5]) +
                                                    np.nanmedian(rss_data[i][j + 5:j + 10])) / 2.

        if args.vorbin_flag == 0:
            for i in np.arange(len(rss_data)):
                if args.sigmaclip_flag == 1:
                    diff = np.diff(rss_data[i])
                    limit = args.sigmaclip_limit * np.nanstd(diff)
                    for j in np.where(diff > limit)[0]:
                        rss_data[i][j - 4:j + 5] = (np.nanmedian(rss_data[i][j - 10:j - 5]) +
                                                    np.nanmedian(rss_data[i][j + 5:j + 10])) / 2.

        rss_head = fits.Header()
        rss_head['SIMPLE'] = True
        rss_head['BITPIX'] = -32
        rss_head['NAXIS'] = 2
        rss_head['NAXIS1'] = rss_data.shape[1]
        rss_head['NAXIS2'] = rss_data.shape[0]
        rss_head['CTYPE1'] = 'WAVELENGTH'
        rss_head['CUNIT1'] = 'Angstrom'
        if args.vorbin_flag == 1:
            rss_head['CDELT1'] = c[1].header['CDELT3']
        if args.vorbin_flag == 0:
            rss_head['CDELT1'] = c[1].header['CD3_3']
        rss_head['DISPAXIS'] = 1
        rss_head['CRVAL1'] = c[1].header['CRVAL3']
        rss_head['CRPIX1'] = c[1].header['CRPIX3']

        rss_ima = fits.HDUList([fits.PrimaryHDU(data=rss_data, header=rss_head),
                                fits.ImageHDU(data=rss_err, header=rss_head, name='ERROR')])

        if args.vorbin_flag == 1:
            f_name = 'blue_vorbin'
        if args.vorbin_flag == 0:
            f_name = 'blue'

        rss_ima.writeto(gal + '_' + f_name + '_RSS.fits', overwrite=True)

        print('')
        print('Running PyParadise best fit')

        if args.el_flag == 1:
            os.system('ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' +
                      fwhm_str + ' --SSP_par parameters_stellar_blue --line_par parameters_eline_blue --parallel ' +
                      str(args.nproc) + ' --verbose')
        else:
            os.system('ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' + fwhm_str +
                      ' --SSP_par parameters_stellar_blue --parallel ' + str(args.nproc) + ' --verbose')

        if args.boot_flag == 1:
            print('')
            print('Running bootstrap models')

            os.system(
                'ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' + fwhm_str +
                ' --SSP_par parameters_stellar_blue --line_par parameters_eline_blue --bootstraps 100 --modkeep 80 '
                '--parallel ' + str(args.nproc) + ' --verbose')

        os.system('mv ' + gal + '*.fits ' + res_dir + '/.')
        os.system('cp excl_blue* lines_blue.fit par_blue.lines parameters_eline_blue '
                                 'parameters_stellar_blue ' + res_dir + '/.')

    # =================== running for APS ===========================

    if args.aps_fit_flag == 1:

        res_dir = gal_dir + '/' + 'pyp_results/APS_' + datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        os.makedirs(res_dir, exist_ok=True)

        if args.vorbin_flag == 1:
            c = fits.open(gal_dir + '/' + gal + '_vorbin_cube.fits')
            vorbin_map = fits.getdata(gal_dir + '/vorbin_map.fits')
            rss_data = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), c[1].data.shape[0]), dtype=np.float32)
            rss_err = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), c[1].data.shape[0]), dtype=np.float32)

            for i in np.unique(vorbin_map[vorbin_map >= 0]).astype(int):
                rss_data[i] = c[1].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
                rss_err[i] = c[2].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
                if args.sigmaclip_flag == 1:
                    diff = np.diff(rss_data[i])
                    limit = args.sigmaclip_limit * np.nanstd(diff)
                    for j in np.where(diff > limit)[0]:
                        rss_data[i][j - 4:j + 5] = (np.nanmedian(rss_data[i][j - 10:j - 5]) +
                                                    np.nanmedian(rss_data[i][j + 5:j + 10])) / 2.

        elif args.vorbin_flag == 0:
            c = fits.open(gal_dir + '/' + gal + '_cube.fits')
            snr_map = fits.getdata(gal_dir + '/SNR_map_aps.fits')

            flux = c[1].data  # shape: (nlambda, ny, nx)
            err = c[2].data

            nl, ny, nx = flux.shape

            yy, xx = np.indices((ny, nx))

            flux_rss = flux.reshape(nl, ny * nx).T
            err_rss = err.reshape(nl, ny * nx).T

            snr_flat = snr_map.reshape(ny * nx)

            x_flat = xx.reshape(ny * nx)
            y_flat = yy.reshape(ny * nx)

            snr_min = 5.0
            mask = snr_flat >= snr_min

            rss_data = flux_rss[mask]
            rss_err = err_rss[mask]

            x_sel = x_flat[mask]
            y_sel = y_flat[mask]

            # rss_data = c[1].data.reshape(c[1].data.shape[2] * c[1].data.shape[1],
            #                              c[1].data.shape[0]) * np.mean(c[5].data[:], axis=0)
            # rss_err = c[2].data.reshape(c[1].data.shape[2] * c[1].data.shape[1],
            #                             c[1].data.shape[0]) * np.mean(c[5].data[:], axis=0)

            for i in np.arange(len(rss_data)):
                if args.sigmaclip_flag == 1:
                    diff = np.diff(rss_data[i])
                    limit = args.sigmaclip_limit * np.nanstd(diff)
                    for j in np.where(diff > limit)[0]:
                        rss_data[i][j - 4:j + 5] = (np.nanmedian(rss_data[i][j - 10:j - 5]) +
                                                    np.nanmedian(rss_data[i][j + 5:j + 10])) / 2.
        else:
            raise ValueError(f"Invalid input: {value}. Expected 0 or 1.")


        # c = fits.open(gal_dir + '/aps_cube_vorbin.fits')
        # vorbin_map = fits.getdata(gal_dir + '/vorbin_map_aps.fits')

        # c = fits.open(gal_dir + '/' + gal + '_vorbin_cube.fits')
        # vorbin_map = fits.getdata(gal_dir + '/vorbin_map.fits')

        if args.resol_flag == 1:
            fwhm_str = gal_dir + '/resol_table_aps.txt'
        if args.resol_flag == 0:
            fwhm_str = str(args.fwhm_aps)

        # create RSS file

        wave = c[1].header['CRVAL3'] + (c[1].header['CDELT3'] * np.arange(c[1].header['NAXIS3']))

        # rss_data = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), len(wave)), dtype=np.float32)
        # rss_err = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), len(wave)), dtype=np.float32)

        # rss_data = np.zeros((int(np.max(np.unique(vorbin_map[vorbin_map >= 0]))) + 1, len(wave)), dtype=np.float32)
        # rss_err = np.zeros((int(np.max(np.unique(vorbin_map[vorbin_map >= 0]))) + 1, len(wave)), dtype=np.float32)

        # for i in np.unique(vorbin_map[vorbin_map >= 0]).astype(int):
        #     rss_data[i] = c[1].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
        #     rss_err[i] = c[2].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
        #     if args.sigmaclip_flag == 1:
        #         diff = np.diff(rss_data[i])
        #         limit = args.sigmaclip_limit * np.nanstd(diff)
        #         for j in np.where(diff > limit)[0]:
        #             rss_data[i][j - 4:j + 5] = (np.nanmedian(rss_data[i][j - 10:j - 5]) +
        #                                         np.nanmedian(rss_data[i][j + 5:j + 10])) / 2.

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
        
        if args.vorbin_flag == 1:
            f_name = 'aps_vorbin'
        if args.vorbin_flag == 0:
            f_name = 'aps'

        rss_ima.writeto(gal + '_' + f_name + '_RSS.fits', overwrite=True)

        if args.vorbin_flag == 0:
            col_x = fits.Column(name='X', format='J', array=x_sel)
            col_y = fits.Column(name='Y', format='J', array=y_sel)

            coords_hdu = fits.BinTableHDU.from_columns([col_x, col_y], name='SPAXEL_COORDS')
            coords_hdu.writeto(gal + '_' + f_name + '_RSS_coords.fits', overwrite=True)

        print('')
        print('Running PyParadise best fit')

        breakpoint()

        if args.el_flag == 1:
            os.system('ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' + fwhm_str +
                      ' --SSP_par parameters_stellar_aps --line_par parameters_eline_aps --parallel ' +
                      str(args.nproc) + ' --verbose')
        else:
            os.system('ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' + fwhm_str +
                      ' --SSP_par parameters_stellar_aps --parallel ' + str(args.nproc) + ' --verbose')

        if args.boot_flag == 1:
            print('')
            print('Running bootstrap models')

            os.system(
                'ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' + fwhm_str +
                ' --SSP_par parameters_stellar_aps --line_par parameters_eline_aps --bootstraps 100 --modkeep 80 '
                '--parallel ' + str(args.nproc) + ' --verbose')

        os.system('mv ' + gal + '*.fits ' + res_dir + '/.')
        os.system('cp excl_aps* lines_aps.fit par_aps.lines parameters_eline_aps parameters_stellar_aps '
                  + res_dir + '/.')
