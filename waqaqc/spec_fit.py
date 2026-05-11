import numpy as np
from astropy.io import fits
import os
from datetime import datetime

# ============================================================
# Utilities
# ============================================================


def sigma_clip_spec(spec, limit_sigma):

    diff = np.diff(spec)
    limit = limit_sigma * np.nanstd(diff)

    for j in np.where(diff > limit)[0]:

        spec[j-4:j+5] = (
            np.nanmedian(spec[j-10:j-5]) +
            np.nanmedian(spec[j+5:j+10])
        ) / 2.

    return spec


def create_rss_from_cube(cube, snr_map=None, vorbin_map=None,
                         apply_sigmaclip=False,
                         sigmaclip_limit=5,
                         scale_flux=False):

    # ========================================================
    # Voronoi case
    # ========================================================

    if vorbin_map is not None:

        bins = np.unique(vorbin_map[vorbin_map >= 0]).astype(int)

        rss_data = np.zeros((len(bins), cube[1].data.shape[0]),
                            dtype=np.float32)

        rss_err = np.zeros_like(rss_data)

        for i in bins:

            yy, xx = np.where(vorbin_map == i)

            rss_data[i] = cube[1].data[:, yy[0], xx[0]]
            rss_err[i] = cube[2].data[:, yy[0], xx[0]]

            if apply_sigmaclip:
                rss_data[i] = sigma_clip_spec(
                    rss_data[i],
                    sigmaclip_limit
                )

        return rss_data, rss_err, None, None

    # ========================================================
    # Normal spaxel case
    # ========================================================

    flux = cube[1].data.copy()
    err = cube[2].data.copy()

    if scale_flux:
        sens = np.mean(cube[5].data[:], axis=0)
        flux *= sens
        err *= sens

    nl, ny, nx = flux.shape

    yy, xx = np.indices((ny, nx))

    flux_rss = flux.reshape(nl, ny * nx).T
    err_rss = err.reshape(nl, ny * nx).T

    snr_flat = snr_map.reshape(ny * nx)

    x_flat = xx.reshape(ny * nx)
    y_flat = yy.reshape(ny * nx)

    mask = snr_flat >= 5.0

    rss_data = flux_rss[mask]
    rss_err = err_rss[mask]

    x_sel = x_flat[mask]
    y_sel = y_flat[mask]

    if apply_sigmaclip:

        for i in range(len(rss_data)):

            rss_data[i] = sigma_clip_spec(
                rss_data[i],
                sigmaclip_limit
            )

    return rss_data, rss_err, x_sel, y_sel


def build_rss_header(cube, rss_data, vorbin_flag):

    h = fits.Header()

    h['SIMPLE'] = True
    h['BITPIX'] = -32
    h['NAXIS'] = 2
    h['NAXIS1'] = rss_data.shape[1]
    h['NAXIS2'] = rss_data.shape[0]
    h['CTYPE1'] = 'WAVELENGTH'
    h['CUNIT1'] = 'Angstrom'
    h['DISPAXIS'] = 1

    if vorbin_flag:
        h['CDELT1'] = cube[1].header['CDELT3']
    else:
        h['CDELT1'] = cube[1].header.get(
            'CD3_3',
            cube[1].header['CDELT3']
        )

    h['CRVAL1'] = cube[1].header['CRVAL3']
    h['CRPIX1'] = cube[1].header['CRPIX3']

    return h


# ============================================================
# Main mode runner
# ============================================================

def run_mode(mode, ob, args, gal, gal_dir, file_dir, stackcubes):

    print(f'\n====================')
    print(f'Running {mode.upper()}')
    print(f'====================')

    mode_cfg = {

        'red': {
            'fit_flag': args.red_fit_flag,
            'cube_file': stackcubes[0],
            'vorbin_cube': 'red_cube_vorbin.fits',
            'vorbin_map': 'vorbin_map_red.fits',
            'snr_map': 'SNR_map_red.fits',
            'fwhm': args.fwhm_red,
            'resol_table': 'resol_table_red.txt',
            'scale_flux': True
        },

        'blue': {
            'fit_flag': args.blue_fit_flag,
            'cube_file': stackcubes[1],
            'vorbin_cube': 'blue_cube_vorbin.fits',
            'vorbin_map': 'vorbin_map_blue.fits',
            'snr_map': 'SNR_map_blue.fits',
            'fwhm': args.fwhm_blue,
            'resol_table': 'resol_table_blue.txt',
            'scale_flux': True
        },

        'aps': {
            'fit_flag': args.aps_fit_flag,
            'cube_file': gal + '_cube.fits',
            'vorbin_cube': gal + '_vorbin_cube.fits',
            'vorbin_map': 'vorbin_map.fits',
            'snr_map': 'SNR_map_aps.fits',
            'fwhm': args.fwhm_aps,
            'resol_table': 'resol_table_aps.txt',
            'scale_flux': False
        }
    }

    cfg = mode_cfg[mode]

    if cfg['fit_flag'] != 1:
        return

    # ========================================================
    # Output directory
    # ========================================================

    res_dir = (
        gal_dir + '/pyp_results/' +
        mode.upper() + '_' +
        datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    )

    os.makedirs(res_dir, exist_ok=True)

    # ========================================================
    # Open cube
    # ========================================================

    if args.vorbin_flag:

        cube = fits.open(gal_dir + '/' + cfg['vorbin_cube'])

        vorbin_map = fits.getdata(
            gal_dir + '/' + cfg['vorbin_map']
        )

        rss_data, rss_err, x_sel, y_sel = create_rss_from_cube(
            cube,
            vorbin_map=vorbin_map,
            apply_sigmaclip=args.sigmaclip_flag,
            sigmaclip_limit=args.sigmaclip_limit
        )

        f_name = f'{mode}_vorbin'

    else:

        if mode in ['red', 'blue']:
            cube = fits.open(file_dir + cfg['cube_file'])
        else:
            cube = fits.open(gal_dir + '/' + cfg['cube_file'])

        snr_map = fits.getdata(
            gal_dir + '/' + cfg['snr_map']
        )

        rss_data, rss_err, x_sel, y_sel = create_rss_from_cube(
            cube,
            snr_map=snr_map,
            apply_sigmaclip=args.sigmaclip_flag,
            sigmaclip_limit=args.sigmaclip_limit,
            scale_flux=cfg['scale_flux']
        )

        f_name = mode

    # ========================================================
    # Resolution
    # ========================================================

    if args.resol_flag:
        fwhm_str = gal_dir + '/' + cfg['resol_table']
    else:
        fwhm_str = str(cfg['fwhm'])

    # ========================================================
    # Save RSS
    # ========================================================

    rss_head = build_rss_header(
        cube,
        rss_data,
        args.vorbin_flag
    )

    rss_hdul = fits.HDUList([

        fits.PrimaryHDU(
            data=rss_data,
            header=rss_head
        ),

        fits.ImageHDU(
            data=rss_err,
            header=rss_head,
            name='ERROR'
        )
    ])

    rss_name = f'{gal}_{f_name}_RSS.fits'

    rss_hdul.writeto(rss_name, overwrite=True)

    # ========================================================
    # Save coordinates
    # ========================================================

    if not args.vorbin_flag:

        coords_hdu = fits.BinTableHDU.from_columns([

            fits.Column(
                name='X',
                format='J',
                array=x_sel
            ),

            fits.Column(
                name='Y',
                format='J',
                array=y_sel
            )

        ], name='SPAXEL_COORDS')

        coords_hdu.writeto(
            f'{gal}_{f_name}_RSS_coords.fits',
            overwrite=True
        )

    # ========================================================
    # Run PyParadise
    # ========================================================

    cmd = (
        f'ParadiseApp.py '
        f'{rss_name} '
        f'{gal}_{f_name} '
        f'{fwhm_str} '
        f'--SSP_par parameters_stellar_{mode}_{ob} '
    )

    if args.el_flag:
        cmd += (
            f'--line_par parameters_eline_{mode}_{ob} '
        )

    cmd += (
        f'--parallel {args.nproc} '
        f'--verbose'
    )

    print('\nRunning PyParadise best fit')
    os.system(cmd)

    # ========================================================
    # Bootstrap
    # ========================================================

    if args.boot_flag:

        boot_cmd = (
            f'ParadiseApp.py '
            f'{rss_name} '
            f'{gal}_{f_name} '
            f'{fwhm_str} '
            f'--SSP_par parameters_stellar_{mode}_{ob} '
            f'--line_par parameters_eline_{mode}_{ob} '
            f'--bootstraps 100 '
            f'--modkeep 80 '
            f'--parallel {args.nproc} '
            f'--verbose'
        )

        print('\nRunning bootstrap models')
        os.system(boot_cmd)

    # ========================================================
    # Move files
    # ========================================================

    os.system(f'mv {gal}*.fits {res_dir}/.')

    os.system(
        f'mv excl_{mode}_{ob}* '
        f'lines_{mode}_{ob}.fit '
        f'par_{mode}_{ob}.lines '
        f'parameters_eline_{mode}_{ob} '
        f'parameters_stellar_{mode}_{ob} '
        f'{res_dir}/.'
    )


# ============================================================
# Main function
# ============================================================

def specs(ob, args):

    file_dir = args.data_path + ob + '/'

    stackcubes = np.sort([
        x for x in os.listdir(file_dir)
        if 'stackcube' in x
    ])

    blue_cube = fits.open(file_dir + stackcubes[1])

    gal = blue_cube[0].header['CCNAME1']

    gal_dir = (
        str(blue_cube[0].header['OBID']) + '_' +
        gal + '_' +
        blue_cube[0].header['MODE']
    )

    for mode in ['red', 'blue', 'aps']:

        run_mode(
            mode,
            ob,
            args,
            gal,
            gal_dir,
            file_dir,
            stackcubes
        )







#
#
# def specs(ob, args):
#
#     file_dir = args.data_path + ob + '/'
#
#     blue_cube = fits.open(file_dir + np.sort([x for x in os.listdir(file_dir) if ('stackcube' in x)])[1])
#
#     gal = blue_cube[0].header['CCNAME1']
#     gal_dir = str(blue_cube[0].header['OBID']) + '_' + gal + '_' + blue_cube[0].header['MODE'] + '/'
#
#     # =================== running for red cube ===========================
#
#     if args.red_fit_flag == 1:
#
#         res_dir = gal_dir + '/' + 'pyp_results/RED_' + datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
#         os.makedirs(res_dir, exist_ok=True)
#
#         # create RSS file
#
#         if args.vorbin_flag == 1:
#             c = fits.open(gal_dir + '/red_cube_vorbin.fits')
#             vorbin_map = fits.getdata(gal_dir + '/vorbin_map_red.fits')
#             rss_data = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), c[1].data.shape[0]), dtype=np.float32)
#             rss_err = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), c[1].data.shape[0]), dtype=np.float32)
#
#             for i in np.unique(vorbin_map[vorbin_map >= 0]).astype(int):
#                 rss_data[i] = c[1].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
#                 rss_err[i] = c[2].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
#                 if args.sigmaclip_flag == 1:
#                     diff = np.diff(rss_data[i])
#                     limit = args.sigmaclip_limit * np.nanstd(diff)
#                     for j in np.where(diff > limit)[0]:
#                         rss_data[i][j - 4:j + 5] = (np.nanmedian(rss_data[i][j - 10:j - 5]) +
#                                                     np.nanmedian(rss_data[i][j + 5:j + 10])) / 2.
#
#         elif args.vorbin_flag == 0:
#             c = fits.open(file_dir + np.sort([x for x in os.listdir(file_dir) if ('stackcube' in x)])[0])
#             snr_map = fits.getdata(gal_dir + '/SNR_map_red.fits')
#
#             flux = c[1].data * np.mean(c[5].data[:], axis=0) # shape: (nlambda, ny, nx)
#             err = c[2].data * np.mean(c[5].data[:], axis=0)
#
#             nl, ny, nx = flux.shape
#
#             yy, xx = np.indices((ny, nx))
#
#             flux_rss = flux.reshape(nl, ny * nx).T
#             err_rss = err.reshape(nl, ny * nx).T
#
#             snr_flat = snr_map.reshape(ny * nx)
#
#             x_flat = xx.reshape(ny * nx)
#             y_flat = yy.reshape(ny * nx)
#
#             snr_min = 5.0
#             mask = snr_flat >= snr_min
#
#             rss_data = flux_rss[mask]
#             rss_err = err_rss[mask]
#
#             x_sel = x_flat[mask]
#             y_sel = y_flat[mask]
#
#             for i in np.arange(len(rss_data)):
#                 if args.sigmaclip_flag == 1:
#                     diff = np.diff(rss_data[i])
#                     limit = args.sigmaclip_limit * np.nanstd(diff)
#                     for j in np.where(diff > limit)[0]:
#                         rss_data[i][j - 4:j + 5] = (np.nanmedian(rss_data[i][j - 10:j - 5]) +
#                                                     np.nanmedian(rss_data[i][j + 5:j + 10])) / 2.
#
#             # rss_data = c[1].data.reshape(c[1].data.shape[2] * c[1].data.shape[1],
#             #                              c[1].data.shape[0]) * np.mean(c[5].data[:], axis=0)
#             # rss_err = c[2].data.reshape(c[1].data.shape[2] * c[1].data.shape[1],
#             #                             c[1].data.shape[0]) * np.mean(c[5].data[:], axis=0)
#
#             # for i in np.arange(len(rss_data)):
#             #     if args.sigmaclip_flag == 1:
#             #         diff = np.diff(rss_data[i])
#             #         limit = args.sigmaclip_limit * np.nanstd(diff)
#             #         for j in np.where(diff > limit)[0]:
#             #             rss_data[i][j - 4:j + 5] = (np.nanmedian(rss_data[i][j - 10:j - 5]) +
#             #                                         np.nanmedian(rss_data[i][j + 5:j + 10])) / 2.
#         else:
#             raise ValueError(f"Invalid input: {args.vorbin_flag}. Expected 0 or 1.")
#
#         if args.resol_flag == 1:
#             fwhm_str = gal_dir + '/resol_table_red.txt'
#         if args.resol_flag == 0:
#             fwhm_str = str(args.fwhm_red)
#
#         rss_head = fits.Header()
#         rss_head['SIMPLE'] = True
#         rss_head['BITPIX'] = -32
#         rss_head['NAXIS'] = 2
#         rss_head['NAXIS1'] = rss_data.shape[1]
#         rss_head['NAXIS2'] = rss_data.shape[0]
#         rss_head['CTYPE1'] = 'WAVELENGTH'
#         rss_head['CUNIT1'] = 'Angstrom'
#         if args.vorbin_flag == 1:
#             rss_head['CDELT1'] = c[1].header['CDELT3']
#         if args.vorbin_flag == 0:
#             rss_head['CDELT1'] = c[1].header['CD3_3']
#         rss_head['DISPAXIS'] = 1
#         rss_head['CRVAL1'] = c[1].header['CRVAL3']
#         rss_head['CRPIX1'] = c[1].header['CRPIX3']
#
#         rss_ima = fits.HDUList([fits.PrimaryHDU(data=rss_data, header=rss_head),
#                                 fits.ImageHDU(data=rss_err, header=rss_head, name='ERROR')])
#
#         if args.vorbin_flag == 1:
#             f_name = 'red_vorbin'
#         if args.vorbin_flag == 0:
#             f_name = 'red'
#
#         rss_ima.writeto(gal + '_' + f_name + '_RSS.fits', overwrite=True)
#
#         if args.vorbin_flag == 0:
#             col_x = fits.Column(name='X', format='J', array=x_sel)
#             col_y = fits.Column(name='Y', format='J', array=y_sel)
#
#             coords_hdu = fits.BinTableHDU.from_columns([col_x, col_y], name='SPAXEL_COORDS')
#             coords_hdu.writeto(gal + '_' + f_name + '_RSS_coords.fits', overwrite=True)
#
#         print('')
#         print('Running PyParadise best fit')
#
#         if args.el_flag == 1:
#             os.system('ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' +
#                       fwhm_str + ' --SSP_par parameters_stellar_red_'+ob+' --line_par parameters_eline_red_'+ob+''
#                       ' --parallel ' + str(args.nproc) + ' --verbose')
#         else:
#             os.system('ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' + fwhm_str +
#                       ' --SSP_par parameters_stellar_red_'+ob+' --parallel ' + str(args.nproc) + ' --verbose')
#
#         if args.boot_flag == 1:
#             print('')
#             print('Running bootstrap models')
#
#             os.system(
#                 'ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' + fwhm_str +
#                 ' --SSP_par parameters_stellar_red_'+ob+' --line_par parameters_eline_red_'+ob+' --bootstraps 100 '
#                 '--modkeep 80 --parallel ' + str(args.nproc) + ' --verbose')
#
#         os.system('mv ' + gal + '*.fits ' + res_dir + '/.')
#         os.system('mv excl_red_'+ob+'* lines_red_'+ob+'.fit par_red_'+ob+'.lines parameters_eline_red_'+ob+' '
#                   'parameters_stellar_red_'+ob+' ' + res_dir + '/.')
#
#     # =================== running for blue cube ===========================
#
#     if args.blue_fit_flag == 1:
#
#         res_dir = gal_dir + '/' + 'pyp_results/BLUE_' + datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
#         os.makedirs(res_dir, exist_ok=True)
#
#         # create RSS file
#
#         if args.vorbin_flag == 1:
#             c = fits.open(gal_dir + '/blue_cube_vorbin.fits')
#             vorbin_map = fits.getdata(gal_dir + '/vorbin_map_blue.fits')
#             rss_data = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), c[1].data.shape[0]), dtype=np.float32)
#             rss_err = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), c[1].data.shape[0]), dtype=np.float32)
#
#             for i in np.unique(vorbin_map[vorbin_map >= 0]).astype(int):
#                 rss_data[i] = c[1].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
#                 rss_err[i] = c[2].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
#                 if args.sigmaclip_flag == 1:
#                     diff = np.diff(rss_data[i])
#                     limit = args.sigmaclip_limit * np.nanstd(diff)
#                     for j in np.where(diff > limit)[0]:
#                         rss_data[i][j - 4:j + 5] = (np.nanmedian(rss_data[i][j - 10:j - 5]) +
#                                                     np.nanmedian(rss_data[i][j + 5:j + 10])) / 2.
#
#         elif args.vorbin_flag == 0:
#             c = fits.open(file_dir + np.sort([x for x in os.listdir(file_dir) if ('stackcube' in x)])[1])
#             snr_map = fits.getdata(gal_dir + '/SNR_map_blue.fits')
#
#             flux = c[1].data * np.mean(c[5].data[:], axis=0) # shape: (nlambda, ny, nx)
#             err = c[2].data * np.mean(c[5].data[:], axis=0)
#
#             nl, ny, nx = flux.shape
#
#             yy, xx = np.indices((ny, nx))
#
#             flux_rss = flux.reshape(nl, ny * nx).T
#             err_rss = err.reshape(nl, ny * nx).T
#
#             snr_flat = snr_map.reshape(ny * nx)
#
#             x_flat = xx.reshape(ny * nx)
#             y_flat = yy.reshape(ny * nx)
#
#             snr_min = 5.0
#             mask = snr_flat >= snr_min
#
#             rss_data = flux_rss[mask]
#             rss_err = err_rss[mask]
#
#             x_sel = x_flat[mask]
#             y_sel = y_flat[mask]
#
#             for i in np.arange(len(rss_data)):
#                 if args.sigmaclip_flag == 1:
#                     diff = np.diff(rss_data[i])
#                     limit = args.sigmaclip_limit * np.nanstd(diff)
#                     for j in np.where(diff > limit)[0]:
#                         rss_data[i][j - 4:j + 5] = (np.nanmedian(rss_data[i][j - 10:j - 5]) +
#                                                     np.nanmedian(rss_data[i][j + 5:j + 10])) / 2.
#
#         if args.resol_flag == 1:
#             fwhm_str = gal_dir + '/resol_table_blue.txt'
#         if args.resol_flag == 0:
#             fwhm_str = str(args.fwhm_blue)
#
#         rss_head = fits.Header()
#         rss_head['SIMPLE'] = True
#         rss_head['BITPIX'] = -32
#         rss_head['NAXIS'] = 2
#         rss_head['NAXIS1'] = rss_data.shape[1]
#         rss_head['NAXIS2'] = rss_data.shape[0]
#         rss_head['CTYPE1'] = 'WAVELENGTH'
#         rss_head['CUNIT1'] = 'Angstrom'
#         if args.vorbin_flag == 1:
#             rss_head['CDELT1'] = c[1].header['CDELT3']
#         if args.vorbin_flag == 0:
#             rss_head['CDELT1'] = c[1].header['CD3_3']
#         rss_head['DISPAXIS'] = 1
#         rss_head['CRVAL1'] = c[1].header['CRVAL3']
#         rss_head['CRPIX1'] = c[1].header['CRPIX3']
#
#         rss_ima = fits.HDUList([fits.PrimaryHDU(data=rss_data, header=rss_head),
#                                 fits.ImageHDU(data=rss_err, header=rss_head, name='ERROR')])
#
#         if args.vorbin_flag == 1:
#             f_name = 'blue_vorbin'
#         if args.vorbin_flag == 0:
#             f_name = 'blue'
#
#         rss_ima.writeto(gal + '_' + f_name + '_RSS.fits', overwrite=True)
#
#         if args.vorbin_flag == 0:
#             col_x = fits.Column(name='X', format='J', array=x_sel)
#             col_y = fits.Column(name='Y', format='J', array=y_sel)
#
#             coords_hdu = fits.BinTableHDU.from_columns([col_x, col_y], name='SPAXEL_COORDS')
#             coords_hdu.writeto(gal + '_' + f_name + '_RSS_coords.fits', overwrite=True)
#
#         print('')
#         print('Running PyParadise best fit')
#
#         if args.el_flag == 1:
#             os.system('ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' +
#                       fwhm_str + ' --SSP_par parameters_stellar_blue_'+ob+' --line_par parameters_eline_blue_'+ob+''
#                       ' --parallel ' + str(args.nproc) + ' --verbose')
#         else:
#             os.system('ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' + fwhm_str +
#                       ' --SSP_par parameters_stellar_blue_'+ob+' --parallel ' + str(args.nproc) + ' --verbose')
#
#         if args.boot_flag == 1:
#             print('')
#             print('Running bootstrap models')
#
#             os.system(
#                 'ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' + fwhm_str +
#                 ' --SSP_par parameters_stellar_blue_'+ob+' --line_par parameters_eline_blue_'+ob+' --bootstraps 100 '
#                 '--modkeep 80 --parallel ' + str(args.nproc) + ' --verbose')
#
#         os.system('mv ' + gal + '*.fits ' + res_dir + '/.')
#         os.system('mv excl_blue_'+ob+'* lines_blue_'+ob+'.fit par_blue_'+ob+'.lines parameters_eline_blue_'+ob+' '
#                   'parameters_stellar_blue_'+ob+' ' + res_dir + '/.')
#
#     # =================== running for APS ===========================
#
#     if args.aps_fit_flag == 1:
#
#         res_dir = gal_dir + '/' + 'pyp_results/APS_' + datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
#         os.makedirs(res_dir, exist_ok=True)
#
#         # create RSS file
#
#         if args.vorbin_flag == 1:
#             c = fits.open(gal_dir + '/' + gal + '_vorbin_cube.fits')
#             vorbin_map = fits.getdata(gal_dir + '/vorbin_map.fits')
#             rss_data = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), c[1].data.shape[0]), dtype=np.float32)
#             rss_err = np.zeros((len(np.unique(vorbin_map[vorbin_map >= 0])), c[1].data.shape[0]), dtype=np.float32)
#
#             for i in np.unique(vorbin_map[vorbin_map >= 0]).astype(int):
#                 rss_data[i] = c[1].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
#                 rss_err[i] = c[2].data[:, np.where(vorbin_map == i)[0][0], np.where(vorbin_map == i)[1][0]]
#                 if args.sigmaclip_flag == 1:
#                     diff = np.diff(rss_data[i])
#                     limit = args.sigmaclip_limit * np.nanstd(diff)
#                     for j in np.where(diff > limit)[0]:
#                         rss_data[i][j - 4:j + 5] = (np.nanmedian(rss_data[i][j - 10:j - 5]) +
#                                                     np.nanmedian(rss_data[i][j + 5:j + 10])) / 2.
#
#         elif args.vorbin_flag == 0:
#             c = fits.open(gal_dir + '/' + gal + '_cube.fits')
#             snr_map = fits.getdata(gal_dir + '/SNR_map_aps.fits')
#
#             flux = c[1].data  # shape: (nlambda, ny, nx)
#             err = c[2].data
#
#             nl, ny, nx = flux.shape
#
#             yy, xx = np.indices((ny, nx))
#
#             flux_rss = flux.reshape(nl, ny * nx).T
#             err_rss = err.reshape(nl, ny * nx).T
#
#             snr_flat = snr_map.reshape(ny * nx)
#
#             x_flat = xx.reshape(ny * nx)
#             y_flat = yy.reshape(ny * nx)
#
#             snr_min = 5.0
#             mask = snr_flat >= snr_min
#
#             rss_data = flux_rss[mask]
#             rss_err = err_rss[mask]
#
#             x_sel = x_flat[mask]
#             y_sel = y_flat[mask]
#
#             for i in np.arange(len(rss_data)):
#                 if args.sigmaclip_flag == 1:
#                     diff = np.diff(rss_data[i])
#                     limit = args.sigmaclip_limit * np.nanstd(diff)
#                     for j in np.where(diff > limit)[0]:
#                         rss_data[i][j - 4:j + 5] = (np.nanmedian(rss_data[i][j - 10:j - 5]) +
#                                                     np.nanmedian(rss_data[i][j + 5:j + 10])) / 2.
#         else:
#             raise ValueError(f"Invalid input: {args.vorbin_flag}. Expected 0 or 1.")
#
#         if args.resol_flag == 1:
#             fwhm_str = gal_dir + '/resol_table_aps.txt'
#         if args.resol_flag == 0:
#             fwhm_str = str(args.fwhm_aps)
#
#         rss_head = fits.Header()
#         rss_head['SIMPLE'] = True
#         rss_head['BITPIX'] = -32
#         rss_head['NAXIS'] = 2
#         rss_head['NAXIS1'] = rss_data.shape[1]
#         rss_head['NAXIS2'] = rss_data.shape[0]
#         rss_head['CTYPE1'] = 'WAVELENGTH'
#         rss_head['CUNIT1'] = 'Angstrom'
#         rss_head['CDELT1'] = c[1].header['CDELT3']
#         rss_head['DISPAXIS'] = 1
#         rss_head['CRVAL1'] = c[1].header['CRVAL3']
#         rss_head['CRPIX1'] = c[1].header['CRPIX3']
#
#         rss_ima = fits.HDUList([fits.PrimaryHDU(data=rss_data, header=rss_head),
#                                 fits.ImageHDU(data=rss_err, header=rss_head, name='ERROR')])
#
#         if args.vorbin_flag == 1:
#             f_name = 'aps_vorbin'
#         if args.vorbin_flag == 0:
#             f_name = 'aps'
#
#         rss_ima.writeto(gal + '_' + f_name + '_RSS.fits', overwrite=True)
#
#         if args.vorbin_flag == 0:
#             col_x = fits.Column(name='X', format='J', array=x_sel)
#             col_y = fits.Column(name='Y', format='J', array=y_sel)
#
#             coords_hdu = fits.BinTableHDU.from_columns([col_x, col_y], name='SPAXEL_COORDS')
#             coords_hdu.writeto(gal + '_' + f_name + '_RSS_coords.fits', overwrite=True)
#
#         print('')
#         print('Running PyParadise best fit')
#
#         if args.el_flag == 1:
#             os.system('ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' + fwhm_str +
#                       ' --SSP_par parameters_stellar_aps_'+ob+' --line_par parameters_eline_aps_'+ob+' --parallel ' +
#                       str(args.nproc) + ' --verbose')
#         else:
#             os.system('ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' + fwhm_str +
#                       ' --SSP_par parameters_stellar_aps_'+ob+' --parallel ' + str(args.nproc) + ' --verbose')
#
#         if args.boot_flag == 1:
#             print('')
#             print('Running bootstrap models')
#
#             os.system(
#                 'ParadiseApp.py ' + gal + '_' + f_name + '_RSS.fits ' + gal + '_' + f_name + ' ' + fwhm_str +
#                 ' --SSP_par parameters_stellar_aps_'+ob+' --line_par parameters_eline_aps_'+ob+' '
#                 '--bootstraps 100 --modkeep 80 --parallel ' + str(args.nproc) + ' --verbose')
#
#         os.system('mv ' + gal + '*.fits ' + res_dir + '/.')
#         os.system('mv excl_aps_'+ob+'* lines_aps_'+ob+'.fit par_aps_'+ob+'.lines parameters_eline_aps_'+ob+' '
#                   'parameters_stellar_aps_'+ob+' '+res_dir+'/.')
