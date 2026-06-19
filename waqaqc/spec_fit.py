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
        spec[j - 4:j + 5] = (
                                    np.nanmedian(spec[j - 10:j - 5]) +
                                    np.nanmedian(spec[j + 5:j + 10])
                            ) / 2.

    return spec


def create_rss_from_cube(cube, vorbin_map=None,
                         apply_sigmaclip=False,
                         sigmaclip_limit=5):
    # ========================================================
    # Voronoi case
    # ========================================================

    if vorbin_map is not None:

        bins = np.unique(vorbin_map[vorbin_map >= 0]).astype(int)

        rss_data = np.zeros((len(bins), cube[1].data.shape[0]),
                            dtype=np.float32)

        rss_err = np.zeros_like(rss_data)

        bin_coords = {
            b: np.argwhere(vorbin_map == b)[0]
            for b in bins
        }

        for k, bin_id in enumerate(bins):
            y, x = bin_coords[bin_id]

            sens = cube[5].data

            rss_data[k] = cube[1].data[:, y, x]
            # rss_err[k] = cube[2].data[:, y, x]

            rss_data[k] *= sens
            # rss_err[k] *= sens

            ivar = cube[2].data[:, y, x]

            ivar /= sens ** 2

            rss_err[k] = np.where(
                ivar > 0,
                1.0 / np.sqrt(ivar),
                np.nan
            )

            rss_data *= 1e20
            rss_err *= 1e20

            if apply_sigmaclip:
                rss_data[k] = sigma_clip_spec(
                    rss_data[k],
                    sigmaclip_limit
                )

        return rss_data, rss_err, None, None


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
        if 'CD3_3' in cube[1].header:
            h['CDELT1'] = cube[1].header['CD3_3']
        else:
            h['CDELT1'] = cube[1].header['CDELT3']

    h['CRVAL1'] = cube[1].header['CRVAL3']
    h['CRPIX1'] = cube[1].header['CRPIX3']

    return h


# ============================================================
# Main mode runner
# ============================================================

def run_mode(mode, ob, args, gal, gal_dir, file_dir, stackcubes):
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
    else:
        print(f'\n====================')
        print(f'Running {mode.upper()}')
        print(f'====================')

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

    cube_name = f'{gal}_{mode}_cube.fits'

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

        flux = cube[1].data.copy()
        err = cube[2].data.copy()
        # ivar = cube[2].data.copy()

        if cfg['scale_flux']:
            sens = cube[5].data

            flux *= sens[:, None, None]
            err *= sens[:, None, None]

            # ivar /= sens[:, None, None] ** 2
            #
            # err = np.where(
            #     ivar > 0,
            #     1.0 / np.sqrt(ivar),
            #     np.nan
            # )
            flux *= 1e20
            err *= 1e20

        # Optional sigma clipping
        if args.sigmaclip_flag:

            for y in range(flux.shape[1]):

                for x in range(flux.shape[2]):
                    flux[:, y, x] = sigma_clip_spec(
                        flux[:, y, x],
                        args.sigmaclip_limit
                    )

        flux_header = cube[1].header.copy()
        err_header = cube[2].header.copy()

        if 'CDELT3' not in flux_header and 'CD3_3' in flux_header:
            flux_header['CDELT3'] = flux_header['CD3_3']

        if 'CDELT3' not in err_header and 'CD3_3' in err_header:
            err_header['CDELT3'] = err_header['CD3_3']

        flux_header['FLUX_NORM'] = 1e-20
        err_header['FLUX_NORM'] = 1e-20

        cube_hdul = fits.HDUList([

            fits.PrimaryHDU(
                data=flux,
                header=flux_header
            ),

            fits.ImageHDU(
                data=err,
                header=err_header,
                name='ERROR'
            )
        ])

        cube_hdul.writeto(cube_name, overwrite=True)

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

    if args.vorbin_flag:

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

        input_name = f'{gal}_{f_name}_RSS.fits'

        rss_hdul.writeto(input_name, overwrite=True)

    else:

        input_name = cube_name

    # ========================================================
    # Run PyParadise
    # ========================================================

    cmd = (
        f'ParadiseApp.py '
        f'{input_name} '
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
            f'{input_name} '
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
