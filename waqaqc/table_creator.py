import numpy as np
from astropy.io import fits
import os
from astropy.table import Table


def fill_cube(vorbin_flag, bins, vorbin_map, contm_file, contr_file,
              eline, shape, coords):

    nl, ny, nx = shape

    contm_data = np.zeros((nl, ny, nx))
    contm_err = np.zeros_like(contm_data)
    contm_badp = np.zeros_like(contm_data)
    contm_norm = np.zeros_like(contm_data)
    contr_data = np.zeros_like(contm_data)

    if eline:
        elinm_data = np.zeros_like(contm_data)
        elinr_data = np.zeros_like(contm_data)

    if vorbin_flag:

        for i, b in enumerate(bins):
            mask = (vorbin_map.T == b)

            contm_data.T[mask] = contm_file[0].data[i]
            contm_err.T[mask] = contm_file[1].data[i]
            contm_badp.T[mask] = contm_file[2].data[i]
            contm_norm.T[mask] = contm_file[3].data[i]
            contr_data.T[mask] = contr_file[0].data[i]

            if eline:
                elinm_data.T[mask] = eline["model"][0].data[i]
                elinr_data.T[mask] = eline["res"][0].data[i]

    else:

        x, y = coords["X"], coords["Y"]

        for i in range(len(x)):
            contm_data[:, y[i], x[i]] = contm_file[0].data[i]
            contm_err[:, y[i], x[i]] = contm_file[1].data[i]
            contm_badp[:, y[i], x[i]] = contm_file[2].data[i]
            contm_norm[:, y[i], x[i]] = contm_file[3].data[i]
            contr_data[:, y[i], x[i]] = contr_file[0].data[i]

            if eline:
                elinm_data[:, y[i], x[i]] = eline["model"][0].data[i]
                elinr_data[:, y[i], x[i]] = eline["res"][0].data[i]

    if eline is not None:

        return contm_data, contm_err, contm_badp, contm_norm, contr_data, elinm_data, elinr_data

    else:

        return contm_data, contm_err, contm_badp, contm_norm, contr_data


def write_outputs(
        res_dir,
        gal,
        mode,
        cube_header,
        map_header,
        contm_data,
        contm_err,
        contm_badp,
        contm_norm,
        contr_data,
        stelt_file,
        tables_maps,
        args,
        elinm_data=None,
        elinr_data=None,
        elint_file=None
):

    # ========================================================
    # Continuum products
    # ========================================================

    n_contm = fits.HDUList([

        fits.PrimaryHDU(
            data=contm_data,
            header=cube_header
        ),

        fits.ImageHDU(
            data=contm_err,
            header=cube_header,
            name='ERROR'
        ),

        fits.ImageHDU(
            data=contm_badp,
            header=cube_header,
            name='BADPIX'
        ),

        fits.ImageHDU(
            data=contm_norm,
            header=cube_header,
            name='NORMALIZE'
        )
    ])

    n_contr = fits.HDUList([

        fits.PrimaryHDU(
            data=contr_data,
            header=cube_header
        )
    ])

    # ========================================================
    # Stellar tables
    # ========================================================

    n_tab_stell = fits.HDUList([

        stelt_file[0].copy(),

        fits.BinTableHDU(
            tables_maps['tab_st'],
            header=stelt_file[1].header
        )
    ])

    # ========================================================
    # Stellar maps
    # ========================================================

    hdu_stelt_maps = fits.HDUList([

        fits.PrimaryHDU()
    ])

    for i, name in enumerate(tables_maps['stelt_maps_n']):

        hdu_stelt_maps.append(

            fits.ImageHDU(
                data=tables_maps['stelt_maps'][i],
                name=name,
                header=map_header
            )
        )

    # ========================================================
    # Base coeff maps
    # ========================================================

    hdu_base_coeff_maps = fits.HDUList([

        fits.PrimaryHDU(),

        fits.BinTableHDU(
            tables_maps['base_coeff_t']
        )
    ])

    for i in range(len(tables_maps['base_coeff_maps'])):

        hdu_base_coeff_maps.append(

            fits.ImageHDU(
                data=tables_maps['base_coeff_maps'][i],
                name=f'Template {i}',
                header=map_header
            )
        )

    # ========================================================
    # Save stellar products
    # ========================================================

    n_contm.writeto(
        f'{res_dir}/{gal}_{mode}_cont_model.fits',
        overwrite=True
    )

    n_contr.writeto(
        f'{res_dir}/{gal}_{mode}_cont_res.fits',
        overwrite=True
    )

    n_tab_stell.writeto(
        f'{res_dir}/{gal}_{mode}_stellar_table.fits',
        overwrite=True
    )

    hdu_stelt_maps.writeto(
        f'{res_dir}/{gal}_{mode}_stellar_maps.fits',
        overwrite=True
    )

    hdu_base_coeff_maps.writeto(
        f'{res_dir}/{gal}_{mode}_base_coeff_maps.fits',
        overwrite=True
    )

    # ========================================================
    # Emission line products
    # ========================================================

    if args.el_flag:

        n_elinm = fits.HDUList([

            fits.PrimaryHDU(
                data=elinm_data,
                header=cube_header
            )
        ])

        n_elinr = fits.HDUList([

            fits.PrimaryHDU(
                data=elinr_data,
                header=cube_header
            )
        ])

        n_tab_eline = fits.HDUList([

            elint_file[0].copy(),

            fits.BinTableHDU(
                tables_maps['tab_el'],
                header=elint_file[1].header
            )
        ])

        hdu_elint_maps = fits.HDUList([

            fits.PrimaryHDU()
        ])

        for i, name in enumerate(
                tables_maps['elint_maps_n']):

            hdu_elint_maps.append(

                fits.ImageHDU(
                    data=tables_maps['elint_maps'][i],
                    name=name,
                    header=map_header
                )
            )

        n_elinm.writeto(
            f'{res_dir}/{gal}_{mode}_eline_model.fits',
            overwrite=True
        )

        n_elinr.writeto(
            f'{res_dir}/{gal}_{mode}_eline_res.fits',
            overwrite=True
        )

        n_tab_eline.writeto(
            f'{res_dir}/{gal}_{mode}_eline_table.fits',
            overwrite=True
        )

        hdu_elint_maps.writeto(
            f'{res_dir}/{gal}_{mode}_eline_maps.fits',
            overwrite=True
        )


def process_tables_and_maps(
        args,
        bins,
        vorbin_map,
        stelt_file,
        elint_file,
        stel_template,
        coords=None
):

    # ========================================================
    # Load tables
    # ========================================================

    stelt_t = Table(stelt_file[1].data)

    if args.el_flag:
        elint_t = Table(elint_file[1].data)

        # remove duplicated fiber=0 rows
        fiber_zero = np.where(elint_t['fiber'] == 0)[0]

        if len(fiber_zero) > 1:
            mask = np.ones(len(elint_t), dtype=bool)
            mask[fiber_zero[1:]] = False
            elint_t = elint_t[mask]

        tab_el = elint_t.copy()

    tab_st = stelt_t.copy()

    # ========================================================
    # Map names
    # ========================================================

    stelt_maps_n = list(stelt_file[1].data.names)
    stelt_maps_n.remove('fiber')
    stelt_maps_n.remove('base_coeff')

    if args.el_flag:
        elint_maps_n = list(elint_file[1].data.names)
        elint_maps_n.remove('fiber')

    # ========================================================
    # Create empty maps
    # ========================================================

    ny, nx = vorbin_map.shape

    stelt_maps = np.full(
        (len(stelt_maps_n), ny, nx),
        np.nan
    )

    if args.el_flag:
        elint_maps = np.full(
            (len(elint_maps_n), ny, nx),
            np.nan
        )

    base_coeff_t = Table(stel_template[1].data)

    base_coeff_maps = np.full(
        (len(base_coeff_t), ny, nx),
        np.nan
    )

    # ========================================================
    # Coordinates
    # ========================================================

    tx = []
    ty = []

    # ========================================================
    # Fill maps
    # ========================================================

    if args.vorbin_flag:

        for row_idx, fiber_id in enumerate(bins):

            mask = (vorbin_map == fiber_id)

            yy, xx = np.where(mask)

            tx.append(xx)
            ty.append(yy)

            # stellar maps
            if np.sum(stelt_t['fiber'] == row_idx):

                for i, name in enumerate(stelt_maps_n):
                    value = stelt_t[name][
                        stelt_t['fiber'] == row_idx
                        ][0]

                    stelt_maps[i][mask] = value

                coeffs = stelt_t['base_coeff'][
                    stelt_t['fiber'] == row_idx
                    ][0]

                for i in range(len(coeffs)):
                    base_coeff_maps[i][mask] = coeffs[i]

            # emission-line maps
            if args.el_flag:

                if np.sum(elint_t['fiber'] == row_idx):

                    for i, name in enumerate(elint_maps_n):
                        value = elint_t[name][
                            elint_t['fiber'] == row_idx
                            ][0]

                        elint_maps[i][mask] = value

    else:

        for row_idx in range(len(stelt_t)):

            x = int(stelt_t['x_cor'][row_idx])
            y = int(stelt_t['y_cor'][row_idx])

            tx.append(x)
            ty.append(y)

            for i, name in enumerate(stelt_maps_n):
                stelt_maps[i][y, x] = \
                    stelt_t[name][row_idx]

            coeffs = stelt_t['base_coeff'][row_idx]

            for i in range(len(coeffs)):
                base_coeff_maps[i][y, x] = coeffs[i]

            if args.el_flag:

                for i, name in enumerate(elint_maps_n):
                    elint_maps[i][y, x] = \
                        elint_t[name][row_idx]

    # ========================================================
    # Expand Voronoi tables
    # ========================================================

    if args.vorbin_flag:

        for row_idx, fiber_id in enumerate(bins):

            n_pix = np.sum(vorbin_map == fiber_id)

            for _ in range(n_pix - 1):

                if np.sum(tab_st['fiber'] == row_idx):
                    tab_st.add_row(
                        tab_st[tab_st['fiber'] == row_idx][0]
                    )

                if args.el_flag:

                    if np.sum(tab_el['fiber'] == row_idx):

                        tab_el.add_row(
                            tab_el[tab_el['fiber'] == row_idx][0]
                        )

    # ========================================================
    # Sort tables
    # ========================================================

    tab_st = tab_st[tab_st.argsort(['fiber'])]

    if args.el_flag:
        tab_el = tab_el[tab_el.argsort(['fiber'])]

    # ========================================================
    # Add coordinates
    # ========================================================

    if args.vorbin_flag:
        ttx = np.concatenate(tx)
        tty = np.concatenate(ty)
    else:
        ttx = tx
        tty = ty

    if 'x_cor' not in tab_st.colnames:
        tab_st.add_column(ttx, name='x_cor', index=0)
    if 'y_cor' not in tab_st.colnames:
        tab_st.add_column(tty, name='y_cor', index=1)

    if args.el_flag:
        if 'x_cor' not in tab_el.colnames:
            tab_el.add_column(ttx, name='x_cor', index=0)
        if 'y_cor' not in tab_el.colnames:
            tab_el.add_column(tty, name='y_cor', index=1)

    # ========================================================
    # Return everything
    # ========================================================

    result = {
        "tab_st": tab_st,
        "stelt_maps": stelt_maps,
        "stelt_maps_n": stelt_maps_n,
        "base_coeff_maps": base_coeff_maps,
        "base_coeff_t": base_coeff_t
    }

    if args.el_flag:

        result["tab_el"] = tab_el
        result["elint_maps"] = elint_maps
        result["elint_maps_n"] = elint_maps_n

    return result


def tab_cre(ob, args):

    # ========================================================
    # Initial setup
    # ========================================================

    file_dir = args.data_path + ob + '/'

    stackcubes = np.sort([
        x for x in os.listdir(file_dir)
        if 'stackcube' in x
    ])

    blue_cube = fits.open(file_dir + stackcubes[1])
    red_cube = fits.open(file_dir + stackcubes[0])

    gal = blue_cube[0].header['CCNAME1']

    gal_dir = (
        str(blue_cube[0].header['OBID']) + '_' +
        gal + '_' +
        blue_cube[0].header['MODE']
    )

    # ========================================================
    # Mode configuration
    # ========================================================

    mode_cfg = {

        'blue': {
            'fit_flag': args.blue_fit_flag,
            'cube': blue_cube,
            'cube_file': 'blue_cube_vorbin.fits',
            'vorbin_map': 'vorbin_map_blue.fits',
            'params': 'parameters_stellar_blue'
        },

        'red': {
            'fit_flag': args.red_fit_flag,
            'cube': red_cube,
            'cube_file': 'red_cube_vorbin.fits',
            'vorbin_map': 'vorbin_map_red.fits',
            'params': 'parameters_stellar_red'
        },

        'aps': {
            'fit_flag': args.aps_fit_flag,
            'cube': gal_dir + '/' + gal + '_cube.fits',
            'cube_file': gal + '_vorbin_cube.fits',
            'vorbin_map': 'vorbin_map.fits',
            'params': 'parameters_stellar_aps'
        }
    }

    # ========================================================
    # Loop over modes
    # ========================================================

    for mode, cfg in mode_cfg.items():

        # ----------------------------------------------------
        # Skip mode if disabled
        # ----------------------------------------------------

        if cfg['fit_flag'] != 1:
            continue

        print('\n====================')
        print(f'Running {mode.upper()}')
        print('====================')

        # ----------------------------------------------------
        # Locate latest result directory
        # ----------------------------------------------------

        pyp_dir = gal_dir + '/pyp_results/'

        res_dir = (
            pyp_dir +
            np.sort([
                x for x in os.listdir(pyp_dir)
                if mode.upper() in x
            ])[-1] +
            '/'
        )

        # ----------------------------------------------------
        # Load WCS cube
        # ----------------------------------------------------

        coords = None

        if args.vorbin_flag:

            cube = fits.open(
                gal_dir + '/' + cfg['cube_file']
            )

            file_n = f'_{mode}_vorbin'

        else:

            if mode == 'aps':
                cube = fits.open(cfg['cube'])
            else:
                cube = cfg['cube']

            file_n = f'_{mode}'

        # ----------------------------------------------------
        # Load PyParadise products
        # ----------------------------------------------------

        contm_file = fits.open(
            res_dir + gal + file_n + '.cont_model.fits'
        )

        contr_file = fits.open(
            res_dir + gal + file_n + '.cont_res.fits'
        )

        stelt_file = fits.open(
            res_dir + gal + file_n + '.stellar_table.fits'
        )

        # ----------------------------------------------------
        # Optional emission line products
        # ----------------------------------------------------

        if args.el_flag:

            elinm_file = fits.open(
                res_dir + gal + file_n + '.eline_model.fits'
            )

            elinr_file = fits.open(
                res_dir + gal + file_n + '.eline_res.fits'
            )

            elint_file = fits.open(
                res_dir + gal + file_n + '.eline_table.fits'
            )

        else:

            elinm_file = None
            elinr_file = None
            elint_file = None

        # ----------------------------------------------------
        # Voronoi map
        # ----------------------------------------------------

        vorbin_map = fits.getdata(
            gal_dir + '/' + cfg['vorbin_map']
        )

        bins = np.unique(
            vorbin_map[vorbin_map >= 0]
        ).astype(int)

        # ----------------------------------------------------
        # Load stellar templates
        # ----------------------------------------------------

        params_stel = open(
            res_dir +
            '/' +
            cfg['params'] +
            '_' +
            ob,
            'r'
        )

        lines = params_stel.readlines()

        stel_template = fits.open(
            args.temp_path +
            lines[1].split()[1]
        )

        # ----------------------------------------------------
        # Reconstruct cubes
        # ----------------------------------------------------

        print('\nReconstructing cubes...')

        if args.vorbin_flag:

            cube_shape = cube[1].data.shape

            if args.el_flag:
                (contm_data, contm_err, contm_badp, contm_norm, contr_data, elinm_data, elinr_data) = \
                    fill_cube(args.vorbin_flag, bins, vorbin_map, contm_file, contr_file,
                              None if not args.el_flag else {"model": elinm_file, "res": elinr_file},
                              cube_shape, coords)

            else:
                (contm_data, contm_err, contm_badp, contm_norm, contr_data) = \
                    fill_cube(args.vorbin_flag, bins, vorbin_map, contm_file, contr_file,
                              None if not args.el_flag else {"model": elinm_file, "res": elinr_file},
                              cube_shape, coords)

        else:

            contm_data = contm_file[0].data
            contm_err = contm_file[1].data
            contm_badp = contm_file[2].data
            contm_norm = contm_file[3].data

            contr_data = contr_file[0].data

            if args.el_flag:
                elinm_data = elinm_file[0].data
                elinr_data = elinr_file[0].data

        # ----------------------------------------------------
        # Process tables + maps
        # ----------------------------------------------------

        print('Processing tables and maps...')

        tables_maps = process_tables_and_maps(

            args=args,
            bins=bins,
            vorbin_map=vorbin_map,
            stelt_file=stelt_file,
            elint_file=elint_file,
            stel_template=stel_template,
            coords=coords
        )

        # ----------------------------------------------------
        # Build cube header
        # ----------------------------------------------------

        cube_head = fits.Header()

        cube_head['SIMPLE'] = True
        cube_head['BITPIX'] = -32
        cube_head['NAXIS'] = 3

        cube_head['NAXIS1'] = contm_data.shape[2]
        cube_head['NAXIS2'] = contm_data.shape[1]
        cube_head['NAXIS3'] = contm_data.shape[0]

        cube_head['CTYPE3'] = 'WAVELENGTH'
        cube_head['CUNIT3'] = 'Angstrom'

        cube_head['CDELT3'] = (
            contm_file[0].header['CDELT1']
        )

        cube_head['DISPAXIS'] = (
            contm_file[0].header['DISPAXIS']
        )

        cube_head['CRVAL3'] = (
            contm_file[0].header['CRVAL1']
        )

        cube_head['CRPIX3'] = (
            contm_file[0].header['CRPIX1']
        )

        cube_head['CRPIX1'] = cube[1].header['CRPIX1']
        cube_head['CRPIX2'] = cube[1].header['CRPIX2']

        cube_head['CRVAL1'] = cube[1].header['CRVAL1']
        cube_head['CRVAL2'] = cube[1].header['CRVAL2']

        cube_head['CDELT1'] = cube[1].header['CDELT1']
        cube_head['CDELT2'] = cube[1].header['CDELT2']

        cube_head['CTYPE1'] = 'RA---TAN'
        cube_head['CTYPE2'] = 'DEC--TAN'

        cube_head['CUNIT1'] = 'deg'
        cube_head['CUNIT2'] = 'deg'

        # ----------------------------------------------------
        # Build map header
        # ----------------------------------------------------

        map_head = fits.Header()

        map_head['SIMPLE'] = True
        map_head['BITPIX'] = -32
        map_head['NAXIS'] = 2

        map_head['NAXIS1'] = vorbin_map.shape[1]
        map_head['NAXIS2'] = vorbin_map.shape[0]

        map_head['DISPAXIS'] = 1

        map_head['CRPIX1'] = cube[1].header['CRPIX1']
        map_head['CRPIX2'] = cube[1].header['CRPIX2']

        map_head['CRVAL1'] = cube[1].header['CRVAL1']
        map_head['CRVAL2'] = cube[1].header['CRVAL2']

        map_head['CDELT1'] = cube[1].header['CDELT1']
        map_head['CDELT2'] = cube[1].header['CDELT2']

        map_head['CTYPE1'] = 'RA---TAN'
        map_head['CTYPE2'] = 'DEC--TAN'

        map_head['CUNIT1'] = 'deg'
        map_head['CUNIT2'] = 'deg'

        # ----------------------------------------------------
        # Write outputs
        # ----------------------------------------------------

        print('Saving outputs...')

        write_outputs(

            res_dir=res_dir,
            gal=gal,
            mode=mode,

            cube_header=cube_head,
            map_header=map_head,

            contm_data=contm_data,
            contm_err=contm_err,
            contm_badp=contm_badp,
            contm_norm=contm_norm,
            contr_data=contr_data,

            stelt_file=stelt_file,

            tables_maps=tables_maps,

            args=args,

            elinm_data=elinm_data if args.el_flag else None,
            elinr_data=elinr_data if args.el_flag else None,

            elint_file=elint_file
        )

        print(f'{mode.upper()} complete.')

