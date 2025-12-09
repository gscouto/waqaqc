from waqaqc import QC_plots, APS_cube, spec_fit, table_creator, pyp_params
import numpy as np
import time
import traceback


def run(args):
    ss_time = time.time()

    for i in np.arange(len(args.ob_list)):

        ob = args.ob_list[i]
        redshift = args.redshift[i]

        print('')
        print('********************************************')
        print('Runner on OB ' + ob)
        print('********************************************')
        print('')

        # ------------------------

        if args.aps_cube:
            try:
                print('')
                print('============================================')
                print('Running APS cube creator')
                print('============================================')
                print('')

                APS_cube.cube_creator(ob, args)

            except Exception as e:
                print(f"❌ [APS_cube] Error processing galaxy {ob}: {e}")
                traceback.print_exc()

        # ------------------------

        if args.qc_plots:
            try:
                print('')
                print('============================================')
                print('Running QC plots')
                print('============================================')
                print('')

                QC_plots.html_plots(ob, redshift, args)

            except Exception as e:
                print(f"❌ [QC_plots] Error processing galaxy {ob}: {e}")
                traceback.print_exc()

        # ------------------------

        if args.pyp_params:
            try:
                print('')
                print('============================================')
                print('Creating PyParadise parameters files')
                print('============================================')
                print('')

                pyp_params.pp(ob, redshift, args)

            except Exception as e:
                print(f"❌ [pyp_params] Error processing galaxy {ob}: {e}")
                traceback.print_exc()

        # ------------------------

        if args.spec_fit:
            try:
                print('')
                print('============================================')
                print('Running PyParadise fitter')
                print('============================================')
                print('')

                spec_fit.specs(ob, args)

            except Exception as e:
                print(f"❌ [spec_fit] Error processing galaxy {ob}: {e}")
                traceback.print_exc()

        # ------------------------

        if args.table_creator:
            try:
                print('')
                print('============================================')
                print('Creating tables from fit')
                print('============================================')
                print('')

                table_creator.tab_cre(ob, args)

            except Exception as e:
                print(f"❌ [table_creator] Error processing galaxy {ob}: {e}")
                traceback.print_exc()

    print('The total run took ' + str(round(time.time() - ss_time, 2)) + ' secs')
