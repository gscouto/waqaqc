from waqaqc import QC_plots, APS_cube, spec_fit, table_creator, pyp_params
import time

gals = ['11114']

ss_time = time.time()

for i in gals:

    print('')
    print('********************************************')
    print('Runner on galaxy '+i)
    print('********************************************')
    print('')

    print('')
    print('============================================')
    print('Running APS cube creator')
    print('============================================')
    print('')

    APS_cube.cube_creator('config_file_'+i+'.env')

    print('')
    print('============================================')
    print('Running QC plots')
    print('============================================')
    print('')

    QC_plots.html_plots('config_file_'+i+'.env')

    print('')
    print('============================================')
    print('Creating PyParadise parameters files')
    print('============================================')
    print('')

    pyp_params.pp('config_file_'+i+'.env')

    print('')
    print('============================================')
    print('Running PyParadise fitter')
    print('============================================')
    print('')
    
    spec_fit.specs('config_file_'+i+'.env')

    print('')
    print('============================================')
    print('Creating tables from fit')
    print('============================================')
    print('')

    table_creator.tab_cre('config_file_'+i+'.env')


print('The total run took ' + str(round(time.time() - ss_time, 2)) + ' secs')
