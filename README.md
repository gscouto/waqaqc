# WAQAQC
WAQAQC is a module package intended to produce QC plots for WEAVE-Apertif Quality Assurance (WAQA) team. It also is a 
wrapper for fitting the PyParadise code to WEAVE Large IFU (LIFU) data.

The PyParadise code can be obtained here: https://github.com/brandherd/PyParadise

To install this package, do a git clone and, inside the waqaqc directory, do a 

```
$ pip install -e .
```

## How to run the module
To run the module, you will need two files that can be found in the 'config' directory within the WAQAQC package: the 
config_file.env and runner.py files. Copy these files to a fresh directory where you will be working. The 
config_file.env contains a few parameters necessary to run the WAQAQC module, while the runner.py is simply a python 
code to run specific tasks within WAQAQC. To use the runner, the user only needs to do:

```
$ python runner.py
```

or within interactive python:

```
$ run runner.py
```

## The runner
The runner is a simple script that will call each of the package modules. You can list the galaxies IDs in the beginning
of the script at 'gals' list, so you can run for several WEAVE targets, given that each target has a specific 
config_file.env. Then it will run every listed module, which are described below. Comment a specific block related to a 
module, and it will skip this process. In principle each module is independent of each other, given that the 
config_file.env reflects that (see more below).

## Package modules
Here we describe in more detail each module:

### APS_cube
This module is responsible for creating the datacubes from the PyAPS WEAVE L2 data, the APS product. As the APS product 
is a fits file containing tables, datacubes may be more user convenient when data handling. This means a datacube format
of both the blue and the red arms spectra joined in a full wavelength range.

The input parameters regarding this module, which can be edited within the config_file.env, are:
- n_proc: number of computer cores to be used while running WAQAQC (this also applies to the other modules)
- file_dir: directory path to the data to be used (not only APS, but other WEAVE data).
- aps_file: APS file name in file_dir (should end with _APS.fits).

The output files of this module are created in a directory named as the target "CNAME _ observation mode (LOWRES, 
HIGHRES) _ OBID" (called targ_dir here for short), created where the WAQAQC package is running. These are:
- APS cube: file named as the CNAME + _cube.fits. This is the APS datacube (after merging the blue and the red arm).
- Vorbin cube: file named as the CNAME + _vorbin_cube.fits. Cube created following the Voronoi binning used at the APS 
procedure.
- Vorbin map: file named as the CNAME + _vorbin_map.fits. A fits map of the Voronoi regions selected by APS.
- APS maps: file named as the CNAME + _APS_maps.fits. A fits file containing all APS products related to the pPXF 
spectral fitting. Each file extension represents a map (ex: FLUX_HA_6562.80).

### QC_plots
This module creates quality control plots of the data obtained for the given target. Several plots are created, such as 
the fiber spectral resolution, sky maps, S/N maps and peak flux variability (to be further detailed). It also performs 
Voronoi binning of both the L1 datacubes, and the APS datacube, in case APS flag is turned on.

The input parameters regarding this module, which can be edited within the config_file.env, are:
- blue(red)_cube: L1 datacubes (stackcube) data file name (located in file_dir).
- aps_flag: flag for making plots for APS or not. 1 = yes, 0 = no. Use it if you do not have APS files (or have not 
created using APS_cube before).
- target_SN: target S/N ratio when performing Voronoi binning. Suggested values are 20 or 30 (for high S/N observations).
- levels: list of level values when plotting the S/N map contours. Suggested are [5, 30].
- blue(red/aps)_wav: central wavelength where S/N is estimated, for each spectral range (blue, red, APS). A window of 
50 Angstrom is created centered in the given wavelength.

The output files of this module are created in the targ_dir, aside from the first two in the list below, which are 
created in the working directory. These are:
- PNG images: file named as the obs date + CNAME + observation mode (LOWRES, HIGHRES) + OBID + _L0(L1/L2).png. For each 
L0, L1 and L2 dataset is created a .png image containing its QC plots. 
- html page: file named as the obs date + CNAME + observation mode (LOWRES, HIGHRES) + OBID + .html. The QC plot images 
are collected into a html page.
- Vorbin cubes: files named as blue/red/aps_cube_vorbin.fits. Datacubes created using the Voronoi binning performed.
- Vorbin maps: files named as vorbin_map_blue/red/aps.fits. Fits maps of the Voronoi regions selected.
- Resolution tables: files named as resol_table_+ L0 data name +.txt, and resol_table_aps/blue/red/mean.txt. Spectra 
resolution is measured by fitting the sky lines in the L0 spectra. These are saved as wavelength and FWHM (in A) 
columns in these files. Each one is created for each single exposure file, and master resolution files are created for 
the blue and the red, and extrapolated for aps. A single mean value is calculated for the three modes (blue, red and 
aps). These files can be later used to run PyParadise.

OBS: the html pages are transferred to minos, a repository environment within AIP. (to be edited)

### pyp_params
This module creates the parameter files necessary to run PyParadise within the WEAVE datacubes, either blue, red or APS. 
See PyParadise manual for more details on these files.

The input parameters regarding this module, which can be edited within the config_file.env, are:
- blue(red/aps)_fit: flag informing which data modes you wish to fit. 1 for yes and 0 for no. You can select more than 
one, or none at all.
- temp_dir: directory path to the stellar templates to be used. Do not use '' and supress the last /.
- temp_file: file name for the stellar template library to be used. See PyParadise manual for more details on how this 
template file should be formatted.
- redshift: inform the target redshift, so the initial velocity guess matches the observed spectra. If the spectra has 
been redshifted corrected (which is the case for APS data), set it to 0.
- disp_max: maximum stellar velocity dispersion to be fitted. Recommended is 500 km/s.
- line_flux: initial guess for emission line flux. This value is not very sensible to the actual value, and should not 
affect strongly in the resulted fitting.
- line_vel: an additive value to the systemic velocity given by the redshift informed. Should be used only if the 
informed redshift does not result in a good fit, otherwise should be set at 0.
- aps(blue/red)_lam_min(max): wavelength coverage for the spectra to be fitted, for each of the data modes. Regions 
outside this interval will be disregarded in the fit.

The output files of this module are created in the working directory. These are:
- parameters_stellar_blue(red/aps): parameters used in the stellar continuum fit.
- parameters_eline_blue(red/aps): parameters used in the emission line fit.
- par_blue(red/aps).lines: parameters for each of the lines fitted in the emission line fit.
- excl_blue(red/aps).cont: continuum regions masked out while doing the stellar fit. These are masked usually due to 
gaps, sky lines or other artifacts.
- excl_blue(red/aps).fit: continuum regions masked out while doing the stellar fit, due to the presence of emission 
lines.

### spec_fit
This module fires PyParadise to run at the Voronoi binned datacubes created. It will first create a RSS file listing all
spectra to be fitted (one per Voronoi region), and then run PyParadise. For more details on how PyParadise works, please 
see its manual.

The input parameters regarding this module, which can be edited within the config_file.env, are:
- resol_flag: a flag indicating if you should use input values (mentioned below), or use the resolution columns created 
from the sky line fits during the QC_plots module, for the instrumental data spectral resolution. 0 is for the input 
values, while 1 is to use the created table.
- fwhm_blue(red/aps): input FWHM for the spectra instrumental resolution, in A. This is only used if resol_flag = 0.
- EL_flag: a flag to inform if you wish to fit emission lines. 1 for yes, and 0 for no.
- boots_flag: a flag for running bootstrapping. 1 for yes, and 0 for no. See PyParadise manual for more details on 
bootstrapping. Consider that uncertainties in the physical parameters are only obtained when running bootstrap.
- cosm_flag: a flag to perform cosmic rejection. PyParadise is sensitive to cosmic rays, or strong artifacts in spectra,
and these should be masked if present. 1 for yes, 0 for no.
- cosm_limit: threshold for cosmic rejection, if turned on. After taking the spectrum derivative, cosm_limit is the 
multiplicative of the derivative stddev. Default value is 30.

The output files of this module are located within a directory called 'pyp_results', inside the targ_dir. The files are 
created inside a directory named after the time of running the code. They are purely outputs from PyParadise. These 
files are: 
- CNAME_blue(red/aps)_vorbin_RSS: data file created from the Voronoi binned datacube. This file lists the unique data 
spectra fitted by PyParadise.
- CNAME_blue(red/aps)_vorbin_stellar(eline)_table: table containing the parameters from the best fit model, for the 
stellar continuum fit and for the emission line fit, respectively. These are listed in a corresponding format to the RSS
file, as well as the files below.
- CNAME_blue(red/aps)_vorbin_cont_model: file list of the resulted modelled spectra for the stellar continuum fit.
- CNAME_blue(red/aps)_vorbin_cont_res: file list of the residual spectra for the stellar continuum fit.
- CNAME_blue(red/aps)_vorbin_eline_model: file list of the resulted modelled spectra for the emission lines fit.
- CNAME_blue(red/aps)_vorbin_eline_res: file list of the residual spectra for the emission lines fit.

### table_creator
This module aims to create datacubes and maps out of the files created from PyParadise. As mentioned above, PyParadise 
runs on a file listing the Voronoi region spectra in order to speed up the fitting procedure. If the user desire to 
visualize the data in maps and datacube format, this module will create those files. 

The input parameters regarding this module, which can be edited within the config_file.env, are:
- template_dir: the path to the stellar templates.

The output files of this module are located within the same directory hosting the PyParadise outputs: 
'pyp_results/mode_date/'. These are:
- CNAME_blue(red/aps)_cont_model: datacube of the resulted modelled spectra for the stellar continuum fit.
- CNAME_blue(red/aps)_cont_res: datacube of the residual spectra for the stellar continuum fit.
- CNAME_blue(red/aps)_eline_model: datacube of the resulted modelled spectra for the emission lines fit.
- CNAME_blue(red/aps)_eline_res: datacube of the residual spectra for the emission lines fit.
- CNAME_blue(red/aps)_stellar(eline)_maps: maps from the different output parameters, for the stellar continuum fit and 
the emission line fit, respectively.
- CNAME_blue(red/aps)_stellar(eline)_table: tables from the different output parameters, for the stellar continuum fit 
and the emission line fit, respectively. The difference between these tables and the output from PyParadise is this have
information for each [x,y] spatial spaxel in the WEAVE datacube.
- CNAME_blue(red/aps)_base_coeff_maps: maps of the contribution of all templates models used in the stellar continuum 
fit. Each template map is given in a fits extension. The maps are listed in the same order as the template file for each 
template.

## The config_file.env

## The weave_gui