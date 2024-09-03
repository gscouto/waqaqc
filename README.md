# WAQAQC
WAQAQC is a module package intended to produce QC plots for WEAVE-Apertif Quality Assurance (WAQA) team. It also is a 
wrapper for fitting the PyParadise code to WEAVE Large IFU (LIFU) data.

The PyParadise code can be obtained here: https://github.com/brandherd/PyParadise

To install this package, do a git clone and, inside the waqaqc directory, do a 

```
$ pip install -e .
```

How to run the module
---------------------
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

The runner
---------------------
The runner is a simple script that will call each of the package modules. You can list the galaxies IDs in the beginning
of the script at 'gals' list, so you can run for several WEAVE targets, given that each target has a specific 
config_file.env. Then it will run every listed module, which are described below. Comment a specific block related to a 
module, and it will skip this process. In principle each module is independent of each other, given that the 
config_file.env reflects that (see more below).

Package modules
---------------------
Here we describe in more detail each module:

1) APS_cube: this module is responsible for creating the datacubes from the PyAPS WEAVE L2 data, the APS product. As the
APS product is a fits file containing tables, datacubes may be more user convenient when data handling.
The input parameters regarding this module, which can be edited within the config_file.env, are:
- n_proc: number of computer cores to be used while running WAQAQC (this also applies to the other modules)
- file_dir: directory path to the data to be used (not only APS, but other WEAVE data).
- aps_file: APS file name in file_dir (should end with _APS.fits).