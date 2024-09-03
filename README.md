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