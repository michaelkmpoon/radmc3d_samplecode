# radmc3d_samplecode

RADMC-3D is a radiative transfer code. For a given star and geometrical distribution dust, it simulates what its image and spectra look like when viewed from a certain angle.

The code in this repository is a model setup file adapted from
https://github.com/dullemond/radmc3d-2.0/blob/master/examples/run_ppdisk_analytic_1/problem_setup.py
to model Dipper star J1604 with its 80 au transition disk and hypothesized misaligned inner disk.

The code is modified to include:

1) Cylindrical to cartesian coordinate tranform to create a radial warp
2) Custom temperature and density profile 
3) Follow PEP 8 style guide for Python

RADMC-3D: https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/
