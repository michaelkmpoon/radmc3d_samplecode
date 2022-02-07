# radmc3d_samplecode

RADMC-3D is a radiative transfer code which for a given star and geometrical distribution dust, simulate what its images and spectra look like when viewed from a certain angle.

The code in this repository is a model setup file adapted from
https://github.com/dullemond/radmc3d-2.0/tree/master/examples/run_ppdisk_analytic_1
to model Dipper star J1604 with its 80 au transition disk and hypothesized misaligned inner disk.

The code is modified to include:

1) cylindrical to cartesial coordinate tranform to create a warp
2) Custom temperature profile 
3) Follow PEP 8 style guide for Python