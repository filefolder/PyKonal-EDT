# Welcome to the *PyKonal-EDT* repository!

# This is a hard fork of the original PyKonal package (https://github.com/malcolmw/pykonal) with serveral significant changes:

1) An Equal Differential Time solver (similar to NonLinLoc) which replaces the Differential Evolution (scipy) solver
2) A SeisComP (via locext) compatible plugin to use in SCOLV
3) A ObsPy-based locator which works with Event objects, optionally using same SC parameters 
4) Various tune-ups and bug fixes


## Installation    
To install this version specifically (which is required for PyVoroTomo v2), either download the package, unzip and install
$ pip install ./pykonal-EDT

or, via pip using the github path
$ pip install git+https://github.com/filefolder/pykonal-EDT.git

## Citation
If you make use of this code in published work, probably best to cite White *et al.* (2020) (hopefully more TBA :)


## References
1. Sethian, J. A. (1996). A fast marching level set method for monotonically advancing fronts. *Proceedings of the National Academy of Sciences, 93*(4), 1591–1595. https://doi.org/10.1073/pnas.93.4.1591
2. White, M. C. A., Fang, H., Nakata, N., & Ben-Zion, Y. (2020). PyKonal: A Python Package for Solving the Eikonal Equation in Spherical and Cartesian Coordinates Using the Fast Marching Method. *Seismological Research Letters, 91*(4), 2378-2389. https://doi.org/10.1785/0220190318

