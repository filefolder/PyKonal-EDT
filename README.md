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

## Choosing a grid spacing and vertical ratio
The tables below show how much traveltime error the grid itself introduces, measured for a 1-D AK135 Earth (P waves) against exact TauP first-arrival times. Traveltime tables were built as in production: spherical grid, `PointSourceSolver` with the source at a surface station, values stored as float32 and read with trilinear interpolation.

- **Spacing h** is the horizontal node interval; **ratio** is vertical/horizontal spacing (dz = ratio × h).
- **P95 error** is the 95th-percentile absolute traveltime error over all test events (excluding events at 0 km depth) and epicentral distances up to the model width (0–90° for global). Event depths: local 2–30 km, regional 5–200 km, global 15–600 km.
- The most accurate ratio at each spacing is in **bold**.
- **Grid size** is one full 3-D traveltime table per station and phase (float32, before compression).
- Errors along diagonal azimuths were similar to or smaller than along the grid axes. S-wave errors are roughly 1.7× larger.
- — means not run (too large).

**Local (200 km wide, 80 km deep)**

| Spacing h (km) | P95 error, ratio 1.0 (ms) | ratio 0.5 | ratio 0.2 | Optimal ratio | Grid size, ratio 1.0 (MB) | ratio 0.5 | ratio 0.2 |
|---:|---:|---:|---:|:---:|---:|---:|---:|
| 1 | **42** | 48 | 65 | 1.0 | 13.1 | 26.0 | 64.8 |
| 3 | **80** | 132 | 191 | 1.0 | 0.52 | 1.0 | 2.5 |
| 5 | 258 | **221** | 314 | 0.5 | 0.11 | 0.22 | 0.54 |
| 10 | 1,206 | **442** | 613 | 0.5 | 0.02 | 0.03 | 0.07 |
| 20 | 4,091 | 1,286 | **1,214** | 0.2 | <0.01 | <0.01 | 0.01 |
| 30 | 7,226 | 3,018 | **1,831** | 0.2 | <0.01 | <0.01 | <0.01 |
| 50 | 13,266 | 6,721 | **3,155** | 0.2 | <0.01 | <0.01 | <0.01 |

**Regional (600 km wide, 300 km deep)**

| Spacing h (km) | P95 error, ratio 1.0 (ms) | ratio 0.5 | ratio 0.2 | Optimal ratio | Grid size, ratio 1.0 (MB) | ratio 0.5 | ratio 0.2 |
|---:|---:|---:|---:|:---:|---:|---:|---:|
| 1 | 42 | **41** | 60 | 0.5 | 435 | 868 | 2,169 |
| 3 | 178 | **121** | 181 | 0.5 | 16.3 | 32.5 | 81.0 |
| 5 | 270 | **205** | 300 | 0.5 | 3.6 | 7.1 | 17.6 |
| 10 | 471 | **395** | 531 | 0.5 | 0.46 | 0.91 | 2.2 |
| 20 | 825 | **742** | 1,015 | 0.5 | 0.06 | 0.12 | 0.29 |
| 30 | 3,509 | **930** | 1,548 | 0.5 | 0.02 | 0.04 | 0.09 |
| 50 | 9,034 | 2,745 | **2,200** | 0.2 | <0.01 | <0.01 | 0.02 |

**Global (20,000 km wide, 3,100 km deep, P to 90°)**

| Spacing h (km) | P95 error, ratio 1.0 (ms) | ratio 0.5 | ratio 0.2 | Optimal ratio | Grid size, ratio 1.0 (MB) | ratio 0.5 | ratio 0.2 |
|---:|---:|---:|---:|:---:|---:|---:|---:|
| 1 | — | — | — | — | — | — | — |
| 3 | 132 | **91** | — | 0.5 | 184,074 | 367,792 | — |
| 5 | **149** | 180 | 197 | 1.0 | 39,764 | 79,464 | 198,563 |
| 10 | **274** | 347 | 417 | 1.0 | 4,981 | 9,946 | 24,841 |
| 20 | 732 | **637** | 906 | 0.5 | 625 | 1,246 | 3,110 |
| 30 | 1,699 | **821** | 1,370 | 0.5 | 187 | 371 | 925 |
| 50 | **1,166** | 1,765 | 2,131 | 1.0 | 40.5 | 80.4 | 200 |

Note: a ratio of 0.2 costs the most nodes but is rarely the most accurate except at coarse spacings, as it biases arrivals early.

## Citation
If you make use of this code in published work, probably best to cite White *et al.* (2020) (hopefully more TBA :)


## References
1. Sethian, J. A. (1996). A fast marching level set method for monotonically advancing fronts. *Proceedings of the National Academy of Sciences, 93*(4), 1591–1595. https://doi.org/10.1073/pnas.93.4.1591
2. White, M. C. A., Fang, H., Nakata, N., & Ben-Zion, Y. (2020). PyKonal: A Python Package for Solving the Eikonal Equation in Spherical and Cartesian Coordinates Using the Fast Marching Method. *Seismological Research Letters, 91*(4), 2378-2389. https://doi.org/10.1785/0220190318

