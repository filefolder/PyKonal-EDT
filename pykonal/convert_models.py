#!/usr/bin/env python3

"""
Conversions between NonLinLoc (NLL) 3-D model grids and PyKonal
(and PyVoroTomo / "PVT") spherical velocity fields.

Command line
------------
    convert_models.py inmodel outmodel [options]

The format of ``inmodel`` is detected automatically and the output is the
*other* format:

    NLL grid  (basename, .hdr or .buf)   ->  pykonal HDF5 (.h5)
    pykonal HDF5 (.h5)                   ->  NLL grid     (outmodel.hdr/.buf)

One run converts one field (e.g. the P model).  Run it again for S.
See ``convert_model.py --help`` for the options.

Requires
--------
nllgrid  https://github.com/claudiodsf/nllgrid
PyKonal-EDT  https://github.com/filefolder/PyKonal-EDT
numpy, scipy, pyproj (pyproj comes in via nllgrid)

Design notes
------------
*   All map projection work is delegated to ``nllgrid.NLLGrid.project`` /
    ``.iproject``.  Those use pyproj with the ellipsoid named in the grid
    header (lcc / tmerc / aeqd / eqc) and apply ``map_rot``.  Re-implementing
    the projections by hand -- especially on a sphere -- guarantees a
    disagreement with the grids NLL itself produced.

*   An NLL grid is a *rectangle in projected x/y*.  A pykonal spherical field
    is a *rectangle in lat/lon*.  These are not the same region, and node
    (i,j,k) of one is not co-located with node (i,j,k) of the other.  The
    conversion is therefore a genuine resampling: build the target grid, map
    every target node back through the projection, and interpolate.  Simply
    transposing and flipping the array (as a naive implementation does)
    shears the model by an amount that grows with distance from the
    projection origin.

*   Target nodes that fall outside the source grid are unavoidable -- the
    lat/lon bounding box of a conic projection's rectangle always spills over
    its edges.  Such nodes are filled by clamping to the nearest edge of the
    source grid rather than by extrapolation, which keeps velocities physical.
    The fraction of clamped nodes is reported.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import nllgrid
import pykonal
import pykonal.transformations
from scipy.interpolate import RegularGridInterpolator

# PyKonal's own Earth radius must be used for every rho <-> depth
# conversion so that geo2sph/sph2geo round-trip exactly.
EARTH_RADIUS = pykonal.constants.EARTH_RADIUS   # 6371.0 km

__all__ = ['NLL2PVT', 'PVT2NLL', 'convert_model', 'detect_format', 'main',
           'nll_header_from_pvt', 'nll_node_coords', 'nll_velocity',
           'geographic_bounds']


# ---------------------------------------------------------------------------
# Grid-value <-> velocity conversion
# ---------------------------------------------------------------------------

def nll_velocity(nll):
    """
    Return the grid values of `nll` as velocity in km/s.

    Handles every NLL model grid type.  ``SLOW_LEN`` stores
    slowness x cell-length, so it requires cubic cells.
    """
    a = np.asarray(nll.array, dtype=np.float64)
    t = nll.type

    if t == 'VELOCITY':
        return a
    if t == 'VELOCITY_METERS':
        return a / 1000.0
    if t == 'SLOWNESS':                      # s/km
        return 1.0 / a
    if t == 'VEL2':                          # (km/s)^2
        return np.sqrt(a)
    if t == 'SLOW2':                         # (s/km)^2
        return 1.0 / np.sqrt(a)
    if t == 'SLOW2_METERS':                  # (s/m)^2
        return 1.0 / (1000.0 * np.sqrt(a))
    if t == 'SLOW_LEN':                      # slowness * cell length [s]
        if not (np.isclose(nll.dx, nll.dy) and np.isclose(nll.dx, nll.dz)):
            raise ValueError(
                f'SLOW_LEN grid requires cubic cells, got '
                f'dx={nll.dx}, dy={nll.dy}, dz={nll.dz}')
        return nll.dx / a

    raise ValueError(
        f'Grid type {t!r} is not a velocity model grid. '
        'Supported: VELOCITY, VELOCITY_METERS, SLOWNESS, VEL2, SLOW2, '
        'SLOW2_METERS, SLOW_LEN.')


def velocity_to_nll(nll, vel):
    """Inverse of :func:`nll_velocity`: velocity in km/s -> `nll.type` units."""
    v = np.asarray(vel, dtype=np.float64)
    t = nll.type

    if t == 'VELOCITY':
        return v
    if t == 'VELOCITY_METERS':
        return v * 1000.0
    if t == 'SLOWNESS':
        return 1.0 / v
    if t == 'VEL2':
        return v ** 2
    if t == 'SLOW2':
        return 1.0 / v ** 2
    if t == 'SLOW2_METERS':
        return 1.0 / (1000.0 * v) ** 2
    if t == 'SLOW_LEN':
        return nll.dx / v

    raise ValueError(f'Unsupported grid type for writing: {t!r}')


# ---------------------------------------------------------------------------
# NLL grid geometry
# ---------------------------------------------------------------------------

def nll_node_coords(nll):
    """
    Node-centre coordinate axes (x, y, z) of an NLL grid, in km.

    Note this is *not* ``nll.get_extent()``, which pads by half a cell on
    each side for plotting and spans ``(n+1)*d`` rather than ``(n-1)*d``.
    """
    x = nll.x_orig + np.arange(nll.nx) * nll.dx
    y = nll.y_orig + np.arange(nll.ny) * nll.dy
    z = nll.z_orig + np.arange(nll.nz) * nll.dz
    return x, y, z


def geographic_bounds(nll, n_edge=401):
    """
    Latitude/longitude bounding box of an NLL grid.

    The image of a projected rectangle is curvilinear, so the extremes do
    not generally lie at the four corners -- for a Lambert grid the extreme
    latitude sits mid-edge and the extreme longitudes at two different
    corners.  The whole boundary is therefore sampled.

    Longitudes are unwrapped relative to ``nll.orig_lon`` so that grids
    crossing the antimeridian give a continuous (possibly >180 deg) range.

    Returns
    -------
    (lat_min, lat_max, lon_min, lon_max) : tuple of float
    """
    x, y, _ = nll_node_coords(nll)
    xs = np.linspace(x[0], x[-1], n_edge)
    ys = np.linspace(y[0], y[-1], n_edge)
    ones = np.ones(n_edge)

    bx = np.concatenate([xs, xs, x[0] * ones, x[-1] * ones])
    by = np.concatenate([y[0] * ones, y[-1] * ones, ys, ys])

    lon, lat = nll.iproject(bx, by)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)

    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError(
            'Projection returned non-finite coordinates on the grid boundary; '
            'check the TRANSFORM line in the NLL header.')

    # Unwrap about the projection origin so the range is continuous.
    lon = nll.orig_lon + (lon - nll.orig_lon + 180.0) % 360.0 - 180.0

    return float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())


def inscribed_bounds(nll, n=600):
    """
    Largest axis-aligned lat/lon box *entirely inside* the NLL grid.

    Whereas :func:`geographic_bounds` returns the enclosing box (which for a
    continental-scale conic grid overshoots the model by 10-20% of its area),
    this returns the biggest box containing no points outside the source
    model, so no velocity has to be invented.  The trade-off is that the
    corners of the real model are discarded.

    Returns
    -------
    (lat_min, lat_max, lon_min, lon_max) : tuple of float
    """
    lat_min, lat_max, lon_min, lon_max = geographic_bounds(nll)
    x, y, _ = nll_node_coords(nll)
    lats = np.linspace(lat_max, lat_min, n)
    lons = np.linspace(lon_min, lon_max, n)
    LAT, LON = np.meshgrid(lats, lons, indexing='ij')
    X, Y = nll.project(LON.ravel(), LAT.ravel())
    inside = ~((X < x[0]) | (X > x[-1]) | (Y < y[0]) | (Y > y[-1]))
    inside = inside.reshape(n, n)

    # maximal all-true rectangle, via the histogram method
    best = (0, None)
    heights = np.zeros(n, dtype=int)
    for i in range(n):
        heights = np.where(inside[i], heights + 1, 0)
        stack = []
        for j in range(n + 1):
            cur = heights[j] if j < n else 0
            start = j
            while stack and stack[-1][1] >= cur:
                s, h = stack.pop()
                area = h * (j - s)
                if area > best[0]:
                    best = (area, (i - h + 1, i, s, j - 1))
                start = s
            stack.append((start, cur))
    if best[1] is None:
        raise ValueError('No inscribed box found; check the projection.')
    i0, i1, j0, j1 = best[1]
    return (float(lats[i1]), float(lats[i0]),
            float(lons[j0]), float(lons[j1]))


# ---------------------------------------------------------------------------
# NLL -> pykonal
# ---------------------------------------------------------------------------

def NLL2PVT(nllfile, pvtfile, npts=None, oversample=1.0,
            bounds='inner', fill='clamp', clip=None, verbose=True):
    """
    Convert a NonLinLoc model grid to a pykonal spherical HDF5 field.

    The output is a regular grid in (radius, colatitude, longitude) covering
    the geographic bounding box of the NLL grid.  Values are trilinearly
    interpolated from the NLL grid after mapping each target node through the
    header's projection.

    Parameters
    ----------
    nllfile : str
        Path to the NLL grid (basename, .hdr or .buf).
    pvtfile : str
        Output path for the pykonal HDF5 file.
    npts : sequence of 3 int, optional
        Target grid size as (n_radius, n_latitude, n_longitude).  By default
        the node count is chosen so that the angular spacing matches the NLL
        cell size at the centre latitude of the grid, which keeps the
        resolution of the source model without aliasing.
    oversample : float
        Multiplier applied to the default `npts`.  Ignored if `npts` is given.
    bounds : {'outer', 'inner'} or 4-tuple
        Geographic extent of the output.  ``'outer'`` (default) is the box
        enclosing the whole NLL grid; because a projected rectangle is
        curvilinear in lat/lon this necessarily includes nodes outside the
        model (~5% for a 1000 km grid, ~14% for a 3000 km one).  ``'inner'``
        is the largest box fully inside the model, so nothing is invented, at
        the cost of discarding the model corners.  A 4-tuple
        ``(lat_min, lat_max, lon_min, lon_max)`` sets it explicitly.
    fill : {'clamp', 'nan'} or float
        What to put at nodes outside the source grid.  ``'clamp'`` takes the
        nearest edge velocity -- reasonable for a small overhang, misleading
        for a continental grid where it extrudes the boundary hundreds of km.
        ``'nan'`` or a float (e.g. a reference-model velocity) makes the
        fabricated region explicit.  Irrelevant when ``bounds='inner'``.

        NOTE: NaN velocities poison a fast-marching solve -- the NaN spreads
        through the whole traveltime field.  Use ``'nan'`` for inspection and
        plotting only, never for a grid you intend to run pykonal on.
    clip : (vmin, vmax), optional
        Clamp velocities into this range, in km/s, before resampling.  Useful
        for pulling sentinel or air-layer values up to something physical
        (e.g. ``clip=(1.5, None)`` to floor the model at the water velocity).
        Either bound may be None.
    verbose : bool
        Print a summary of the conversion.

    Returns
    -------
    pykonal.fields.ScalarField3D
    """
    nll = nllgrid.NLLGrid(nllfile)
    if nll.array is None:
        raise ValueError(f'No buffer data read from {nllfile}')
    if min(nll.nx, nll.ny, nll.nz) < 2:
        raise ValueError(
            f'Grid is degenerate ({nll.nx}x{nll.ny}x{nll.nz}); 2-D NLL grids '
            'cannot be converted to a 3-D spherical field.')

    x, y, z = nll_node_coords(nll)
    vel = nll_velocity(nll)
    if clip is not None:
        vmin, vmax = clip
        n_clipped = int(np.count_nonzero(
            ((vel < vmin) if vmin is not None else False) |
            ((vel > vmax) if vmax is not None else False)))
        vel = np.clip(vel, vmin, vmax)
        if verbose and n_clipped:
            print(f'  clipped {n_clipped} source nodes '
                  f'({100 * n_clipped / vel.size:.2f}%) into {clip} km/s')
    if not np.all(np.isfinite(vel)) or np.any(vel <= 0):
        warnings.warn('Source grid contains non-positive or non-finite '
                      'velocities; these will propagate to the output.')

    if bounds == 'outer':
        lat_min, lat_max, lon_min, lon_max = geographic_bounds(nll)
    elif bounds == 'inner':
        lat_min, lat_max, lon_min, lon_max = inscribed_bounds(nll)
    else:
        lat_min, lat_max, lon_min, lon_max = (float(v) for v in bounds)
    dep_min, dep_max = float(z[0]), float(z[-1])

    # -- target grid size ---------------------------------------------------
    if npts is None:
        lat_mid = 0.5 * (lat_min + lat_max)
        km_per_deg = np.pi * EARTH_RADIUS / 180.0
        dlat = nll.dy / km_per_deg
        dlon = nll.dx / (km_per_deg * max(np.cos(np.radians(lat_mid)), 1e-6))
        n_lat = int(np.ceil(oversample * (lat_max - lat_min) / dlat)) + 1
        n_lon = int(np.ceil(oversample * (lon_max - lon_min) / dlon)) + 1
        n_rad = int(np.ceil(oversample * (nll.nz - 1))) + 1
        npts = (n_rad, n_lat, n_lon)
    npts = np.asarray(npts, dtype=int)
    if np.any(npts < 2):
        raise ValueError(f'npts must be >= 2 in every dimension, got {npts}')
    n_rad, n_lat, n_lon = (int(v) for v in npts)

    # -- target axes --------------------------------------------------------
    # pykonal spherical axes are ascending: rho (up), theta (north->south),
    # phi (west->east).  So depth descends and latitude descends.
    depths = np.linspace(dep_max, dep_min, n_rad)
    lats = np.linspace(lat_max, lat_min, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)

    # -- resample -----------------------------------------------------------
    interp = RegularGridInterpolator(
        (x, y, z), vel, method='linear',
        bounds_error=False, fill_value=None)

    LAT, LON = np.meshgrid(lats, lons, indexing='ij')       # (n_lat, n_lon)
    X, Y = nll.project(LON.ravel(), LAT.ravel())            # projection is
    X = np.asarray(X, dtype=np.float64)                     # depth-independent,
    Y = np.asarray(Y, dtype=np.float64)                     # so do it once

    outside = ((X < x[0]) | (X > x[-1]) | (Y < y[0]) | (Y > y[-1]))
    X = np.clip(X, x[0], x[-1])       # clamp to nearest edge rather than
    Y = np.clip(Y, y[0], y[-1])       # extrapolate

    if fill == 'nan':
        warnings.warn(
            'fill="nan" produces a velocity field that will break a '
            'fast-marching solve (NaN propagates through the traveltime '
            'field). Use it for inspection only.')
    if fill == 'clamp':
        fill_value = None
    elif fill == 'nan':
        fill_value = np.nan
    else:
        fill_value = float(fill)

    values = np.empty((n_rad, n_lat, n_lon), dtype=np.float64)
    zq = np.clip(depths, z[0], z[-1])
    out2d = outside.reshape(n_lat, n_lon)
    for i, zi in enumerate(zq):
        pts = np.column_stack([X, Y, np.full(X.shape, zi)])
        layer = interp(pts).reshape(n_lat, n_lon)
        if fill_value is not None:
            layer = np.where(out2d, fill_value, layer)
        values[i] = layer

    # -- build the pykonal field -------------------------------------------
    # Keep longitudes in [0, 360) as pykonal expects a positive phi axis.
    lon_ref = lon_min + 360.0 if lon_min < 0 else lon_min
    if lon_ref + (lon_max - lon_min) > 360.0 and lon_min < 0:
        warnings.warn('Grid straddles the prime meridian; the phi axis will '
                      'span negative longitudes. Check downstream use.')
        lon_ref = lon_min

    pvt = pykonal.fields.ScalarField3D(coord_sys='spherical')
    pvt.min_coords = pykonal.transformations.geo2sph(
        (lat_max, lon_ref, dep_max))           # rho_min, theta_min, phi_min
    pvt.node_intervals = np.array([
        (dep_max - dep_min) / (n_rad - 1),
        np.radians(lat_max - lat_min) / (n_lat - 1),
        np.radians(lon_max - lon_min) / (n_lon - 1),
    ])
    pvt.npts = npts
    pvt.values = np.ascontiguousarray(values)

    if os.path.isfile(pvtfile):
        os.remove(pvtfile)
    pvt.to_hdf(pvtfile)

    if verbose:
        pct = 100.0 * outside.mean()
        print(f'NLL2PVT: {nll.type} {nll.nx}x{nll.ny}x{nll.nz} '
              f'({nll.proj_name}, {nll.proj_ellipsoid}) -> '
              f'{n_rad}x{n_lat}x{n_lon}')
        print(f'  lat  {lat_min:.4f} .. {lat_max:.4f}')
        print(f'  lon  {lon_min:.4f} .. {lon_max:.4f}')
        print(f'  dep  {dep_min:.3f} .. {dep_max:.3f} km')
        print(f'  vel  {np.nanmin(values):.4f} .. {np.nanmax(values):.4f} km/s')
        how = {'clamp': 'edge-clamped'}.get(fill, f'filled with {fill}')
        print(f'  {pct:.2f}% of nodes outside the source grid ({how})')

    return pvt


# ---------------------------------------------------------------------------
# pykonal -> NLL
# ---------------------------------------------------------------------------

def pvt_geographic_extent(pvt):
    """
    (lat_min, lat_max, lon_min, lon_max, dep_min, dep_max) of a pykonal
    spherical field, in degrees / km.  Longitudes are as stored (0..360).
    """
    rho0, th0, ph0 = pvt.min_coords
    rho1, th1, ph1 = pvt.max_coords
    return (90.0 - np.degrees(th1), 90.0 - np.degrees(th0),
            float(np.degrees(ph0)), float(np.degrees(ph1)),
            EARTH_RADIUS - rho1, EARTH_RADIUS - rho0)


def nll_header_from_pvt(pvt, dx=None, proj_name='AZIMUTHAL_EQUIDIST',
                        proj_ellipsoid='WGS-84', map_rot=0.0,
                        first_std_paral=None, second_std_paral=None,
                        nll_type='SLOW_LEN', float_type='FLOAT'):
    """
    Build an empty NLL grid (header only) that covers a pykonal field.

    Used by :func:`PVT2NLL` when no template grid is given.  The recipe
    follows ``xyz_to_nll.convert``: the projection origin is the centre of
    the lat/lon box, the perimeter of the box is pushed through the
    projection to find the x/y extent, the extent is made symmetric about
    the origin and padded by one cell, and cells are cubic (NonLinLoc's
    Grid2Time requires that).  The z axis is node-aligned at the field's
    shallowest depth.

    Parameters
    ----------
    pvt : pykonal.fields.ScalarField3D
    dx : float, optional
        Cubic cell size in km.  Default: the field's latitude node spacing,
        snapped to the nearest 0.5 km when within 1% (otherwise rounded to
        0.01 km), which for a field made by ``xyz_to_pvt`` recovers the
        ``dy`` that was asked for.  If the field has finer vertical
        than horizontal spacing (e.g. dx=5, dz=2.5) the vertical detail is
        lost at the default; pass a smaller `dx` to keep it.
    first_std_paral, second_std_paral : float, optional
        LAMBERT only.  Default to 1/6 and 5/6 of the latitude range.
    """
    lat_min, lat_max, lon_min, lon_max, dep_min, dep_max = \
        pvt_geographic_extent(pvt)

    if dx is None:
        dx = float(pvt.node_intervals[1]) * EARTH_RADIUS
        snapped = round(dx * 2.0) / 2.0          # 4.99 -> 5.0, 8.99 -> 9.0
        dx = snapped if abs(snapped - dx) < 0.01 * dx else round(dx, 2)
    dx = float(dx)
    if dx <= 0:
        raise ValueError(f'cell size must be positive, got {dx}')

    grd = nllgrid.NLLGrid()
    grd.float_type = float_type
    grd.type = nll_type
    grd.dx = grd.dy = grd.dz = dx
    grd.proj_name = proj_name
    grd.proj_ellipsoid = proj_ellipsoid
    grd.map_rot = map_rot
    if proj_name == 'LAMBERT':
        span = lat_max - lat_min
        grd.first_std_paral = (lat_min + span / 6.0
                               if first_std_paral is None else first_std_paral)
        grd.second_std_paral = (lat_max - span / 6.0
                                if second_std_paral is None else second_std_paral)
    lon_mid = 0.5 * (lon_min + lon_max)
    grd.orig_lon = (lon_mid + 180.0) % 360.0 - 180.0
    grd.orig_lat = 0.5 * (lat_min + lat_max)

    # perimeter of the lat/lon box -> projected extent
    n = 401
    t = np.linspace(0.0, 1.0, n)
    lons = np.concatenate([lon_min + t * (lon_max - lon_min)] * 2 +
                          [np.full(n, lon_min), np.full(n, lon_max)])
    lats = np.concatenate([np.full(n, lat_min), np.full(n, lat_max)] +
                          [lat_min + t * (lat_max - lat_min)] * 2)
    px, py = grd.project(lons, lats)
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    if not (np.all(np.isfinite(px)) and np.all(np.isfinite(py))):
        raise ValueError(
            f'{proj_name} projection failed on the field boundary; try a '
            'different --proj for this region.')

    half_x = max(abs(px.min()), abs(px.max())) + dx
    half_y = max(abs(py.min()), abs(py.max())) + dx
    grd.nx = int(np.ceil(2.0 * half_x / dx)) + 1
    grd.ny = int(np.ceil(2.0 * half_y / dx)) + 1
    grd.nz = int(np.ceil((dep_max - dep_min) / dx)) + 1
    grd.x_orig = -0.5 * (grd.nx - 1) * dx
    grd.y_orig = -0.5 * (grd.ny - 1) * dx
    grd.z_orig = float(dep_min)
    return grd


def PVT2NLL(pvtfile, nllfile_template=None, outfile=None, verbose=True,
            header_kw=None):
    """
    Convert a pykonal spherical HDF5 field to a NonLinLoc grid.

    The NLL header (geometry, projection, grid type) comes from an existing
    grid, if `nllfile_template` is given, or is built from the field itself
    (see :func:`nll_header_from_pvt`).  Values are resampled from the
    spherical field onto the NLL nodes, which is the exact inverse operation
    of :func:`NLL2PVT` (not a transpose).

    Parameters
    ----------
    pvtfile : str
        Path to the pykonal HDF5 file.
    nllfile_template : str, optional
        Existing NLL grid used as the header template.  If omitted, a header
        is generated and `outfile` is required.
    outfile : str, optional
        Output basename.  With a template this defaults to the template
        basename with ``.v2`` inserted after the first dot-separated
        component.
    verbose : bool
    header_kw : dict, optional
        Keyword arguments for :func:`nll_header_from_pvt` (``dx``,
        ``proj_name``, ...).  Ignored when a template is given.

    Returns
    -------
    nllgrid.NLLGrid
    """
    pvt = pykonal.fields.read_hdf(pvtfile)
    if nllfile_template is not None:
        nll = nllgrid.NLLGrid(nllfile_template)
    else:
        if outfile is None:
            raise ValueError('outfile is required when no template is given')
        nll = nll_header_from_pvt(pvt, **(header_kw or {}))

    rho = np.linspace(pvt.min_coords[0], pvt.max_coords[0], pvt.npts[0])
    theta = np.linspace(pvt.min_coords[1], pvt.max_coords[1], pvt.npts[1])
    phi = np.linspace(pvt.min_coords[2], pvt.max_coords[2], pvt.npts[2])

    interp = RegularGridInterpolator(
        (rho, theta, phi), np.asarray(pvt.values, dtype=np.float64),
        method='linear', bounds_error=False, fill_value=None)

    x, y, z = nll_node_coords(nll)
    X, Y = np.meshgrid(x, y, indexing='ij')                 # (nx, ny)
    lon, lat = nll.iproject(X.ravel(), Y.ravel())
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)

    # Match the phi branch used by the field.
    phi_mid = np.degrees(0.5 * (phi[0] + phi[-1]))
    lon = phi_mid + (lon - phi_mid + 180.0) % 360.0 - 180.0

    th = np.radians(90.0 - lat)
    ph = np.radians(lon)
    th_c = np.clip(th, theta[0], theta[-1])
    ph_c = np.clip(ph, phi[0], phi[-1])
    outside = (th != th_c) | (ph != ph_c)

    vel = np.empty((nll.nx, nll.ny, nll.nz), dtype=np.float64)
    for k, zk in enumerate(z):
        r = np.clip(EARTH_RADIUS - zk, rho[0], rho[-1])
        pts = np.column_stack([np.full(th_c.shape, r), th_c, ph_c])
        vel[:, :, k] = interp(pts).reshape(nll.nx, nll.ny)

    nll.array = velocity_to_nll(nll, vel).astype(np.float64)

    if outfile is None:
        dirname = os.path.dirname(nll.basename)
        parts = os.path.basename(nll.basename).split('.')
        parts.insert(1, 'v2')
        outfile = os.path.join(dirname, '.'.join(parts))
    nll.basename = outfile

    nll.write_hdr_file()
    nll.write_buf_file()

    if verbose:
        pct = 100.0 * outside.mean()
        print(f'PVT2NLL: {tuple(int(n) for n in pvt.npts)} -> '
              f'{nll.nx}x{nll.ny}x{nll.nz} {nll.type}  [{outfile}]')
        print(f'  {nll.proj_name} ({nll.proj_ellipsoid}) origin '
              f'{nll.orig_lat:.4f}, {nll.orig_lon:.4f}; '
              f'cell {nll.dx:g} x {nll.dy:g} x {nll.dz:g} km')
        print(f'  vel  {vel.min():.4f} .. {vel.max():.4f} km/s')
        print(f'  {pct:.2f}% of nodes outside the spherical field '
              '(edge-clamped)')

    return nll


# ---------------------------------------------------------------------------
# Format detection and command line
# ---------------------------------------------------------------------------

_HDF5_MAGIC = b'\x89HDF\r\n\x1a\n'
_PVT_EXTS = ('.h5', '.hdf5', '.hdf')
_NLL_PROJECTIONS = ('AZIMUTHAL_EQUIDIST', 'LAMBERT', 'TRANS_MERC', 'SIMPLE')


def _strip_nll_ext(path):
    for ext in ('.hdr', '.buf'):
        if path.endswith(ext):
            return path[:-len(ext)]
    return path


def detect_format(path):
    """
    Work out whether `path` is a pykonal HDF5 field or an NLL grid.

    * a file that starts with the HDF5 signature            -> ``'pvt'``
    * a basename (or .hdr / .buf name) with both
      ``<base>.hdr`` and ``<base>.buf`` on disk              -> ``'nll'``

    The file *content* is checked, not the extension, so a pykonal file
    called ``foo.mod`` is still recognised.

    Returns
    -------
    (fmt, path) : (str, str)
        `path` is returned as the HDF5 file for 'pvt', or as the NLL
        basename (extension stripped) for 'nll'.

    Raises
    ------
    FileNotFoundError, ValueError
    """
    if os.path.isfile(path):
        with open(path, 'rb') as fp:
            if fp.read(len(_HDF5_MAGIC)) == _HDF5_MAGIC:
                return 'pvt', path

    base = _strip_nll_ext(path)
    have_hdr = os.path.isfile(base + '.hdr')
    have_buf = os.path.isfile(base + '.buf')
    if have_hdr and have_buf:
        return 'nll', base
    if have_hdr or have_buf:
        missing = base + ('.buf' if have_hdr else '.hdr')
        raise FileNotFoundError(
            f'{path!r} looks like an NLL grid but {missing} is missing')
    if os.path.exists(path):
        raise ValueError(
            f'{path!r} is neither an HDF5 (pykonal) file nor an NLL '
            '.hdr/.buf grid')
    raise FileNotFoundError(f'no such model: {path!r} (tried {base}.hdr/.buf)')


def convert_model(inmodel, outmodel, nll_opts=None, pvt_opts=None,
                  verbose=True):
    """
    Convert `inmodel` to the other format, detected from its content.

    Parameters
    ----------
    inmodel : str
        NLL grid (basename / .hdr / .buf) or pykonal HDF5 file.
    outmodel : str
        Output.  For an NLL result this is the basename (``.hdr``/``.buf``
        are added; if given they are stripped).  For a pykonal result a
        ``.h5`` extension is appended unless one is already present.
    nll_opts : dict, optional
        pykonal -> NLL only.  ``template`` (existing NLL grid for the
        header) and/or the :func:`nll_header_from_pvt` keywords.
    pvt_opts : dict, optional
        NLL -> pykonal only.  Keyword arguments for :func:`NLL2PVT`.

    Returns
    -------
    (fmt_in, fmt_out, output_path)
    """
    nll_opts = dict(nll_opts or {})
    pvt_opts = dict(pvt_opts or {})
    fmt, src = detect_format(inmodel)

    if fmt == 'nll':
        if nll_opts:
            raise ValueError('header options (template, dx, proj, ...) only '
                             'apply when converting pykonal -> NLL')
        dst = outmodel if outmodel.lower().endswith(_PVT_EXTS) \
            else outmodel + '.h5'
        if verbose:
            print(f'Input {src!r} is an NLL grid -> writing pykonal {dst!r}')
        _prepare_output(dst, verbose)
        NLL2PVT(src, dst, verbose=verbose, **pvt_opts)
        return 'nll', 'pvt', dst

    if pvt_opts:
        raise ValueError('bounds / fill / clip / oversample options only '
                         'apply when converting NLL -> pykonal')
    dst = _strip_nll_ext(outmodel)
    template = nll_opts.pop('template', None)
    if template is not None:
        template = _strip_nll_ext(template)
        if nll_opts:
            raise ValueError('--template supplies the whole header; it '
                             'cannot be combined with dx/proj/... options')
    if verbose:
        print(f'Input {src!r} is a pykonal field -> writing NLL grid {dst!r}')
    _prepare_output(dst + '.hdr', verbose)
    PVT2NLL(src, template, outfile=dst, verbose=verbose,
            header_kw=nll_opts or None)
    return 'pvt', 'nll', dst


def _prepare_output(path, verbose):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    if verbose and os.path.exists(path):
        print(f'  (overwriting existing {path})')


def _fill_arg(text):
    if text in ('clamp', 'nan'):
        return text
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r}: expected 'clamp', 'nan' or a velocity in km/s")


def _build_parser():
    ap = argparse.ArgumentParser(
        prog='convert_model.py',
        description='Convert a velocity model between NonLinLoc (NLL) and '
                    'pykonal (PVT) formats.  The input format is detected '
                    'automatically and the output is the other one.  One '
                    'field (e.g. P) is converted per run.',
        epilog='examples:\n'
               '  convert_model.py WA_9km.P.mod WA_9km.P.h5        '
               'NLL -> pykonal\n'
               '  convert_model.py WA_5km.P.h5 WA_5km.P.mod        '
               'pykonal -> NLL (header generated)\n'
               '  convert_model.py WA_5km.P.h5 out/P.mod --template '
               'old.P.mod\n'
               '                                                   '
               'pykonal -> NLL (header copied)',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inmodel', help='NLL grid (basename, .hdr or .buf) or '
                                    'pykonal HDF5 file')
    ap.add_argument('outmodel', help='output; NLL basename, or pykonal '
                                     '.h5 file (".h5" appended if missing)')
    ap.add_argument('-q', '--quiet', action='store_true',
                    help='suppress the conversion summary')

    g = ap.add_argument_group(
        'pykonal -> NLL options',
        'The NLL header is built from the field unless --template is given.')
    g.add_argument('--template', metavar='NLL',
                   help='existing NLL grid to copy the header '
                        '(geometry, projection, type) from')
    g.add_argument('--dx', type=float, metavar='KM',
                   help='cubic cell size in km (default: the field\'s '
                        'latitude node spacing)')
    g.add_argument('--proj', choices=_NLL_PROJECTIONS,
                   help='map projection (default: AZIMUTHAL_EQUIDIST)')
    g.add_argument('--ellipsoid', metavar='NAME',
                   help='projection ellipsoid (default: WGS-84)')
    g.add_argument('--map-rot', type=float, metavar='DEG',
                   help='map rotation, degrees CW (default: 0)')
    g.add_argument('--first-paral', type=float, metavar='LAT',
                   help='LAMBERT first standard parallel')
    g.add_argument('--second-paral', type=float, metavar='LAT',
                   help='LAMBERT second standard parallel')
    g.add_argument('--float-type', choices=('FLOAT', 'DOUBLE'),
                   help='NLL buffer precision (default: FLOAT)')

    g = ap.add_argument_group(
        'NLL -> pykonal options',
        'Control the lat/lon box and how nodes outside the NLL grid are '
        'filled.')
    g.add_argument('--bounds', choices=('inner', 'outer'),
                   help="'inner' (default): largest lat/lon box fully inside "
                        "the NLL grid; 'outer': box enclosing the whole grid")
    g.add_argument('--bbox', nargs=4, type=float,
                   metavar=('LATMIN', 'LATMAX', 'LONMIN', 'LONMAX'),
                   help='explicit lat/lon box; overrides --bounds')
    g.add_argument('--fill', type=_fill_arg, metavar='MODE',
                   help="nodes outside the NLL grid: 'clamp' (default, "
                        "nearest edge), 'nan' (inspection only!) or a "
                        "velocity in km/s")
    g.add_argument('--clip', nargs=2, type=float, metavar=('VMIN', 'VMAX'),
                   help='clamp source velocities into this range (km/s) '
                        'before resampling')
    g.add_argument('--oversample', type=float, metavar='F',
                   help='multiply the default node count by F (default: 1)')
    return ap


def main(argv=None):
    ap = _build_parser()
    a = ap.parse_args(argv)

    nll_opts = {}
    if a.template is not None:
        nll_opts['template'] = a.template
    for key, val in (('dx', a.dx), ('proj_name', a.proj),
                     ('proj_ellipsoid', a.ellipsoid), ('map_rot', a.map_rot),
                     ('first_std_paral', a.first_paral),
                     ('second_std_paral', a.second_paral),
                     ('float_type', a.float_type)):
        if val is not None:
            nll_opts[key] = val

    pvt_opts = {}
    if a.bbox is not None:
        pvt_opts['bounds'] = tuple(a.bbox)
    elif a.bounds is not None:
        pvt_opts['bounds'] = a.bounds
    if a.fill is not None:
        pvt_opts['fill'] = a.fill
    if a.clip is not None:
        pvt_opts['clip'] = tuple(a.clip)
    if a.oversample is not None:
        pvt_opts['oversample'] = a.oversample

    try:
        convert_model(a.inmodel, a.outmodel, nll_opts=nll_opts,
                      pvt_opts=pvt_opts, verbose=not a.quiet)
    except (OSError, ValueError) as exc:
        ap.exit(1, f'convert_model.py: error: {exc}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())