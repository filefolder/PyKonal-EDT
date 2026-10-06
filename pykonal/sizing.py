#!/usr/bin/env python3
"""
sizing.py

Search-box sizing and depth handling, shared by the SeisComP locator
(pykonal_locext.py) and the ObsPy relocator (pykonal_obspy.py).

This module exists so the two drivers (obspy and seiscomp) cannot diverge.
Everything that decides WHERE the locator searches lives here: 
the geometry-based sigma, the measured posterior sigma, the floors, the cap,
the depth clamp and the confidence scaling. The drivers are responsible only
for getting picks and stations into a locator and for writing results out
in their own format.

No SeisComP and no ObsPy imports -- numpy and the standard library only.

Coordinate conventions
----------------------
Grid-axis order is (r, theta, phi) for spherical grids, so axis 0 is DEPTH
and increasing depth means DECREASING r. For cartesian it is (x, y, z) and
axis 2 is depth. Every quantity in this module in "grid-axis order" follows
that; anything in "config order" is [lat/north, lon/east, depth] as written
in the JSON. The two are different and mixing them up has been the source of
several bugs, so conversions go through _config_to_grid_order() and the depth
axis is always addressed via depth_axis().
"""

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    "SearchBox", "chi_scale", "auto_search_delta", "depth_axis",
    "resolve_depth_bounds", "clamp_depth_to_bounds", "size_auto_box",
    "size_fixed_box", "AUTO_DEFAULTS", "ellipsoid_orientation",
    "quakeml_rotation_matrix", "EDGE_DENSITY_LIMIT", "describe_search_box",
    "box_limit_report", "widen_truncated_axes", "cap_delta_grid",
    "suggest_search_delta_km", "describe_widening",
]


AUTO_DEFAULTS = {
    "auto_search_k_sigma": 4.0,
    "auto_search_cond_max": 1.0e6,
    "auto_search_max_km": [200.0, 200.0, 60.0],
    "auto_search_min_km": [2.0, 2.0, 5.0],
    "auto_search_fallback_sigma_km": [15.0, 15.0, 15.0],
    "auto_search_depth_dist_factor": 0.5,
    "auto_search_posterior_nsamples": 512,
    "auto_search_posterior_min_ess": 20.0,
    "auto_search_provisional_km": [150.0, 150.0, 60.0],
}

_KM_PER_DEG = 111.195


@dataclass
class SearchBox:
    """
    The result of sizing. `delta_grid` and `bounds` drive locate();
    `posterior_delta` drives sample_posterior() and is deliberately NOT
    depth-clamped, so a pinned or clamped depth still gets an honest
    uncertainty measured over the full extent.
    """
    delta_grid: np.ndarray
    posterior_delta: np.ndarray
    bounds: Optional[np.ndarray] = None
    sigma_km: Optional[np.ndarray] = None
    half_widths_km: Optional[np.ndarray] = None
    depth_pinned: bool = False
    provisional_solution: Optional[np.ndarray] = None
    info: Dict = field(default_factory=dict)

    def delta_full(self, delta_t: float) -> np.ndarray:
        """The 4-vector locate() wants."""
        return np.append(np.asarray(self.delta_grid, dtype=float), float(delta_t))



# Small shared helpers
# ---------------------------------------------------------------------------
def depth_axis(coord_sys: str) -> int:
    """Index of the depth-carrying axis in grid coordinates."""
    return 0 if coord_sys == "spherical" else 2


def axis_names(coord_sys: str) -> Tuple[str, str, str]:
    """Human labels in grid-axis order, for log messages."""
    return (("depth", "north-south", "east-west") if coord_sys == "spherical"
            else ("north-south", "east-west", "depth"))


def _config_to_grid_order(values, coord_sys: str) -> np.ndarray:
    """[lat, lon, depth] (config order) -> grid-axis order."""
    v = np.asarray(values, dtype=float)
    return v[[2, 0, 1]] if coord_sys == "spherical" else v


def _grid_order_to_delta(transform, hw_grid_order, at):
    hw = np.asarray(hw_grid_order, dtype=float)
    if transform.coord_sys == "spherical":
        return np.asarray(transform.delta_to_grid(hw[1], hw[2], hw[0], at=at),
                          dtype=float)
    return np.asarray(transform.delta_to_grid(hw[0], hw[1], hw[2]), dtype=float)


def _cfg(cfg, key):
    return cfg.get(key, AUTO_DEFAULTS[key]) if cfg else AUTO_DEFAULTS[key]


def _noop(*_a, **_k):
    pass


def _pin_half_width(transform, at, tolerance_km):
    """
    Depth half-width in GRID units for a pinned depth.

    Not zero: TraveltimeInventory.read rejects min >= max, and a small
    non-zero window also lets the solver report an honest depth sensitivity
    rather than a hard pin. The value is the operator's configured
    tolerance, not an epsilon -- 2 km and 1 m are very different claims
    about how firmly the depth is held.
    """
    tol = max(float(tolerance_km), 1e-3)
    iz = depth_axis(transform.coord_sys)
    return abs(float(np.asarray(
        transform.delta_to_grid(0.0, 0.0, tol, at=at))[iz]))


def chi_scale(percent: float, dof: int) -> float:
    """
    Number of sigmas enclosing `percent` of a `dof`-dimensional Gaussian.

    dof=3 for the confidence ellipsoid, 2 for the horizontal ellipse, 1 for
    a single marginal (latitude, longitude, depth). These differ a lot: at
    68.27% they are 1.878, 1.515 and 1.000 respectively, so quoting a
    1-sigma semi-axis as a "68% ellipsoid" understates it by ~47%.

    Exact for dof 1 and 2; Wilson-Hilferty for dof 3 (error < 0.5% over the
    range of levels anyone reports). No scipy dependency.
    """
    p = min(max(percent / 100.0, 1e-6), 1.0 - 1e-9)
    nd = NormalDist()
    if dof == 1:
        return nd.inv_cdf(0.5 * (1.0 + p))
    if dof == 2:
        return math.sqrt(-2.0 * math.log(1.0 - p))
    z = nd.inv_cdf(p)
    return math.sqrt(
        dof * (1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))) ** 3
    )


# Geometry-based sigma
# ---------------------------------------------------------------------------
def auto_search_delta(locator, initial_xyz, arrival_keys, sigma_s,
                      coord_sys, k_reach=3.5, cond_max=1.0e6):
    """
    Geometry-based search half-widths from the predicted location
    covariance C = sigma^2 (G^T G)^-1. Each row of G is one arrival's
    traveltime spatial gradient at the trial hypocenter (the ray slowness
    vector) plus a unit origin-time column.

    Spherical grids: axis 0 is r (radius, km; depth = R - r), axes 1-2 are
    colatitude and longitude in RADIANS. The finite-difference step is one
    node interval per axis (a flat step in radians would land off-grid),
    and the angular derivatives are converted to per-km via the local
    metric (d/dtheta / r, d/dphi / (r sin theta)) so all three columns of G
    are d(tt)/d(distance_km). The returned half-widths are converted back
    to grid units for the locator.

    NOTE: C assumes INDEPENDENT residuals. Where the velocity model does
    not fit, residuals trend with distance, the effective number of
    independent observations is well below N, and this underestimates the
    true spread -- worst on depth, which trades against origin time. That
    is why size_auto_box() also measures a posterior and takes the larger.

    Returns (delta_grid, info). info has: sigma (per-axis 1-sigma in km, in
    grid-axis order), cond (of G^T G), n (rows used), weak (cond>cond_max).
    Returns (None, info) if too few gradients or the inverse fails.
    """
    spherical = (coord_sys == "spherical")
    r0 = float(initial_xyz[0]) if spherical else None
    theta0 = float(initial_xyz[1]) if spherical else None
    rows = []
    info = {"sigma": None, "cond": float("inf"), "n": 0, "weak": True}

    for key in arrival_keys:
        field_ = locator.traveltimes.get(key) if hasattr(
            locator, "traveltimes") else None
        if field_ is None:
            continue
        node = np.asarray(field_.node_intervals, dtype=float)
        grad, ok = [], True
        for ax in range(3):
            h = float(node[ax])
            if not np.isfinite(h) or h <= 0:
                ok = False
                break
            xp = np.asarray(initial_xyz, dtype=float).copy()
            xm = np.asarray(initial_xyz, dtype=float).copy()
            xp[ax] += h
            xm[ax] -= h
            try:
                tp = field_.value(xp)
                tm = field_.value(xm)
            except Exception:
                ok = False
                break
            if not (np.isfinite(tp) and np.isfinite(tm)):
                ok = False
                break
            d = (tp - tm) / (2.0 * h)
            if spherical:                       # radians -> per-km
                if ax == 1:
                    d /= r0
                elif ax == 2:
                    d /= (r0 * max(math.sin(theta0), 1e-6))
            grad.append(d)
        if ok and len(grad) == 3:
            rows.append(grad + [1.0])           # unit origin-time column

    info["n"] = len(rows)
    if len(rows) < 4:
        return None, info

    G = np.array(rows)
    GtG = G.T @ G
    try:
        cond = float(np.linalg.cond(GtG))
        C = sigma_s ** 2 * np.linalg.inv(GtG)
    except np.linalg.LinAlgError:
        return None, info

    sigma_km = np.sqrt(np.clip(np.diag(C)[:3], 0.0, None))
    info["cond"] = cond
    info["sigma"] = sigma_km
    info["weak"] = cond > cond_max
    hw_km = k_reach * sigma_km
    if spherical:
        delta_grid = np.array([
            hw_km[0],
            hw_km[1] / r0,
            hw_km[2] / (r0 * max(math.sin(theta0), 1e-6)),
        ], dtype=float)
    else:
        delta_grid = hw_km
    return delta_grid, info


# Depth bounds and clamping
# ---------------------------------------------------------------------------
def resolve_depth_bounds(fixed_depth, depth_min, depth_max):
    """
    A pinned depth IS a depth bound of zero width. Expressing it that way
    means the clamp is the only depth logic anywhere -- there is no separate
    fixed-depth path to keep in sync with it, which is what previously let
    the SeisComP locator move an operator-pinned depth by several km.

    Returns (depth_min, depth_max, pinned).
    """
    if fixed_depth is not None:
        d = float(fixed_depth)
        return d, d, True
    return (None if depth_min is None else float(depth_min),
            None if depth_max is None else float(depth_max),
            False)


def clamp_depth_to_bounds(transform, initial, delta, dep0_km,
                          depth_min, depth_max, lat0=0.0, lon0=0.0, *,
                          min_half_width_km=1e-3, log=None):
    """
    Restrict the depth search to [depth_min, depth_max], returned as explicit
    asymmetric grid bounds of shape (2, 4).

    The anchor is NEVER moved. Re-expressing a truncated interval as
    centre +/- half-width reproduces the same interval but shifts the centre
    off the anchor; on an axis the data does not constrain, the solution then
    tracks the centre, and fed back into a repeat relocation that walks the
    hypocenter deeper a few km at a time. The recursion has a fixed point at
    depth_min + half_width regardless of the data, which is exactly what was
    observed.

    Returns None when no bound applies.
    """
    log = log or _noop
    if depth_min is None and depth_max is None:
        return None

    iz = depth_axis(transform.coord_sys)
    initial = np.asarray(initial, dtype=float)
    delta = np.asarray(delta, dtype=float)

    # current depth half-width in km
    if transform.coord_sys == "spherical":
        hw_km = abs(float(delta[iz]))          # axis 0 is r, already km
    else:
        hw_km = abs(float(delta[iz]))

    lo = -np.inf if depth_min is None else float(depth_min)
    hi = np.inf if depth_max is None else float(depth_max)
    if lo > hi:
        raise ValueError(
            f"depth_min ({lo} km) is greater than depth_max ({hi} km)")
    d_lo = max(lo, dep0_km - hw_km)
    d_hi = min(hi, dep0_km + hw_km)
    if d_hi < d_lo:
        # No overlap. The bound on the anchor's side is necessarily finite.
        if dep0_km < lo:
            d_lo, d_hi = lo, min(hi, lo + 2.0 * hw_km)
        else:
            d_lo, d_hi = max(lo, hi - 2.0 * hw_km), hi
        log(f"depth clamp: anchor {dep0_km:.1f} km +/- {hw_km:.1f} km lies "
            f"outside the allowed depth range [{lo:g}, {hi:g}] km; searching "
            f"{d_lo:.1f} to {d_hi:.1f} km against the nearer bound")

    lo_grid = (initial[:3] - delta[:3]).copy()
    hi_grid = (initial[:3] + delta[:3]).copy()

    # depth increases downward, but axis 0 of a spherical grid is RADIUS,
    # which decreases downward -- convert both ends and order them rather
    # than assuming a sign.
    g1 = float(np.asarray(transform.geo_to_grid(lat0, lon0, d_lo))[iz])
    g2 = float(np.asarray(transform.geo_to_grid(lat0, lon0, d_hi))[iz])
    lo_grid[iz], hi_grid[iz] = min(g1, g2), max(g1, g2)

    # A zero-width depth bound (an operator-pinned depth) must stay > 0:
    # TraveltimeInventory.read rejects min >= max. Widen to the caller's
    # tolerance rather than an arbitrary epsilon, so the interval the solver
    # sees matches the tolerance the operator configured.
    _tol = max(float(min_half_width_km), 1e-3)
    _tol_grid = abs(float(np.asarray(
        transform.delta_to_grid(0.0, 0.0, _tol, at=initial[:3]))[iz]))
    if hi_grid[iz] - lo_grid[iz] < 2.0 * _tol_grid:
        mid = 0.5 * (lo_grid[iz] + hi_grid[iz])
        lo_grid[iz], hi_grid[iz] = mid - _tol_grid, mid + _tol_grid

    asym = abs((d_hi - dep0_km) - (dep0_km - d_lo))
    log(f"depth clamp: search interval {d_lo:.1f} to {d_hi:.1f} km "
        f"(anchor {dep0_km:.1f} km, half-width {hw_km:.1f} km); anchor "
        f"retained, interval asymmetric by {asym:.1f} km")

    t_lo = float(initial[3] - delta[3]) if len(initial) > 3 else 0.0
    t_hi = float(initial[3] + delta[3]) if len(initial) > 3 else 0.0
    return np.stack([np.append(lo_grid, t_lo), np.append(hi_grid, t_hi)])


# Floors and cap
# ---------------------------------------------------------------------------
def apply_floors_and_cap(hw_km, sigma_km, coord_sys, cfg, *,
                         provisional_offset_km=None,
                         min_station_dist_km=None, log=None):
    """
    Raise the half-widths to the floors, then cap them. Order matters: a
    floor must never exceed auto_search_max_km, which bounds the grid volume.

    Floor 1 (coverage): the box must contain the provisional solution. The
      provisional pass searches wide and the real pass searches k*sigma
      around the SAME anchor, with nothing otherwise checking that the second
      contains the first. Where pass 1 moved the event a long way, a tight
      box centred back on the anchor cannot reach it, and since the search is
      seeded at the anchor the result is a confident location AT the anchor.

    Floor 2 (depth resolution): depth is not resolvable much better than the
      nearest station offset, because beyond roughly one focal depth the
      takeoff angle stops changing with depth and the arrival constrains
      origin time instead. Heuristic, not derived. It saturates against the
      cap for regional networks (0.5 x a 236 km nearest station is 118 km),
      which is harmless but means the floor log fires on every such event --
      consider auto_search_depth_dist_factor: 0 if that is all your network
      ever sees.

    Floor 3 (absolute): a hard minimum, so no path can produce a box narrower
      than the model and grid can meaningfully resolve.
    """
    log = log or _noop
    hw = np.asarray(hw_km, dtype=float).copy()
    names = axis_names(coord_sys)
    iz = depth_axis(coord_sys)

    hw_cov = np.zeros(3)
    if provisional_offset_km is not None:
        hw_cov = 1.5 * np.asarray(provisional_offset_km, dtype=float) \
            + np.asarray(sigma_km, dtype=float)

    hw_geom = np.zeros(3)
    if min_station_dist_km is not None and np.isfinite(min_station_dist_km):
        hw_geom[iz] = float(_cfg(cfg, "auto_search_depth_dist_factor")) \
            * float(min_station_dist_km)

    hw_abs = _config_to_grid_order(_cfg(cfg, "auto_search_min_km"), coord_sys)

    raw = hw.copy()
    hw = np.maximum.reduce([hw, hw_cov, hw_geom, hw_abs])
    floored = [n for n, a, b in zip(names, raw, hw) if b > a + 1e-6]
    if floored:
        log("search_delta auto: half-width floored on " + ", ".join(floored)
            + " (k*sigma would have given "
            + "/".join(f"{v:.1f}" for v in raw) + " km, using "
            + "/".join(f"{v:.1f}" for v in hw) + " km)")

    cap_km = _config_to_grid_order(_cfg(cfg, "auto_search_max_km"), coord_sys)
    capped = [n for n, a, c in zip(names, hw, cap_km) if a > c + 1e-6]
    hw = np.minimum(hw, cap_km)
    if capped:
        log("search_delta auto: half-width capped by auto_search_max_km on "
            + ", ".join(capped) + " (using "
            + "/".join(f"{v:.1f}" for v in hw) + " km)")
    return hw


# Top-level sizing
# ---------------------------------------------------------------------------
def size_fixed_box(transform, initial, delta_t, search_delta_km, dep0_km, *,
                   fixed_depth=None, fixed_depth_tolerance=2.0,
                   depth_min=None, depth_max=None,
                   lat0=0.0, lon0=0.0, log=None):
    """
    An explicitly configured box (search_delta_km as a list), with the same
    depth handling the auto path uses.
    """
    log = log or _noop
    depth_min, depth_max, pinned = resolve_depth_bounds(
        fixed_depth, depth_min, depth_max)

    hw_grid_order = _config_to_grid_order(search_delta_km,
                                          transform.coord_sys)
    delta_grid = _grid_order_to_delta(transform, hw_grid_order, np.asarray(initial, float)[:3])
    posterior_delta = delta_grid.copy()      # never depth-clamped

    if pinned:
        delta_grid = delta_grid.copy()
        delta_grid[depth_axis(transform.coord_sys)] = _pin_half_width(
            transform, np.asarray(initial, float)[:3], fixed_depth_tolerance)

    full_initial = np.asarray(initial, dtype=float)
    full_delta = np.append(delta_grid, float(delta_t))
    bounds = clamp_depth_to_bounds(transform, full_initial, full_delta,
                                   dep0_km, depth_min, depth_max, lat0, lon0,
                                   min_half_width_km=fixed_depth_tolerance,
                                   log=log)
    return SearchBox(delta_grid=delta_grid, posterior_delta=posterior_delta,
                     bounds=bounds, half_widths_km=hw_grid_order,
                     depth_pinned=pinned,
                     info={"mode": "fixed"})


def size_auto_box(locator, transform, initial, delta_t, *,
                  arrival_keys, dep0_km, lat0, lon0,
                  cfg=None, method="edt", alpha=0.01,
                  default_pick_error=0.1, posterior_seed=None,
                  station_distances_km=None,
                  fixed_depth=None, fixed_depth_tolerance=2.0,
                  depth_min=None, depth_max=None,
                  log=None, stage=None):
    """
    Two-pass geometry-based sizing.

    The box must scale with the ACTUAL location uncertainty, which is set by
    how well the model fits -- the a posteriori data variance
    s^2 = sum(r^2)/(N-M) -- not by the nominal pick error, which is only a
    floor. The misfit is unknown before locating, so:

      pass 1: locate in a wide provisional box, measure the rms residual and
              sample a cheap posterior;
      pass 2: size from max(geometry prediction, measured posterior), apply
              the floors and the cap, then clamp depth.

    `station_distances_km` is an optional {(net, sta): distance_km} or a bare
    sequence of distances, used only for floor 2.

    Returns a SearchBox. The caller performs pass 2 itself.
    """
    log = log or _noop
    stage = stage or _noop
    cfg = cfg or {}
    coord_sys = transform.coord_sys
    k = float(_cfg(cfg, "auto_search_k_sigma"))
    initial = np.asarray(initial, dtype=float)
    initial_xyz = initial[:3]

    # ---- pass 1: wide provisional box
    prov = _grid_order_to_delta(
        transform, _config_to_grid_order(_cfg(cfg, "auto_search_provisional_km"),
                                         coord_sys), initial_xyz)
    prov = np.minimum(prov, _grid_order_to_delta(
        transform, _config_to_grid_order(_cfg(cfg, "auto_search_max_km"),
                                         coord_sys), initial_xyz))
    locator.read_traveltimes(min_coords=(initial_xyz - prov),
                             max_coords=(initial_xyz + prov))
    stage("provisional read_traveltimes")
    prov_soln = np.asarray(
        locator.locate(initial, np.append(prov, delta_t), alpha, method),
        dtype=float)
    stage("provisional locate (pass 1)")

    prov_res = locator.residuals(prov_soln)
    if prov_res:
        rv = np.array(list(prov_res.values()), dtype=float)
        s_data = float(np.sqrt(np.sum(rv ** 2) / max(len(rv) - 4, 1)))
    else:
        s_data = float(default_pick_error)
    s_eff = max(s_data, float(default_pick_error))

    # ---- measured sigma from a cheap posterior in the provisional box
    meas_sig = None
    prov_n = int(_cfg(cfg, "auto_search_posterior_nsamples"))
    if method == "edt" and prov_n > 0:
        try:
            pp = locator.sample_posterior(prov_soln[:3], prov, nsamples=prov_n,
                                          nscatter=0, seed=posterior_seed,
                                          rounds=2)
            s = np.sqrt(np.clip(
                np.diag(np.asarray(pp["covariance_km"], float)), 0.0, None))
            pess = float(pp.get("ess", float("nan")))
            pmc = pp.get("sigma_rel_mcse")
            pmc = float("nan") if pmc is None else float(pmc)
            ess_min = float(_cfg(cfg, "auto_search_posterior_min_ess"))
            if np.all(np.isfinite(s)) and s.min() > 0 and pess >= ess_min \
                    and (not np.isfinite(pmc) or pmc < 0.5):
                meas_sig = s
                edge = np.asarray(pp.get("at_search_edge", [False] * 3),
                                  dtype=bool)
                if edge.any():
                    hit = ", ".join(n for n, e in zip(axis_names(coord_sys), edge)
                                    if e)
                    log(f"search_delta auto: provisional solution sits against "
                        f"the searched edge on {hit} -- the likelihood is "
                        f"monotone there and the data do not locate that axis "
                        f"at all")
            else:
                log(f"search_delta auto: provisional posterior not usable for "
                    f"sizing (ess={pess:.0f}, mcse={pmc:.2f}); using the "
                    f"geometry prediction")
        except Exception as exc:
            log(f"search_delta auto: provisional posterior failed ({exc}); "
                f"using the geometry prediction")
    stage("provisional posterior")

    # ---- geometry prediction
    _adx, ainfo = auto_search_delta(
        locator, initial_xyz, list(arrival_keys), s_eff, coord_sys,
        k_reach=k, cond_max=float(_cfg(cfg, "auto_search_cond_max")))

    sig_pred = ainfo.get("sigma")
    if sig_pred is None:
        # Fall back to a wide SIGMA, never to a wide box: a box assigned
        # directly bypasses the cap, the floors and the measured posterior.
        if meas_sig is not None:
            sigma = meas_sig
            why = "geometry unusable; sized from the measured provisional posterior"
        else:
            sigma = _config_to_grid_order(
                _cfg(cfg, "auto_search_fallback_sigma_km"), coord_sys)
            why = "geometry unusable and no measured posterior; wide default sigma"
        log(f"search_delta auto: {why} (insufficient or near-singular "
            f"gradients, s_data={s_data:.2f}s)")
    elif meas_sig is not None:
        # Take the LARGER per axis: the failure mode is one-sided. A box
        # narrower than the peak truncates the pdf and understates the
        # uncertainty; an oversized box only costs sampling density.
        sigma = np.maximum(sig_pred, meas_sig)
    else:
        sigma = sig_pred

    hw = k * np.asarray(sigma, dtype=float)

    # ---- floors and cap
    prov_geo = transform.grid_to_geo(prov_soln[:3])
    off_lat = abs(float(prov_geo[0]) - lat0) * _KM_PER_DEG
    # wrap the longitude difference into [-180, 180]: across the antimeridian
    dlon = (float(prov_geo[1]) - lon0 + 180.0) % 360.0 - 180.0
    off_lon = abs(dlon) * _KM_PER_DEG * math.cos(math.radians(lat0))
    off_dep = abs(float(prov_geo[2]) - dep0_km)
    offset = (np.array([off_dep, off_lat, off_lon]) if coord_sys == "spherical"
              else np.array([off_lat, off_lon, off_dep]))

    dmin_km = None
    if station_distances_km:
        vals = (list(station_distances_km.values())
                if isinstance(station_distances_km, dict)
                else list(station_distances_km))
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if vals:
            dmin_km = min(vals)

    hw = apply_floors_and_cap(hw, sigma, coord_sys, cfg,
                              provisional_offset_km=offset,
                              min_station_dist_km=dmin_km, log=log)

    delta_grid = _grid_order_to_delta(transform, hw, initial_xyz)

    cd = ainfo.get("cond", float("inf"))
    cd_s = "undefined" if not np.isfinite(cd) else "%.1e" % cd
    weak = "" if not ainfo.get("weak") else \
        f" [WEAK geometry: cond={cd_s}, depth poorly constrained]"
    ratio = ("" if (meas_sig is None or sig_pred is None) else
             " measured/predicted sigma ratio "
             + "/".join(f"{m / max(p, 1e-6):.1f}"
                        for m, p in zip(meas_sig, sig_pred)))
    lab = ((sigma[1], sigma[2], sigma[0], hw[1], hw[2], hw[0])
           if coord_sys == "spherical"
           else (sigma[0], sigma[1], sigma[2], hw[0], hw[1], hw[2]))
    log(f"search_delta auto: data_std={s_data:.2f}s "
        f"(floor {float(default_pick_error):.2f}s) -> 1-sigma(km) "
        f"lat={lab[0]:.2f} lon={lab[1]:.2f} dep={lab[2]:.2f}, k={k:.1f} -> "
        f"half-widths(km) lat={lab[3]:.1f} lon={lab[4]:.1f} dep={lab[5]:.1f} "
        f"(cond={cd_s}, n={ainfo['n']}){weak}{ratio}")

    # ---- depth: pin and/or clamp
    depth_min, depth_max, pinned = resolve_depth_bounds(
        fixed_depth, depth_min, depth_max)

    # The posterior delta keeps the FULL depth extent even when the depth is
    # pinned or clamped. The bounds pin the LOCATION; suppressing the
    # uncertainty too would discard a real number -- an analyst fixing depth
    # at 10 km needs to know whether the data tolerate 10 +/- 2 or 10 +/- 30.
    posterior_delta = delta_grid.copy()

    if pinned:
        delta_grid = delta_grid.copy()
        delta_grid[depth_axis(coord_sys)] = _pin_half_width(
            transform, initial_xyz, fixed_depth_tolerance)

    bounds = clamp_depth_to_bounds(transform, initial,
                                   np.append(delta_grid, delta_t),
                                   dep0_km, depth_min, depth_max, lat0, lon0,
                                   min_half_width_km=fixed_depth_tolerance,
                                   log=log)

    return SearchBox(
        delta_grid=delta_grid, posterior_delta=posterior_delta, bounds=bounds,
        sigma_km=np.asarray(sigma, dtype=float), half_widths_km=hw,
        depth_pinned=pinned, provisional_solution=prov_soln,
        info={"mode": "auto", "s_data": s_data, "s_eff": s_eff,
              "sigma_predicted": sig_pred, "sigma_measured": meas_sig,
              "cond": cd, "n_gradients": ainfo["n"], "weak": ainfo.get("weak"),
              "min_station_dist_km": dmin_km},
    )



# Ellipsoid help for QML
# ---------------------------------------------------------------------------

PLUNGE_SIGN = -1.0   # -1: plunge positive down (seismological). +1: literal.


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def quakeml_rotation_matrix(azimuth_deg, plunge_deg, rotation_deg):
    """Columns are the major, minor and intermediate axes in (N, E, D)."""
    return (_rz(math.radians(azimuth_deg))
            @ _ry(PLUNGE_SIGN * math.radians(plunge_deg))
            @ _rx(math.radians(rotation_deg)))


# covariance_km frame at the solution: (r_hat, theta_hat, phi_hat) = (up, south, east)
_LOCAL_TO_NED = np.array([[ 0.0, -1.0, 0.0],   # north = -south
                          [ 0.0,  0.0, 1.0],   # east
                          [-1.0,  0.0, 0.0]])  # down  = -up


def ellipsoid_orientation(cov_km, coord_sys="spherical"):
    """
    (semi_axes_km [major, intermediate, minor],
     plunge_deg, azimuth_deg, rotation_deg), 1-sigma lengths.
    Scale the lengths by chi_scale(level, 3) before publishing.
    """
    if coord_sys != "spherical":
        raise NotImplementedError(
            "ellipsoid_orientation needs a spherical grid: a cartesian "
            "grid's axes have no defined relation to north/east/down")

    c = _LOCAL_TO_NED @ np.asarray(cov_km, dtype=float) @ _LOCAL_TO_NED.T
    evals, evecs = np.linalg.eigh(c)
    order = np.argsort(evals)[::-1]
    semi = np.sqrt(np.clip(evals[order], 0.0, None))
    X = evecs[:, order[0]].copy()          # major
    Y = evecs[:, order[2]].copy()          # minor

    # Axes are sign-free: point the major axis down (plunge 0..90). A
    # horizontal major axis is ambiguous, so take the end with azimuth 0..180.
    if X[2] < -1e-9 or (abs(X[2]) <= 1e-9 and
                         (X[1] < 0 or (X[1] == 0 and X[0] < 0))):
        X = -X

    plunge = math.degrees(math.asin(min(max(X[2], -1.0), 1.0)))
    plunge = plunge if abs(plunge) > 1e-9 else 0.0
    if math.hypot(X[0], X[1]) > 1e-9:
        azimuth = math.degrees(math.atan2(X[1], X[0])) % 360.0
    else:
        azimuth = 0.0                      # vertical: azimuth is undefined

    # Minor and intermediate directions at rotation 0, then the angle of the
    # actual minor axis from that reference, about X.
    R0 = quakeml_rotation_matrix(azimuth, plunge, 0.0)
    Y0, Z0 = R0[:, 1], R0[:, 2]
    rotation = math.degrees(math.atan2(np.dot(Y, Z0), np.dot(Y, Y0)))
    rotation = round(rotation, 9) % 180.0  # Y is sign-free too
    return semi, plunge, azimuth, rotation



# Operator-facing wording, shared by both drivers
# ---------------------------------------------------------------------------
# Posterior density at the edge of the sampled box, relative to its peak,
# above which an axis counts as truncated. For a Gaussian marginal the ratio
# is exp(-k^2/2) for a box at k sigma, so 0.05 is a box at ~2.45 sigma, where
# the reported sigma is ~5% short. Must match box_limited in locate.pyx.
EDGE_DENSITY_LIMIT = 0.05


def describe_search_box(transform, initial, delta_grid, bounds, auto,
                        fixed_depth=None, fixed_depth_tolerance_km=2.0):
    """One line describing the volume searched, in km."""
    initial = np.asarray(initial, dtype=float)
    delta_grid = np.abs(np.asarray(delta_grid, dtype=float))
    iz = depth_axis(transform.coord_sys)
    if transform.coord_sys == "spherical":
        r, theta = float(initial[0]), float(initial[1])
        ns_km = delta_grid[1] * r
        ew_km = delta_grid[2] * r * max(math.sin(theta), 1e-6)
    else:
        ns_km, ew_km = delta_grid[0], delta_grid[1]
    _, _, dep0 = transform.grid_to_geo(initial[:3])
    if bounds is not None:
        b = np.asarray(bounds, dtype=float)
        depths = []
        for g in (b[0][:3], b[1][:3]):
            x = initial[:3].copy()
            x[iz] = g[iz]
            depths.append(transform.grid_to_geo(x)[2])
        d_top, d_bot = min(depths), max(depths)
    else:
        d_top, d_bot = dep0 - delta_grid[iz], dep0 + delta_grid[iz]
    if fixed_depth is not None:
        depth_txt = (f"depth fixed at {float(fixed_depth):.1f} km "
                     f"(+/-{float(fixed_depth_tolerance_km):.1f})")
    else:
        depth_txt = f"depth {d_top:.0f} to {d_bot:.0f} km"
    return (f"search box ({'auto' if auto else 'fixed'}): north-south "
            f"+/-{ns_km:.0f} km, east-west +/-{ew_km:.0f} km, {depth_txt}")


def cap_delta_grid(transform, at, cfg=None):
    """auto_search_max_km (config order) as grid-unit half-widths at `at`."""
    cap = _config_to_grid_order(_cfg(cfg or {}, "auto_search_max_km"),
                                transform.coord_sys)
    return np.abs(_grid_order_to_delta(transform, cap, np.asarray(at, float)[:3]))


def widen_truncated_axes(posterior, sampled_delta_grid, cap_grid,
                         coord_sys="spherical", limit=EDGE_DENSITY_LIMIT,
                         target_k=4.0, min_factor=1.5, max_factor=3.0):
    """
    Half-widths for a second posterior pass, or None if none is needed.

    Only HORIZONTAL axes whose edge density exceeds `limit` and that are
    still below the cap are widened; depth truncation means the arrivals
    do not constrain depth, which a bigger box does not fix. The factor
    comes from the Gaussian relation edge = exp(-k^2/2): the box currently
    sits at k_eff = sqrt(-2 ln edge) sigma, so scale it by target_k / k_eff,
    kept within [min_factor, max_factor] because EDT tails are heavier than
    Gaussian and a single estimate can undershoot.

    Returns (new_delta_grid or None, widened_axis_indices,
    capped_axis_indices); new_delta_grid is None when nothing was widened.
    """
    edge = posterior.get("edge_density")
    if edge is None:
        return None, [], []
    edge = np.asarray(edge, dtype=float)
    iz = depth_axis(coord_sys)
    delta = np.abs(np.asarray(sampled_delta_grid, dtype=float)).copy()
    cap = np.abs(np.asarray(cap_grid, dtype=float))
    widened, capped = [], []
    for a in range(3):
        if a == iz or edge[a] <= limit:
            continue
        k_eff = math.sqrt(-2.0 * math.log(min(max(edge[a], 1e-12), 0.999)))
        factor = min(max(target_k / k_eff, min_factor), max_factor)
        new = min(delta[a] * factor, cap[a])
        if new < delta[a] * 1.05:          # already at (or within 5% of) the cap
            capped.append(a)
            continue
        delta[a] = new
        widened.append(a)
    return (delta if widened else None), widened, capped


def describe_widening(old_delta_grid, new_delta_grid, widened, metric_scale,
                      coord_sys="spherical"):
    """Log line for an automatic widening pass."""
    names = axis_names(coord_sys)
    scale = np.asarray(metric_scale, dtype=float)
    old_km = np.abs(np.asarray(old_delta_grid, dtype=float)) * scale
    new_km = np.abs(np.asarray(new_delta_grid, dtype=float)) * scale
    return ("search box widened for the uncertainty estimate (the posterior "
            "reached its edge): " + ", ".join(
                f"{names[a]} +/-{old_km[a]:.0f} -> +/-{new_km[a]:.0f} km"
                for a in widened))


def suggest_search_delta_km(posterior, k=6.0, min_km=(2.0, 2.0, 5.0),
                            coord_sys="spherical"):
    """
    A fixed search_delta_km (config order: north, east, depth) that would
    contain this posterior: k * sigma per axis, floored at min_km. Depth is
    only meaningful when the arrivals constrain it.
    """
    sig = np.sqrt(np.clip(np.diag(np.asarray(
        posterior["covariance_km"], dtype=float)), 0.0, None))
    iz = depth_axis(coord_sys)
    horiz = [a for a in range(3) if a != iz]
    north, east, depth = sig[horiz[0]], sig[horiz[1]], sig[iz]
    return [float(np.ceil(max(k * v, m))) for v, m in
            zip((north, east, depth), min_km)]


def box_limit_report(posterior, sampled_delta_grid, auto, k_sigma=None,
                     coord_sys="spherical", limit=EDGE_DENSITY_LIMIT):
    """
    Interpret sample_posterior's edge_density for operators.

    Returns a dict:
      warning  : str or None -- horizontal axes the box truncates, with the fix
      note     : str or None -- depth unconstrained (no box fixes that)
      detail   : str         -- per-axis numbers for a verbose log
      truncated_horizontal : list of axis names
      depth_unconstrained  : bool
    Horizontal truncation is fixed by a wider box; a truncated depth axis
    means the arrivals do not constrain depth.
    """
    names = axis_names(coord_sys)
    iz = depth_axis(coord_sys)
    horiz_axes = [a for a in range(3) if a != iz]
    edge = posterior.get("edge_density")
    scale = np.asarray(posterior.get("metric_scale", np.ones(3)), dtype=float)
    half_km = np.abs(np.asarray(sampled_delta_grid, dtype=float)) * scale
    sig_km = np.sqrt(np.clip(np.diag(np.asarray(
        posterior["covariance_km"], dtype=float)), 0.0, None))
    fill = posterior.get("box_fill")
    out = dict(warning=None, note=None, detail="", truncated_horizontal=[],
               depth_unconstrained=False)
    if edge is None:
        out["detail"] = "posterior vs search box: edge_density unavailable"
        return out
    edge = np.asarray(edge, dtype=float)
    order = horiz_axes + [iz]
    out["detail"] = "posterior vs search box: " + "; ".join(
        f"{names[a]} sigma {sig_km[a]:.1f} km, sampled half-width "
        f"{half_km[a]:.0f} km, edge density {edge[a]:.3f}"
        + (f", fill {float(fill[a]):.2f}" if fill is not None else "")
        for a in order)

    horiz = [a for a in horiz_axes if edge[a] > limit]
    if horiz:
        out["truncated_horizontal"] = [names[a] for a in horiz]
        which = " and ".join(names[a] for a in horiz)
        pct = "/".join(f"{100 * edge[a]:.0f}%" for a in horiz)
        sugg = suggest_search_delta_km(posterior, coord_sys=coord_sys)
        if auto:
            # auto has already widened what it could: what remains is at
            # auto_search_max_km.
            fix = (f"raise auto_search_max_km on that axis; this event needs "
                   f"at least {sugg[0]:g}/{sugg[1]:g} km north/east")
        else:
            fix = (f"use a larger search_delta_km, at least "
                   f"[{sugg[0]:g}, {sugg[1]:g}, {sugg[2]:g}], or \"auto\"")
        out["warning"] = (
            f"{which} uncertainty is likely understated: the posterior is "
            f"still {pct} of its peak at the edge of the search box. To fix, "
            f"{fix}.")
    if edge[iz] > limit:
        out["depth_unconstrained"] = True
        out["note"] = (
            f"depth is not constrained by these arrivals: the posterior is "
            f"still {100 * edge[iz]:.0f}% of its peak at the edge of the "
            f"+/-{half_km[iz]:.0f} km depth range sampled, so the reported "
            f"depth uncertainty (+/-{sig_km[iz]:.0f} km) is a lower bound.")
    return out

