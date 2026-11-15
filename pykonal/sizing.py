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
    "SearchBox", "chi_scale", "depth_axis",
    "resolve_depth_bounds", "clamp_depth_to_bounds", "size_auto_box",
    "size_fixed_box", "AUTO_DEFAULTS", "ellipsoid_orientation",
    "quakeml_rotation_matrix", "EDGE_DENSITY_LIMIT", "describe_search_box",
    "box_limit_report", "widen_truncated_axes", "cap_delta_grid",
    "suggest_search_delta_km", "describe_widening", "AUTO_REMOVED",
    "removed_auto_keys", "grid_depth_bounds", "linearised_sigma",
    "size_posterior_box", "EDT_WIDTH_FACTOR",
]


AUTO_DEFAULTS = {
    # Half-widths [north, east, depth] km of the volume the location is
    # searched in, and the cap on the uncertainty estimate's box.
    "auto_search_max_km": [150.0, 150.0, 60.0],
    # The uncertainty estimate is sampled within this many (EDT) sigma.
    "auto_search_k_sigma": 4.0,
    "auto_search_min_km": [2.0, 2.0, 5.0],
    "auto_search_depth_dist_factor": 0.5,
}

# Keys the two-pass sizing used. Accepted and ignored, with a note, so a
# config written for an earlier version keeps working.
AUTO_REMOVED = {
    "auto_search_provisional_km":
        "the location is searched once, in auto_search_max_km",
    "auto_search_cond_max":
        "the geometry prediction it guarded is no longer used",
    "auto_search_fallback_sigma_km":
        "with no usable prediction the whole searched box is sampled",
    "auto_search_posterior_nsamples":
        "there is no provisional posterior any more",
    "auto_search_posterior_min_ess":
        "there is no provisional posterior any more",
}

# Width of the EDT posterior relative to the linearised (weighted
# least-squares) sigma at the same solution. Measured on synthetic events
# whose noise matched the assumed error model: 1.4-1.5 on one-sided
# geometry, 1.5-1.6 with six stations, 1.7-1.9 with fifteen. The linearised
# sigma only has to put the sampling box in the right place -- an axis the
# posterior still reaches the edge of is widened afterwards -- so this is
# the top of that range, not a tuning knob.
EDT_WIDTH_FACTOR = 2.0


def removed_auto_keys(cfg):
    """[(key, why)] for removed auto_search_* keys present in `cfg`."""
    return [(k, AUTO_REMOVED[k]) for k in (cfg or {}) if k in AUTO_REMOVED]

_KM_PER_DEG = 111.195


@dataclass
class SearchBox:
    """
    The result of sizing. `delta_grid` and `bounds` drive locate();
    `posterior_delta` drives sample_posterior() and is deliberately NOT
    depth-clamped, so a pinned or clamped depth still gets an honest
    uncertainty measured over the full extent. In auto mode it is None
    until the event has been located: the box for the uncertainty estimate
    is sized at the solution (size_posterior_box).
    """
    delta_grid: np.ndarray
    posterior_delta: Optional[np.ndarray] = None
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


# Linearised sigma at the solution
# ---------------------------------------------------------------------------
def linearised_sigma(locator, soln_xyz, coord_sys):
    """
    Per-axis 1-sigma (km, grid-axis order) of the weighted least-squares
    location covariance C = (G^T W G)^-1 at `soln_xyz`.

    Each row of G is one arrival's traveltime gradient at the solution, in
    s/km in the local frame, plus a unit origin-time column, so origin time
    is eliminated rather than assumed known. W holds 1/variance with the
    variance the posterior itself uses: pick error^2 + (alpha * traveltime)^2,
    floored at the locator's variance floor and at the a-posteriori residual
    variance (see EQLocator.sample_posterior).

    This sizes the box the uncertainty estimate is sampled in; it is not the
    reported uncertainty. Measured against sampled posteriors it is within a
    few percent in the median but under-covers on its own (57-65% inside its
    nominal 68% ellipsoid), because the problem is not linear over a region
    several sigma wide.

    Gradients are central differences one node interval wide, one-sided
    where a neighbour falls outside the loaded grid. Arrivals with no
    traveltime at the solution are skipped.

    Returns (sigma_km, info). sigma_km is None when fewer than four arrivals
    have a gradient or G^T W G cannot be inverted; info has n (rows used).
    """
    spherical = (coord_sys == "spherical")
    x0 = np.asarray(soln_xyz, dtype=float)[:3].copy()
    if spherical:
        metric = np.array([1.0, x0[0],
                           x0[0] * max(abs(math.sin(x0[1])), 1e-6)])
    else:
        metric = np.ones(3)
    info = {"n": 0}

    arrivals = locator.arrivals
    fields_ = locator.traveltimes
    pick_errors = locator.pick_errors or {}
    default_err = float(locator.default_pick_error)
    alpha = float(locator.alpha)

    rows, tts, sig, res = [], [], [], []
    for key, t_obs in arrivals.items():
        field_ = fields_.get(key)
        if field_ is None:
            continue
        try:
            tt0 = float(field_.value(x0))
        except Exception:
            continue
        if not np.isfinite(tt0) or tt0 > 9999:
            continue
        node = np.asarray(field_.node_intervals, dtype=float)
        grad = []
        for ax in range(3):
            h0 = float(node[ax])
            if not np.isfinite(h0) or h0 <= 0:
                break
            d = None
            # A full node interval first; a quarter of one when both
            # neighbours are outside the loaded grid, which happens on the
            # depth axis of a pinned-depth search (a slab a few km thick).
            for h in (h0, 0.25 * h0):
                xp, xm = x0.copy(), x0.copy()
                xp[ax] += h
                xm[ax] -= h
                tp, tm = float(field_.value(xp)), float(field_.value(xm))
                okp = np.isfinite(tp) and tp <= 9999
                okm = np.isfinite(tm) and tm <= 9999
                if okp and okm:
                    d = (tp - tm) / (2.0 * h)
                elif okp:
                    d = (tp - tt0) / h
                elif okm:
                    d = (tt0 - tm) / h
                if d is not None:
                    break
            if d is None:
                break
            grad.append(d / metric[ax])
        if len(grad) != 3:
            continue
        rows.append(grad + [1.0])
        tts.append(tt0)
        sig.append(float(pick_errors.get(key, default_err)))
        res.append(float(t_obs) - tt0)

    info["n"] = len(rows)
    if len(rows) < 4:
        return None, info

    tts, sig, res = np.array(tts), np.array(sig), np.array(res)
    floor = float(locator.variance_floor)
    if len(res) > 4:
        r = res - np.median(res)
        floor = max(floor, float(np.sum(r * r) / (len(res) - 4)))
    var = np.maximum(np.maximum(sig, 1e-5) ** 2 + (alpha * tts) ** 2, floor)

    G = np.array(rows)
    try:
        C = np.linalg.inv(G.T @ (G / var[:, None]))
    except np.linalg.LinAlgError:
        return None, info
    d = np.diag(C)[:3]
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        return None, info
    return np.sqrt(d), info


def grid_depth_bounds(locator, transform, arrival_keys, at_xyz,
                      depth_min=None, depth_max=None, max_grids=8):
    """
    Fill an unset depth_min / depth_max from the traveltime grids.

    A depth the grids do not reach has no traveltime, so searching it only
    wastes samples: an unbounded auto box anchored at 10 km ran from 23 km
    ABOVE the surface to 43 km below it. Bounds the operator set are
    returned unchanged; an unset one becomes the vertical extent common to
    the grids. Grids built from one velocity model share that extent, so at
    most `max_grids` are inspected (three small reads each), spread across
    the arrivals so both phases are represented.

    Returns (depth_min, depth_max, from_grid); from_grid is True when
    either value came from the grids.
    """
    if depth_min is not None and depth_max is not None:
        return depth_min, depth_max, False
    inventory = getattr(locator, "traveltime_inventory", None)
    if inventory is None:
        return depth_min, depth_max, False
    iz = depth_axis(transform.coord_sys)
    keys = [k for k in arrival_keys if inventory.has("/".join(k))]
    if not keys:
        return depth_min, depth_max, False
    step = max(1, len(keys) // max_grids)
    tops, bottoms = [], []
    for key in keys[::step][:max_grids]:
        try:
            g = inventory.f5["/".join(key)]
            lo = float(g["min_coords"][iz])
            hi = lo + (int(g["npts"][iz]) - 1) * float(g["node_intervals"][iz])
        except Exception:
            continue
        ends = []
        for v in (lo, hi):
            x = np.asarray(at_xyz, dtype=float)[:3].copy()
            x[iz] = v
            ends.append(float(transform.grid_to_geo(x)[2]))
        tops.append(min(ends))
        bottoms.append(max(ends))
    if not tops:
        return depth_min, depth_max, False
    top, bottom = max(tops), min(bottoms)      # common to all inspected
    if top >= bottom:
        return depth_min, depth_max, False
    return (top if depth_min is None else depth_min,
            bottom if depth_max is None else depth_max, True)


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

    Floor 1 (coverage): only when `provisional_offset_km` is given. It was
      needed while a second, tighter search was centred back on the anchor;
      the box sized here is now centred on the solution, so size_posterior_box
      does not pass it.

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
        log("auto box: half-width floored on " + ", ".join(floored)
            + " (k*sigma would have given "
            + "/".join(f"{v:.1f}" for v in raw) + " km, using "
            + "/".join(f"{v:.1f}" for v in hw) + " km)")

    cap_km = _config_to_grid_order(_cfg(cfg, "auto_search_max_km"), coord_sys)
    capped = [n for n, a, c in zip(names, hw, cap_km) if a > c + 1e-6]
    hw = np.minimum(hw, cap_km)
    if capped:
        log("auto box: half-width capped by auto_search_max_km on "
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
                  arrival_keys=None, dep0_km, lat0, lon0, cfg=None,
                  fixed_depth=None, fixed_depth_tolerance=2.0,
                  depth_min=None, depth_max=None, log=None, **_unused):
    """
    The volume an "auto" location is searched in: auto_search_max_km around
    the anchor, with the depth pin and depth bounds applied.

    The event is located ONCE, in this box. Earlier versions located twice:
    a provisional pass here, then a second pass in a k*sigma box centred
    back on the anchor, sized from a provisional posterior and a geometry
    prediction. Measured on synthetic events the second pass moved the
    solution by a median of 10-30 m against sigmas of 1-10 km, and once in
    40 one-sided events it left the solution 36 km away on a worse misfit,
    so it cost a second search and bought nothing. The geometry prediction
    never set the box either: it came out at 0.6-0.7 of the final sigma and
    the larger of the two estimates was always the provisional posterior.

    The box for the uncertainty estimate is sized afterwards, at the
    solution: see size_posterior_box.

    Extra keyword arguments the two-pass version took (method, alpha,
    default_pick_error, posterior_seed, station_distances_km, stage) are
    accepted and ignored.

    Returns a SearchBox with posterior_delta None.
    """
    log = log or _noop
    cfg = cfg or {}
    coord_sys = transform.coord_sys
    initial = np.asarray(initial, dtype=float)
    initial_xyz = initial[:3]

    hw = _config_to_grid_order(_cfg(cfg, "auto_search_max_km"), coord_sys)
    delta_grid = np.abs(_grid_order_to_delta(transform, hw, initial_xyz))

    depth_min, depth_max, pinned = resolve_depth_bounds(
        fixed_depth, depth_min, depth_max)
    if pinned:
        delta_grid = delta_grid.copy()
        delta_grid[depth_axis(coord_sys)] = _pin_half_width(
            transform, initial_xyz, fixed_depth_tolerance)

    bounds = clamp_depth_to_bounds(transform, initial,
                                   np.append(delta_grid, delta_t),
                                   dep0_km, depth_min, depth_max, lat0, lon0,
                                   min_half_width_km=fixed_depth_tolerance,
                                   log=log)
    return SearchBox(delta_grid=delta_grid, posterior_delta=None,
                     bounds=bounds, half_widths_km=hw, depth_pinned=pinned,
                     info={"mode": "auto"})


def size_posterior_box(locator, transform, soln_xyz, *, cfg=None,
                       min_station_dist_km=None, log=None):
    """
    Half-widths (grid units) of the box the uncertainty estimate is sampled
    in, centred on the solution, for "auto" mode.

    auto_search_k_sigma * EDT_WIDTH_FACTOR * the linearised sigma at the
    solution, raised to the depth-resolution and absolute floors and capped
    by auto_search_max_km (apply_floors_and_cap). When the linearised sigma
    is unusable -- fewer than four gradients, or a singular geometry -- the
    cap itself is returned: the whole searched volume is sampled, which
    costs sampling density but cannot truncate anything.

    The box only has to be roughly right. sample_posterior treats it as a
    rejection boundary, and relocate() widens any axis the posterior still
    reaches the edge of.

    Returns (delta_grid, info); info has sigma_km (or None), half_widths_km
    and n_gradients.
    """
    log = log or _noop
    cfg = cfg or {}
    coord_sys = transform.coord_sys
    soln_xyz = np.asarray(soln_xyz, dtype=float)[:3]
    k = float(_cfg(cfg, "auto_search_k_sigma"))
    cap_km = _config_to_grid_order(_cfg(cfg, "auto_search_max_km"), coord_sys)

    sigma, linfo = linearised_sigma(locator, soln_xyz, coord_sys)
    if sigma is None:
        hw = np.asarray(cap_km, dtype=float).copy()
        log(f"uncertainty box: no usable linearised sigma "
            f"({linfo['n']} gradient(s)); sampling the whole searched box")
    else:
        hw = apply_floors_and_cap(k * EDT_WIDTH_FACTOR * sigma, sigma,
                                  coord_sys, cfg,
                                  min_station_dist_km=min_station_dist_km,
                                  log=log)
        iz = depth_axis(coord_sys)
        h = [a for a in range(3) if a != iz]
        log(f"uncertainty box: linearised 1-sigma(km) north-south="
            f"{sigma[h[0]]:.2f} east-west={sigma[h[1]]:.2f} "
            f"depth={sigma[iz]:.2f} (n={linfo['n']}) -> half-widths(km) "
            f"north-south={hw[h[0]]:.1f} east-west={hw[h[1]]:.1f} "
            f"depth={hw[iz]:.1f}")
    delta_grid = np.abs(_grid_order_to_delta(transform, hw, soln_xyz))
    return delta_grid, {"sigma_km": sigma, "half_widths_km": hw,
                        "n_gradients": linfo["n"]}


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
                         target_k=4.0, min_factor=1.5, max_factor=3.0,
                         widen_depth=False):
    """
    Half-widths for a second posterior pass, or None if none is needed.

    Axes whose edge density exceeds `limit` and that are still below the
    cap are widened. Depth is included only with widen_depth=True, which
    the auto path sets: its depth box comes from a linearised sigma that
    can be too small, so reaching the edge does not yet mean the arrivals
    cannot constrain depth. Once depth is at the cap it does. The factor
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
        if (a == iz and not widen_depth) or edge[a] <= limit:
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

