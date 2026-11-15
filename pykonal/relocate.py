"""
The relocation sequence shared by every pykonal driver.

  * reads its input format into plain Python (arrivals, pick errors, station
    coordinates, a starting hypocentre, settings),
  * opens the traveltime inventory and an EQLocator (grid building and file
    locking stay with the driver),
  * calls relocate(), and
  * writes the returned Relocation in its output format.

Sequence
--------
 1. configure the locator (alpha, pick errors, EDT options, seed, stations)
 2. choose the solve set (solve_n_closest)
 3. size the search box (sizing.size_auto_box / size_fixed_box); depth
    bounds the config leaves open default to the extent of the grids
 4. locate, once
 5. residuals for EVERY arrival, reading grids for those outside the solve set
 6. iterative outlier rejection, relocating on the solve set
 7. quality metrics
 8. posterior (EDT only): in auto mode size the box at the solution
    (sizing.size_posterior_box), read grids for the sampled volume, sample
    until the effective sample size is reached, widen truncated axes in
    auto mode, interpret, diagnose

Coordinates: spherical grids only, as produced by transformations.GeoTransform.
Time: arrival times and the starting origin time must share one reference
(epoch seconds, or seconds after the first pick). Relative times are better
conditioned; the library copes with epochs.

"""
import math
import time as _time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import sizing

__all__ = ["RelocationError", "Relocation", "DEFAULTS", "relocate",
           "FIRST_ARRIVAL_PHASES", "normalize_phase_map", "map_phase",
           "low_ess_message", "ESS_MIN", "ESS_TARGET",
           "resolve_posterior_nsamples"]

Key = Tuple[str, str, str]          # (network, station, phase)

DEPTH_AXIS = 0                      # spherical grid: axis 0 is r, i.e. depth

# Defaults for settings a caller does not supply. Drivers normally pass their
# own resolved values; these only fill gaps.
DEFAULTS = {
    "method": "edt",
    "alpha": 0.01,
    "default_pick_error": 0.1,          # scalar: locator fallback, sizing floor
    "edt_ot_wt": True,
    "edt_exponent": None,
    "search_delta_km": "auto",          # "auto" or [north, east, depth] km
    "search_delta_t": 5.0,
    "depth_min": None,
    "depth_max": None,
    "fixed_depth_tolerance_km": 2.0,
    "solve_n_closest": 0,
    "outlier_cutoff_s": 3.0,
    "outlier_mad_factor": 4.0,
    "outlier_min_arrivals": 6,
    # "auto": draw until the effective sample size reaches ESS_TARGET
    # (starting at NSAMPLES_AUTO_START, doubling, at most
    # NSAMPLES_SUGGEST_MAX). An integer draws exactly that many; 0 skips
    # the uncertainty estimate.
    "posterior_nsamples": "auto",
    "posterior_rounds": 2,              # ignored by the default qmc proposal
    "posterior_seed": None,
    "log_search_detail": False,
    # auto_search_* keys are read by sizing from the same dict
}


# Effective sample size below which the uncertainty estimate is flagged.
ESS_MIN = 50
# Target effective sample size: what posterior_nsamples "auto" draws until
# it reaches, and what a suggested larger fixed posterior_nsamples aims for.
# At 200 the Monte Carlo error on a reported sigma is about 5%.
ESS_TARGET = 200
# First draw in "auto" mode, and the most "auto" will draw or a suggestion
# will name (beyond it, sampling costs several seconds per event).
NSAMPLES_AUTO_START = 1024
NSAMPLES_SUGGEST_MAX = 65536


def resolve_posterior_nsamples(value):
    """
    (nsamples, ess_target) for a posterior_nsamples setting.

    "auto" or None -> (NSAMPLES_AUTO_START, ESS_TARGET): draw, check the
    effective sample size, double until it is reached. An integer -> that
    many samples and no target; 0 disables the uncertainty estimate.
    """
    if value is None or (isinstance(value, str)
                         and value.strip().lower() == "auto"):
        return NSAMPLES_AUTO_START, float(ESS_TARGET)
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"posterior_nsamples must be \"auto\" or an "
                         f"integer, got {value!r}") from None
    return max(n, 0), None


def low_ess_message(ess, nsamples, mcse=None, auto=False):
    """
    Operator message for a poorly resolved uncertainty estimate.

    ESS grows roughly in proportion to posterior_nsamples (measured on a
    low-ESS event: 72, 297 and 1188 at 4096, 16384 and 65536 samples, with
    the depth sigma varying 11.7%, 6.0% and 1.3% between seeds). Shrinking
    the box barely helps (74 -> 104 when the depth half-width went from 20
    to 8 km) and cuts off the posterior, so the only advice is more samples.
    """
    ess = max(float(ess), 1.0)
    if auto:
        # "auto" already drew up to its limit; there is nothing to raise.
        spread = "" if mcse is None or not np.isfinite(mcse) else (
            f" (sigma may vary by about {100 * float(mcse):.0f}% between "
            f"runs)")
        return (f"posterior: effective sample size {ess:.0f} after "
                f"{int(nsamples)} samples, the most posterior_nsamples "
                f"\"auto\" draws - uncertainties are poorly resolved{spread}")
    want = int(nsamples) * ESS_TARGET / ess
    suggest = min(1 << int(math.ceil(math.log2(max(want, 1.0)))),
                  NSAMPLES_SUGGEST_MAX)
    spread = "" if mcse is None or not np.isfinite(mcse) else (
        f" (sigma may vary by about {100 * float(mcse):.0f}% between runs)")
    text = (f"posterior: effective sample size {ess:.0f} of {int(nsamples)} - "
            f"uncertainties are poorly resolved{spread}; ")
    if suggest > int(nsamples):
        text += (f"raise posterior_nsamples to about {suggest} (effective "
                 f"sample size grows roughly in proportion)")
    else:
        text += "raise posterior_nsamples"
    return text


# ---------------------------------------------------------------------------
# Phase labels
# ---------------------------------------------------------------------------
# The traveltime grids hold FIRST-ARRIVAL times (the eikonal solution is the
# earliest P or S at each point), so only picks of first arrivals may be
# compared with them. Pg and Pn both qualify: the grid gives whichever
# arrives first. Later phases -- depth phases (pP, sP), reflections (PmP,
# SmS), core phases, Lg, Rg, T -- arrive seconds late and must not be used.
# Matching is EXACT and case-sensitive: lowercase p/s (used by some agencies
# for local direct phases) are accepted while pP/sP are not.
FIRST_ARRIVAL_PHASES = {
    "P": "P", "Pg": "P", "Pn": "P", "Pb": "P", "P*": "P", "p": "P",
    "S": "S", "Sg": "S", "Sn": "S", "Sb": "S", "S*": "S", "s": "S",
}


def normalize_phase_map(phase_map):
    """
    Validate the config's phase_map ({label: "P" | "S" | null}) once.

    It is consulted BEFORE the built-in list, so it can add a label your
    network uses for a first arrival ({"Pup": "P"}), or exclude a built-in
    one with null ({"P*": null}). Raises ValueError for any other value:
    a label mapped to e.g. "Pg" would silently find no traveltime grid.
    """
    if not phase_map:
        return {}
    if not isinstance(phase_map, dict):
        raise ValueError(f"phase_map must be an object {{label: \"P\" | "
                         f"\"S\" | null}}, got {phase_map!r}")
    out = {}
    for label, grid_phase in phase_map.items():
        if grid_phase is None:
            out[str(label).strip()] = None
            continue
        gp = str(grid_phase).strip().upper()
        if gp not in ("P", "S"):
            raise ValueError(f"phase_map[{label!r}] = {grid_phase!r}: must "
                             f"be \"P\", \"S\" or null")
        out[str(label).strip()] = gp
    return out


def map_phase(label, phase_map=None):
    """
    The grid phase ("P" or "S") a pick/arrival label locates with, or None
    when the label is not a first arrival and must not be used.
    `phase_map` should already have been through normalize_phase_map().
    """
    label = (label or "").strip()
    if not label:
        return None
    if phase_map and label in phase_map:
        return phase_map[label]
    return FIRST_ARRIVAL_PHASES.get(label)


class RelocationError(RuntimeError):
    """The event cannot be relocated with the data and settings given."""


@dataclass
class Relocation:
    """Everything a driver needs to write its output."""
    hypocenter: np.ndarray                  # grid (r, theta, phi, t0)
    latitude: float
    longitude: float
    depth_km: float
    time: float                             # same reference as the arrivals
    residuals: Dict[Key, float]             # every remaining arrival with a traveltime
    solve_keys: List[Key]                   # arrivals the location was solved with
    excluded: List[Key]                     # rejected outliers
    uncovered: List[Key]                    # have a grid, but no traveltime at the solution
    quality: Dict
    box: sizing.SearchBox
    depth_pinned: bool
    posterior: Optional[Dict] = None        # sample_posterior output, or None
    posterior_delta: Optional[np.ndarray] = None   # half-widths finally sampled
    limits: Optional[Dict] = None           # sizing.box_limit_report output
    outlier_cutoff_s: Optional[float] = None
    compute_seconds: float = 0.0
    messages: List[Tuple[str, str]] = field(default_factory=list)  # (level, text)
    no_grid: List[Key] = field(default_factory=list)  # no grid in the inventory at all


def _setting(settings, key):
    """settings[key], falling back to DEFAULTS when absent or null."""
    v = settings.get(key)
    return DEFAULTS.get(key) if v is None else v


def _locate(locator, initial, delta, alpha, method, bounds):
    if bounds is None:
        return np.asarray(locator.locate(initial, delta, alpha, method), dtype=float)
    return np.asarray(locator.locate(initial, delta, alpha, method, bounds=bounds),
                      dtype=float)


def _set_arrivals(locator, arrivals, pick_errors):
    locator.clear_arrivals()
    locator.add_arrivals(arrivals)
    locator.pick_errors = {k: v for k, v in pick_errors.items() if k in arrivals}


def _residuals_all(locator, soln, arrivals, pick_errors, solve_arrivals,
                   read_lo, read_hi):
    """
    Residuals for every arrival at `soln`, then restore the solve set.

    locate() loads grids only for the solve set, so arrivals outside it need
    their grids read (over the locate volume) or they get no residual and
    look as if their grid did not cover the solution.
    """
    if len(solve_arrivals) == len(arrivals):
        return locator.residuals(soln)
    _set_arrivals(locator, arrivals, pick_errors)
    locator.read_traveltimes(min_coords=np.asarray(read_lo, dtype=float),
                             max_coords=np.asarray(read_hi, dtype=float))
    out = locator.residuals(soln)
    _set_arrivals(locator, solve_arrivals, pick_errors)
    return out


def relocate(locator, transform, arrivals, pick_errors, stations, initial,
             settings=None, *, fixed_depth=None, posterior=True,
             log: Optional[Callable[[str], None]] = None,
             depth_advice: str = "",
             missing_grid_advice: str = "") -> Relocation:
    """
    Relocate one event on an open EQLocator.

    Parameters
    ----------
    locator : EQLocator, already open on the traveltime inventory. The
        caller sets `expected_missing` if it knows some grids are absent.
    transform : transformations.GeoTransform
    arrivals : {(net, sta, phase): arrival time, s}
    pick_errors : {(net, sta, phase): 1-sigma pick uncertainty, s}
    stations : {(net, sta): (lat, lon, elevation_m)}; may be empty, in which
        case quality metrics, solve_n_closest and the auto depth floor that
        needs station distances are unavailable.
    initial : (lat, lon, depth_km, t0) -- the search anchor. t0 in the same
        reference as `arrivals`.
    settings : dict of the keys in DEFAULTS plus any auto_search_* keys
        (the pykonal_locext.json keys; extra keys are ignored).
    fixed_depth : pin the depth (km); searched within fixed_depth_tolerance_km.
    posterior : estimate uncertainties (EDT only).
    log : receives operator messages (str). Detail messages go to it too
        when settings["log_search_detail"] is true. Every message is also
        recorded in Relocation.messages with its level.
    depth_advice : text appended to the "depth is not constrained" note,
        e.g. the driver's own fixed-depth option (" Consider --fixed-depth.").
    missing_grid_advice : text appended to the one-off "no traveltime grid"
        message, e.g. how this driver builds grids.

    The caller's dicts are not modified.
    """
    settings = dict(settings or {})
    messages: List[Tuple[str, str]] = []
    sink = log or (lambda _m: None)
    show_detail = bool(_setting(settings, "log_search_detail"))

    def say(text, level="info"):
        messages.append((level, text))
        prefix = {"warning": "WARNING: ", "note": "NOTE: "}.get(level, "")
        sink(prefix + text)

    def detail(text):
        messages.append(("detail", text))
        if show_detail:
            sink(text)

    arrivals = dict(arrivals)
    pick_errors = {k: float(v) for k, v in pick_errors.items() if k in arrivals}
    stations = dict(stations or {})

    method = str(_setting(settings, "method"))
    alpha = float(_setting(settings, "alpha"))
    default_err = float(_setting(settings, "default_pick_error"))
    search_delta_km = _setting(settings, "search_delta_km")
    auto = isinstance(search_delta_km, str) and search_delta_km.lower() == "auto"
    if not auto:
        search_delta_km = [float(v) for v in search_delta_km]
    delta_t = float(_setting(settings, "search_delta_t"))
    seed = settings.get("posterior_seed")
    seed = None if seed is None else int(seed)
    tol_km = float(_setting(settings, "fixed_depth_tolerance_km"))
    depth_min, depth_max = settings.get("depth_min"), settings.get("depth_max")
    if len(arrivals) < 4:
        raise RelocationError(f"only {len(arrivals)} arrivals, need at least 4")

    # Arrivals whose station-phase has no grid at all. Reported once here and
    # registered as expected, so the locator does not warn again on every
    # traveltime read (sizing, locate, posterior).
    inventory = getattr(locator, "traveltime_inventory", None)
    no_grid: List[Key] = []
    if inventory is not None:
        no_grid = sorted(k for k in arrivals if not inventory.has("/".join(k)))
    if no_grid:
        known = set(getattr(locator, "expected_missing", None) or ())
        locator.expected_missing = known | {"/".join(k) for k in no_grid}
        say(f"No traveltime grid for {len(no_grid)} arrival(s): "
            + ", ".join(".".join(k) for k in no_grid)
            + ". They are not used and are reported with weight 0."
            + missing_grid_advice, "warning")
        if len(arrivals) - len(no_grid) < 4:
            raise RelocationError(
                f"only {len(arrivals) - len(no_grid)} arrivals have traveltime "
                f"grids, need at least 4")

    lat0, lon0, dep0, t0 = (float(v) for v in initial)
    if fixed_depth is not None:
        dep0 = float(fixed_depth)            # anchor on the pin
    initial_grid = np.append(transform.geo_to_grid(lat0, lon0, dep0), t0)

    # ---- 1. locator configuration
    locator.alpha = alpha
    locator.default_pick_error = default_err
    if hasattr(locator, "edt_ot_wt"):
        locator.edt_ot_wt = bool(_setting(settings, "edt_ot_wt"))
    if settings.get("edt_exponent") is not None and hasattr(locator, "edt_exponent"):
        locator.edt_exponent = float(settings["edt_exponent"])
    if seed is not None and hasattr(locator, "locate_seed"):
        locator.locate_seed = seed
    if stations:
        locator.stations = {k: transform.geo_to_grid(v[0], v[1], -v[2] / 1000.0)
                            for k, v in stations.items()}
    # distances on the grid sphere (transformations.geo_distance_azimuth),
    # the metric the traveltime grids and their max_dist crop use
    station_dist_km = {k: transform.distance_azimuth(lat0, lon0, v[0], v[1])[0]
                       for k, v in stations.items()}

    # ---- 2. solve set
    n_closest = int(_setting(settings, "solve_n_closest") or 0)
    solve_arrivals = arrivals
    if n_closest > 0 and len(arrivals) > n_closest:
        ranked, unranked = [], []
        for key in arrivals:
            if key[:2] in station_dist_km:
                ranked.append((station_dist_km[key[:2]], key))
            else:
                unranked.append(key)
        ranked.sort(key=lambda t: t[0])
        keep = {k for _, k in ranked[:n_closest]}
        solve_arrivals = {k: v for k, v in arrivals.items() if k in keep}
        say(f"solve_n_closest={n_closest}: solving with the "
            f"{len(solve_arrivals)} closest arrival(s); "
            f"{len(arrivals) - len(solve_arrivals)} farther arrival(s) kept "
            f"for residual reporting only"
            + (f" ({len(unranked)} unrankable, no coords)" if unranked else ""))
    solve_arrivals = dict(solve_arrivals)
    _set_arrivals(locator, solve_arrivals, pick_errors)

    # ---- 3. search box
    for key, why in sizing.removed_auto_keys(settings):
        detail(f"{key} is no longer used ({why}); you can delete it from "
               f"your config")
    # A depth bound the config leaves open defaults to where the grids end.
    # A pinned depth is its own bound, so nothing needs filling in.
    if fixed_depth is None:
        depth_min, depth_max, from_grid = sizing.grid_depth_bounds(
            locator, transform, list(solve_arrivals), initial_grid[:3],
            depth_min, depth_max)
        if from_grid:
            detail(f"depth search limited to the traveltime grids: "
                   f"{depth_min:.1f} to {depth_max:.1f} km")
    if auto:
        box = sizing.size_auto_box(
            locator, transform, initial_grid, delta_t,
            arrival_keys=list(solve_arrivals), dep0_km=dep0, lat0=lat0,
            lon0=lon0, cfg=settings,
            fixed_depth=fixed_depth, fixed_depth_tolerance=tol_km,
            depth_min=depth_min, depth_max=depth_max, log=detail)
    else:
        box = sizing.size_fixed_box(
            transform, initial_grid, delta_t, search_delta_km, dep0,
            fixed_depth=fixed_depth, fixed_depth_tolerance=tol_km,
            depth_min=depth_min, depth_max=depth_max, lat0=lat0, lon0=lon0,
            log=detail)
    if fixed_depth is not None:
        say(f"fixed depth {float(fixed_depth):.2f} km, search tolerance "
            f"+/-{max(tol_km, 1e-3):.2f} km"
            + ("" if box.posterior_delta is None else
               f"; posterior depth half-width "
               f"{abs(float(box.posterior_delta[DEPTH_AXIS])):.1f} km"))
    say(sizing.describe_search_box(transform, initial_grid, box.delta_grid,
                                   box.bounds, auto, fixed_depth=fixed_depth,
                                   fixed_depth_tolerance_km=tol_km))
    delta_full = box.delta_full(delta_t)
    if box.bounds is not None:
        read_lo, read_hi = box.bounds[0][:3], box.bounds[1][:3]
    else:
        read_lo = initial_grid[:3] - box.delta_grid
        read_hi = initial_grid[:3] + box.delta_grid

    # ---- 4-5. locate, residuals for all
    t_compute = _time.time()
    soln = _locate(locator, initial_grid, delta_full, alpha, method, box.bounds)
    residuals = _residuals_all(locator, soln, arrivals, pick_errors,
                               solve_arrivals, read_lo, read_hi)

    # ---- 6. outlier rejection
    # EDT already down-weights inconsistent picks when locating, but they
    # inflate the RMS, and a mutually consistent group of bad picks can build
    # a competing peak. Cutoff max(outlier_cutoff_s, outlier_mad_factor * MAD);
    # never below outlier_min_arrivals; worst first; up to 3 passes.
    cut_abs = float(_setting(settings, "outlier_cutoff_s"))
    cut_mad = float(_setting(settings, "outlier_mad_factor"))
    min_keep = max(int(_setting(settings, "outlier_min_arrivals")), 4)
    excluded: List[Key] = []
    cut = None
    if cut_abs > 0 and len(arrivals) > min_keep:
        for _pass in range(3):
            if not residuals:
                break
            vals = np.array(list(residuals.values()), dtype=float)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med))) * 1.4826
            cut = max(cut_abs, cut_mad * mad)
            bad = sorted((k for k, r in residuals.items() if abs(r - med) > cut),
                         key=lambda k: -abs(residuals[k] - med))
            bad = bad[:max(0, len(arrivals) - min_keep)]
            if not bad:
                break
            for k in bad:
                arrivals.pop(k, None)
                pick_errors.pop(k, None)
                solve_arrivals.pop(k, None)
            excluded.extend(bad)
            _set_arrivals(locator, solve_arrivals, pick_errors)
            soln = _locate(locator, initial_grid, delta_full, alpha, method,
                           box.bounds)
            residuals = _residuals_all(locator, soln, arrivals, pick_errors,
                                       solve_arrivals, read_lo, read_hi)
        if excluded:
            say(f"rejected {len(excluded)} outlier arrival(s) (cutoff "
                f"{cut:.2f} s): " + ", ".join(".".join(k) for k in excluded))

    # The search cannot leave its box. A solution resting against a
    # horizontal face is the edge of the box, not a minimum: say so rather
    # than report it as a location. (Depth is bounded by the grids or by the
    # operator's limits, where stopping at the bound is legitimate.)
    span = np.asarray(read_hi, dtype=float) - np.asarray(read_lo, dtype=float)
    at_face = [name for ax, name in ((1, "north-south"), (2, "east-west"))
               if span[ax] > 0 and min(soln[ax] - read_lo[ax],
                                       read_hi[ax] - soln[ax]) < 0.01 * span[ax]]
    if at_face:
        say(f"the solution lies on the {' and '.join(at_face)} edge of the "
            f"search box, so the event is probably outside it: "
            + ("raise auto_search_max_km" if auto else
               "use a larger search_delta_km")
            + " or start from a closer origin.", "warning")

    uncovered = sorted(set(arrivals) - set(residuals) - set(no_grid))
    if uncovered:
        say(f"{len(uncovered)} arrival(s) have no traveltime at the solution "
            f"and did not contribute to the fit (grid does not cover the "
            f"hypocenter): " + ", ".join(".".join(k) for k in uncovered)
            + ". These are reported with weight 0, the same as rejected "
              "outliers.")

    # ---- 7. quality
    quality = locator.quality(soln)
    lat, lon, depth_km = transform.grid_to_geo(soln[:3])

    result = Relocation(
        hypocenter=soln, latitude=lat, longitude=lon, depth_km=depth_km,
        time=float(soln[3]), residuals=residuals, solve_keys=list(solve_arrivals),
        excluded=excluded, uncovered=uncovered, no_grid=no_grid,
        quality=quality, box=box,
        depth_pinned=box.depth_pinned, outlier_cutoff_s=cut, messages=messages)

    # ---- 8. posterior
    try:
        nsamples, ess_target = resolve_posterior_nsamples(
            settings.get("posterior_nsamples", DEFAULTS["posterior_nsamples"]))
    except ValueError as err:
        raise RelocationError(str(err)) from None
    if posterior and method == "edt" and nsamples > 0:
        _posterior(result, locator, transform, settings, auto, seed, nsamples,
                   ess_target, say, detail, depth_advice)
    result.compute_seconds = _time.time() - t_compute
    return result


def _posterior(result, locator, transform, settings, auto, seed, nsamples,
               ess_target, say, detail, depth_advice=""):
    soln, box = result.hypocenter, result.box
    rounds = int(_setting(settings, "posterior_rounds"))
    if auto:
        # Sized at the solution from the linearised covariance of the grids
        # locate() left in memory (the solve set, over the searched volume).
        mind = result.quality.get("min_station_dist", float("nan"))
        try:
            delta, pinfo = sizing.size_posterior_box(
                locator, transform, soln[:3], cfg=settings,
                min_station_dist_km=(float(mind) if np.isfinite(mind)
                                     else None),
                log=detail)
        except Exception as exc:               # never lose a location
            delta = sizing.cap_delta_grid(transform, soln[:3], settings)
            pinfo = {"sigma_km": None}
            detail(f"uncertainty box: sizing failed ({exc}); sampling the "
                   f"whole searched box")
        box.posterior_delta = np.array(delta, dtype=float)
        box.sigma_km = pinfo.get("sigma_km")
        box.info.update(posterior_box=pinfo)
    delta = np.abs(np.asarray(box.posterior_delta, dtype=float)).copy()

    def sample(half, widened=False):
        # Read grids for exactly the sampled volume: the grids in memory
        # cover the locate box, which is centred on the anchor and may be
        # depth-pinned, and a sample with no traveltime is silently rejected.
        locator.read_traveltimes(min_coords=soln[:3] - half,
                                 max_coords=soln[:3] + half)
        # box_fill (informational) is measured against the search box
        # horizontally and the sampled extent in depth, since the location
        # box may be pinned; after widening, against the widened box.
        if widened or auto:
            # auto: the box sampled IS the reference; the location search
            # box is the whole of auto_search_max_km and says nothing here
            search_hw = np.array(half, dtype=float)
        else:
            search_hw = np.abs(np.asarray(box.delta_grid, dtype=float)).copy()
            search_hw[DEPTH_AXIS] = half[DEPTH_AXIS]
        return locator.sample_posterior(soln[:3], half, nsamples=nsamples,
                                        nscatter=0, seed=seed, rounds=rounds,
                                        search_delta=search_hw,
                                        ess_target=ess_target,
                                        max_nsamples=NSAMPLES_SUGGEST_MAX)

    try:
        post = sample(delta)
    except Exception as exc:                   # never lose a location
        say(f"posterior sampling failed: {exc}", "error")
        return

    # "auto" settles the box without operator input: an axis the posterior
    # still runs into is widened (up to auto_search_max_km) and sampled
    # again. Depth is included: its starting width is a linearised estimate
    # that can be too small, so it only counts as unconstrained once it is
    # still truncated at the cap.
    if auto:
        cap = sizing.cap_delta_grid(transform, soln[:3], settings)
        for _pass in range(3):
            new_d, widened, _capped = sizing.widen_truncated_axes(
                post, delta, cap, widen_depth=True)
            if new_d is None:
                break
            try:
                post_new = sample(new_d, widened=True)
            except Exception as exc:
                say(f"widened posterior pass failed: {exc}", "error")
                break
            say(sizing.describe_widening(delta, new_d, widened,
                                         post["metric_scale"]))
            delta, post = new_d, post_new

    limits = sizing.box_limit_report(
        post, delta, auto=auto,
        k_sigma=settings.get("auto_search_k_sigma",
                             sizing.AUTO_DEFAULTS["auto_search_k_sigma"]))
    post["box_limit_report"] = limits
    detail(limits["detail"])
    if limits["warning"]:
        say(limits["warning"], "warning")
    if limits["note"]:
        say(limits["note"] + depth_advice, "note")

    corr = post.get("depth_time_correlation")
    if corr is not None and np.isfinite(corr) and abs(corr) > 0.9:
        say(f"depth and origin time are correlated at {corr:+.2f}: they trade "
            f"off almost completely, so neither is independently determined. "
            f"Depth is only as good as the origin time here.")
    ess = float(post.get("ess", float("nan")))
    low_ess = np.isfinite(ess) and ess < ESS_MIN
    mcse = post.get("sigma_rel_mcse")
    if (not low_ess and mcse is not None and np.isfinite(mcse)
            and mcse > 0.15):
        say(f"sigma has a Monte Carlo standard error of {100 * mcse:.0f}% -- "
            f"raise posterior_nsamples if you need the uncertainty itself "
            f"to be precise")
    prop = str(post.get("proposal", "unknown"))
    if "fallback" in prop:
        say(f"posterior: curvature-based proposal unusable, fell back to "
            f"{prop}; uncertainties are less reliable")
    if low_ess:
        say(low_ess_message(ess, int(post.get("nsamples", nsamples)), mcse,
                            auto=ess_target is not None))

    result.posterior, result.posterior_delta, result.limits = post, delta, limits
