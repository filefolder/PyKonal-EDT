#!/usr/bin/env python3

"""
In-Python manual relocation for ObsPy Event objects.

    from obspy import read_events, read_inventory
    import pykonal.pykonal_obspy as P

    cat = read_events("events.xml")
    inv = read_inventory("stations.xml")      # optional: stations_csv in the config also works
    for i, ev in enumerate(cat):
        cat[i] = P.relocate(ev, "pykonal_locext.json", station_inventory=inv, log=print)
    cat.write("relocated.xml", format="QUAKEML")

relocate() returns a relocated COPY of the event; pass inplace=True to modify
the event in the catalogue directly. Settings come from the same JSON config
pykonal_locext reads; keyword arguments override it for one call.

The relocated hypocenter is appended as a NEW Origin and set as
`event.preferred_origin_id`; the input origin is left untouched so the
relocation stays auditable. Pass `replace_origin=True` to overwrite instead.
"""

import copy
import json
import math
import os
import warnings
from typing import Dict, Optional

import numpy as np

from obspy import UTCDateTime
from obspy.core.event import (
    Arrival, ConfidenceEllipsoid, Origin, OriginQuality, OriginUncertainty,
    QuantityError, ResourceIdentifier,
)
from obspy.geodetics import kilometer2degrees

from pykonal.locate import EQLocator
from pykonal.transformations import GeoTransform, geo_distance_azimuth
from pykonal import sizing
from pykonal import relocate as pkrelocate
from pykonal.inventory import (ensure_traveltimes_sharded, fit_requests_to_grid,
                               inventory_lock, resolve_max_dist,
                               DEFAULT_MAX_DIST_KM)


__all__ = ["relocate", "RelocationError"]


class RelocationError(RuntimeError):
    """Raised when an event cannot be relocated."""

# ---------------------------------------------------------------------------
# Configuration
#
# Same JSON file as the SeisComP locator (pykonal_locext.json), so one config
# drives both tools. Keyword arguments to relocate() override the file.
# ---------------------------------------------------------------------------
CONFIG_DEFAULTS = {
    "method": "edt",
    "alpha": 0.01,
    "edt_ot_wt": True,
    "edt_exponent": None,
    "default_pick_error": 0.1,        # scalar, or {"P": 0.08, "S": 0.16}
    "search_delta_km": [25.0, 25.0, 25.0],
    "search_delta_t": 5.0,
    "depth_min": None,
    "depth_max": None,
    "default_depth": None,
    "min_picks": 4,
    "posterior_nsamples": 4096,
    "posterior_seed": None,
    "confidence_level": 68.0,
    "stations_csv": None,
    "earth_model_id": None,
    "max_dist_km": DEFAULT_MAX_DIST_KM,   # null in the config = no limit
    "solve_n_closest": 0,
    "outlier_cutoff_s": 3.0,
    "outlier_mad_factor": 4.0,
    "outlier_min_arrivals": 6,
    "suppress_warnings": [],          # config keys to stay quiet about
    "velocity_models": None,          # {"P": "vp.h5", "S": "vs.h5"}
    "traveltime_dir": None,           # sharded inventory directory
    "ensure_nproc": None,
    "fixed_depth_tolerance_km": 2.0,
    "phase_map": None,
    "build_missing": True,             # compute absent grids from velocity_models
    "log_search_detail": False,        # sizing internals and per-axis numbers
    "_raw_config": {},
}

# Recognised in pykonal_locext.json but not acted on here. Warned about once
# rather than ignored silently, so a shared config does not quietly behave
# differently between the two tools.
# Keys recognised in pykonal_locext.json whose absence here is harmless --
# they describe SeisComP-side or logging behaviour with no ObsPy equivalent,
# so warning about them on every call is noise, not information.
CONFIG_QUIET = {
    "search_anchor": "the anchor is always the event's own origin",
    "grid_center": "removed; ignored by both tools (pykonal_locext logs a "
                   "note asking for it to be deleted)",
    "log_arrival_detail": "use the returned origin's arrivals instead",
}

# Recognised in pykonal_locext.json but not acted on here. Warned about once
# rather than ignored silently, so a shared config cannot quietly behave
# differently between the two tools. Add a key to `suppress_warnings` in the
# config to silence one without editing this file.
CONFIG_UNSUPPORTED = {
}
_AUTO_PREFIX = "auto_search_"


def load_config(config):
    """
    Load a pykonal_locext-style JSON config. Accepts a path or a dict.
    Keys beginning with '_' are treated as documentation and dropped.

    Unused keys are warned about once, so a config shared with the SeisComP
    locator cannot quietly behave differently here. Add a key to
    `suppress_warnings` in the config to silence it; keys in CONFIG_QUIET
    are silent already.
    """
    if config is None:
        return {}
    if isinstance(config, dict):
        raw = dict(config)
    else:
        path = os.fspath(config)
        try:
            with open(path) as fh:
                raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise RelocationError(
                f"{path!r} is not a JSON config ({exc}). The second positional "
                f"argument to relocate() is the config; pass the sharded "
                f"inventory directory as traveltime_dir=...") from exc
    cfg = {k: v for k, v in raw.items() if not k.startswith("_")}

    quiet = set(CONFIG_QUIET) | set(raw.get("suppress_warnings") or [])
    unsupported = [k for k in cfg
                   if k in CONFIG_UNSUPPORTED and k not in quiet]
    auto = [k for k in cfg if k.startswith(_AUTO_PREFIX)]
    if unsupported:
        warnings.warn(
            "config keys not used by pykonal_obspy: "
            + "; ".join(f"{k} ({CONFIG_UNSUPPORTED[k]})" for k in unsupported))
    unknown = [k for k in cfg
               if k not in CONFIG_DEFAULTS and k not in CONFIG_UNSUPPORTED
               and k not in CONFIG_QUIET and k not in quiet
               and not k.startswith(_AUTO_PREFIX)]
    bad_auto = [k for k in auto if k not in sizing.AUTO_DEFAULTS]
    if bad_auto:
        warnings.warn(f"unrecognised auto_search_* keys ignored: "
                      f"{', '.join(sorted(bad_auto))}")
    if unknown:
        warnings.warn(f"unrecognised config keys ignored: {', '.join(sorted(unknown))}")
    return cfg


def _settings(config, overrides):
    """Merge defaults < config file < explicit keyword arguments."""
    raw = load_config(config)
    out = dict(CONFIG_DEFAULTS)
    out.update({k: v for k, v in raw.items() if k in CONFIG_DEFAULTS})
    # auto_search_* keys are consumed by pykonal_sizing, not by this module,
    # so pass the raw dict through rather than enumerating them here -- one
    # less place to forget a key when the sizing module gains an option.
    out["_raw_config"] = raw
    out.update({k: v for k, v in overrides.items() if v is not None})

    sd = out["search_delta_km"]
    if isinstance(sd, str):
        if sd.lower() != "auto":
            raise RelocationError(
                f"search_delta_km must be 'auto' or a list, got {sd!r}")
        out["search_delta_km"] = "auto"
    elif len(sd) != 3:
        raise RelocationError("search_delta_km must have three entries "
                              "[north_km, east_km, depth_km]")
    return out


chi_scale = sizing.chi_scale
_DEPTH_AXIS = 0
_AXIS_NAMES = ("depth", "north-south", "east-west")

def _read_stations_csv(path):
    """NET,STA,lat,lon,elev_m -- same format the SeisComP locator reads."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            net, sta, lat, lon, elev = (p.strip() for p in parts[:5])
            if net.lower() == "network":
                continue
            try:
                out[(net, sta)] = (float(lat), float(lon), float(elev))
            except ValueError:
                warnings.warn(f"could not parse station line: {line}")
    return out


# ObsPy extraction
# ---------------------------------------------------------------------------
def _map_phase(label, phase_map: Optional[Dict[str, str]]) -> Optional[str]:
    """Grid phase ("P"/"S") for a first-arrival label, else None. The list
    of accepted labels is shared with pykonal_locext: see
    relocate.FIRST_ARRIVAL_PHASES."""
    return pkrelocate.map_phase(label, phase_map)


def _pick_phase(pick, phase_map: Optional[Dict[str, str]]) -> Optional[str]:
    return _map_phase(pick.phase_hint, phase_map)


def _phase_error(default_pick_error, phase, label=None):
    """default_pick_error may be a scalar or a per-phase dict, as in the
    SeisComP config ({"P": 0.08, "S": 0.16})."""
    if isinstance(default_pick_error, dict):
        # an exact label entry ("Sg") wins, as in pykonal_locext
        if label and label in default_pick_error:
            return float(default_pick_error[label])
        return float(default_pick_error.get(
            phase, default_pick_error.get("P", 0.1)))
    return float(default_pick_error)


def _pick_rank(pick):
    """Sort key among duplicate picks for one station-phase: manual before
    automatic (or unset), then earliest. Earliest is the right tie-break for
    first-arrival grids (Pg vs Pn at one station), but it must not let an
    early automatic mis-pick beat the analyst's manual pick."""
    return (0 if pick.evaluation_mode == "manual" else 1,
            pick.time.timestamp)


def _pick_key(pick, phase):
    """(net, sta, phase) for a pick, or None if it cannot be used."""
    if phase is None or pick.time is None:
        return None
    wid = pick.waveform_id
    if wid is None or not wid.network_code or not wid.station_code:
        return None
    return (wid.network_code, wid.station_code, phase)


def _collect_arrivals(event, origin, phase_map, default_pick_error):
    """
    Build {(net, sta, phase): epoch_seconds} plus per-key pick errors and the
    originating Pick objects.

    Picks come from the origin's arrivals when it has any that resolve to a
    pick (so the analyst's association is respected), otherwise from
    event.picks. The analyst's decisions are honoured:

      * a pick with evaluation_status "rejected" is never used;
      * an input arrival with time_weight 0 is never used;
      * an arrival's own phase label takes precedence over the pick's
        phase_hint (the association is the later, deliberate decision);
      * only first-arrival labels are located with (P, Pg, Pn, ... see
        relocate.FIRST_ARRIVAL_PHASES, extendable through phase_map);
      * duplicates for one (net, sta, phase) key: manual beats automatic,
        then the earliest wins.

    Returns (arrivals, pick_errors, sources, inactive, notes):
      sources  : {key: Pick} for the picks used to locate
      inactive : [(key, Pick, reason)] for associated arrivals that are kept
                 in the output with weight 0 but not used (input weight 0,
                 rejected pick, not a first-arrival phase). Only arises from
                 the origin's arrivals. For a non-first-arrival phase,
                 key[2] is the original label (e.g. "pP"), so the output
                 arrival keeps it.
      notes    : human-readable lines for the driver's log
    """
    picks_by_id = {str(p.resource_id): p for p in event.picks}

    candidates = []          # (key, pick)
    inactive = []
    notes = []
    from_origin = False
    n_unlabelled = 0
    ignored_labels = {}      # label -> count, fallback path only
    labels = {}              # id(pick) -> label it was located under
    if origin is not None and origin.arrivals:
        for arr in origin.arrivals:
            p = picks_by_id.get(str(arr.pick_id))
            if p is None:
                continue
            from_origin = True
            # The arrival's label wins whenever it has one: an association
            # that relabels a pick (hint "P" -> arrival "pP") is deliberate.
            label = (arr.phase or "").strip() or (p.phase_hint or "").strip()
            if not label:
                n_unlabelled += 1
                continue
            phase = _map_phase(label, phase_map)
            if phase is None:
                # Not a first arrival (pP, PmP, Lg, ...): keep it in the
                # output under its own label, weight 0, rather than dropping
                # the analyst's association.
                key = _pick_key(p, label)
                if key is not None:
                    inactive.append((key, p, f"phase {label} is not a first "
                                              f"arrival"))
                continue
            key = _pick_key(p, phase)
            if key is None:
                continue
            labels[id(p)] = label
            if p.evaluation_status == "rejected":
                inactive.append((key, p, "pick evaluation status is rejected"))
            elif arr.time_weight is not None and float(arr.time_weight) == 0.0:
                inactive.append((key, p, "weight 0 in the input origin"))
            else:
                candidates.append((key, p))

    if not from_origin:
        n_rejected = 0
        for p in event.picks:
            phase = _pick_phase(p, phase_map)
            if phase is None:
                label = (p.phase_hint or "").strip()
                if label:
                    ignored_labels[label] = ignored_labels.get(label, 0) + 1
                else:
                    n_unlabelled += 1
                continue
            key = _pick_key(p, phase)
            if key is None:
                continue
            labels[id(p)] = p.phase_hint.strip()
            if p.evaluation_status == "rejected":
                n_rejected += 1
                continue
            candidates.append((key, p))
        if n_rejected:
            notes.append(f"ignored {n_rejected} pick(s) with evaluation "
                         f"status 'rejected'")

    if ignored_labels:
        notes.append("ignored picks that are not first arrivals: "
                     + ", ".join(f"{lab} x{n}" for lab, n
                                 in sorted(ignored_labels.items()))
                     + " (add a label to phase_map to use it)")
    if n_unlabelled:
        notes.append(f"ignored {n_unlabelled} pick(s) with no phase label")

    # one pick per key
    by_key = {}
    for key, p in candidates:
        by_key.setdefault(key, []).append(p)
    arrivals, pick_errors, sources = {}, {}, {}
    n_dup = 0
    for key, plist in by_key.items():
        p = min(plist, key=_pick_rank)
        n_dup += len(plist) - 1
        arrivals[key] = float(p.time.timestamp)
        unc = None
        if p.time_errors is not None:
            unc = p.time_errors.uncertainty
            if unc is None and p.time_errors.upper_uncertainty is not None \
                    and p.time_errors.lower_uncertainty is not None:
                unc = 0.5 * (p.time_errors.upper_uncertainty
                             + p.time_errors.lower_uncertainty)
        pick_errors[key] = (float(unc) if unc and unc > 0
                            else _phase_error(default_pick_error, key[2],
                                              labels.get(id(p))))
        sources[key] = p
    if n_dup:
        notes.append(f"{n_dup} duplicate pick(s) for an already-picked "
                     f"station-phase ignored (manual preferred, then "
                     f"earliest)")
    if inactive:
        notes.append(f"{len(inactive)} associated arrival(s) not used and "
                     f"kept with weight 0: " + ", ".join(
                         f"{'.'.join(k)} ({why})" for k, _, why in inactive[:8])
                     + ("..." if len(inactive) > 8 else ""))
    return arrivals, pick_errors, sources, inactive, notes


def _station_coords(station_inventory, arrivals, reference_time):
    """
    {(net, sta): (lat, lon, elev_m)} from an ObsPy Inventory, for the stations
    in `arrivals` only. Returns whatever it finds; reporting gaps is the
    caller's job, since a CSV may still supply them.

    Walks the inventory tree rather than calling get_coordinates(), which
    needs a fully-specified SEED id (no wildcards) and a channel to exist --
    neither is guaranteed for a station-only inventory.
    """
    out = {}
    if station_inventory is None:
        return out
    wanted = {(net, sta) for (net, sta, _phase) in arrivals}
    for network in station_inventory.networks:
        for station in network.stations:
            key = (network.code, station.code)
            if key not in wanted:
                continue
            # prefer an epoch that brackets the event
            if key in out and not _epoch_active(station, reference_time):
                continue
            if station.latitude is None or station.longitude is None:
                continue
            out[key] = (float(station.latitude), float(station.longitude),
                        float(station.elevation or 0.0))
    return out


def _resolve_stations(arrivals, station_inventory, stations_csv,
                      reference_time):
    """
    Station coordinates from whichever sources are available.

    Neither source is required -- an event relocates without any station
    coordinates at all, it just loses azimuthal gap, minimum distance and the
    per-arrival distance/azimuth fields. stations_csv alone is a complete
    answer; the ObsPy Inventory is a convenience for people who already have
    one loaded.

    Both are consulted and merged per station. The inventory wins where it has
    an entry, because it carries operating epochs and the CSV does not, but a
    partial inventory no longer suppresses the CSV for the stations it is
    missing -- which it used to, since the CSV was only read when the
    inventory produced nothing at all.
    """
    wanted = {(net, sta) for (net, sta, _phase) in arrivals}
    out = {}

    if stations_csv:
        csv_all = _read_stations_csv(stations_csv)
        out.update({k: v for k, v in csv_all.items() if k in wanted})

    from_inv = _station_coords(station_inventory, arrivals, reference_time)
    out.update(from_inv)                      # inventory takes precedence

    missing = sorted(wanted - set(out))
    if missing:
        if not out:
            warnings.warn(
                "no station coordinates available (set stations_csv in the "
                "config or pass station_inventory): azimuthal gap, minimum "
                "distance and per-arrival distance/azimuth will be omitted, "
                "and max_dist_km cannot be applied")
        else:
            warnings.warn(
                f"no coordinates for {len(missing)} station(s): "
                + ", ".join(f"{n}.{s}" for n, s in missing[:8])
                + ("..." if len(missing) > 8 else "")
                + ". Distance and azimuth will be omitted for these arrivals.")
    return out


def _epoch_active(station, when):
    if station.start_date is not None and when < station.start_date:
        return False
    if station.end_date is not None and when > station.end_date:
        return False
    return True


# Uncertainty conversion
# ---------------------------------------------------------------------------
def _horizontal_uncertainty(cov_km, coord_sys="spherical"):
    """
    (sigma_north_km, sigma_east_km, semi_major_km, semi_minor_km, azimuth_deg)
    from the posterior covariance in the local km frame.

    For a spherical grid that frame's axes are (r_hat, theta_hat, phi_hat) =
    (up, south, east), so north = -theta and the north/east cross-term takes
    a sign flip.
    """
    cov_km = np.asarray(cov_km, dtype=float)
    if coord_sys == "spherical":
        var_n, var_e = cov_km[1, 1], cov_km[2, 2]
        cov_ne = -cov_km[1, 2]
    else:
        var_n, var_e = cov_km[0, 0], cov_km[1, 1]
        cov_ne = cov_km[0, 1]

    h = np.array([[var_n, cov_ne], [cov_ne, var_e]], dtype=float)
    evals, evecs = np.linalg.eigh(h)
    evals = np.clip(evals, 0.0, None)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    semi_major, semi_minor = np.sqrt(evals[0]), np.sqrt(evals[1])
    # azimuth of the major axis, measured clockwise from north
    azimuth = math.degrees(math.atan2(evecs[1, 0], evecs[0, 0])) % 180.0
    return (math.sqrt(max(var_n, 0.0)), math.sqrt(max(var_e, 0.0)),
            float(semi_major), float(semi_minor), float(azimuth))


# Main entry point
# ---------------------------------------------------------------------------
def relocate(event,
             config=None,
             *,
             station_inventory=None,
             method=None,
             search_delta_km=None,
             search_delta_t=None,
             alpha=None,
             default_pick_error=None,
             fixed_depth=None,
             fixed_depth_tolerance_km=None,
             depth_min=None,
             depth_max=None,
             default_depth=None,
             min_picks=None,
             phase_map=None,
             posterior=True,
             posterior_nsamples=None,
             posterior_seed=None,
             confidence_level=None,
             stations_csv=None,
             earth_model_id=None,
             max_dist_km=None,
             velocity_models=None,
             traveltime_dir=None,
             ensure_nproc=None,
             build_missing=None,
             edt_ot_wt=None,
             edt_exponent=None,
             replace_origin=False,
             inplace=False,
             log=None,
             author="PyKonal-EDT",
             agency_id=None):
    """
    Relocate an ObsPy Event and return it with the new hypocenter attached.

    Settings come from a pykonal_locext-style JSON config (the same file the
    SeisComP locator reads), with keyword arguments overriding it:

        ev = relocate(ev, "pykonal_locext.json", station_inventory=inv)
        ev = relocate(ev, cfg, fixed_depth=10.0)      # override one setting

    Config keys honoured: method, alpha,
    edt_ot_wt, edt_exponent, default_pick_error (scalar or per-phase dict),
    search_delta_km, search_delta_t, depth_min, depth_max, default_depth,
    min_picks, posterior_nsamples, posterior_seed, confidence_level,
    stations_csv, earth_model_id, max_dist_km, phase_map. Keys the SeisComP
    locator uses but this module does not are warned about on load.

    Parameters that are not config keys
    -----------------------------------
    station_inventory : obspy Inventory, optional. Not required -- the
        config's stations_csv is a complete alternative, and both are merged
        per station when present (the inventory wins, since it carries
        operating epochs). With neither, the relocation still runs; only the
        distance/azimuth and gap fields are omitted.
    max_dist_km : float, optional. Station-event distance limit in km. None
        (the default) means "use the config", where an absent key gives
        1000 km and an explicit null gives no limit. Pass float("inf") to
        ask for no limit from here.
    fixed_depth : float, pin the depth (km). A zero-width depth bound, so the
        depth axis is not searched. The reported depth uncertainty is kept:
        it becomes the posterior spread about the pinned value.
    log : callable, optional. Receives the sizing diagnostics the SeisComP
        locator writes to its log (box sizing, floors, cap, depth clamp,
        search-edge warnings). Defaults to discarding them; pass `print` or
        a logger method to see them.
    build_missing : bool, optional. Compute any traveltime grids that are
        absent, into the sharded inventory at `traveltime_dir`, using
        `velocity_models`. None (default) takes the config's build_missing
        key, which defaults to true -- the same behaviour as pykonal_locext
        reading the same file. Each missing grid is one FMM solve (seconds
        to minutes, once per station-phase); the start of a build is logged,
        or raised as a UserWarning when no `log` is given, so a long first
        relocation is not mistaken for a hang. Without velocity_models
        nothing can be built and those arrivals get weight 0.
    replace_origin : overwrite the input origin instead of appending.
    inplace : modify the passed event rather than a deep copy.

    Returns
    -------
    obspy.core.event.Event
    """
    log_discarded = log is None
    log = log or (lambda _m: None)
    st = _settings(config, dict(
        method=method, search_delta_km=search_delta_km,
        search_delta_t=search_delta_t, alpha=alpha,
        default_pick_error=default_pick_error, depth_min=depth_min,
        depth_max=depth_max, default_depth=default_depth, min_picks=min_picks,
        fixed_depth_tolerance_km=fixed_depth_tolerance_km,
        phase_map=phase_map, posterior_nsamples=posterior_nsamples,
        posterior_seed=posterior_seed, confidence_level=confidence_level,
        stations_csv=stations_csv, earth_model_id=earth_model_id,
        max_dist_km=max_dist_km, velocity_models=velocity_models,
        traveltime_dir=traveltime_dir, ensure_nproc=ensure_nproc,
        edt_ot_wt=edt_ot_wt, edt_exponent=edt_exponent,
        build_missing=build_missing))
    build_missing = bool(st["build_missing"])

    method = st["method"]
    alpha = float(st["alpha"])
    min_picks = int(st["min_picks"])
    try:
        phase_map = pkrelocate.normalize_phase_map(st["phase_map"])
    except ValueError as err:
        raise RelocationError(f"config: {err}") from None
    search_delta_km = ("auto" if st["search_delta_km"] == "auto"
                       else [float(v) for v in st["search_delta_km"]])
    search_delta_t = float(st["search_delta_t"])
    depth_min, depth_max = st["depth_min"], st["depth_max"]
    confidence_level = float(st["confidence_level"])
    try:
        max_dist = resolve_max_dist(st)      # None = no limit
    except ValueError as err:
        raise RelocationError(str(err)) from None

    ev = event if inplace else copy.deepcopy(event)

    origin = None
    try:
        origin = ev.preferred_origin()
    except Exception:
        origin = None
    if origin is None and ev.origins:
        origin = ev.origins[-1]
    if origin is None:
        raise RelocationError("event has no origin to start from")
    if origin.latitude is None or origin.longitude is None:
        raise RelocationError("starting origin has no latitude/longitude")

    if origin.depth is not None:
        dep0_km = float(origin.depth) / 1000.0
    elif st["default_depth"] is not None:
        dep0_km = float(st["default_depth"])
    else:
        dep0_km = 0.0
    t0 = float(origin.time.timestamp) if origin.time is not None else None

    arrivals, pick_errors, pick_sources, inactive, notes = _collect_arrivals(
        ev, origin, phase_map, st["default_pick_error"])
    for line in notes:
        log(line)
    if len(arrivals) < min_picks:
        raise RelocationError(
            f"only {len(arrivals)} usable picks, need {min_picks}"
            + (f" ({len(inactive)} more associated but disabled)"
               if inactive else ""))

    if t0 is None:                       # no starting time: use earliest pick
        t0 = min(arrivals.values()) - 1.0

    stations_geo = _resolve_stations(arrivals, station_inventory,
                                     st["stations_csv"], UTCDateTime(t0))

    # max_dist_km: drop arrivals whose station is beyond the cutoff. A 1D
    # model that fits the crust does not fit Pn at 1000 km, and those
    # arrivals carry systematic error that drags the solution.
    dropped = []
    if max_dist is not None and stations_geo:
        cutoff = max_dist
        for key in list(arrivals):
            sta = stations_geo.get(key[:2])
            if sta is None:
                continue
            # grid-sphere distance: the metric inventory.add used to crop
            # the grids to max_dist, so this filter agrees with them
            d_km = geo_distance_azimuth(float(origin.latitude),
                                        float(origin.longitude),
                                        sta[0], sta[1])[0]
            if d_km > cutoff:
                dropped.append(key)
                arrivals.pop(key), pick_errors.pop(key, None)
        if dropped:
            warnings.warn(f"max_dist_km={cutoff:g}: dropped {len(dropped)} "
                          f"arrival(s) beyond the cutoff")
        if len(arrivals) < min_picks:
            raise RelocationError(
                f"only {len(arrivals)} picks within max_dist_km={cutoff:g}, "
                f"need {min_picks}")

    transform = GeoTransform()

    # ---- traveltime grids: use what exists, optionally build what does not
    # With no log, still print the start of a build as a plain message:
    # an FMM solve per missing grid can take minutes and must not look like a hang
    build_log = log if not log_discarded else (
        lambda m: print(m, flush=True) if str(m).startswith("computing") else None)
    inventory_path, expected_missing = _ensure_grids(
        arrivals, stations_geo, transform, st, max_dist,
        build_missing=build_missing, log=build_log)

    # Settings for the shared sequence: the raw config (auto_search_* keys
    # and anything else sizing reads) overlaid with this driver's resolved
    # values. The locator's scalar pick-error fallback is the P value.
    rsettings = dict(st.get("_raw_config", {}))
    rsettings.update({k: st[k] for k in pkrelocate.DEFAULTS if k in st})
    rsettings["search_delta_km"] = search_delta_km
    rsettings["default_pick_error"] = _phase_error(st["default_pick_error"], "P")

    with inventory_lock(inventory_path, shared=True), \
         EQLocator(str(inventory_path)) as locator:
        locator.expected_missing = expected_missing
        try:
            result = pkrelocate.relocate(
                locator, transform, arrivals, pick_errors, stations_geo,
                (float(origin.latitude), float(origin.longitude), dep0_km, t0),
                rsettings, fixed_depth=fixed_depth, posterior=posterior,
                log=log,
                missing_grid_advice=_missing_grid_advice(
                    build_missing, bool(st.get("velocity_models"))))
        except pkrelocate.RelocationError as exc:
            raise RelocationError(str(exc)) from exc

    for level, text in result.messages:
        # failures always warn; missing grids warn only when log messages
        # are being discarded, so they are shown once either way
        if level == "error" or (log_discarded
                                and text.startswith("No traveltime grid")):
            warnings.warn(text)

    post = result.posterior
    lat, lon, depth_km = result.latitude, result.longitude, result.depth_km
    soln, residuals, quality = result.hypocenter, result.residuals, result.quality
    depth_pinned = result.depth_pinned

    new_origin = _build_origin(
        ev, origin, lat, lon, depth_km, float(soln[3]), residuals, quality,
        post, pick_sources, stations_geo, depth_pinned, fixed_depth,
        author, agency_id, confidence_level, st["earth_model_id"],
        solve_keys=set(result.solve_keys), inactive=inactive,
        not_used=_not_used_reasons(result, dropped))

    if replace_origin:
        ev.origins = [o for o in ev.origins
                      if str(o.resource_id) != str(origin.resource_id)]
    ev.origins.append(new_origin)
    ev.preferred_origin_id = new_origin.resource_id
    return ev


def _ensure_grids(arrivals, stations_geo, transform,
                  st, max_dist, *, build_missing, log):
    """
    Resolve the traveltime inventory path, building missing station-phase
    grids first when asked to.

    Building needs build_missing AND velocity_models. Each missing grid is
    one FMM solve -- seconds to minutes -- so the caller routes the start of
    a build to a warning when it has no log.
    """
    vel_models = st.get("velocity_models")
    tt_dir = st.get("traveltime_dir")
    if not tt_dir:
        raise RelocationError("No traveltime_dir configured: set it to the sharded inventory.")
    master = os.path.join(str(tt_dir), "inventory.h5")
    if not (build_missing and vel_models):
        if not os.path.exists(master):
            why = ("build_missing is false" if vel_models
                   else "no velocity_models are configured")
            raise RelocationError(
                f"Sharded traveltime index {master} does not exist and "
                f"{why}, so it cannot be built. Configure velocity_models "
                f"with build_missing true, or point traveltime_dir at an "
                f"existing sharded inventory.")
        return master, set()

    # A grid cannot be solved without knowing where the station is. This is
    # stricter than locating, which tolerates missing coordinates and merely
    # drops the distance and azimuth fields.
    requests = {}
    no_coords = []
    for key in arrivals:
        sta = stations_geo.get(key[:2])
        if sta is None:
            no_coords.append(key)
            requests[key] = None
        else:
            requests[key] = transform.geo_to_grid(sta[0], sta[1], -sta[2] / 1000.0)
    if no_coords:
        warnings.warn(
            f"no coordinates for {len(no_coords)} station-phase(s), so their "
            f"grids cannot be built: "
            + ", ".join(".".join(k) for k in no_coords[:8])
            + ("..." if len(no_coords) > 8 else ""))

    # arrivals that can never get a grid: no coords, no model for the
    # phase, or dropped by the off-grid filter (diff before/after, since
    # fit_requests_to_grid logs but does not return the excluded keys)
    unbuildable = {k for k, v in requests.items() if v is None}
    _phases = {p.upper() for p in vel_models}
    unbuildable |= {k for k in requests if k[2].upper() not in _phases}
    _before = set(requests)
    requests = fit_requests_to_grid(
        requests, vel_models,
        vertical_axis=_DEPTH_AXIS, log=log)
    unbuildable |= _before - set(requests)

    report = ensure_traveltimes_sharded(
        tt_dir, requests, vel_models,
        max_dist=max_dist,
        nproc=st.get("ensure_nproc"), log=log)

    if report["computed"]:
        log(f"computed {len(report['computed'])} new traveltime grid(s)")
    if report["skipped"]:
        warnings.warn(
            f"{len(report['skipped'])} station-phase(s) have no traveltime "
            f"grid and will not contribute: "
            + ", ".join(".".join(k) for k in report["skipped"][:8])
            + ("..." if len(report["skipped"]) > 8 else ""))
    return report["master"], {"/".join(k) for k in unbuildable}


def _missing_grid_advice(build_missing, have_models):
    """Tail of the 'No traveltime grid' message: what would fix it."""
    if not have_models:
        return (" Configure velocity_models (with build_missing true) to "
                "compute them.")
    if not build_missing:
        return (" Set build_missing to true in the config (or pass "
                "build_missing=True) to compute them.")
    return (" They could not be built (no station coordinates, no velocity "
            "model for the phase, or off the model grid).")


def _not_used_reasons(result, dropped_far):
    """{key: why it did not contribute}, for the weight-0 arrival comments."""
    why = {}
    for k in dropped_far:
        why[k] = "beyond max_dist_km"
    for k in result.no_grid:
        why[k] = "no traveltime grid"
    for k in result.uncovered:
        why[k] = "traveltime grid does not cover the solution"
    for k in result.excluded:
        why[k] = "rejected as an outlier"
    solve = set(result.solve_keys)
    for k in result.residuals:
        if k not in solve:
            why[k] = "report only (not among the solve_n_closest stations)"
    return why


def _build_origin(ev, src_origin, lat, lon, depth_km, t0, residuals, quality,
                  post, pick_sources, stations_geo, depth_pinned,
                  fixed_depth, author, agency_id, confidence_level, earth_model_id,
                  *, solve_keys, inactive=(), not_used=None):
    """
    solve_keys : arrivals the hypocentre was solved with. Only these get
        weight 1; residuals for the rest (solve_n_closest report-only set)
        are still written, with weight 0.
    inactive : [(key, pick, reason)] associated arrivals the analyst
        disabled; written with weight 0 and the reason as a comment.
    not_used : {key: reason} for the other weight-0 arrivals.
    """
    not_used = not_used or {}
    o = Origin()
    o.resource_id = ResourceIdentifier(prefix="smi:local/pykonal/origin")
    o.time = UTCDateTime(t0)
    o.latitude = _r(lat)
    o.longitude = _r(lon)
    o.depth = _r(depth_km * 1000.0)            # QuakeML depth is metres
    o.depth_type = "operator assigned" if depth_pinned else "from location"
    o.method_id = ResourceIdentifier("smi:local/pykonal/edt")
    o.earth_model_id = ResourceIdentifier(
        earth_model_id if earth_model_id else "smi:local/pykonal/traveltimes")
    o.evaluation_mode = "manual"
    o.evaluation_status = "preliminary"
    o.creation_info = {"author": author, "creation_time": UTCDateTime()}
    if agency_id:
        o.creation_info["agency_id"] = agency_id

    # ---- arrivals
    def _arrival(key, pick, weight, residual=None, reason=None):
        arr = Arrival()
        arr.resource_id = ResourceIdentifier(prefix="smi:local/pykonal/arrival")
        arr.pick_id = pick.resource_id
        arr.phase = key[2]
        arr.time_weight = weight
        if residual is not None:
            arr.time_residual = _r(residual)
        if reason:
            arr.comments.append(_comment(f"not used: {reason}"))
        sta = stations_geo.get(key[:2])
        if sta is not None:
            _km, dist_deg, az = geo_distance_azimuth(lat, lon, sta[0], sta[1])
            arr.distance = _r(dist_deg)
            arr.azimuth = _r(az, 1)
        o.arrivals.append(arr)

    for key, pick in pick_sources.items():
        used = key in residuals and key in solve_keys
        _arrival(key, pick, 1.0 if used else 0.0, residuals.get(key),
                 None if used else not_used.get(
                     key, "no traveltime at the solution"))
    for key, pick, reason in inactive:
        _arrival(key, pick, 0.0, reason=reason)

    # ---- quality
    q = OriginQuality()
    q.used_phase_count = int(quality.get("nobs", len(residuals)))
    q.associated_phase_count = len(pick_sources) + len(inactive)
    q.standard_error = _r(quality.get("rms"))

    # Azimuthal gap from locator.quality(), the same value pykonal_locext
    # reports: it is computed over the solve set only (rejected outliers,
    # uncovered arrivals and the solve_n_closest report-only set do not
    # shrink it), and it includes the sector crossing 0/360 deg.
    gap = quality.get("azimuthal_gap", np.nan)
    if gap is not None and np.isfinite(gap):
        q.azimuthal_gap = _r(gap, 2)

    mind = quality.get("min_station_dist", np.nan)
    if np.isfinite(mind):
        q.minimum_distance = _r(kilometer2degrees(mind))
    o.quality = q

    # ---- uncertainty
    if post is not None:
        cov_km = np.asarray(post["covariance_km"], dtype=float)
        iz = sizing.depth_axis("spherical")
        sig_n, sig_e, semi_major, semi_minor, az_major = \
            _horizontal_uncertainty(cov_km, "spherical")
        # Reported even when the depth was fixed: this is the data's opinion
        # of the pinned depth, not a claim that the depth was solved for.
        sig_d = math.sqrt(max(cov_km[iz, iz], 0.0))

        # Scale to the requested confidence level. The three factors differ
        # a lot -- at 68% they are 1.000 (marginal), 1.515 (2-D ellipse) and
        # 1.878 (3-D ellipsoid) -- so a single sigma cannot be reused for
        # all three. Matches pykonal_locext so the two tools agree.
        k1 = chi_scale(confidence_level, 1)
        k2 = chi_scale(confidence_level, 2)
        k3 = chi_scale(confidence_level, 3)

        # lat/lon uncertainties are DEGREES here (5 dp is about 1 m);
        # depth and the ellipse/ellipsoid are METRES (1 dp is 10 cm).
        o.latitude_errors = QuantityError(
            uncertainty=_r(kilometer2degrees(sig_n * k1), 5),
            confidence_level=confidence_level)
        o.longitude_errors = QuantityError(
            uncertainty=_r(kilometer2degrees(
                sig_e * k1 / max(math.cos(math.radians(lat)), 1e-4)), 5),
            confidence_level=confidence_level)
        o.depth_errors = QuantityError(uncertainty=_r(sig_d * k1 * 1000.0, 1),
                                       confidence_level=confidence_level)
        ts = post.get("time_sigma")
        if ts is not None and np.isfinite(ts):
            o.time_errors = QuantityError(uncertainty=_r(float(ts) * k1),
                                          confidence_level=confidence_level)

        ou = OriginUncertainty()
        ou.preferred_description = "uncertainty ellipse"
        ou.confidence_level = confidence_level
        ou.min_horizontal_uncertainty = _r(semi_minor * k2 * 1000.0, 1)
        ou.max_horizontal_uncertainty = _r(semi_major * k2 * 1000.0, 1)
        ou.azimuth_max_horizontal_uncertainty = _r(az_major, 1)
        ou.horizontal_uncertainty = _r(semi_major * k2 * 1000.0, 1)
        semi, plunge, az3, rot = sizing.ellipsoid_orientation(cov_km)
        if np.all(np.isfinite(semi)):
            semi = semi * k3 * 1000.0        # 3-dof scaling, km -> m
            ou.confidence_ellipsoid = ConfidenceEllipsoid(
                semi_major_axis_length=_r(semi[0], 1),
                semi_intermediate_axis_length=_r(semi[1], 1),
                semi_minor_axis_length=_r(semi[2], 1),
                major_axis_plunge=_r(plunge, 1),
                major_axis_azimuth=_r(az3, 1),
                # Written: QuakeML requires it (the file is schema-invalid
                # without), and it shares the plunge sign convention that
                # the plunge above already commits to (sizing.PLUNGE_SIGN,
                # positive down). Verified by rebuilding the covariance from
                # (azimuth, plunge, rotation).
                major_axis_rotation=_r(rot, 1),
            )
        o.origin_uncertainty = ou

        limits = post.get("box_limit_report")
        if limits is not None:
            if limits["warning"]:
                o.comments.append(_comment(limits["warning"]))
            if limits["note"]:
                o.comments.append(_comment(limits["note"]))
        # at_search_edge means the curvature stencil ran off the LOADED
        # traveltime grids -- a grid-coverage problem, not box truncation.
        edge = np.asarray(post.get("at_search_edge", [False] * 3), dtype=bool)
        if edge.any():
            hit = ", ".join(n for n, e in zip(_AXIS_NAMES, edge) if e)
            o.comments.append(_comment(
                f"traveltime grids end within the posterior on {hit}: the "
                f"uncertainty on that axis may be understated"))
        dtc = post.get("depth_time_correlation")
        if dtc is not None and np.isfinite(dtc) and abs(dtc) > 0.9:
            o.comments.append(_comment(
                f"depth and origin time correlate at {dtc:+.2f}: they trade "
                f"off almost completely and neither is independently "
                f"determined"))

    if depth_pinned:
        _du = (o.depth_errors.uncertainty
               if o.depth_errors is not None else None)
        _pin = float(fixed_depth) if fixed_depth is not None else None
        _head = (f"depth pinned at {_pin:.2f} km by operator; solved "
                 f"{depth_km:.2f} km within the fixed_depth_tolerance window"
                 if _pin is not None and abs(depth_km - _pin) > 1e-9 else
                 f"depth fixed at {depth_km:.2f} km by operator")
        o.comments.append(_comment(
            _head + "; depth axis searched only within the pin's tolerance "
            "window. The reported depth uncertainty"
            + (f" ({_du / 1000.0:.2f} km)" if _du is not None else "")
            + " is the posterior spread of the likelihood about that pinned "
              "depth -- how well the data support it -- not a solved-for "
              "depth error."))

    return o


def _r(value, places=6):
    """
    Round a value to 'places' decimals to keep file sizes down.
    """
    if value is None:
        return None
    v = float(value)
    return None if not math.isfinite(v) else round(v, places)


def _comment(text):
    from obspy.core.event import Comment
    return Comment(text=text,
                   resource_id=ResourceIdentifier(prefix="smi:local/pykonal/comment"))