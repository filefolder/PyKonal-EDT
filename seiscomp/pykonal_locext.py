#!/usr/bin/env python3
"""
SeisComP LocExt wrapper for the PyKonal-EDT locator.

SeisComP's ExternalLocator (plugin: locext) calls this script with a
SeisComP XML EventParameters document on stdin (one origin + its picks)
and expects a SeisComP XML document containing a single relocated Origin
on stdout.

Configure in global.cfg (or scolv.cfg):

    plugins = ${plugins}, locext
    ExternalLocator.profiles = pykonalEDT:"/path/to/python /path/to/pykonal_locext.py --config /path/to/pykonal_locext.json"

LocExt may additionally pass:
    --max-dist=X                cut-off distance (degrees) — applied if
                                station coordinates are configured
    --fixed-depth=X             fix depth to X km
    --ignore-initial-location   anchor the search on the centroid of the
                                stations with arrivals instead of the
                                input origin (search_anchor "stations")

All diagnostics go to stderr (visible in scolv's process manager / logs);
stdout carries ONLY the result XML.
"""

import time as _time
#_T_PROC = _time.time()

import argparse
import json
import math
import os
import sys
import tempfile
import traceback

import numpy as np

_REAL_STDOUT = sys.stdout

import seiscomp.core
import seiscomp.datamodel
import seiscomp.io

from pykonal.locate import EQLocator
from pykonal import constants as pk_constants
from pykonal.transformations import GeoTransform, geo_distance_azimuth
from pykonal.relocate import (relocate as pk_relocate, RelocationError,
                              map_phase, normalize_phase_map)
from pykonal.sizing import chi_scale, ellipsoid_orientation
from pykonal.inventory import (ensure_traveltimes_sharded, fit_requests_to_grid,
                               inventory_lock, resolve_max_dist,
                               describe_max_dist)


# Mean Earth radius (km) for angular <-> arc-length conversions on the
# spherical grid. Grids are always spherical (pykonal geo2sph)
MEAN_RADIUS = 6371.


# Decimal places for numbers written to the output Origin.
#
# The posterior is a Monte Carlo estimate and the solver works to sub-node
# precision, so both produce floats with far more digits than carry meaning.
# Full repr in the XML implies a precision the method does not have and makes
# files awkward to diff.
#
# NOTE the units, which differ from the QuakeML side: SeisComP carries
# lat/lon/depth uncertainty and horizontalUncertainty in KILOMETRES, while
# the confidence ellipsoid is in METRES. The digit counts below are chosen
# for equivalent physical precision, not to match the QuakeML numbers.
ROUND_DP = {
    "coord": 6,          # lat/lon degrees (~0.1 m)
    "depth": 6,          # km
    "coord_unc": 3,      # km (1 m)
    "depth_unc": 3,      # km (1 m)
    "horiz_unc": 3,      # km (1 m)
    "ellipsoid": 1,      # m (10 cm)
    "azimuth": 1,        # degrees
    "plunge": 1,         # degrees
    "gap": 2,            # degrees
    "distance": 6,       # degrees (~0.1 m)
    "time": 6,           # seconds
    "residual": 6,       # seconds
}


def _round_out(value, kind):
    """Round an output value; returns None if it is not finite."""
    v = float(value)
    return None if not math.isfinite(v) else round(v, ROUND_DP[kind])


def gc_distance_azimuth(lat1, lon1, lat2, lon2):
    """Great-circle distance (degrees of arc) and azimuth 1->2 (degrees), on
    the traveltime grid sphere (geocentric latitude): the metric the grids
    and their max_dist crop use. See transformations.geo_distance_azimuth."""
    _km, dist_deg, az = geo_distance_azimuth(lat1, lon1, lat2, lon2)
    return dist_deg, az


def log(msg):
    print(f"[pykonal_locext] {msg}", file=sys.stderr, flush=True)


def _set_enum(setter, *names):
    """
    Set a seiscomp enum-valued field robustly across versions. For each
    candidate attribute name on seiscomp.datamodel that exists, try the
    setter; keep the first value the setter accepts. A name may resolve to
    a constant from a DIFFERENT enum, which the setter rejects (e.g.
    "enum value out of range") — those are skipped. Returns True on
    success, False if no candidate worked.
    """
    for name in names:
        value = getattr(seiscomp.datamodel, name, None)
        if value is None:
            continue
        try:
            setter(value)
            return True
        except Exception:
            continue
    return False



# --------------------------------------------------------------------------
# XML I/O helpers
# --------------------------------------------------------------------------


def read_input():
    """
    Read EventParameters from stdin and extract everything needed into
    plain Python objects.

    Returns
    -------
    origin_info : dict
        latitude, longitude, depth (km, may be None), time (seismic.Time),
        publicID.
    arrivals : list of dict
        one per arrival: network, station, phase, pick_id, time
        (seismic.Time), time_uncertainty (float or None), weight (float),
        used (bool).
    keepalive : the EventParameters object (hold a reference until done).
    """
    data = sys.stdin.buffer.read()
    return read_input_from_bytes(data)


def read_input_from_bytes(data):
    """
    Same as read_input() but parses EventParameters from an in-memory
    bytes object instead of stdin. Used by the batch driver so many events
    can be relocated in one process. See read_input for the segfault note
    on keeping the parent EventParameters alive.
    """
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        f.write(data)
        path = f.name

    ar = seiscomp.io.XMLArchive()
    if not ar.open(path):
        raise RuntimeError("Could not open input XML")
    obj = ar.readObject()
    ar.close()
    try:
        os.unlink(path)
    except OSError:
        pass

    ep = seiscomp.datamodel.EventParameters.Cast(obj)
    if ep is None:
        raise RuntimeError("Input is not an EventParameters document")
    if ep.originCount() < 1:
        raise RuntimeError("Input contains no origin")

    # index picks by publicID for arrival lookup
    picks = {}
    for i in range(ep.pickCount()):
        pick = ep.pick(i)
        picks[pick.publicID()] = pick

    origin = ep.origin(0)

    def _opt(getter):
        """Return getter().value() or None if the quantity is unset."""
        try:
            return getter().value()
        except Exception:
            return None

    origin_info = {
        "publicID": origin.publicID(),
        "latitude": origin.latitude().value(),
        "longitude": origin.longitude().value(),
        "depth": _opt(origin.depth),
        "time": origin.time().value(),
        "event_id": _event_id_for_origin(ep, origin.publicID()),
    }

    arrivals = []
    for i in range(origin.arrivalCount()):
        arr = origin.arrival(i)
        pid = arr.pickID()
        pick = picks.get(pid)
        if pick is None:
            continue

        try:
            weight = arr.weight()
        except Exception:
            weight = 1.0

        # phase: prefer the arrival's phase, else the pick's hint
        phase = None
        try:
            phase = arr.phase().code()
        except Exception:
            try:
                phase = pick.phaseHint().code()
            except Exception:
                phase = "P"
        if not phase:
            phase = "P"

        try:
            time_uncertainty = pick.time().uncertainty()
        except Exception:
            time_uncertainty = None

        wf = pick.waveformID()
        arrivals.append({
            "network": wf.networkCode(),
            "station": wf.stationCode(),
            "phase": phase,
            "pick_id": pid,
            "time": pick.time().value(),
            "time_uncertainty": time_uncertainty,
            "weight": weight,
        })

    return origin_info, arrivals, ep


def _event_id_for_origin(ep, origin_id):
    """
    publicID of the event this origin belongs to, or None. scolv does not
    always include the Event in the document it hands a LocExt locator, so
    absence is normal; any API difference is treated the same way.
    """
    try:
        events = [ep.event(i) for i in range(ep.eventCount())]
    except Exception:
        return None
    for ev in events:
        try:
            if ev.preferredOriginID() == origin_id:
                return ev.publicID()
            for j in range(ev.originReferenceCount()):
                if ev.originReference(j).originID() == origin_id:
                    return ev.publicID()
        except Exception:
            continue
    if len(events) == 1:            # a single event in the document
        try:
            return events[0].publicID()
        except Exception:
            pass
    return None


def _banner(label, width=60):
    """'====label====' padded to `width`, at least two '=' each side."""
    label = str(label) if label else ""
    if not label:
        return "=" * width
    if len(label) + 4 >= width:
        return f"=={label}=="
    return f"{label:=^{width}}"


def write_output(origin, wrap_eventparameters=False):
    """
    Write the relocated origin to the REAL stdout as a SeisComP XML
    document containing the Origin DIRECTLY under <seiscomp>, i.e.

        <seiscomp ...>
          <Origin publicID="..."> ... </Origin>
        </seiscomp>

    This is exactly what the LocExt plugin expects ("a SeisComP XML
    document just containing an origin"). NOTE: the origin must NOT be
    wrapped in <EventParameters> — scolv's reader looks for a top-level
    Origin and, if it finds EventParameters instead, rejects the result
    with "no origin in result document". Writing the Origin object itself
    (not an EventParameters) via XMLArchive produces the correct layout.

    Uses _REAL_STDOUT (captured before we redirected fd 1 to stderr in
    main), so nothing any library printed to stdout can corrupt the
    single XML document the plugin reads from our stdout.
    """
    xml_bytes = origin_to_bytes(origin, wrap_eventparameters=wrap_eventparameters)
    _REAL_STDOUT.buffer.write(xml_bytes)
    _REAL_STDOUT.flush()


def origin_to_bytes(origin, wrap_eventparameters=False):
    """Serialize the relocated Origin to SeisComP XML bytes.

    By default writes a BARE <Origin> directly under <seiscomp>, which is
    what scolv's LocExt reader requires. With wrap_eventparameters=True the
    Origin is placed inside an <EventParameters> element, producing a
    self-contained document that scdispatch can merge/associate directly
    (no separate wrapping step). The arrivals still reference existing pick
    publicIDs; the picks themselves are not emitted, so the target database
    must already hold them (true for relocation of an existing catalog)."""
    if wrap_eventparameters:
        ep = seiscomp.datamodel.EventParameters()
        ep.add(origin)
        obj = ep
        expect = b"<EventParameters"
    else:
        obj = origin
        expect = b"<Origin"
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="r+") as f:
        path = f.name
    ar = seiscomp.io.XMLArchive()
    ar.setFormattedOutput(True)
    if not ar.create(path):
        raise RuntimeError("Could not create output XML")
    ar.writeObject(obj)
    ar.close()
    with open(path, "rb") as f:
        xml_bytes = f.read()
    if not xml_bytes.lstrip().startswith(b"<?xml"):
        raise RuntimeError("generated output does not begin with an XML declaration")
    if expect not in xml_bytes:
        raise RuntimeError(f"generated output missing expected element {expect!r}")
    try:
        os.unlink(path)
    except OSError:
        pass
    return xml_bytes


# --------------------------------------------------------------------------
# main locator logic
# --------------------------------------------------------------------------
def main():
    # Capture the real stdout for the final XML, then redirect fd 1 to fd 2
    # so that anything pykonal / h5py / solvers / warnings might print goes
    # to stderr (visible in scolv logs) and can never corrupt the single
    # XML document the locext plugin reads from our stdout.
    global _REAL_STDOUT
    # Redirect fd 1 -> fd 2 exactly once so stray library output can't
    # corrupt the XML on stdout. Guarded so batch drivers that call main()
    # in a loop don't re-dup on every event.
    if not getattr(main, "_stdout_redirected", False):
        _REAL_STDOUT = os.fdopen(os.dup(1), "w")
        os.dup2(2, 1)
        main._stdout_redirected = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-dist", type=float, default=None)
    parser.add_argument("--fixed-depth", type=float, default=None)
    parser.add_argument("--ignore-initial-location", action="store_true")
    parser.add_argument(
        "--wrap-eventparameters", action="store_true",
        help="wrap the output Origin in an <EventParameters> element so the "
             "document can be fed straight to scdispatch (-O merge) for "
             "standalone/batch relocation. The DEFAULT is a bare <Origin>, "
             "which is what scolv's LocExt reader requires; only use this "
             "flag when driving the plugin outside scolv.")
    args, unknown = parser.parse_known_args()
    if unknown:
        log(f"ignoring unknown args: {unknown}")

    with open(args.config) as f:
        cfg = json.load(f)
    # max_dist_km: absent -> 1000 km; null -> no limit (explicit only).
    # Resolved once here and used for both the arrival filter and the grid
    # builds, so the two can never disagree.
    try:
        max_dist_km = resolve_max_dist(cfg)
        # phase_map: extra first-arrival labels, or null to exclude a
        # built-in one; shared with pykonal_obspy (relocate.map_phase)
        phase_map = normalize_phase_map(cfg.get("phase_map"))
    except ValueError as err:
        raise RelocationError(f"config: {err}") from None

    transform = GeoTransform()
    method = cfg.get("method", "edt")
    alpha = float(cfg.get("alpha", 0.01))
    alpha = np.maximum(alpha,1e-5)

    # Per-phase default pick uncertainty (seconds), used ONLY when a pick
    # carries no stated time uncertainty of its own. A phase->floor map:
    # keys may be exact phase names ("Sg") or a single letter ("S") matching
    # any phase starting with it, with exact names taking precedence. S
    # arrivals are typically less precise than P, so raise the S value if
    # desired. Picks that DO carry their own uncertainty always use that.
    default_pick_error = cfg.get("default_pick_error", {"P": 0.1, "S": 0.1})
    if not isinstance(default_pick_error, dict):
        # tolerate a legacy scalar: apply it to all phases
        default_pick_error = {"P": float(default_pick_error),
                              "S": float(default_pick_error)}
    default_pick_error = {str(k): float(v)
                          for k, v in default_pick_error.items()}
    # a single scalar for event-level uses (auto-box floor, pykonal global
    # fallback): the smallest phase floor, so it never over-inflates.
    _default_err_scalar = (min(default_pick_error.values())
                           if default_pick_error else 0.1)

    def _default_err_for_phase(phase):
        if phase in default_pick_error:
            return default_pick_error[phase]             # exact match wins
        if phase and phase[0] in default_pick_error:
            return default_pick_error[phase[0]]          # letter match
        return _default_err_scalar
    delta_km = cfg.get("search_delta_km", [30.0, 30.0, 15.0])
    _auto_delta = isinstance(delta_km, str) and delta_km.lower() == "auto"
    if _auto_delta:
        delta_km = [50.0, 50.0, 25.0]   # placeholder; recomputed per event
    delta_t = float(cfg.get("search_delta_t", 10.0))
    nsamples = int(cfg.get("posterior_nsamples", 2048))
    # Number of adaptive importance-sampling rounds inside sample_posterior.
    # The solver refines its proposal to the weighted posterior moments each
    # round (2 is the library default and is plenty for a smooth ellipsoid);
    # raise for very skewed posteriors, lower to 1 for speed. This is the
    # solver's OWN adaptation -- the wrapper no longer resizes the box.
    posterior_rounds = int(cfg.get("posterior_rounds", 2))
    # Fixed seed for posterior sampling so repeated relocations of the
    # same event give IDENTICAL uncertainties (the DE search is already
    # seeded separately). Set posterior_seed to null in the config for
    # independent draws each time (e.g. to gauge sampling variability).
    _pseed = cfg.get("posterior_seed", None)
    posterior_seed = None if _pseed is None else int(_pseed)
    # Confidence level (percent) for the reported ellipsoid and
    # uncertainties. 68 matches NonLinLoc, whose confidence ellipsoid is
    # the 68% THREE-DIMENSIONAL region -- 1.878 sigma, not 1 sigma. Earlier
    # versions of this plugin emitted bare 1-sigma semi-axes while calling
    # them a confidence ellipsoid, which made pykonal ellipsoids look about
    # half the linear size of NLL ellipsoids for an identical posterior.
    confidence_level = float(cfg.get("confidence_level", 68.0))

    stations = {}
    if cfg.get("stations_csv"):
        # CSV: NET,STA,lat,lon,elev_m — used for arrival distance/azimuth
        # and the --max-dist filter. Optional but recommended.
        with open(cfg["stations_csv"]) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    net, sta, lat, lon, elev = line.split(",")[:5]
                    if net.lower() == "network": continue
                    stations[(net, sta)] = (float(lat), float(lon), float(elev))
                except Exception as e:
                    print(f"could not parse station line: {line} \n {e}")

    # origin_info: plain dict; arrivals_in: list of plain dicts; ep is a
    # keepalive reference (see read_input) — hold it until main returns.
    origin_info, arrivals_in, _ep_keepalive = read_input()

    # ---------------- initial location (search-box center)
    #
    # The search volume is (anchor +/- search_delta_km):
    #   search_anchor = "origin" (default): the input origin.
    #   search_anchor = "stations": centroid of the stations with arrivals,
    #       at default_depth. A pick-derived anchor that does not depend on
    #       the origin scolv feeds back, so repeated relocations are
    #       idempotent; needs a box wide enough to reach the event.
    # --ignore-initial-location selects "stations" for this run.
    # (The former "grid_center" anchor was removed: a fixed point is too
    # far from most events on a regional model for the capped search box
    # to reach them.)
    search_anchor = str(cfg.get("search_anchor", "origin")).strip().lower()
    if search_anchor == "grid_center":
        log("NOTE: search_anchor \"grid_center\" has been removed; using "
            "\"stations\" (the station centroid) instead. Set search_anchor "
            "to \"stations\" or \"origin\" in the config.")
        search_anchor = "stations"
    elif search_anchor not in ("origin", "stations"):
        raise RelocationError(
            f"config: search_anchor must be \"origin\" or \"stations\", got "
            f"{cfg.get('search_anchor')!r}")
    if "grid_center" in cfg:
        log("NOTE: the grid_center config key has been removed and is "
            "ignored; you can delete it from the config.")
    if args.ignore_initial_location:
        search_anchor = "stations"

    if search_anchor == "stations" and stations:
        # spherical centroid of the stations with arrivals: a stable,
        # pick-derived anchor independent of the origin scolv feeds back
        _used = {(a["network"], a["station"]) for a in arrivals_in}
        _lats = [stations[k][0] for k in _used if k in stations]
        _lons = [stations[k][1] for k in _used if k in stations]
        if _lats:
            # spherical centroid (mean of unit position vectors): a plain
            # mean of longitudes sends stations at 179E and 179W to lon 0,
            # on the far side of the planet from the network
            _ph = np.radians(_lons)
            _th = np.radians(90.0 - np.asarray(_lats))
            _x = float(np.mean(np.sin(_th) * np.cos(_ph)))
            _y = float(np.mean(np.sin(_th) * np.sin(_ph)))
            _z = float(np.mean(np.cos(_th)))
            lat0 = 90.0 - math.degrees(math.atan2(math.hypot(_x, _y), _z))
            lon0 = math.degrees(math.atan2(_y, _x))
        else:
            log("NOTE: search_anchor \"stations\" needs station coordinates "
                "for the arrivals (stations_csv) and none were found; "
                "anchoring on the input epicentre instead.")
            lat0 = origin_info["latitude"]
            lon0 = origin_info["longitude"]
        dep0 = float(cfg.get("default_depth", 10.0))
    else:
        if search_anchor == "stations":      # no stations_csv at all
            log("NOTE: search_anchor \"stations\" needs stations_csv; none "
                "loaded, so anchoring on the input origin instead.")
        lat0 = origin_info["latitude"]
        lon0 = origin_info["longitude"]
        dep0 = origin_info["depth"]
        if dep0 is None:
            dep0 = float(cfg.get("default_depth", 10.0))

    depth_min = cfg.get("depth_min")
    depth_max = cfg.get("depth_max")
    # --fixed-depth pins the anchor depth; the tolerance window about it is
    # the whole constraint (size_fixed_box turns the pin into an explicit
    # asymmetric bound). Config clamps are dropped so the two can't fight —
    # resolve_depth_bounds would let the pin win anyway.
    if args.fixed_depth is not None:
        dep0 = float(args.fixed_depth)
        depth_min = depth_max = None

    # Diagnostic: fingerprint the INPUT to this call. If two successive
    # relocations differ, comparing these lines shows whether scolv fed
    # back a changed initial origin (different dep0/lat0/lon0) or the same
    # input (=> any output difference is the locator, not the input).
    import hashlib as _hashlib
    _afp = _hashlib.md5(
        repr(sorted(
            (a["network"], a["station"], a["phase"], a["pick_id"])
            for a in arrivals_in
        )).encode()
    ).hexdigest()[:8]
    # Event boundary marker. A run interleaves many events in one stderr
    # stream, so this gives a single unambiguous line to search or split on
    # without altering the content lines below it.
    log(_banner(origin_info.get("event_id") or origin_info.get("publicID")))
    log(f"input: initial=({lat0:.4f}, {lon0:.4f}, dep {dep0:.1f} km), "
        f"{len(arrivals_in)} arrivals, fingerprint={_afp}, "
        f"anchor={search_anchor}, "
        f"fixed_depth={args.fixed_depth}")

    # ---------------- collect arrivals
    # Reference epoch: earliest pick time, so locator times stay small.
    pick_list = []
    # Associated arrivals whose phase is not a first arrival (pP, PmP, Lg,
    # ...). The grids hold first-arrival times, so these cannot be located
    # with; they are written back with weight 0 under their own label so
    # the association made in scolv is not lost.
    later_phase = []       # (network, station, label, pick_id)
    pick_labels = {}       # pick_id -> original label, for default errors
    for a in arrivals_in:
        if a["weight"] == 0.0:
            continue  # operator disabled this arrival in scolv
        label = str(a["phase"] or "").strip()
        phase = map_phase(label, phase_map)
        if phase is None:
            later_phase.append((a["network"], a["station"], label or "?",
                                a["pick_id"]))
            continue
        key = (a["network"], a["station"], phase)
        pick_labels[a["pick_id"]] = label
        pick_list.append(
            (key, a["pick_id"], a["time"], a["time_uncertainty"], a["weight"])
        )
    if later_phase:
        log(f"{len(later_phase)} arrival(s) are not first-arrival phases and "
            f"are kept with weight 0, not located with: "
            + ", ".join(f"{n}.{s} {lab}" for n, s, lab, _ in later_phase)
            + " (add a label to phase_map in the config to use it)")

    # Distance census (always on): make it explicit how far each arrival's
    # station is, and exclude from the solve any beyond max_dist_km (none
    # when it is null). Done here, by distance, rather than left to the
    # grids' coverage, so the rule holds whatever limit the stored grids
    # were built with.
    _far_keys = set()
    if stations:
        _too_far = []
        _dists = []
        for key, pid, t, terr, wt in pick_list:
            sta = stations.get(key[:2])
            if sta is None:
                _dists.append((key, None))
                continue
            d_deg, _ = gc_distance_azimuth(lat0, lon0, sta[0], sta[1])
            d_km = d_deg * 111.195
            _dists.append((key, d_km))
            if max_dist_km is not None and d_km > max_dist_km:
                _too_far.append((key, d_km))
        _no_coord = [k for k, d in _dists if d is None]
        if _too_far:
            # Excluded from the solve explicitly rather than left to the
            # grids' NaN mask: a grid built for a LARGER max_dist is kept
            # when the setting is lowered (it is a superset, and rebuilding
            # it would cost an FMM solve for nothing), so the mask alone no
            # longer enforces this limit. They stay in the output with
            # weight 0, like rejected outliers. Same rule as pykonal_obspy.
            log(f"{len(_too_far)} arrival(s) beyond max_dist_km="
                f"{describe_max_dist(max_dist_km)} "
                f"excluded from the solve (kept in the output, weight 0): "
                + ", ".join(f"{'.'.join(k)}@{d:.0f}km" for k, d in _too_far))
            _far_keys = {k for k, _ in _too_far}
        if _no_coord:
            log(f"{len(_no_coord)} arrival(s) with no station coordinates in "
                f"stations_csv (distance unknown): "
                + ", ".join('.'.join(k) for k in _no_coord))

    if len(pick_list) < 4:
        _n_off = len(arrivals_in) - len(pick_list)
        raise RelocationError(
            f"only {len(pick_list)} usable pick(s) of {len(arrivals_in)} "
            f"arrival(s)"
            + (f" ({_n_off} disabled with weight 0 or not first-arrival "
               f"phases)" if _n_off else "")
            + "; at least 4 are needed to locate. Add or re-enable picks.")


    # --max-dist filter (degrees), if station coordinates are available
    if args.max_dist is not None and stations:
        kept = []
        for key, pid, t, terr, wt in pick_list:
            sta = stations.get(key[:2])
            if sta is None:
                kept.append((key, pid, t, terr, wt))
                continue
            d, _ = gc_distance_azimuth(lat0, lon0, sta[0], sta[1])
            if d <= args.max_dist:
                kept.append((key, pid, t, terr, wt))
        log(f"--max-dist={args.max_dist}: kept {len(kept)}/{len(pick_list)} picks")
        pick_list = kept
        if len(pick_list) < 4:
            raise RelocationError(f"only {len(pick_list)} picks within "
                                f"--max-dist={args.max_dist}; need >= 4")

    epoch = min(t for _, _, t, _, _ in pick_list)
    arrivals = {}
    pick_errors = {}
    pick_ids = {}
    arrival_weights = {}   # scolv weight per key, echoed back on output
    _n_dup = 0
    for key, pid, t, terr, wt in pick_list:
        if key in arrivals:
            _n_dup += 1
            log(f"duplicate arrival for {'.'.join(key)}; keeping first "
                f"(same network/station/phase key)")
            continue
        arrivals[key] = (t - epoch).length()  # TimeSpan -> seconds
        # Pick uncertainty comes from the pick's own time uncertainty, or
        # default_pick_error. Input arrival weights are deliberately NOT
        # folded in: they arrive as a mixture of unrelated quantities
        # (SeisComP 1/2^index quality classes such as 0.125, and the
        # previous locator's internal weights which can exceed 1), so
        # there is no consistent scale to map onto an uncertainty. Doing so
        # silently deleted low-weight picks and sharpened others past their
        # stated error, flattening the EDT surface and destabilising the
        # solution. Weight 0 still excludes a pick (handled above), and the
        # input weight is echoed back on output unchanged.
        # default error: an exact label entry ("Sg") if the config has
        # one, else the grid phase's ("S") -- not the label's first letter,
        # which would send lowercase "s" to the smallest (P) default
        _lab = pick_labels.get(pid)
        pick_errors[key] = (float(terr) if terr else
                            default_pick_error[_lab]
                            if _lab in default_pick_error
                            else _default_err_for_phase(key[2]))
        pick_ids[key] = pid
        arrival_weights[key] = float(wt) if wt else 1.0

    # Reconcile the counts so it is never a mystery where arrivals went
    # between input and location: input -> (weight-0 excluded) -> distance
    # census -> (duplicate-key collapsed) -> unique keys located. Any
    # further reduction below is the off-grid filter and outlier rejection,
    # which log separately.
    _n_zero = sum(1 for a in arrivals_in if a["weight"] == 0.0)
    _n_later = len(later_phase)
    for key in _far_keys:              # beyond max_dist_km: output only
        arrivals.pop(key, None)
        pick_errors.pop(key, None)
    _n_far = len(_far_keys)
    if len(arrivals) < 4:
        raise RelocationError(
            f"only {len(arrivals)} usable arrival(s) within max_dist_km="
            f"{describe_max_dist(max_dist_km)}; at least 4 are needed to "
            f"locate.")
    log(f"arrival census: {len(arrivals_in)} input, "
        f"{_n_zero} weight-0 excluded, {_n_later} not first arrivals, "
        f"{_n_far} beyond max_dist_km, "
        f"{_n_dup} duplicate-key collapsed, "
        f"{len(arrivals)} unique station-phase keys to locate")

    grid_stations = {}
    for (net, sta), (slat, slon, selev) in stations.items():
        grid_stations[(net, sta)] = transform.geo_to_grid(slat, slon, -selev / 1000.0)

    # ------------- build / extend the traveltime inventory on demand -------
    # Precomputed grids are canonical; before locating, any station/phase
    # in the pick set that lacks a stored grid is solved once (from the
    # configured velocity_models) and persisted, so this and every later
    # run find it. This also BUILDS the inventory from nothing on the very
    # first event if it does not yet exist. Serialized across processes via
    # <dir>.lock. Requires "velocity_models" in the config AND station
    # coordinates (stations_csv); without either, grids cannot be computed
    # and their arrivals are simply excluded.
    # Storage: one HDF5 file per station-phase in traveltime_dir, plus a
    # small inventory.h5 master of external links that pykonal reads
    # transparently. Grids are fingerprinted with the velocity model that
    # produced them and rebuilt automatically when that model changes.
    shard_dir = cfg.get("traveltime_dir")
    if not shard_dir:
        raise RelocationError(
            "traveltime_dir is required: the directory holding the sharded "
            "per-station-phase grids (single-file inventories are no longer "
            "supported)")
    inventory_path = os.path.join(shard_dir, "inventory.h5")

    velocity_models = cfg.get("velocity_models", {})
    # build_missing (default true): compute absent grids from
    # velocity_models before locating. false = locate with the grids that
    # exist; arrivals without one get weight 0.
    build_missing = bool(cfg.get("build_missing", True))
    if velocity_models and not build_missing:
        log("build_missing is false: missing traveltime grids will not be "
            "computed")
    if velocity_models and build_missing:
        requests = {
            (net, sta, phase): grid_stations.get((net, sta))
            for (net, sta, phase) in arrivals
        }
        missing_coords = sorted(
            f"{net}.{sta}" for (net, sta, ph), c in requests.items()
            if c is None
        )
        if missing_coords:
            log(f"no station coordinates for {len(missing_coords)} "
                f"station(s): {', '.join(missing_coords)}; their grids "
                f"cannot be built (add them to stations_csv)")

        # Drop stations that fall outside the velocity grid (excluded) and
        # clamp those above the datum onto the grid top. Logs both outcomes.
        requests = fit_requests_to_grid(requests, velocity_models,
            vertical_axis=0, log=log)

        if not os.path.isdir(shard_dir):
            log(f"traveltime directory {shard_dir} does not exist yet; "
                f"creating it")
        _t_build = _time.time()
        report = ensure_traveltimes_sharded(
            shard_dir, requests, velocity_models,
            max_dist=max_dist_km,
            nproc=cfg.get("ensure_nproc"), log=log)


        if report["computed"]:
            _elapsed = (
                f" in {_time.time() - _t_build:.1f} s"
                if _t_build is not None else ""
            )
            log(f"computed and stored {len(report['computed'])} new "
                f"traveltime grid(s){_elapsed}: "
                f"{', '.join('.'.join(k) for k in report['computed'])}")
        if report["skipped"]:
            log(f"could not compute grids for "
                f"{', '.join('.'.join(k) for k in report['skipped'])} "
                f"(missing coords or per-phase velocity model); their "
                f"arrivals will be excluded")
    elif not os.path.exists(inventory_path):
        _why = ("build_missing is false" if velocity_models
                else "no 'velocity_models' are configured")
        raise RelocationError(
            f"sharded traveltime inventory {inventory_path} does not exist "
            f"and {_why}, so it cannot be built. Configure velocity_models "
            f"(and stations_csv) with build_missing true, or run "
            f"pykonal_prepopulate.py.")

    # -----------------------------------------------------------------------

    # Shared lock in both modes: single-file writes take an exclusive lock
    # elsewhere, and the sharded master is swapped atomically (os.replace),
    # so readers never see a partial index.
    with inventory_lock(inventory_path, shared=True), \
         EQLocator(inventory_path,
                   coord_sys=transform.coord_sys) as locator:
        # The relocation sequence (search box, locate, residuals, outlier
        # rejection, quality, posterior with automatic widening) is shared
        # with pykonal_obspy in pykonal.relocate.
        if "edt_reg" in cfg:
            log("NOTE: edt_reg has been removed; it is ignored. EDT_OT_WT "
                "(edt_ot_wt) does the same job without the outlier "
                "sensitivity. You can delete the key from your config.")
        _settings = dict(cfg)
        _settings.update(
            method=method, alpha=float(alpha),
            default_pick_error=_default_err_scalar,
            search_delta_km=("auto" if _auto_delta
                             else [float(d) for d in delta_km]),
            search_delta_t=delta_t, posterior_nsamples=nsamples,
            posterior_rounds=posterior_rounds, posterior_seed=posterior_seed,
            depth_min=depth_min, depth_max=depth_max,
            edt_ot_wt=bool(cfg.get("edt_ot_wt", True)),
            fixed_depth_tolerance_km=float(
                cfg.get("fixed_depth_tolerance_km", 2.0)),
            solve_n_closest=int(cfg.get("solve_n_closest", 0)),
            outlier_cutoff_s=float(cfg.get("outlier_cutoff_s", 3.0)),
            outlier_mad_factor=float(cfg.get("outlier_mad_factor", 4.0)),
            outlier_min_arrivals=int(cfg.get("outlier_min_arrivals", 6)),
            log_search_detail=bool(cfg.get("log_search_detail", False)))
        # the anchor depth and origin time share the arrivals' reference:
        # seconds after the earliest pick
        _reloc = pk_relocate(
            locator, transform, arrivals, pick_errors, stations,
            (lat0, lon0, dep0, 0.0), _settings, fixed_depth=args.fixed_depth,
            posterior=(method == "edt" and nsamples > 0), log=log,
            depth_advice=" Consider --fixed-depth.",
            missing_grid_advice=(
                " Configure velocity_models (with build_missing true) to "
                "compute them." if not velocity_models else
                " Set build_missing to true in the config to compute them."
                if not build_missing else
                " They could not be built (no station coordinates, no "
                "velocity model for the phase, or off the model grid)."))

        soln = _reloc.hypocenter
        residuals = _reloc.residuals
        # Arrivals the hypocentre was actually solved with. With
        # solve_n_closest > 0, residuals ALSO holds the farther arrivals,
        # computed for reporting only; they must not be marked as used.
        solve_keys = set(_reloc.solve_keys)
        quality = _reloc.quality
        posterior = _reloc.posterior
        _compute_seconds = _reloc.compute_seconds
        # spherical grid axis order (r, theta, phi) = (depth, lat, lon)
        _ilon, _ilat, _iz = 2, 1, 0

        # traveltimes for per-arrival predicted times were already read
        lat, lon, dep = transform.grid_to_geo(soln[:3])
        # NOTE: the solved depth is reported as-is, NOT snapped to
        # args.fixed_depth. With fixed_depth_tolerance_km > 0 the solver is
        # allowed off the pin, and the residuals, arrival distances and
        # origin time all correspond to the solved depth; overwriting it
        # would make the reported depth inconsistent with the rest of the
        # origin. The pin is enforced by the search box, not by the output.
        t0 = seiscomp.core.Time(epoch) + seiscomp.core.TimeSpan(float(soln[3]))

        # Uncertainties in km from the EDT posterior covariance, at the
        # configured confidence level, computed here so they can be reported
        # inline with the solution and reused below when populating the
        # origin. None if no posterior.
        # The library now reports covariance_km: the posterior covariance
        # already expressed in km^2 in the local orthonormal frame
        # (up, south, east for spherical grids). Prefer
        # it. The fallback path reproduces the old hand-rolled scaling so
        # this script still runs against a pykonal that has not been
        # rebuilt -- but note that only the fallback's per-axis sigmas were
        # ever correct; its ellipsoid was an eigendecomposition of a matrix
        # mixing km with radians.
        lat_km = lon_km = dz = None
        cov_km = None
        if posterior is not None:
            cov_km = posterior.get("covariance_km")
            if cov_km is None:
                _cov = posterior["covariance"]
                _r0 = pk_constants.EARTH_RADIUS - dep
                _s = np.array([1.0, _r0,
                               _r0 * math.sin(math.radians(90.0 - lat))])
                cov_km = np.asarray(_cov) * np.outer(_s, _s)
            cov_km = np.asarray(cov_km, dtype=float)
            # _ilon / _ilat / _iz are set above, before posterior sampling.
            # marginal (1-dof) uncertainties at the requested level
            _k1 = chi_scale(confidence_level, 1)
            lat_km = _k1 * math.sqrt(max(cov_km[_ilat, _ilat], 0.0))
            lon_km = _k1 * math.sqrt(max(cov_km[_ilon, _ilon], 0.0))
            dz     = _k1 * math.sqrt(max(cov_km[_iz, _iz], 0.0))

    if lat_km is not None:
        _ess_txt = ""
        try:
            _ess_txt = f" ess={float(posterior['ess']):.0f}"
        except Exception:
            pass
        _t_txt = ""
        try:
            _ts = float(posterior["time_sigma"])
            if np.isfinite(_ts):
                _t_txt = f", t0 \u00b1{_ts:.2f} s"
        except Exception:
            pass
        _unc = (f"lat {lat:.4f}\u00b0 \u00b1{lat_km:.2f} km, "
                f"lon {lon:.4f}\u00b0 \u00b1{lon_km:.2f} km, "
                f"depth {dep:.1f} \u00b1{dz:.2f} km{_t_txt} "
                f"({confidence_level:.0f}% marginal){_ess_txt}")
    else:
        _unc = (f"lat {lat:.4f}\u00b0, lon {lon:.4f}\u00b0, "
                f"depth {dep:.1f} km (uncertainties not estimated)")
    # Per-arrival detail (opt-in): distance, predicted traveltime and the
    # implied apparent velocity. Because EDT locates on DIFFERENTIAL times,
    # a systematic traveltime error largely cancels in the location and is
    # absorbed by the origin time -- so the hypocenter can match another
    # locator while absolute residuals stay large. Comparing the apparent
    # velocity here against your velocity model is the direct check: if it
    # is consistently higher than the model, the grids were built from a
    # model that is too fast (the usual cause of large positive residuals).
    if cfg.get("log_arrival_detail") and residuals:
        log("per-arrival detail (dist_km, tt_pred_s, apparent_km_s, resid_s):")
        # compute distance for each arrival first so the table can be sorted
        # closest-station-first; arrivals with no coordinates sort last.
        _detail = []
        for key in residuals:
            _sta = stations.get(key[:2])
            if _sta is None:
                continue
            _ddeg, _ = gc_distance_azimuth(lat, lon, _sta[0], _sta[1])
            _detail.append((_ddeg * 111.195, key))
        for _d, key in sorted(_detail, key=lambda t: t[0]):
            # predicted traveltime = observed_offset - t0_offset - residual
            _ttp = arrivals[key] - float(soln[3]) - residuals[key]
            _app = (_d / _ttp) if _ttp > 0.01 else float("nan")
            log(f"    {'.'.join(key):18s} {_d:7.1f} {_ttp:7.2f} "
                f"{_app:7.2f} {residuals[key]:+7.2f}")

    # Residual breakdown: EDT is outlier-robust, so it can produce a good
    # hypocenter while a few bad picks still carry large residuals. Those
    # picks inflate the plain (unweighted) RMS reported above even though
    # the location barely used them. Report the worst offenders and the
    # RMS with them removed, so a high RMS can be attributed to specific
    # picks rather than to a poor location.
    if residuals:
        _r = sorted(residuals.items(), key=lambda kv: -abs(kv[1]))
        _vals = np.array([v for _, v in _r], dtype=float)
        _n_big = int(np.sum(np.abs(_vals) > 1.0))
        _worst = ", ".join(
            f"{'.'.join(k)}={v:+.2f}s" for k, v in _r[:3]
        )
        if len(_vals) > 3:
            _rms_trim = float(np.sqrt(np.mean(np.sort(_vals ** 2)[:-2])))
            log(f"residuals: worst {_worst}; "
                f"{_n_big}/{len(_vals)} exceed 1.0 s; "
                f"rms excluding 2 worst = {_rms_trim:.3f}s")
        else:
            log(f"residuals: worst {_worst}")

    log(f"solution: {_unc} "
        f"rms={quality['rms']:.3f}s nobs={quality['nobs']} "
        f"(computed in {_compute_seconds:.2f} s)")

    # ---------------- build the output Origin
    origin = seiscomp.datamodel.Origin.Create()
    origin.setLatitude(seiscomp.datamodel.RealQuantity(_round_out(lat, "coord")))
    origin.setLongitude(seiscomp.datamodel.RealQuantity(_round_out(lon, "coord")))
    origin.setDepth(seiscomp.datamodel.RealQuantity(_round_out(dep, "depth")))
    # Origin time, with its uncertainty when the posterior supplied one.
    # Previously the time was emitted bare, implying an exactness it never
    # had -- on geometry without a near-source station the origin time
    # trades off almost completely against depth (see the correlation
    # logged above) and can be uncertain by seconds.
    _tq = seiscomp.datamodel.TimeQuantity(t0)
    if posterior is not None:
        _ts = posterior.get("time_sigma")
        if _ts is not None and np.isfinite(_ts):
            try:
                _tq.setUncertainty(_round_out(_ts, "time"))
            except Exception:
                pass
    origin.setTime(_tq)
    origin.setMethodID(f"pykonal-{method.upper()}")
    origin.setEarthModelID(cfg.get("earth_model_id", "pykonal"))
    _set_enum(origin.setEvaluationMode,
              "AUTOMATIC", "EvaluationMode_AUTOMATIC")
    if args.fixed_depth is not None:
        _set_enum(origin.setDepthType,
                  "OPERATOR_ASSIGNED", "OriginDepthType_OPERATOR_ASSIGNED")

    ci = seiscomp.datamodel.CreationInfo()
    ci.setCreationTime(seiscomp.core.Time.GMT())
    ci.setAuthor("PyKonal-EDT")
    origin.setCreationInfo(ci)

    # arrivals with residuals
    #
    # SeisComP validates Arrival.distance on serialization: an Arrival with
    # no distance set throws "Arrival.distance is not set" and aborts the
    # whole output. That happens for any station missing from stations_csv
    # (no coordinates -> no distance). Such arrivals were already excluded
    # from the LOCATION, so we simply do not emit an Arrival for them here;
    # emitting a distance-less one would crash the plugin. Logged so the
    # dropped station is visible.
    used = 0
    _no_dist = []
    for key, pid in pick_ids.items():
        sta = stations.get(key[:2])
        if sta is None:
            _no_dist.append(key)
            continue  # cannot set distance -> would crash serialization
        arr = seiscomp.datamodel.Arrival()
        arr.setPickID(pid)
        arr.setPhase(seiscomp.datamodel.Phase(key[2]))
        if key in residuals:
            # residual reported for every arrival that has one, including
            # the solve_n_closest report-only set
            arr.setTimeResidual(_round_out(residuals[key], "residual"))
        if key in residuals and key in solve_keys:
            arr.setWeight(arrival_weights.get(key, 1.0))
            arr.setTimeUsed(True)
            used += 1
        else:
            arr.setWeight(0.0)
            arr.setTimeUsed(False)
        dist, az = gc_distance_azimuth(lat, lon, sta[0], sta[1])
        arr.setDistance(_round_out(dist, "distance"))  # degrees of arc
        arr.setAzimuth(_round_out(az, "azimuth"))
        origin.add(arr)
    for net, sta_code, label, pid in later_phase:
        sta = stations.get((net, sta_code))
        if sta is None:
            _no_dist.append((net, sta_code, label))
            continue
        arr = seiscomp.datamodel.Arrival()
        arr.setPickID(pid)
        arr.setPhase(seiscomp.datamodel.Phase(label))
        arr.setWeight(0.0)
        arr.setTimeUsed(False)
        dist, az = gc_distance_azimuth(lat, lon, sta[0], sta[1])
        arr.setDistance(_round_out(dist, "distance"))
        arr.setAzimuth(_round_out(az, "azimuth"))
        origin.add(arr)
    if _no_dist:
        log(f"{len(_no_dist)} arrival(s) omitted from output (no station "
            f"coordinates, cannot set Arrival.distance): "
            + ", ".join('.'.join(k) for k in _no_dist))

    # quality
    q = seiscomp.datamodel.OriginQuality()
    q.setUsedPhaseCount(used)
    q.setAssociatedPhaseCount(len(pick_ids) + len(later_phase))
    if np.isfinite(quality["rms"]):
        q.setStandardError(_round_out(quality["rms"], "residual"))
    if np.isfinite(quality["azimuthal_gap"]):
        q.setAzimuthalGap(_round_out(quality["azimuthal_gap"], "gap"))
    if np.isfinite(quality["min_station_dist"]):
        q.setMinimumDistance(_round_out(math.degrees(
            float(quality["min_station_dist"]) / MEAN_RADIUS), "distance"))
    origin.setQuality(q)

    # uncertainty ellipsoid from the EDT posterior
    if posterior is not None:
        # Lengths and orientation from one decomposition of cov_km
        # (sizing.ellipsoid_orientation, QuakeML Tait-Bryan convention).
        semi_km, _plunge, _az, _rot = ellipsoid_orientation(cov_km)

        _k3 = chi_scale(confidence_level, 3)
        # TODO(units): assumes SeisComP stores semi-axis lengths in metres.
        # SeisComP's NonLinLoc plugin documents a direct mapping from NLL's
        # km lengths. Check one event through scxmldump + sccnv against the
        # ObsPy driver before trusting this factor.
        semi = semi_km * _k3 * 1000.0  # km -> m

        ce = seiscomp.datamodel.ConfidenceEllipsoid()
        ce.setSemiMajorAxisLength(_round_out(semi[0], "ellipsoid"))
        ce.setSemiIntermediateAxisLength(_round_out(semi[1], "ellipsoid"))
        ce.setSemiMinorAxisLength(_round_out(semi[2], "ellipsoid"))
        ce.setMajorAxisAzimuth(_round_out(_az, "azimuth"))
        ce.setMajorAxisPlunge(_round_out(_plunge, "plunge"))
        # majorAxisRotation is written: QuakeML requires it, and leaving it
        # unset does not avoid the plunge-sign question -- the rotation
        # uses the same convention (sizing.PLUNGE_SIGN, positive down) that
        # the plunge above already commits to. The three angles rebuild the
        # posterior covariance exactly. Confirm the convention once against
        # NonLinLoc's plunge for the same event.
        ce.setMajorAxisRotation(_round_out(_rot, "azimuth"))

        ou = seiscomp.datamodel.OriginUncertainty()
        ou.setConfidenceEllipsoid(ce)
        # State the level explicitly. Without it a consumer cannot tell a
        # 1-sigma ellipsoid from a 68% or 95% one, and they differ by
        # factors of 1.88 and 2.80 respectively.
        try:
            ou.setConfidenceLevel(float(confidence_level))
        except Exception:
            pass
        # The enum constant for "confidence ellipsoid" varies by seiscomp
        # version, and a module attribute of the right NAME may belong to a
        # different enum (setter then rejects it as "out of range"). So we
        # try each candidate against the setter and keep the first accepted;
        # if none work, leave preferredDescription unset (the ellipsoid is
        # still attached and used by scolv regardless).
        # Try to set preferredDescription = confidence ellipsoid; on some
        # seiscomp builds no accepted enum value exists, which is fine —
        # the ellipsoid is attached regardless, so we don't warn about it.
        _set_enum(
            ou.setPreferredDescription,
            "CONFIDENCE_ELLIPSOID", "ConfidenceEllipsoid",
            "OU_CONFIDENCE_ELLIPSOID",
            "OriginUncertaintyDescription_CONFIDENCE_ELLIPSOID",
        )
        # Per-coordinate uncertainties in KM (SeisComP's convention for
        # lat/lon/depth uncertainty is kilometers, not degrees). These
        # populate origin.latitude/longitude/depth .uncertainty(), which is
        # what scolv's location view displays. They are MARGINAL (1-dof)
        # intervals at `confidence_level`, so at the 68% default they are
        # very nearly plain 1-sigma; the ellipsoid above is the 3-dof
        # region at the same level and is correspondingly wider. Computed
        # inline with the solution above from covariance_km.
        # horizontalUncertainty (km): the semi-major axis of the HORIZONTAL
        # marginal ellipse at the requested level -- not max(lat, lon). When
        # the horizontal errors are correlated (routine for a one-sided
        # network, where the ellipse is a long diagonal ridge) the true
        # semi-major exceeds both marginal sigmas, sometimes by a lot.
        _ch = cov_km[np.ix_([_ilon, _ilat], [_ilon, _ilat])]
        _hmax = float(np.sqrt(max(np.linalg.eigvalsh(_ch)[-1], 0.0)))
        ou.setHorizontalUncertainty(
            _round_out(_hmax * chi_scale(confidence_level, 2), "horiz_unc"))
        origin.setUncertainty(ou)

        # latitude / longitude uncertainties (km) — mirror what we do for
        # depth so scolv shows all three
        _lat = origin.latitude()
        _lat.setUncertainty(_round_out(lat_km, "coord_unc"))
        origin.setLatitude(_lat)

        _lon = origin.longitude()
        _lon.setUncertainty(_round_out(lon_km, "coord_unc"))
        origin.setLongitude(_lon)

        # depth uncertainty (km)
        d = origin.depth()
        d.setUncertainty(_round_out(dz, "depth_unc"))
        origin.setDepth(d)

    write_output(origin, wrap_eventparameters=args.wrap_eventparameters)


if __name__ == "__main__":
    try:
        main()
    except RelocationError as err:
        # expected: say why in one line, no traceback
        log(f"cannot locate: {err}")
        sys.exit(1)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
