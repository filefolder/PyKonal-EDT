"""
The ObsPy driver (pykonal.pykonal_obspy.relocate): Event in, Event out.

Skipped when ObsPy is not installed. The traveltime directory is a real
sharded inventory (P and S, 4 stations) built once in a temp directory.
"""
import io
import json
import warnings

import numpy as np
import pytest

obspy = pytest.importorskip("obspy")

from obspy import UTCDateTime, read_events  # noqa: E402
from obspy.core.event import (Catalog, Event, Origin, Pick,  # noqa: E402
                              QuantityError, WaveformStreamID)
from obspy.core.inventory import (Channel, Inventory, Network, Site,  # noqa: E402
                                  Station)

from pykonal import pykonal_obspy as P  # noqa: E402
from pykonal.fields import ScalarField3D  # noqa: E402
from pykonal.inventory import (ensure_traveltimes_sharded,  # noqa: E402
                               fit_requests_to_grid)
from pykonal.transformations import GeoTransform  # noqa: E402

T0 = UTCDateTime("2024-03-01T12:00:00")
STAS = [("AU", "KMBL", -30.8, 121.4), ("AU", "FORT", -31.9, 122.1),
        ("WG", "WBS23", -31.2, 121.9), ("WG", "WBR22", -31.6, 121.2)]
TRUE = (-31.39, 121.82, 6.0)
VP, VS = 5.8, 3.4
TOL = 2.0          # default fixed_depth_tolerance_km


def _model(path, speed):
    """23 x 25 x 25 spherical grid: depth 0..44 km, lat -34..-28, lon 119..125.

    0.25 deg x 2 km on purpose: before the near field was sized to the
    far-field cells, some of these stations got no traveltimes at all."""
    R = 6371.0
    f = ScalarField3D(coord_sys="spherical")
    f.min_coords = [R - 44.0, np.radians(90.0 + 28.0), np.radians(119.0)]
    f.node_intervals = [2.0, np.radians(0.25), np.radians(0.25)]
    f.npts = [23, 25, 25]
    f.values = np.full(tuple(f.npts), speed)
    f.to_hdf(str(path), overwrite=True)
    return str(path)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("obspy")
    models = {"P": _model(tmp / "vp.h5", VP), "S": _model(tmp / "vs.h5", VS)}
    tt_dir = str(tmp / "tt_dir")
    T = GeoTransform()
    # stations 300 m above the model top, clamped onto it as the drivers do
    req = {(n, s, ph): T.geo_to_grid(la, lo, -0.3)
           for n, s, la, lo in STAS for ph in ("P", "S")}
    req = fit_requests_to_grid(req, models, vertical_axis=0)
    ensure_traveltimes_sharded(tt_dir, req, models, max_dist=1000.0)
    return dict(tmp=tmp, tt_dir=tt_dir, models=models)


def _picks_for(site):
    """Picks from the stored grids for an event at TRUE (exact traveltimes)."""
    from pykonal.inventory import TraveltimeInventory
    T = GeoTransform()
    hypo = T.geo_to_grid(*TRUE)
    out = []
    with TraveltimeInventory(f"{site['tt_dir']}/inventory.h5") as inv:
        for n, s, *_ in STAS:
            for ph in ("P", "S"):
                out.append((n, s, ph, float(inv.read(f"{n}/{s}/{ph}").value(hypo))))
    return out


@pytest.fixture(scope="module")
def picks(site):
    out = _picks_for(site)
    # guard: a fixture built on empty grids is what let the old version of
    # this file "pass" without locating anything
    assert all(np.isfinite(tt) and tt > 0 for *_, tt in out), out
    return out


def make_event(picks, n=None):
    ev = Event()
    o = Origin(time=T0, latitude=-31.3, longitude=121.7, depth=5000.0)
    ev.origins.append(o)
    ev.preferred_origin_id = o.resource_id
    for net, sta, ph, tt in picks[:n]:
        ev.picks.append(Pick(
            time=T0 + 1.0 + tt, phase_hint=ph,
            waveform_id=WaveformStreamID(network_code=net, station_code=sta,
                                         channel_code="BHZ"),
            time_errors=QuantityError(uncertainty=0.05)))
    return ev


def make_inventory(codes=None, elevation=300.0):
    nets = {}
    for net, sta, la, lo in STAS:
        if codes is not None and (net, sta) not in codes:
            continue
        n = nets.setdefault(net, Network(code=net, stations=[]))
        s = Station(code=sta, latitude=la, longitude=lo, elevation=elevation,
                    site=Site(name=sta), creation_date=UTCDateTime("2000-01-01"))
        s.channels = [Channel(code="BHZ", location_code="", latitude=la, longitude=lo,
                              elevation=elevation, depth=0.0)]
        n.stations.append(s)
    return Inventory(networks=list(nets.values()), source="test")


@pytest.fixture(scope="module")
def inv():
    return make_inventory()


def _run(site, picks, inv, **kw):
    kw.setdefault("posterior_seed", 1)
    return P.relocate(make_event(picks), traveltime_dir=site["tt_dir"],
                      station_inventory=inv, **kw)


# ----------------------------------------------------------------- location
def test_free_depth_origin(site, picks, inv):
    ev = _run(site, picks, inv, depth_min=-3.0, depth_max=40.0)
    o = ev.preferred_origin()
    assert len(ev.origins) == 2
    assert o.latitude == pytest.approx(TRUE[0], abs=0.02)
    assert o.longitude == pytest.approx(TRUE[1], abs=0.02)
    assert abs(o.time - (T0 + 1.0)) < 0.5
    assert len(o.arrivals) == 8 and o.quality.used_phase_count == 8
    assert o.quality.azimuthal_gap is not None and o.quality.minimum_distance > 0
    assert all(a.distance is not None and a.azimuth is not None for a in o.arrivals)
    ou = o.origin_uncertainty
    assert 0 < ou.min_horizontal_uncertainty <= ou.max_horizontal_uncertainty
    assert o.depth_errors.uncertainty > 0


@pytest.mark.parametrize("pin", [0.0, 5.0, 12.5])
def test_fixed_depth(site, picks, inv, pin):
    o = _run(site, picks, inv, fixed_depth=pin, depth_min=-3.0,
             depth_max=40.0).preferred_origin()
    assert abs(o.depth / 1000 - pin) <= TOL + 1e-9
    assert o.depth_type == "operator assigned"
    assert o.depth_errors.uncertainty > 0


def test_input_event_is_not_mutated(site, picks, inv):
    src = make_event(picks)
    before = str(src.origins[0].resource_id)
    ev = P.relocate(src, traveltime_dir=site["tt_dir"], station_inventory=inv,
                    fixed_depth=5.0)
    assert len(src.origins) == 1
    assert len(ev.origins) == 2 and str(ev.origins[0].resource_id) == before
    assert str(ev.preferred_origin_id) == str(ev.origins[1].resource_id)


def test_replace_origin(site, picks, inv):
    ev = _run(site, picks, inv, fixed_depth=5.0, replace_origin=True)
    assert len(ev.origins) == 1


def test_quakeml_round_trip(site, picks, inv):
    ev = _run(site, picks, inv, fixed_depth=7.0)
    buf = io.BytesIO()
    Catalog(events=[ev]).write(buf, format="QUAKEML")
    buf.seek(0)
    ob = read_events(buf)[0].preferred_origin()
    o = ev.preferred_origin()
    assert ob.depth == pytest.approx(o.depth)
    assert ob.depth_type == "operator assigned"
    assert len(ob.arrivals) == len(o.arrivals)
    assert ob.depth_errors.uncertainty == pytest.approx(o.depth_errors.uncertainty)


# ------------------------------------------------------------------- config
@pytest.fixture
def cfg_file(site, tmp_path):
    cfg = {"traveltime_dir": site["tt_dir"], "alpha": 0.02,
           "default_pick_error": {"P": 0.08, "S": 0.16},
           "search_delta_km": [30.0, 30.0, 20.0], "search_delta_t": 10.0,
           "depth_min": -3.0, "depth_max": 40.0, "default_depth": 10.0,
           "posterior_nsamples": 2048, "posterior_seed": 8675,
           "confidence_level": 90.0, "earth_model_id": "my_3d_model",
           "max_dist_km": 1000.0, "_help": "dropped", "search_anchor": "origin",
           "velocity_models": {"P": site["models"]["P"]}, "nonsense": 1}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def test_config_file_drives_the_run(picks, inv, cfg_file):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o = P.relocate(make_event(picks), cfg_file, station_inventory=inv).preferred_origin()
    msgs = [str(x.message) for x in w]
    assert o.earth_model_id.id == "my_3d_model"
    assert o.latitude_errors.confidence_level == 90.0
    assert o.origin_uncertainty.confidence_level == 90.0
    assert any("nonsense" in m for m in msgs), msgs
    assert not any("_help" in m for m in msgs)


@pytest.mark.filterwarnings("ignore:unrecognised config keys")
def test_keyword_arguments_override_the_config(picks, inv, cfg_file):
    o = P.relocate(make_event(picks), cfg_file, station_inventory=inv,
                   fixed_depth=7.5, confidence_level=68.0).preferred_origin()
    assert abs(o.depth / 1000 - 7.5) <= TOL + 1e-9
    assert o.latitude_errors.confidence_level == 68.0
    assert o.depth_errors.uncertainty > 0


def test_auto_search_box(site, picks, inv):
    cfg = {"traveltime_dir": site["tt_dir"], "search_delta_km": "auto",
           "depth_min": -3.0, "depth_max": 40.0, "auto_search_k_sigma": 4.0,
           "auto_search_max_km": [50., 50., 40.], "auto_search_min_km": [10., 10., 20.]}
    msgs = []
    o = P.relocate(make_event(picks), cfg, station_inventory=inv,
                   log=msgs.append).preferred_origin()
    assert any("search box (auto)" in m for m in msgs), msgs
    assert any("depth -3" in m for m in msgs), msgs
    assert o.latitude == pytest.approx(TRUE[0], abs=0.02)

    o = P.relocate(make_event(picks), cfg, station_inventory=inv,
                   fixed_depth=6.0).preferred_origin()
    assert abs(o.depth / 1000 - 6.0) <= TOL + 1e-9
    assert o.depth_errors.uncertainty > 0


def test_removed_auto_keys_warn(site, picks, inv):
    cfg = {"traveltime_dir": site["tt_dir"], "search_delta_km": "auto",
           "auto_search_cond_max": 1e6}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        P.relocate(make_event(picks), cfg, station_inventory=inv)
    assert any("auto_search_cond_max" in str(x.message) for x in w)


# ----------------------------------------------------------------- stations
@pytest.fixture
def stations_csv(tmp_path):
    p = tmp_path / "stations.csv"
    p.write_text("# Network,Station,Latitude,Longitude,Elevation_m\n"
                 + "\n".join(f"{n},{s},{la},{lo},300" for n, s, la, lo in STAS))
    return str(p)


def test_stations_from_csv_alone(site, picks, stations_csv):
    o = P.relocate(make_event(picks), {"traveltime_dir": site["tt_dir"],
                                       "stations_csv": stations_csv}).preferred_origin()
    assert all(a.distance is not None for a in o.arrivals)
    assert o.quality.azimuthal_gap is not None


def test_csv_fills_gaps_in_a_partial_inventory(site, picks, stations_csv):
    o = P.relocate(make_event(picks), {"traveltime_dir": site["tt_dir"],
                                       "stations_csv": stations_csv},
                   station_inventory=make_inventory({("AU", "KMBL")})).preferred_origin()
    assert all(a.distance is not None for a in o.arrivals)


def test_no_station_coordinates_still_relocates(site, picks):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o = P.relocate(make_event(picks),
                       {"traveltime_dir": site["tt_dir"]}).preferred_origin()
    assert all(a.distance is None for a in o.arrivals)
    assert o.quality.azimuthal_gap is None
    assert any("no station coordinates" in str(x.message) for x in w)


# ------------------------------------------------------------------- errors
@pytest.mark.parametrize("cfg_kind, want", [("no_models", "no velocity_models"),
                                            ("no_dir", "No traveltime_dir")])
def test_build_missing_guards(tmp_path, site, picks, inv, cfg_kind, want):
    # an empty traveltime_dir, so there is something to build
    cfg = ({"traveltime_dir": str(tmp_path / "empty")} if cfg_kind == "no_models"
           else {"velocity_models": {"P": site["models"]["P"]}})
    with pytest.raises(P.RelocationError, match=want):
        P.relocate(make_event(picks), cfg, station_inventory=inv, build_missing=True)


def test_no_traveltimes_is_an_error(picks, inv):
    with pytest.raises(P.RelocationError):
        P.relocate(make_event(picks), {}, station_inventory=inv)


def test_too_few_picks_is_an_error(site, picks, inv):
    with pytest.raises(P.RelocationError):
        P.relocate(make_event(picks, n=2), traveltime_dir=site["tt_dir"],
                   station_inventory=inv)


def test_build_missing_computes_new_grids(tmp_path, site, picks, inv):
    tt_dir = str(tmp_path / "fresh")
    o = P.relocate(make_event(picks), {"traveltime_dir": tt_dir,
                                       "velocity_models": site["models"]},
                   station_inventory=inv, build_missing=True,
                   posterior_seed=1).preferred_origin()
    assert o.quality.used_phase_count == 8
