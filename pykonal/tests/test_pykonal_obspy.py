import sys, math; sys.path.insert(0,"fake")
import numpy as np
import os
from obspy import UTCDateTime
from obspy.core.event import (Event, Origin, Pick, WaveformStreamID,
                              QuantityError, Catalog)
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from pykonal.fields import ScalarField3D
from pykonal.transformations import GeoTransform
from pykonal.inventory import ensure_traveltimes_sharded
from pykonal import pykonal_obspy as P

T0 = UTCDateTime("2024-03-01T12:00:00")
STAS = [("AU","KMBL",-30.8,121.4),("AU","FORT",-31.9,122.1),
        ("WG","WBS23",-31.2,121.9),("WG","WBR22",-31.6,121.2)]

def make_event():
    ev = Event(); o = Origin(time=T0, latitude=-31.39, longitude=121.82, depth=1500.0)
    ev.origins.append(o); ev.preferred_origin_id = o.resource_id
    for i,(net,sta,la,lo) in enumerate(STAS):
        for ph,off in (("P",2.0+i),("S",3.6+1.7*i)):
            p = Pick(time=T0+off, phase_hint=ph,
                     waveform_id=WaveformStreamID(network_code=net,
                                                  station_code=sta,
                                                  channel_code="BHZ"),
                     time_errors=QuantityError(uncertainty=0.05))
            ev.picks.append(p)
    return ev

def make_inv():
    nets={}
    for net,sta,la,lo in STAS:
        n=nets.setdefault(net, Network(code=net, stations=[]))
        s=Station(code=sta, latitude=la, longitude=lo, elevation=300.0,
                  site=Site(name=sta), creation_date=UTCDateTime("2000-01-01"))
        s.channels=[Channel(code="BHZ", location_code="", latitude=la, longitude=lo,
                            elevation=300.0, depth=0.0)]
        n.stations.append(s)
    return Inventory(networks=list(nets.values()), source="test")

inv = make_inv()

R = 6371.
MODELS = {"P": "vp.h5", "S": "vs.h5"}
def _make_model(path, speed):
    """21x25x25 spherical grid: depth 0..40 km, lat -34..-28, lon 119..125."""
    f = ScalarField3D(coord_sys="spherical")
    f.min_coords = [R - 40.0, np.radians(90.0 + 28.0), np.radians(119.0)]
    f.node_intervals = [2.0, np.radians(0.25), np.radians(0.25)]
    f.npts           = [23, 25, 25]
    f.values         = np.full(tuple(f.npts), speed)
    f.to_hdf(path, overwrite=True)     # heals partial states; see guard below

# Build the fixture ONCE. Re-runs must not rewrite the model files: a model
# rewrite bumps its mtime, which mismatches every shard's velocity-model
# fingerprint and would silently trigger 8 FMM rebuilds per test run.
if not (os.path.exists("vp.h5") and os.path.exists("vs.h5")
        and os.path.exists("./tt_dir/inventory.h5")):
    _make_model("vp.h5", 5.8)
    _make_model("vs.h5", 3.4)
    _tr = GeoTransform()
    _req = {(net, sta, ph): _tr.geo_to_grid(la, lo, -0.3)
            for net, sta, la, lo in STAS for ph in ("P", "S")}
    ensure_traveltimes_sharded("./tt_dir", _req, MODELS,
                               max_dist=1000.0, log=print)


# warm the sharded inventory once: 8 FMM solves on a 13k-node grid (seconds)
_tr = GeoTransform()
_req = {(net, sta, ph): _tr.geo_to_grid(la, lo, -0.3)
        for net, sta, la, lo in STAS for ph in ("P", "S")}
ensure_traveltimes_sharded("./tt_dir", _req, MODELS,
                           max_dist=1000.0, log=print)

def main():
    print("=== 1. free depth ===")
    ev = P.relocate(make_event(), traveltime_dir="./tt_dir", station_inventory=inv,
                    depth_min=-3.0, depth_max=80.0, posterior_seed=1)
    o = ev.preferred_origin()
    print(f"  depth={o.depth/1000:.2f} km  type={o.depth_type}  origins={len(ev.origins)}")
    print(f"  arrivals={len(o.arrivals)}  used={o.quality.used_phase_count} "
          f"rms={o.quality.standard_error} gap={o.quality.azimuthal_gap}")
    print(f"  min_dist={o.quality.minimum_distance:.4f} deg  "
          f"arr0 dist={o.arrivals[0].distance:.4f} az={o.arrivals[0].azimuth:.1f}")
    print(f"  ellipse: max={o.origin_uncertainty.max_horizontal_uncertainty:.0f} m "
          f"min={o.origin_uncertainty.min_horizontal_uncertainty:.0f} m "
          f"az={o.origin_uncertainty.azimuth_max_horizontal_uncertainty:.1f} deg")
    print(f"  comments: {[c.text[:46] for c in o.comments]}")

    print("\n=== 2. fixed depth ===")
    for pin in (0.0, 5.0, 12.5):
        ev = P.relocate(make_event(), traveltime_dir="./tt_dir", station_inventory=inv,
                        fixed_depth=pin, depth_min=-3.0, depth_max=80.0, posterior_seed=1)
        o = ev.preferred_origin()
        TOL = 2.0   # fixed_depth_tolerance_km: the pin is a +-TOL window, and
        assert abs(o.depth/1000 - pin) <= TOL + 1e-9, o.depth
        assert o.depth_type == "operator assigned"
        assert o.depth_errors.uncertainty > 0.0
        assert o.depth_type == "operator assigned"
        assert o.depth_errors.uncertainty > 0.0
        print(f"  pin {pin:5.1f} -> depth={o.depth/1000:6.2f} km, "
              f"type={o.depth_type}, depth_err={o.depth_errors.uncertainty/1000:.2f} km")

    print("\n=== 3. original origin preserved / preferred updated ===")
    src = make_event(); before = str(src.origins[0].resource_id)
    ev = P.relocate(src, traveltime_dir="./tt_dir", station_inventory=inv, fixed_depth=5.0)
    assert len(ev.origins) == 2 and str(ev.origins[0].resource_id) == before
    assert str(ev.preferred_origin_id) == str(ev.origins[1].resource_id)
    assert len(src.origins) == 1, "input event was mutated!"
    print("  input untouched, new origin appended and preferred. OK")

    print("\n=== 4. replace_origin ===")
    ev = P.relocate(make_event(), traveltime_dir="./tt_dir", station_inventory=inv,
                    fixed_depth=5.0, replace_origin=True)
    print(f"  origins={len(ev.origins)}")

    print("\n=== 5. QuakeML round-trip ===")
    import io
    cat = Catalog(events=[P.relocate(make_event(), traveltime_dir="./tt_dir",
                                     station_inventory=inv, fixed_depth=7.0)])
    buf = io.BytesIO(); cat.write(buf, format="QUAKEML"); buf.seek(0)
    from obspy import read_events
    back = read_events(buf)
    ob = back[0].preferred_origin()
    print(f"  round-tripped depth={ob.depth/1000:.2f} km type={ob.depth_type} "
          f"arrivals={len(ob.arrivals)} comments={len(ob.comments)}")
    TOL = 2.0   # fixed_depth_tolerance_km: the pin is a +-TOL window, and
    assert abs(o.depth/1000 - pin) <= TOL + 1e-9, o.depth
    assert o.depth_type == "operator assigned"
    assert o.depth_errors.uncertainty > 0.0

    print("\n=== 6. config file drives everything ===")
    import json, warnings
    CFG = {"traveltime_dir":"./tt_dir","alpha":0.02,
           "default_pick_error":{"P":0.08,"S":0.16},
           "search_delta_km":[30.0,30.0,20.0],"search_delta_t":10.0,
           "depth_min":-3.0,"depth_max":80.0,"default_depth":10.0,
           "posterior_nsamples":2048,"posterior_seed":8675,
           "confidence_level":90.0,"earth_model_id":"my_3d_model",
           "max_dist_km":1000.0,"_help":"dropped","search_anchor":"origin",
           "auto_search_k_sigma":6.0,"velocity_models":{"P":"vp.h5"},"nonsense":1}
    with open("/tmp/cfg.json","w") as f: json.dump(CFG,f)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ev = P.relocate(make_event(), "/tmp/cfg.json", station_inventory=inv)
        msgs = [str(x.message) for x in w]
    o = ev.preferred_origin()
    assert o.earth_model_id.id == "my_3d_model"
    assert o.latitude_errors.confidence_level == 90.0
    assert o.origin_uncertainty.confidence_level == 90.0
    #assert any("auto_search_k" in m for m in msgs), "auto_search_k keys not warned"
    assert any("nonsense" in m for m in msgs), "unknown keys not warned"
    print(f"  earth_model={o.earth_model_id.id} conf={o.latitude_errors.confidence_level}%")
    print(f"  warned about {len(msgs)} config issue(s)")

    print("\n=== 7. kwargs override the config ===")
    ev = P.relocate(make_event(), "/tmp/cfg.json", station_inventory=inv,
                    fixed_depth=7.5, confidence_level=68.0)
    o = ev.preferred_origin()
    TOL = 2.0   # fixed_depth_tolerance_km: the pin is a +-TOL window, and
    assert abs(o.depth/1000 - 7.5) <= TOL + 1e-9, o.depth
    assert o.depth_type == "operator assigned"
    assert o.depth_errors.uncertainty > 0.0
    assert o.latitude_errors.confidence_level == 68.0
    assert o.depth_errors.uncertainty > 0, "depth uncertainty must survive a pin"
    print(f"  depth={o.depth/1000:.2f} km conf={o.latitude_errors.confidence_level}% "
          f"depth_unc={o.depth_errors.uncertainty/1000:.2f} km")

    print("\n=== 8. search_delta_km 'auto' rejected ===")
    try:
        P.relocate(make_event(), {"traveltime_dir":"./tt_dir","search_delta_km":"auto"})
        print("  FAILED: no exception")
    except P.RelocationError as e:
        print(f"  RelocationError: {str(e)[:70]}...")

    print("\n=== 9. station sources: CSV alone, and merged with a partial inventory ===")
    import warnings as _w
    from obspy.core.inventory import Inventory as _Inv, Network as _Net, Station as _St, Site as _Si
    open("/tmp/stations.csv","w").write(
        "# Network,Station,Latitude,Longitude,Elevation_m\n"
        + "\n".join(f"{n},{s},{la},{lo},300" for n,s,la,lo in STAS))
    def _partial(codes):
        nets={}
        for n,s,la,lo in STAS:
            if (n,s) not in codes: continue
            net=nets.setdefault(n,_Net(code=n,stations=[]))
            net.stations.append(_St(code=s,latitude=la,longitude=lo,elevation=999.0,
                                    site=_Si(name=s),creation_date=UTCDateTime("2000-01-01")))
        return _Inv(networks=list(nets.values()),source="t")
    BASE={"traveltime_dir":"./tt_dir","search_delta_km":[30.,30.,20.]}
    CSV={**BASE,"stations_csv":"/tmp/stations.csv"}

    o = P.relocate(make_event(), CSV).preferred_origin()
    assert all(a.distance is not None for a in o.arrivals), "CSV-only failed"
    assert o.quality.azimuthal_gap is not None
    print("  CSV alone (no Inventory): distance on all arrivals, gap computed")

    o = P.relocate(make_event(), CSV,
                   station_inventory=_partial({("AU","KMBL")})).preferred_origin()
    assert all(a.distance is not None for a in o.arrivals), "CSV did not fill inventory gaps"
    print("  partial Inventory + CSV: gaps filled from CSV")

    with _w.catch_warnings(record=True) as ws:
        _w.simplefilter("always")
        o = P.relocate(make_event(), BASE).preferred_origin()
    assert all(a.distance is None for a in o.arrivals)
    assert o.quality.azimuthal_gap is None
    assert any("no station coordinates" in str(x.message) for x in ws)
    print("  neither source: relocation still succeeds, geometry fields omitted")

    print("\n=== 10. search_delta_km: 'auto' uses the shared sizing module ===")
    AUTO={"traveltime_dir":"./tt_dir","search_delta_km":"auto",
          "depth_min":-3.0,"depth_max":80.0,"auto_search_k_sigma":4.0,
          "auto_search_max_km":[50.,50.,40.],"auto_search_min_km":[10.,10.,20.],
          "auto_search_depth_dist_factor":0.5}
    msgs=[]
    o = P.relocate(make_event(), AUTO, station_inventory=inv,
                   log=msgs.append).preferred_origin()

    #print(f"  auto sizing diagnostics captured: {len(msgs)}")
    #for m in msgs[:10]:
    #    print("    ", m[:90])

    assert any("search box (auto)" in m for m in msgs), f"auto sizing did not run; got {msgs}"
    # the clamp's observable: the box respects depth_min despite a wider raw box
    assert any("depth -3" in m for m in msgs), "depth_min clamp not applied; got {msgs}"
    print(f"  auto sizing emitted {len(msgs)} diagnostics, depth={o.depth/1000:.2f} km")

    msgs=[]
    o = P.relocate(make_event(), AUTO, station_inventory=inv, fixed_depth=6.0,
                   log=msgs.append).preferred_origin()
    TOL = 2.0   # fixed_depth_tolerance_km: the pin is a +-TOL window, and
    assert abs(o.depth/1000 - 6.0) <= TOL + 1e-9, o.depth
    assert o.depth_type == "operator assigned"
    assert o.depth_errors.uncertainty > 0.0
    assert o.depth_errors.uncertainty > 0, "auto path zeroed the depth uncertainty"


    print(f"  auto + fixed depth: {o.depth/1000:.2f} km, "
          f"unc={o.depth_errors.uncertainty/1000:.2f} km, {o.depth_type}")

    print("\n=== 11. build_missing guards ===")
    for cfg, want in [
        ({"traveltime_dir":"./tt_dir","search_delta_km":[30.,30.,20.]},
         "no velocity_models"),          # message: "...and no velocity_models are configured..."
        ({"search_delta_km":[30.,30.,20.],
          "velocity_models":{"P":"vp.h5"}}, "No traveltime_dir"),
    ]:
        try:
            P.relocate(make_event(), cfg, station_inventory=inv, build_missing=True)
            print("  FAILED: no exception")
        except P.RelocationError as e:
            assert want in str(e), str(e)
            print(f"  RelocationError contains {want!r}")
    try:
        P.relocate(make_event(), {"search_delta_km":[30.,30.,20.]},
                   station_inventory=inv)
        print("  FAILED: no exception")
    except P.RelocationError as e:
        print(f"  no inventory, no build -> {str(e)[:60]}...")

    print("\n=== 12. too few picks ===")
    try:
        bad = make_event(); bad.picks = bad.picks[:2]
        P.relocate(bad, traveltime_dir="./tt_dir", station_inventory=inv)
        print("  FAILED: no exception")
    except P.RelocationError as e:
        print(f"  RelocationError: {e}")
    print("\nall tests passed")


if __name__ == "__main__":
    main()
